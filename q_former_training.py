import numpy as np
import transformers
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch
import glob
from PIL import Image
from options import args_parser
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import re
import os
from tqdm import tqdm
from peft import LoraConfig, get_peft_model
import difflib
from transformers.utils import logging
logging.set_verbosity_error()
from torchmetrics import Accuracy, Precision, Recall, F1Score
import pytz
import wandb
from datetime import datetime

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class MultimodalClassifier(torch.nn.Module):
    def __init__(self):
        super(MultimodalClassifier, self).__init__()
        self.classifier = torch.nn.Linear(768, 4)

    def forward(self, x):
        # Get features from BLIP-2
        return self.classifier(x)

def save_checkpoint(model, classifier, optimizer, epoch, step, acc, device):
    filename = "BLIP2_Q_FORMER_epoch_"+str(epoch)+"_acc_"+str(acc)+".pth"
    filename_classifier = "Classifier_epoch_"+str(epoch)+"_acc_"+str(acc)+".pth"

    print("Saving weights to {}".format(filename))
    model.to("cpu")
    torch.save(model.state_dict(), filename)
    print(f"Checkpoint saved to {filename}")
    model.to(device)

    print("Saving weights to {}".format(filename_classifier))
    classifier.to("cpu")
    torch.save(classifier.state_dict(), filename_classifier)
    print(f"Checkpoint saved to {filename_classifier}")
    classifier.to(device)

def gen_inputs(encoding, answer):
    #out_start = processor(text=f"{answer}", padding=False, truncation=False, return_tensors="pt")

    labels = answer
    # labels[labels == t5_tokenizer.pad_token_id] = -100 

    return {**encoding, "labels": (torch.ones(1)*cls_dict[labels]).long()}
    
    # return {**encoding, "decoder_input_ids": decoder_input_ids.squeeze(0),
    #     "labels": labels.squeeze(0)}

def remove_numbers(input_string):
    return re.sub(r'\d+', '', input_string)


class ImageCaptioningDataset(Dataset):
    def __init__(self, paths, processor):
        self.dataset = paths
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item_path = self.dataset[idx]
        item_text = remove_numbers(item_path.split("/")[-1])[:-4].replace("_", " ").rstrip().lstrip()
        raw_image = Image.open(item_path).convert("RGB")
        # prompt = f"Question: how many cats are there? Answer:"
        prompt = f'''Question: Knowing that the Blue label is recyclable,
        Green label is for composting,
        Black label is non-recyclable,
        and Yellow label needs to be recycled at a specific location,
        which label does this {item_text} belong to? Answer:'''
        # prompt = f"Knowing that the Blue label is recyclable, Green is for composting, Black is non-recyclable, and TTR needs to be recycled at a specific location, which category of recycling does this {item_text} belong to?"
        encoding = processor(images=raw_image, text=prompt, return_tensors="pt", max_length=100, padding="max_length", padding_side = "left")
        # encoding['pixel_values'] = encoding['pixel_values'].squeeze(0)
        # encoding['input_ids'] = f"Knowing that the Blue label is recyclable, Green is for composting, Black is non-recyclable, and TTR needs to be recycled at a specific location, which category of recycling does this {item_text} belong to?"#encoding['input_ids'].squeeze(0)
        # encoding['input_ids'] = f"Question: how many cats are there? Answer:"
        # encoding['attention_mask'] = encoding['attention_mask'].squeeze(0)
        
        label = item_path.split('/')[-2]
        
        if label=="TTR":
            label="Yellow"
        
        outs = gen_inputs(encoding, label)
        
        return outs

def collate_fn(batch):
    # pad the input_ids and attention_mask
    processed_batch = {}
    for key in batch[0].keys():
        if key != "input_ids" and key!='labels':
            processed_batch[key] = torch.stack([example[key] for example in batch])
        elif key=='labels':
            # text_labels = processor.tokenizer(
            #     [example["labels"] for example in batch], padding=True, return_tensors="pt")            
            processed_batch["labels"] = torch.stack([example[key] for example in batch])#text_labels["input_ids"]
        else:
            # text_inputs = [example["input_ids"] for example in batch]
            text_inputs = torch.stack([example["input_ids"] for example in batch])

            # print(type(batch))  # Output: <class 'int'>
            # for element in text_inputs:
            #     print("element: ", element)

            # processed_batch["input_ids"] = text_inputs["input_ids"]
            processed_batch["input_ids"] = text_inputs
            # processed_batch["attention_mask"] = text_inputs["attention_mask"]
    
    processed_batch['input_ids'] = processed_batch['input_ids'].squeeze(1)
    processed_batch['attention_mask'] = processed_batch['attention_mask'].squeeze(1)
    processed_batch['pixel_values'] = processed_batch['pixel_values'].squeeze(1)
      
    return processed_batch

def get_n_params(model):
    pp=0
    for p in [p for p in model.parameters() if p.requires_grad]:#list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def find_closest_string(target, string_list):

    match = re.search(r'Answer:\s*(.*)', target)
    if match:
        target = match.group(1)

    if len(target) <= 3:
        if "Yel" in target:
            return string_list[3]
        elif "Blu" in target:
            return string_list[0]
        elif "Gre" in target:
            return string_list[1]
        elif "Bla" in target:
            return string_list[2]
        
    closest_match = difflib.get_close_matches(target, string_list, n=1)
    return closest_match[0] if closest_match else string_list[0]

def calculate_acc(model, classifier, loader, device, processor, dataset):

    outs=[]
    preds=[]
    total_batches = int(len(loader))+1
    for idx, b in enumerate(loader):
        with torch.no_grad():
            batch_dev={}
            for key in b.keys():
                batch_dev[key]=b[key].to(device)
        
            y_true = batch_dev['labels']
            del batch_dev['labels']
            outputs = model(**batch_dev)
            x = outputs['qformer_outputs'].last_hidden_state[:,0,:]
            out = classifier(x)
            outs.append(out.detach().cpu().argmax(1))
            preds.append(y_true.detach().cpu())

            print(f"Batch in eval:{idx}/{total_batches}", end="\r")

    ytrue_ = torch.tensor(np.array(preds)).view(-1)
    outs_ = torch.tensor(np.array(outs)).view(-1)
    precision = Precision(task="multiclass",num_classes=4, average='macro')
    recall = Recall(task="multiclass",num_classes=4, average='macro')
    f1 = F1Score(task="multiclass",num_classes=4, average='macro')
    accuracy = Accuracy(task="multiclass",num_classes=4)
    print("Len of ytrue_: ", len(ytrue_))
    print("Len of outs_: ", len(outs_))
    print("Results for dataset: ", dataset)
    print(f"Acc:{accuracy(outs_.to('cpu'), ytrue_.to('cpu')).item()}, recall:{recall(outs_.to('cpu'),ytrue_.to('cpu')).item()}, precision:{precision(outs_.to('cpu'), ytrue_.to('cpu')).item()},\
            f1: {f1(outs_.to('cpu'), ytrue_.to('cpu')).item()}")

    return accuracy(outs_.to('cpu'), ytrue_.to('cpu')).item()

args = args_parser()
timezone = pytz.timezone('America/Edmonton')
now = datetime.now(timezone)
date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
print("Starting W&B...")
run = wandb.init(
    config=args,
    project="BLIP 2",
    name="Date QFORMER: " + str(date_time)
)

device="cuda:0"
processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b", cache_dir="/scratch")

model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", device_map="cpu", cache_dir="/scratch")
wandb.watch(model)


ims = glob.glob(args.dataset_folder_name+"/*/*")
im=ims[0]

_batch_size = args.batch_size
_workers = 32

dataset = ImageCaptioningDataset(ims, processor)
loader_train = DataLoader(dataset, batch_size=_batch_size, num_workers=_workers, collate_fn=collate_fn, shuffle=True)

# Define training parameters
num_epochs = args.epochs

classifier = MultimodalClassifier()
model.to(device)
classifier.to(device)
print("went to device")

# Let's define the LoraConfig
config = LoraConfig(
r=32,
lora_alpha=8,
lora_dropout=0.05,
bias="none",
    target_modules=["q_proj", "k_proj"])

model = get_peft_model(model, config)
model.print_trainable_parameters()

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters: {total_params}")
print(f"Trainable parameters: {trainable_params}")

accumulation_steps = 8  # Number of steps to accumulate gradients

optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad] + [p for p in classifier.parameters()],
            lr=5e-4, eps=1e-05)

classifier.train()
model.train()
print("model in train mode")

#EVAL
ims_val = glob.glob(args.dataset_folder_name_val+"/*/*")
dataset_val = ImageCaptioningDataset(ims_val, processor)
test_loader_val = DataLoader(dataset_val, batch_size=_batch_size, num_workers=_workers, collate_fn=collate_fn, shuffle=False)

# Example usage
strings = ['Blue', 'Green', 'Black', 'Yellow']
cls_dict = {
"Blue": 0,
"Green": 1,
"Black": 2,
"Yellow":   3}

print_loss=0
criterion = torch.nn.CrossEntropyLoss()

max_val_accuracy = 0.0
best_epoch = 0
for epoch in range(num_epochs):
    
    model.train()
    classifier.train()
    with tqdm(total=len(loader_train), desc=f"Epoch {epoch + 1}/{num_epochs}") as pbar:
        total_loss = 0
        for step, batch in enumerate(loader_train):
            batch_device={}
            for key in batch.keys():
                batch_device[key]=batch[key].to(device)        
            optimizer.zero_grad()

            # print("keys: ", batch_device.keys())
            # Forward pass
            # print("input_ids on training:", processor.tokenizer.batch_decode(batch_device['input_ids'], skip_special_tokens=True))
            # print("labels on training:", processor.tokenizer.batch_decode(batch_device['labels'], skip_special_tokens=True))
            # print("attention_mask on training:", batch_device['attention_mask'])
            # print("shape pixel_values:", batch_device['pixel_values'].shape)
            y_true = batch_device['labels']
            del batch_device['labels']
            
            outputs = model(**batch_device)
            x = outputs['qformer_outputs'].last_hidden_state[:,0,:]
            out = classifier(x)

            loss = criterion(out, y_true.squeeze(1))
            loss = loss / accumulation_steps  # Normalize the loss
            loss.backward()
            total_loss+=loss.item()
            print_loss+=loss.item()

            # Perform optimizer step every accumulation_steps
            if (step + 1) % accumulation_steps == 0:
                optimizer.step()  # Update parameters
                f_loss = print_loss/accumulation_steps
                pbar.set_postfix(loss=f_loss)                
                print_loss=0

            pbar.update(1)
            
        avg_loss = total_loss/step
        print("loss", avg_loss)
        
        if (step + 1) % accumulation_steps != 0:
            optimizer.step()  # Update parameters for any remaining accumulated gradients
            pbar.set_postfix(loss=loss.item())  # Log the loss after the final update
   
    model.eval()
    classifier.eval()
    train_acc = calculate_acc(model, classifier, loader_train, device, processor, "Train")
    val_accuracy = calculate_acc(model, classifier, test_loader_val, device, processor, "Validation")
    
    if val_accuracy > max_val_accuracy:
        print("Best model obtained based on Val Acc. Saving it!")
        save_checkpoint(model, classifier, optimizer, epoch, step, val_accuracy, device)
        max_val_accuracy = val_accuracy
        best_epoch = epoch
    else:
        print("Not saving model on epoch {}, best Val Acc so far on epoch {}: {:.3f}".format(epoch, best_epoch,
                                                                                                max_val_accuracy))
    
    wandb.log({'train_loss_avg': avg_loss,
    'train_accuracy_history': train_acc,
    'val_accuracy_history': val_accuracy,
    'max_val_acc_percentage': max_val_accuracy*100})