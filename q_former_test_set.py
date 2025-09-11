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
from sklearn.metrics import classification_report
from torchmetrics.classification import ConfusionMatrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

os.environ["TOKENIZERS_PARALLELISM"] = "false"

strings = ['Blue', 'Green', 'Black', 'Yellow']
cls_dict = {
"Blue":  0,
"Green": 1,
"Black": 2,
"Yellow":3}

classes = ["Black", "Blue", "Green", "Yellow"]

class MultimodalClassifier(torch.nn.Module):
    def __init__(self):
        super(MultimodalClassifier, self).__init__()
        self.classifier = torch.nn.Linear(768, 4)

    def forward(self, x):
        # Get features from BLIP-2
        return self.classifier(x)

def generate_report_and_image(test_report_dict,test_accuracy, conf_matrix):
        
    dataframe = pd.DataFrame.from_dict(test_report_dict)
    dataframe.to_csv(os.path.join("./QFORMER_report_test_set_acc_{:.2f}.csv".format(test_accuracy)), index=True)

    df_cm = pd.DataFrame(conf_matrix, index=classes,
                         columns=classes)

    plt.rcParams.update({'font.size': 16})
    plt.figure(figsize=(10, 5))
    sn.heatmap(df_cm, annot=True, cmap='viridis', fmt='g')
    plt.savefig(
        os.path.join('./conf_matrix_QFORMER_model_test_set_acc_{:.2f}.png'.format(test_accuracy)))

    print("Test accuracy: {:.2f} %".format(test_accuracy))
    print("Test Report:")
    print(test_report)

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

    len_test_set = 2000
    outs=[]
    preds=[]
    gt_labels=[]
    correct = 0
    total_batches = int(len(loader))+1
    confmat = ConfusionMatrix(task="multiclass", num_classes=4)
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
            
            outs_2 = torch.cat(outs).view(-1)
            preds_2 = torch.cat(preds).view(-1)

            correct = torch.sum(torch.eq(outs_2, preds_2)).item()
            
            print("Running test accuracy: {:.3f} %".format(100*(correct/len_test_set)))

            print(f"Batch in test:{idx}/{total_batches}", end="\r")

    ytrue_ = torch.tensor(np.array(outs)).view(-1)
    outs_ = torch.tensor(np.array(preds)).view(-1)
    precision = Precision(task="multiclass",num_classes=4, average='macro')
    recall = Recall(task="multiclass",num_classes=4, average='macro')
    f1 = F1Score(task="multiclass",num_classes=4, average='macro')
    accuracy = Accuracy(task="multiclass",num_classes=4)
    print("Len of ytrue_: ", len(ytrue_))
    print("Len of outs_: ", len(outs_))
    print("Results for dataset: ", dataset)
    print(f"Acc:{accuracy(outs_.to('cpu'), ytrue_.to('cpu')).item()}, recall:{recall(outs_.to('cpu'),ytrue_.to('cpu')).item()}, precision:{precision(outs_.to('cpu'), ytrue_.to('cpu')).item()},\
            f1: {f1(outs_.to('cpu'), ytrue_.to('cpu')).item()}")

    all_preds = outs_
    all_labels = ytrue_
    test_acc = 100 * (correct/len_test_set)
    conf_matrix = confmat(torch.tensor(all_labels), torch.tensor(all_preds))
    print(conf_matrix)

    report = classification_report(torch.tensor(all_labels).cpu(),
                                   torch.tensor(all_preds).cpu(),
                                   target_names=classes)

    report_dict = classification_report(torch.tensor(all_labels).cpu(),
                                        torch.tensor(all_preds).cpu(),
                                        target_names=classes, output_dict=True)

    return test_acc, report, report_dict, conf_matrix

args = args_parser()

device="cuda:0"

processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b", cache_dir="/scratch")

model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", device_map="cpu", cache_dir="/scratch")

ims = glob.glob(args.dataset_folder_name+"/*/*")
im=ims[0]

_batch_size = 16
_workers = 32

dataset = ImageCaptioningDataset(ims, processor)
loader_test_set = DataLoader(dataset, batch_size=_batch_size, num_workers=_workers, collate_fn=collate_fn, shuffle=True)

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

model_name = args.model_path
loaded_model = torch.load(model_name)

try:
    model.load_state_dict(loaded_model['model_state_dict'])
except:
    print("error loading, trying again directly")
    model.load_state_dict(loaded_model)

classifier.load_state_dict(torch.load("../classifier_epoch_9_acc_0.8855.pth"))

classifier.eval()
model.eval()
print("model in eval mode")

test_accuracy, test_report, test_report_dict, conf_matrix = calculate_acc(model, classifier, loader_test_set, device, processor, "Test")

generate_report_and_image(test_report_dict,test_accuracy, conf_matrix)
