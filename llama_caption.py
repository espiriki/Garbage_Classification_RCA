import os
import csv
from PIL import Image
from tqdm import tqdm
from options import args_parser

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import AutoProcessor, AutoModelForVision2Seq, MllamaForConditionalGeneration

from huggingface_hub import login
import os

os.environ["HF_HUB_DISABLE_XET"] = "1"

login(token="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")

args = args_parser()

# CONFIG
IMAGE_DIR = args.dataset_folder_name
OUTPUT_CSV = "image_descriptions_train.csv"
PROMPT = "can you please describe this image in just one sentence?"
MODEL_NAME = "meta-llama/Llama-3.2-11B-Vision-Instruct"  # Replace with the correct one
BATCH_SIZE = 1

# DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Load processor and model
# processor = AutoProcessor.from_pretrained(MODEL_NAME, cache_dir="/scratch")
# model = AutoModelForVision2Seq.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, cache_dir="/scratch").to(device)
# model.eval()

model = MllamaForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

processor = AutoProcessor.from_pretrained(MODEL_NAME)


# Dataset class
class ImageFolderDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_paths = []
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, root_dir)
                    self.image_paths.append((full_path, rel_path))  # (absolute_path, relative_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        full_path, rel_path = self.image_paths[idx]
        try:
            image = Image.open(full_path).convert("RGB")
        except Exception as e:
            print(f"Error loading {rel_path}: {e}")
            image = Image.new("RGB", (224, 224), color="white")
        return image, rel_path


# Collate function for batching
def collate_fn(batch):
    images, filenames = zip(*batch)
    inputs = processor(
        text=[PROMPT] * len(images),
        images=list(images),
        return_tensors="pt",
        padding=True
    )
    return inputs, filenames


# # Load dataset
# dataset = ImageFolderDataset(IMAGE_DIR)
# dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

image_files = []
for root, _, files in os.walk(IMAGE_DIR):
    for file in files:
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, IMAGE_DIR)  # for CSV
            image_files.append((full_path, rel_path))  # (absolute_path, relative_path)

# Inference loop
results = []

messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": PROMPT}
    ]}
]

def ensure_quoted(s: str) -> str:
    s = s.strip()  # optional: remove surrounding whitespace/newlines
    if not s.startswith('"'):
        s = '"' + s
    if not s.endswith('"'):
        s = s + '"'
    return s

# Open CSV once in append mode and write header if needed
need_header = not os.path.exists(OUTPUT_CSV) or os.path.getsize(OUTPUT_CSV) == 0
with open(OUTPUT_CSV, mode="a", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    if need_header:
        writer.writerow(["filename", "description"])
        f.flush()

    # Inference loop with streaming writes
    for full_path, rel_path in tqdm(image_files, desc="Generating"):
        try:
            image = Image.open(full_path).convert("RGB")

            # Build chat-style inputs
            input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(
                image,
                input_text,
                add_special_tokens=False,
                return_tensors="pt"
            ).to(model.device)

            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=200)
                description = processor.decode(output[0], skip_special_tokens=True)
                description = description[72:].strip()  # remove the prompt part
                description = ensure_quoted(description)
                print("description: ", description)


            # Write one row immediately
            writer.writerow([rel_path, description])
            f.flush()  # ensure data hits disk promptly

        except Exception as e:
            err_msg = f"ERROR: {e}"
            print(f"Error on {rel_path}: {e}")
            writer.writerow([rel_path, err_msg])
            f.flush()

print(f"\n✅ Done! Descriptions saved to {OUTPUT_CSV}")
