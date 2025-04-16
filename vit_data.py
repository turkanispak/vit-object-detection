# vit_data.py - AU-AIR 2019 DataLoader + Transforms for DETR

import os
import json
import logging
from PIL import Image, ImageDraw
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import matplotlib.pyplot as plt

# --- Configuration ---
IMAGE_DIR = os.path.join("auair2019", "images")
ANNOTATION_DIR = os.path.join("processed_auair2019")
LOG_PATH = os.path.join(ANNOTATION_DIR, 'vit_data.log')
SAVE_SAMPLES = os.path.join(ANNOTATION_DIR, 'vit_data_samples')

os.makedirs(SAVE_SAMPLES, exist_ok=True)

logging.basicConfig(filename=LOG_PATH, level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

def log_print(msg):
    print(msg)
    logger.info(msg)

# --- Transforms ---
train_transform = A.Compose([
    A.LongestMaxSize(512),
    A.PadIfNeeded(512, 512, border_mode=0),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.GaussNoise(p=0.3),
    A.Affine(scale=(0.9, 1.1), translate_percent=0.05, rotate=10, p=0.5),
    ToTensorV2()
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

val_transform = A.Compose([
    A.LongestMaxSize(512),
    A.PadIfNeeded(512, 512, border_mode=0),
    ToTensorV2()
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

# --- Dataset Class ---
class AUAirDataset(Dataset):
    def __init__(self, split):
        self.split = split
        self.ann_path = os.path.join(ANNOTATION_DIR, f"{split}_annotations.json")
        self.data = self.load_annotations()
        self.images = list(self.data.keys())
        self.transform = train_transform if split == 'train' else val_transform

        log_print(f"Loaded {split} split -> {len(self.images)} images")

    def load_annotations(self):
        with open(self.ann_path, 'r') as f:
            raw = json.load(f)
        data = defaultdict(list)
        for ann in raw['annotations']:
            for box in ann['bbox']:
                data[ann['image_name']].append({
                    'x_min': box['left'],
                    'y_min': box['top'],
                    'x_max': box['left'] + box['width'],
                    'y_max': box['top'] + box['height'],
                    'label': box['class']
                })
        return data

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(IMAGE_DIR, img_name)
        img = np.array(Image.open(img_path).convert("RGB"))

        bboxes = self.data[img_name]
        if not bboxes:  # just in case
            bboxes = [{'x_min':1,'y_min':1,'x_max':2,'y_max':2,'label':0}]

        boxes = [[b['x_min'], b['y_min'], b['x_max'], b['y_max']] for b in bboxes]
        labels = [b['label'] for b in bboxes]

        transformed = self.transform(image=img, bboxes=boxes, class_labels=labels)

        # Convert to COCO format for DETR
        boxes = torch.tensor(transformed['bboxes'], dtype=torch.float)
        boxes[:, 2] -= boxes[:, 0]  # width
        boxes[:, 3] -= boxes[:, 1]  # height

        target = {
            "boxes": boxes,
            "class_labels": torch.tensor(transformed['class_labels'], dtype=torch.long),
            "image_id": torch.tensor([idx])
        }

        return {
            "pixel_values": transformed['image'],
            "labels": target
        }

# --- Collate for variable bbox counts ---
def collate_fn(batch):
    pixel_values = [item["pixel_values"] for item in batch]
    labels = [item["labels"] for item in batch]
    return {"pixel_values": torch.stack(pixel_values), "labels": labels}

# --- Build Dataloader ---
def build_dataloader(split, batch_size=4, shuffle=True):
    dataset = AUAirDataset(split)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

# --- Optional Visualization ---
def visualize_sample(dataset, idx):
    sample = dataset[idx]
    img = sample['pixel_values'].permute(1,2,0).numpy()
    labels = sample['labels']

    img = (img * 255).astype(np.uint8)
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)

    for box, label in zip(labels['boxes'], labels['class_labels']):
        x, y, w, h = box.tolist()
        draw.rectangle([x, y, x+w, y+h], outline='red', width=2)
        draw.text((x, y-10), f"Class {label.item()}", fill='yellow')

    img.save(os.path.join(SAVE_SAMPLES, f"{dataset.split}_sample_{idx}.png"))
    log_print(f"Saved {dataset.split} sample {idx}")

# --- Usage Example ---
if __name__ == "__main__":
    train_loader = build_dataloader('train', batch_size=4, shuffle=True)
    val_loader = build_dataloader('val', batch_size=4, shuffle=False)

    log_print("Dataloaders Ready.")

    # Optional - visualize 3 random train samples
    dataset = AUAirDataset('train')
    for idx in np.random.choice(len(dataset), 3, replace=False):
        visualize_sample(dataset, idx)
