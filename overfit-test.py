import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DeformableDetrForObjectDetection, DeformableDetrImageProcessor
from PIL import Image
from torchvision.transforms import functional as TF
from tqdm import tqdm
from pathlib import Path

# --- Paths
ROOT = Path("C:/Users/turka/OneDrive/Desktop/vit-object-detection")
IMAGE_DIR = ROOT / "auair2019/images"
ANNOT_PATH = ROOT / "auair2019/annotations.json"

# --- Define Categories in consistent order
CATEGORIES = ['Human', 'Car', 'Truck', 'Van', 'M.bike', 'Bicycle', 'Bus', 'Trailer']
CATEGORY_TO_ID = {cat: i for i, cat in enumerate(CATEGORIES)}

# --- Convert AU-AIR JSON to COCO-style
def convert_to_coco(annot_path, output_path):
    with open(annot_path) as f:
        data = json.load(f)

    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": name} for i, name in enumerate(CATEGORIES)]
    }

    ann_id = 1
    for img_id, item in enumerate(data['annotations']):
        fname = item['image_name']
        coco["images"].append({
            "id": img_id,
            "file_name": fname,
            "width": 640,
            "height": 512,
        })

        for box in item["bbox"]:
            category = box["class"]
            if category not in CATEGORY_TO_ID:
                continue
            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": CATEGORY_TO_ID[category],
                "bbox": [box["left"], box["top"], box["width"], box["height"]],
                "area": box["width"] * box["height"],
                "iscrowd": 0,
            })
            ann_id += 1

    with open(output_path, "w") as f:
        json.dump(coco, f, indent=2)

# --- Only run once
converted_annot_path = ROOT / "processed_auair2019/coco_annotations.json"
converted_annot_path.parent.mkdir(exist_ok=True)
if not converted_annot_path.exists():
    convert_to_coco(ANNOT_PATH, converted_annot_path)

# --- Dataset
class AUAirDataset(Dataset):
    def __init__(self, img_dir, coco_json, processor):
        with open(coco_json) as f:
            coco = json.load(f)

        self.img_dir = img_dir
        self.images = coco['images']
        self.annotations = coco['annotations']
        self.processor = processor

        # Map image_id to annotations
        self.id2anns = {}
        for ann in self.annotations:
            self.id2anns.setdefault(ann['image_id'], []).append(ann)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_info = self.images[idx]
        img_path = self.img_dir / image_info['file_name']
        image = Image.open(img_path).convert("RGB")

        anns = self.id2anns.get(image_info["id"], [])
        boxes = [ann['bbox'] for ann in anns]
        labels = [ann['category_id'] for ann in anns]

        encoding = self.processor(images=image, annotations={"image_id": 0, "annotations": [{"bbox": b, "category_id": l} for b, l in zip(boxes, labels)]}, return_tensors="pt")
        return {
            'pixel_values': encoding['pixel_values'].squeeze(0),
            'labels': encoding['labels'][0]
        }

# --- HuggingFace model & processor
processor = DeformableDetrImageProcessor.from_pretrained("SenseTime/deformable-detr")
# Load pretrained model (with original 91-class COCO head)
model = DeformableDetrForObjectDetection.from_pretrained("SenseTime/deformable-detr")

# Replace the classification head to match AU-AIR's 8 classes
model.class_labels_classifier = torch.nn.Linear(model.config.hidden_size, 8)

torch.nn.init.normal_(model.class_labels_classifier.weight, std=0.02)
torch.nn.init.zeros_(model.class_labels_classifier.bias)


# --- Dataloader
dataset = AUAirDataset(IMAGE_DIR, converted_annot_path, processor)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=lambda x: {
    'pixel_values': torch.stack([s['pixel_values'] for s in x]),
    'labels': [s['labels'] for s in x]
})

# --- Training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
model.train().cuda()

for epoch in range(10):
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/10")
    for batch in pbar:
        pixel_values = batch['pixel_values'].cuda()
        labels = [{k: v.cuda() for k, v in t.items()} for t in batch['labels']]

        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        pbar.set_postfix(loss=loss.item())

# --- Save model
model.save_pretrained(ROOT / "trained_detr_model")
processor.save_pretrained(ROOT / "trained_detr_model")
