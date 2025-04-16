# eda.py - AU-AIR 2019 Dataset EDA 

import os
import json
import logging
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw
import pandas as pd
import numpy as np
import random

# Create output folders
os.makedirs('eda', exist_ok=True)
os.makedirs('eda/samples', exist_ok=True)

# Setup Logging
logging.basicConfig(filename='eda/eda.log', level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

def log_print(msg):
    print(msg)
    logger.info(msg)

# Paths
IMAGE_DIR = os.path.join('auair2019', 'images')
ANNOTATION_PATH = os.path.join('auair2019', 'annotations.json')

def load_annotations(annotation_path):
    with open(annotation_path, 'r') as f:
        data = json.load(f)
    return data

def parse_auair_annotations(data):
    # AU-AIR: categories = list of strings
    category_id_to_name = {idx: name for idx, name in enumerate(data['categories'])}
    annotations = data['annotations']
    return category_id_to_name, annotations

def analyze_annotations(data):
    log_print("Analyzing annotations...")

    category_id_to_name, annotations = parse_auair_annotations(data)

    # --- Dataset Basic Info ---
    total_images = len(annotations)
    total_boxes = sum(len(ann['bbox']) for ann in annotations)
    num_categories = len(category_id_to_name)

    log_print(f"Total images: {total_images}")
    log_print(f"Total bounding boxes: {total_boxes}")
    log_print(f"Number of categories: {num_categories}")
    log_print(f"Categories: {list(category_id_to_name.values())}")

    # --- Category Distribution ---
    category_counts = Counter()
    for ann in annotations:
        for box in ann['bbox']:
            category_counts[category_id_to_name[box['class']]] += 1

    plt.figure(figsize=(10, 5))
    sns.barplot(x=list(category_counts.keys()), y=list(category_counts.values()))
    plt.xticks(rotation=45)
    plt.title('Object Category Distribution')
    plt.tight_layout()
    plt.savefig('eda/category_distribution.png')
    plt.close()

    pd.DataFrame(category_counts.items(), columns=['Category', 'Count']).to_csv('eda/category_counts.csv', index=False)
    log_print("Saved category distribution.")

    # --- Bounding Boxes Per Image ---
    bbox_per_image = [len(ann['bbox']) for ann in annotations]

    plt.figure(figsize=(10, 5))
    sns.histplot(bbox_per_image, bins=30, kde=False)
    plt.title('Bounding Boxes per Image')
    plt.tight_layout()
    plt.savefig('eda/bbox_per_image.png')
    plt.close()

    log_print("Saved bbox per image plot.")

    # --- Bounding Box Area Distribution ---
    bbox_areas = []
    bbox_widths, bbox_heights = [], []

    for ann in annotations:
        for box in ann['bbox']:
            w, h = box['width'], box['height']
            bbox_areas.append(w * h)
            bbox_widths.append(w)
            bbox_heights.append(h)

    plt.figure(figsize=(10, 5))
    sns.histplot(bbox_areas, bins=50, kde=True)
    plt.title('Bounding Box Area Distribution')
    plt.tight_layout()
    plt.savefig('eda/bbox_area_distribution.png')
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.scatter(bbox_widths, bbox_heights, alpha=0.5)
    plt.title('BBox Width vs Height')
    plt.xlabel('Width')
    plt.ylabel('Height')
    plt.tight_layout()
    plt.savefig('eda/bbox_width_height_scatter.png')
    plt.close()

    log_print("Saved bbox area & scatter plots.")

def visualize_sample_images(data, n=5):
    log_print("Saving annotated sample images...")

    category_id_to_name, annotations = parse_auair_annotations(data)

    # Pick random N + extreme cases
    selected = random.sample(annotations, n)

    # Extreme case 1: Most bbox
    selected.append(max(annotations, key=lambda x: len(x['bbox'])))

    # Extreme case 2: Smallest bbox
    smallest = min(
        (box for ann in annotations for box in ann['bbox']),
        key=lambda b: b['width'] * b['height']
    )
    smallest_image = next(ann for ann in annotations if smallest in ann['bbox'])
    selected.append(smallest_image)

    # Extreme case 3: Largest bbox
    largest = max(
        (box for ann in annotations for box in ann['bbox']),
        key=lambda b: b['width'] * b['height']
    )
    largest_image = next(ann for ann in annotations if largest in ann['bbox'])
    selected.append(largest_image)

    for idx, ann in enumerate(selected):
        img_path = os.path.join(IMAGE_DIR, ann['image_name'])
        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)

        for box in ann['bbox']:
            left, top, w, h = box['left'], box['top'], box['width'], box['height']
            cls_name = category_id_to_name[box['class']]
            draw.rectangle([(left, top), (left + w, top + h)], outline="red", width=2)
            draw.text((left, max(0, top-10)), cls_name, fill="yellow")

        img.save(f'eda/samples/sample_{idx+1}.png')

    log_print(f"Saved {len(selected)} sample images in eda/samples/")

def main():
    data = load_annotations(ANNOTATION_PATH)
    analyze_annotations(data)
    visualize_sample_images(data)
    log_print("EDA completed successfully.")

if __name__ == "__main__":
    main()
