# dataset_prep.py - AU-AIR 2019 Dataset Cleaning & Preparation Pipeline
# Goal: Prepare clean, balanced, ready-to-train dataset for object detection (model-agnostic)

import os
import json
import argparse
import random
import logging
from collections import Counter, defaultdict
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from eda import analyze_annotations, visualize_sample_images, load_annotations, log_print

# --- Configuration ---
RANDOM_SEED = 42
SPLIT_RATIOS = [0.7, 0.2, 0.1]  # train, val, test split

# STEP 1: Flatten annotations for easier processing

def flatten_bboxes(data):
    """Extract all bounding boxes from annotations into a single DataFrame."""
    bboxes = []
    for ann in data['annotations']:
        for box in ann['bbox']:
            bboxes.append({
                'image_name': ann['image_name'],
                'left': box['left'],
                'top': box['top'],
                'width': box['width'],
                'height': box['height'],
                'area': box['width'] * box['height'],
                'class': box['class']
            })
    return pd.DataFrame(bboxes)

# STEP 2: Clean bad annotations (tiny, huge, weird aspect ratios)

def clean_annotations(df, min_area, max_area, min_ar, max_ar):
    df['aspect_ratio'] = df['width'] / (df['height'] + 1e-6)
    before = len(df)
    df = df[(df['area'] >= min_area) & (df['area'] <= max_area)]
    df = df[(df['aspect_ratio'] >= min_ar) & (df['aspect_ratio'] <= max_ar)]
    after = len(df)
    log_print(f"Removed {before - after} bad bboxes (Area or Aspect Ratio filtering)")
    return df

# STEP 3.5: Handle Class Imbalance

def handle_class_imbalance(df, car_class_id=1, car_target=20000, min_target=3000):
    """
    Downsample 'Car' class and upsample rare classes
    """
    # Downsample Car
    car_df = df[df['class'] == car_class_id]
    non_car_df = df[df['class'] != car_class_id]

    before = len(car_df)
    if before > car_target:
        car_df = car_df.sample(n=car_target, random_state=RANDOM_SEED)
        log_print(f"Downsampled 'Car' from {before} -> {len(car_df)} bboxes")
    else:
        log_print(f"No downsampling for 'Car', only {before} bboxes")

    df_balanced = pd.concat([non_car_df, car_df]).reset_index(drop=True)

    # Upsample rare classes
    final_dfs = []
    for cls in df_balanced['class'].unique():
        cls_df = df_balanced[df_balanced['class'] == cls]
        if len(cls_df) < min_target:
            cls_df = cls_df.sample(n=min_target, replace=True, random_state=RANDOM_SEED)
            log_print(f"Upsampled class {cls} to {min_target} bboxes")
        final_dfs.append(cls_df)

    return pd.concat(final_dfs).reset_index(drop=True)

# STEP 4: Group cleaned bboxes back by image

def group_annotations(df):
    grouped = defaultdict(list)
    for _, row in df.iterrows():
        grouped[row['image_name']].append({
            'left': row['left'],
            'top': row['top'],
            'width': row['width'],
            'height': row['height'],
            'class': row['class']
        })
    return grouped

# STEP 5: Analyze split distributions

def analyze_split_distribution(df, split_name, output_dir, class_map):
    counts = df['class'].map(class_map).value_counts()

    plt.figure(figsize=(10, 5))
    sns.barplot(x=counts.index, y=counts.values)
    plt.title(f'{split_name.upper()} Split Class Distribution')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{split_name}_class_distribution.png'))
    plt.close()

    counts.to_csv(os.path.join(output_dir, f'{split_name}_class_counts.csv'))
    log_print(f"Saved {split_name} class distribution plot & csv.")

# STEP 6: Deterministic stratified split with rare combos to train

def stratified_split(df, output_dir, data):
    """
    Step 6: Deterministic stratified split
    - Stratify based on unique class combinations (only if they appear >=2 times)
    - Rare combinations (1 sample only) are added directly to Train or Val
    - No data loss, maximum balance, robust against any weird distributions
    """

    # --- STEP 6A: Prepare stratification keys per image ---
    img_to_class = df.groupby('image_name')['class'].apply(list)

    # Sort class list per image -> turn into string key
    all_classes_raw = [','.join(sorted(map(str, set(cls_list)))) for cls_list in img_to_class.values]

    # Count unique combinations
    df_strat = pd.DataFrame({
        'image_name': img_to_class.index,
        'stratify_key': all_classes_raw
    })

    counts = df_strat['stratify_key'].value_counts()

    # Stratifiable = combinations with >=2 occurrences
    strat_df = df_strat[df_strat['stratify_key'].isin(counts[counts > 1].index)]

    # Rare combos = only 1 occurrence
    non_strat_df = df_strat[df_strat['stratify_key'].isin(counts[counts == 1].index)]

    log_print(f"Images in Stratifiable Pool: {len(strat_df)}")
    log_print(f"Images with Unique Class Combos (sent to Train): {len(non_strat_df)}")

    # --- STEP 6B: Stratified Train/Test split (from stratifiable pool) ---
    train_imgs, temp_imgs, _, temp_classes = train_test_split(
        strat_df['image_name'].tolist(),
        strat_df['stratify_key'].tolist(),
        train_size=SPLIT_RATIOS[0],
        random_state=RANDOM_SEED,
        stratify=strat_df['stratify_key'].tolist()
    )

    # Add rare-combo images directly to train
    train_imgs += non_strat_df['image_name'].tolist()

    # --- STEP 6C: Handle Val/Test split carefully ---
    temp_imgs_array = np.array(temp_imgs)
    temp_classes_array = np.array(temp_classes)

    # Count class combo frequencies in temp pool
    temp_strat_keys = pd.Series(temp_classes).value_counts()

    # Images that can still be stratified (combo appears >=2 times)
    temp_stratifiable_idx = [i for i, key in enumerate(temp_classes) if temp_strat_keys[key] > 1]

    # Rare combo images in temp -> send to val
    temp_non_stratifiable_idx = [i for i in range(len(temp_imgs)) if i not in temp_stratifiable_idx]

    # Proper stratified val/test split
    val_imgs, test_imgs = train_test_split(
        temp_imgs_array[temp_stratifiable_idx].tolist(),
        train_size=SPLIT_RATIOS[1] / (SPLIT_RATIOS[1] + SPLIT_RATIOS[2]),
        random_state=RANDOM_SEED,
        stratify=temp_classes_array[temp_stratifiable_idx].tolist()
    )

    # Rare combo images directly to val
    val_imgs += temp_imgs_array[temp_non_stratifiable_idx].tolist()

    log_print(f"Added {len(temp_non_stratifiable_idx)} rare-combo images directly to Val set")

    # --- STEP 6D: Save splits ---
    splits = {
        'train': train_imgs,
        'val': val_imgs,
        'test': test_imgs
    }

    class_map = {idx: name for idx, name in enumerate(data['categories'])}

    for split, imgs in splits.items():
        split_anns = [{"image_name": img, "bbox": grouped_ann[img]} for img in imgs]

        with open(os.path.join(output_dir, f"{split}_annotations.json"), 'w') as f:
            json.dump({'categories': data['categories'], 'annotations': split_anns}, f, indent=2)

        log_print(f"Saved {split} annotations -> {len(imgs)} images")

        # Analyze per-split class distributions
        df_split = df[df['image_name'].isin(imgs)]
        analyze_split_distribution(df_split, split, output_dir, class_map)

# --- Main Entry ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--min_area_q", type=float, default=0.001)
    parser.add_argument("--max_area_q", type=float, default=0.999)
    parser.add_argument("--min_ar", type=float, default=0.1)
    parser.add_argument("--max_ar", type=float, default=10)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(filename=os.path.join(args.output_dir, 'prep.log'), level=logging.INFO, format='%(message)s')

    log_print("=== Dataset Preparation Started ===")

    # STEP 0: Load annotations
    data = load_annotations(args.input_path)

    # STEP 1: Run EDA before cleaning
    analyze_annotations(data)
    visualize_sample_images(data)

    df = flatten_bboxes(data)
    log_print(f"Initial BBoxes: {len(df)}")

    # STEP 2: Clean annotations
    min_area = df['area'].quantile(args.min_area_q)
    max_area = df['area'].quantile(args.max_area_q)
    log_print(f"Area Thresholds -> Min: {min_area}, Max: {max_area}")

    df = clean_annotations(df, min_area, max_area, args.min_ar, args.max_ar)

    # STEP 3.5: Handle Class Imbalance
    df = handle_class_imbalance(df, car_class_id=1, car_target=20000, min_target=3000)

    # STEP 4: Group back
    grouped_ann = group_annotations(df)
    grouped_ann = {img: boxes for img, boxes in grouped_ann.items() if len(boxes) > 0}
    df = df[df['image_name'].isin(grouped_ann.keys())]

    # STEP 5: Split
    stratified_split(df, args.output_dir, data)

    # STEP 6: EDA after cleaning
    clean_data = {'categories': data['categories'], 'annotations': [{"image_name": k, "bbox": v} for k, v in grouped_ann.items()]}
    analyze_annotations(clean_data)
    visualize_sample_images(clean_data)

    log_print("=== Dataset Preparation Completed ===")
