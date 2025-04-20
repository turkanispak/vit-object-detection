# detr_evaluate.py – Evaluation for DETR AU-AIR 2019
import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from tqdm import tqdm
from torchvision.ops import box_convert
import wandb

from vit_data import build_dataloader
from models.detr import build
from util.box_ops import box_cxcywh_to_xyxy
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
from PIL import Image, ImageDraw

# ===========
# COCO Evaluator Helpers
# ===========
def coco_format_predictions(outputs, targets, image_ids):
    coco_results = []

    for i, (pred_logits, pred_boxes) in enumerate(zip(outputs['pred_logits'], outputs['pred_boxes'])):
        scores = pred_logits.softmax(-1)[..., :-1].max(-1)
        labels = pred_logits.softmax(-1)[..., :-1].argmax(-1)

        for score, label, box in zip(scores.values, labels, pred_boxes):
            x, y, w, h = box.tolist()
            coco_results.append({
                "image_id": image_ids[i],
                "category_id": label.item(),
                "bbox": [x, y, w, h],
                "score": score.item()
            })

    return coco_results


def evaluate_predictions(coco_gt, coco_dt, split, output_dir):
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Now coco_eval is ready ✅
    metrics = {
        "mAP@[IoU=0.50:0.95]": coco_eval.stats[0],
        "mAP@50": coco_eval.stats[1],
        "mAR@100": coco_eval.stats[8]
    }

    # Add per-class AP extraction
    category_wise_ap = {}
    for idx, cat in enumerate(coco_gt.getCatIds()):
        precision = coco_eval.eval['precision'][:, :, idx, 0, -1]  # IoU, Recall, Cat, Area, maxDets
        precision = precision[precision > -1]
        ap = np.mean(precision) * 100 if precision.size > 0 else 0.0
        cat_name = coco_gt.loadCats(cat)[0]['name']
        category_wise_ap[cat_name] = round(ap, 2)

    metrics.update(category_wise_ap)

    # Save metrics
    metrics_path = os.path.join(output_dir, split, "metrics.json")
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)

    # Plot PR curve
    precisions = coco_eval.eval['precision'][0, :, 0, 0, -1]  # IoU=0.5, all cat, area=all, maxDets=100
    recall = coco_eval.params.recThrs

    plt.figure(figsize=(7, 5))
    plt.plot(recall, precisions)
    plt.title(f"PR Curve - {split}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(True)

    pr_path = os.path.join(output_dir, split, "pr_curve.png")
    plt.savefig(pr_path)
    plt.close()

    return metrics


# ===========
# Visualize Predictions
# ===========
def overlay_predictions(img_tensor, pred_boxes, pred_labels, gt_boxes, gt_labels, class_names):
    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    img = (img * 255).astype(np.uint8)
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)

    # GT in green
    for box, label in zip(gt_boxes, gt_labels):
        x, y, w, h = box.tolist()
        draw.rectangle([x, y, x + w, y + h], outline='green', width=2)
        draw.text((x, y), f"GT {class_names[label]}", fill="green")

    # Predictions in red
    for box, label in zip(pred_boxes, pred_labels):
        x, y, w, h = box.tolist()
        draw.rectangle([x, y, x + w, y + h], outline='red', width=2)
        draw.text((x + 5, y + 5), f"Pred {class_names[label]}", fill="red")

    return img_pil


def visualize_sample_predictions(model, dataloader, device, output_dir, class_names):
    model.eval()
    samples_saved = 0
    for batch in dataloader:
        imgs = batch["pixel_values"].to(device)
        targets = batch["labels"]
        with torch.no_grad():
            outputs = model(imgs)

        pred_logits = outputs['pred_logits'].softmax(-1)[..., :-1]
        pred_labels = pred_logits.argmax(-1)
        pred_boxes = outputs['pred_boxes']

        for i in range(len(imgs)):
            if samples_saved >= 5:
                return
            image_id = targets[i]['image_id'].item()
            pred_box = box_cxcywh_to_xyxy(pred_boxes[i]).cpu()
            gt_box = targets[i]['boxes'].cpu()
            img = overlay_predictions(
                imgs[i],
                pred_box,
                pred_labels[i].cpu(),
                gt_box,
                targets[i]['labels'].cpu(),
                class_names
            )
            sample_path = os.path.join(output_dir, f"sample_{image_id}.png")
            img.save(sample_path)
            samples_saved += 1


# ===========
# Main Evaluation Script
# ===========
def main(args):
    device = torch.device(args.device)

    # Load model
    model, _, _ = build(args)
    model.to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Run eval for each split
    for split in args.eval_splits:
        print(f"\n Evaluating on {split} split")
        dataloader = build_dataloader(split, batch_size=args.batch_size, shuffle=False)

        coco_predictions = []
        all_targets = []
        all_image_ids = []

        for batch in tqdm(dataloader, desc=f"[{split}] Inference"):
            imgs = batch["pixel_values"].to(device)
            targets = batch["labels"]
            with torch.no_grad():
                outputs = model(imgs)

            # Convert boxes to COCO format
            boxes = outputs['pred_boxes'].cpu()
            boxes = box_cxcywh_to_xyxy(boxes)
            for i, box_set in enumerate(boxes):
                image_id = targets[i]['image_id'].item()
                pred_logits = outputs['pred_logits'][i].cpu()
                scores, labels = pred_logits.softmax(-1)[..., :-1].max(-1)
                for score, label, box in zip(scores, labels, box_set):
                    x_min, y_min, x_max, y_max = box.tolist()
                    coco_predictions.append({
                        "image_id": image_id,
                        "category_id": int(label),
                        "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                        "score": float(score)
                    })
                all_image_ids.append(image_id)
                all_targets.append(targets[i])

        # Save COCO predictions
        coco_pred_path = os.path.join(args.output_dir, split, "coco_predictions.json")
        os.makedirs(os.path.dirname(coco_pred_path), exist_ok=True)
        with open(coco_pred_path, "w") as f:
            json.dump(coco_predictions, f)

        # Build GT COCO-style for pycocotools
        coco_gt = COCO()
        coco_gt.dataset = {
            "images": [{"id": t["image_id"].item()} for t in all_targets],
            "annotations": [],
            "categories": [{"id": i, "name": str(i)} for i in range(args.num_classes)]
        }
        ann_id = 1
        for target in all_targets:
            boxes = target["boxes"].cpu().numpy()
            labels = target["labels"].cpu().numpy()
            image_id = target["image_id"].item()
            for box, label in zip(boxes, labels):
                coco_gt.dataset["annotations"].append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": int(label),
                    "bbox": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                    "area": float(box[2] * box[3]),
                    "iscrowd": 0
                })
                ann_id += 1
        coco_gt.createIndex()

        coco_dt = coco_gt.loadRes(coco_predictions)

        # Evaluate + Save
        metrics = evaluate_predictions(coco_gt, coco_dt, split, args.output_dir)

        # Optional sample visualizations
        sample_dir = os.path.join(args.output_dir, split, "samples")
        os.makedirs(sample_dir, exist_ok=True)
        class_names = [str(i) for i in range(args.num_classes)]
        visualize_sample_predictions(model, dataloader, device, sample_dir, class_names)

        # Optional W&B logging
        if args.wandb:
            wandb.init(project=args.wandb_project, name=f"detr_eval_{split}", config=vars(args))
            wandb.log(metrics)
            wandb.log({f"{split}_pr_curve": wandb.Image(os.path.join(args.output_dir, split, "pr_curve.png"))})
            wandb.finish()


# =========== CLI ===========

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone DETR Evaluation")

    # === Required ===
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--eval_splits", nargs="+", default=["val", "test"], help="Splits to evaluate")
    parser.add_argument("--output_dir", type=str, default="evaluation/", help="Directory to save results")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu)")
    parser.add_argument("--batch_size", type=int, default=4, help="Evaluation batch size")

    # === Model-Specific (MUST MATCH training config) ===
    parser.add_argument('--num_classes', type=int, default=8)
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--dilation', action='store_true')
    parser.add_argument('--position_embedding', type=str, default='sine', choices=['sine', 'learned'])
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_queries', type=int, default=100)
    parser.add_argument('--aux_loss', action='store_true')
    parser.add_argument('--masks', action='store_true')  # for compatibility

    # === Transformer ===
    parser.add_argument('--enc_layers', type=int, default=6)
    parser.add_argument('--dec_layers', type=int, default=6)
    parser.add_argument('--dim_feedforward', type=int, default=2048)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--nheads', type=int, default=8)
    parser.add_argument('--pre_norm', action='store_true')

    # === Hungarian Matcher ===
    parser.add_argument('--set_cost_class', type=float, default=1)
    parser.add_argument('--set_cost_bbox', type=float, default=5)
    parser.add_argument('--set_cost_giou', type=float, default=2)

    # === Loss Coefficients ===
    parser.add_argument('--bbox_loss_coef', type=float, default=5)
    parser.add_argument('--giou_loss_coef', type=float, default=2)
    parser.add_argument('--eos_coef', type=float, default=0.1)

    # === Learning Rate to detect freezing ===
    parser.add_argument('--lr_backbone', type=float, default=1e-5)

    # === Optional Logging ===
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='vit-object-detection')

    args = parser.parse_args()
    main(args)
