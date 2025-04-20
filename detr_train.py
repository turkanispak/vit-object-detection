# detr_train.py — DETR Training on AU-AIR 2019

import os
import sys
import time
import json
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import wandb

from detr_config import get_args
from vit_data import build_dataloader
from models.detr import build
from models.matcher import HungarianMatcher
from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from util.utils import save_checkpoint, MetricLogger


def main():
    # =====================
    # CONFIG & LOGGING
    # =====================
    args = get_args()

    # File Logger
    logging.basicConfig(
        filename=os.path.join(args.log_dir, 'train.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler(sys.stdout))

    logger.info("Starting DETR Training")
    logger.info(f"Arguments: {vars(args)}")

    # W&B Logger
    best_loss = float('inf')  # Track best loss
    if args.wandb:
        wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))

    # =====================
    # DATA
    # =====================
    train_split = "train"
    val_split = "val"
    train_loader = build_dataloader(train_split, batch_size=args.batch_size, shuffle=True)
    val_loader = build_dataloader(val_split, batch_size=args.batch_size, shuffle=False)

    logger.info("Data loaded.")

    # =====================
    # MODEL
    # =====================
    model, criterion, postprocessors = build(args)
    model.to(args.device)
    criterion.to(args.device)

    if args.freeze_backbone:
        for param in model.backbone.parameters():
            param.requires_grad = False
        logger.info("Backbone frozen!")


    # =====================
    # LOSS + OPTIMIZER
    # =====================
    matcher = HungarianMatcher(cost_class=1, cost_bbox=5, cost_giou=2)
    weight_dict = {"loss_ce": 1, "loss_bbox": 5, "loss_giou": 2}
    losses = ["labels", "boxes", "cardinality"]

    criterion = SetCriterion(args.num_classes, matcher, weight_dict, eos_coef=0.1, losses=losses)
    criterion.to(args.device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    # =====================
    # TRAINING LOOP
    # =====================
    for epoch in range(args.epochs):
        model.train()
        metric_logger = MetricLogger()
        logger.info(f"Epoch {epoch+1}/{args.epochs}")

        for batch in tqdm(train_loader, desc=f"[Epoch {epoch+1}] Training"):
            samples = batch["pixel_values"].to(args.device)
            targets = [{k: v.to(args.device) for k, v in t.items()} for t in batch["labels"]]


            outputs = model(samples)
            loss_dict = criterion(outputs, targets)
            losses = sum(loss_dict[k] * weight_dict.get(k, 1) for k in loss_dict.keys())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            metric_logger.update(loss=losses.item(), **loss_dict)

            if args.wandb:
                wandb.log({f"train/{k}": v.item() for k, v in loss_dict.items()}, step=epoch)

        logger.info(f"Epoch {epoch+1} Summary:")
        logger.info(metric_logger)

        # =====================
        # CHECKPOINT
        # =====================
        if (epoch + 1) % 10 == 0:
            ckpt_path = os.path.join(args.ckpt_dir, f"detr_epoch_{epoch+1}.pth")
            save_checkpoint(
                state={
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch + 1
                },
                is_best=False,
                checkpoint_dir=args.ckpt_dir,
                filename=f"detr_epoch_{epoch+1}.pth"
            )
            logger.info(f"Saved checkpoint to {ckpt_path}")

        # Save best model based on lowest total loss
        if losses.item() < best_loss:
            best_loss = losses.item()
            save_checkpoint(
                state={
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch + 1
                },
                is_best=True,
                checkpoint_dir=args.ckpt_dir,
                filename="best_model.pth"
            )
            logger.info(f"Best model saved at Epoch {epoch+1} with loss {best_loss:.4f}")


        lr_scheduler.step()

    logger.info("Training Complete!")
    if args.wandb:
        wandb.finish()


# --- SetCriterion: Core Loss ---
class SetCriterion(nn.Module):
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        self.empty_weight = torch.ones(num_classes + 1)
        self.empty_weight[-1] = self.eos_coef
        self.loss_ce = nn.CrossEntropyLoss(weight=self.empty_weight)

    def forward(self, outputs, targets):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        indices = self.matcher(outputs_without_aux, targets)
        loss_dict = {}

        # Classification loss
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        loss_ce = self.loss_ce(src_logits[idx], target_classes_o)
        loss_dict['loss_ce'] = loss_ce

        # Bounding box L1 loss
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_bbox = nn.functional.l1_loss(src_boxes, target_boxes, reduction='none')
        loss_dict['loss_bbox'] = loss_bbox.sum() / src_boxes.size(0)

        # GIoU loss
        giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_boxes),
                                                  box_cxcywh_to_xyxy(target_boxes)))
        loss_dict['loss_giou'] = giou.sum() / src_boxes.size(0)

        return loss_dict

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx


if __name__ == "__main__":
    main()
