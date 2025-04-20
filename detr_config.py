# detr_config.py — Configuration for DETR Training Pipeline

import argparse
import os
import torch
from datetime import datetime


def get_args():
    parser = argparse.ArgumentParser(description="Train DETR on AU-AIR 2019")

    # --- Dataset + Paths ---
    parser.add_argument('--data_dir', type=str, default='auair2019/images')
    parser.add_argument('--ann_train', type=str, default='processed_auair2019/train_annotations.json')
    parser.add_argument('--ann_val', type=str, default='processed_auair2019/val_annotations.json')
    parser.add_argument("--ann_test", type=str, default="processed_auair2019/test_annotations.json", help="Path to test annotations")
    parser.add_argument('--output_dir', type=str, default='outputs')

    # --- Model Architecture ---
    parser.add_argument('--num_classes', type=int, default=8)
    parser.add_argument('--pretrained_path', type=str, default='detr-r50-e632da11.pth')
    parser.add_argument('--freeze_backbone', action='store_true')
    parser.add_argument('--backbone', default='resnet50', type=str)
    parser.add_argument('--dilation', action='store_true')
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'))
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--num_queries', default=100, type=int)
    parser.add_argument('--aux_loss', action='store_true')
    parser.add_argument('--masks', action='store_true')  # we don't use this, but keep it for compatibility

    # --- Transformer Specific ---
    parser.add_argument('--enc_layers', default=6, type=int)
    parser.add_argument('--dec_layers', default=6, type=int)
    parser.add_argument('--dim_feedforward', default=2048, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--nheads', default=8, type=int)
    parser.add_argument('--pre_norm', action='store_true') 
    
    # --- Hungarian Matcher ---
    parser.add_argument('--set_cost_class', default=1, type=float)
    parser.add_argument('--set_cost_bbox', default=5, type=float)
    parser.add_argument('--set_cost_giou', default=2, type=float)

    # --- Loss Weights ---
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float)

    # --- Training ---
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_backbone', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    # --- Logging ---
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='vit-object-detection')

    # --- System ---
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    # === Derived ===
    args = parser.parse_args()
    now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    args.run_name = f"detr_run_{now}"
    args.log_dir = os.path.join(args.output_dir, 'logs', args.run_name)
    args.ckpt_dir = os.path.join(args.output_dir, 'checkpoints', args.run_name)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    return args
