"""
train_exp3.py — Experiment 3: Joint Training (COD10K + CAMO + ACD1K)
COMP 9130 Capstone: Military Camouflage Object Detection
Author: Binger Yu (Data & Preprocessing Lead)

Usage (Colab):
    !python src/train_exp3.py \
        --data_root  data/ \
        --splits_dir splits/ \
        --output_dir outputs/exp3/ \
        --lr         6e-5 \
        --acd1k_w    8.0 \
        --epochs     50 \
        --batch_size 8

Hyperparameter sweep — run three times to find best LR + ACD1K weight:
    lr         : [1e-4, 6e-5, 1e-5]
    acd1k_w    : [4.0, 8.0, 16.0]
    (fix batch_size=8, epochs=30 for sweep; re-run winner for 50 epochs)
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import SegformerForSemanticSegmentation
import torch.nn.functional as F

# ── Allow running from repo root or src/ ─────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from dataset import build_dataloader


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics(preds, masks):
    """
    Compute mIoU, F1 (Dice), MAE for a batch.
    preds : FloatTensor [B, 1, H, W] — probabilities after sigmoid
    masks : FloatTensor [B, 1, H, W] — binary ground truth {0, 1}
    Returns dict of scalar averages over the batch.
    """
    preds_bin = (preds > 0.5).float()

    inter = (preds_bin * masks).sum(dim=(1, 2, 3))
    union = preds_bin.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3)) - inter

    iou_fg  = (inter + 1e-6) / (union + 1e-6)
    iou_bg  = ((1 - preds_bin) * (1 - masks)).sum(dim=(1, 2, 3))
    union_bg = (1 - preds_bin).sum(dim=(1, 2, 3)) + (1 - masks).sum(dim=(1, 2, 3)) - iou_bg
    iou_bg  = (iou_bg + 1e-6) / (union_bg + 1e-6)
    miou    = ((iou_fg + iou_bg) / 2).mean().item()

    dice = (2 * inter + 1e-6) / (preds_bin.sum(dim=(1, 2, 3)) +
                                   masks.sum(dim=(1, 2, 3)) + 1e-6)
    f1   = dice.mean().item()

    mae = (preds - masks).abs().mean().item()

    return {'mIoU': miou, 'F1': f1, 'MAE': mae}


# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────

def build_model():
    """
    Load SegFormer-B2 pretrained on ADE20K, replace head with binary output.
    """
    model = SegformerForSemanticSegmentation.from_pretrained(
        'nvidia/segformer-b2-finetuned-ade-512-512',
        num_labels=1,
        ignore_mismatched_sizes=True,
    )
    return model


def forward_pass(model, images, masks, input_size=512):
    """
    Run forward pass; upsample logits to input_size; return loss + probs.
    """
    outputs   = model(pixel_values=images, labels=None)
    logits    = outputs.logits                          # [B, 1, H/4, W/4]
    upsampled = F.interpolate(logits, size=(input_size, input_size),
                               mode='bilinear', align_corners=False)
    probs     = torch.sigmoid(upsampled)

    # Binary cross-entropy + Dice loss
    bce  = F.binary_cross_entropy_with_logits(upsampled, masks)
    inter = (probs * masks).sum()
    dice_loss = 1 - (2 * inter + 1) / (probs.sum() + masks.sum() + 1)
    loss = bce + dice_loss

    return loss, probs


# ──────────────────────────────────────────────────────────────────────────────
# Train / validation loops
# ──────────────────────────────────────────────────────────────────────────────

def run_epoch(model, loader, optimizer, device, train=True, accum_steps=1):
    model.train() if train else model.eval()
    total_loss = 0.0
    all_miou, all_f1, all_mae = [], [], []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for i, batch in enumerate(loader):
            images = batch['image'].to(device)
            masks  = batch['mask'].to(device)

            loss, probs = forward_pass(model, images, masks)

            if train:
                loss = loss / accum_steps
                loss.backward()
                if (i + 1) % accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

            total_loss += loss.item()
            m = compute_metrics(probs.detach(), masks)
            all_miou.append(m['mIoU'])
            all_f1.append(m['F1'])
            all_mae.append(m['MAE'])

    n = len(loader)
    return {'loss': total_loss/n, 'mIoU': np.mean(all_miou),
            'F1': np.mean(all_f1), 'MAE': np.mean(all_mae)}


# ──────────────────────────────────────────────────────────────────────────────
# Main training function
# ──────────────────────────────────────────────────────────────────────────────

def train(args):
    # ── Reproducibility ───────────────────────────────────────────────────
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Save run config ───────────────────────────────────────────────────
    config = vars(args)
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    print(f'Config saved → {output_dir}/config.json')
    print(json.dumps(config, indent=2))

    # ── Dataloaders ───────────────────────────────────────────────────────
    print('\n[DataLoaders]')

    # Override ACD1K weight in dataset.py sampler via monkeypatch
    import dataset as ds_module
    _orig_sampler = ds_module._build_weighted_sampler
    def _patched_sampler(concat_dataset):
        weights = []
        for ds in concat_dataset.datasets:
            w = args.acd1k_w if ds.dataset_name == 'ACD1K' else 1.0
            weights.extend([w] * len(ds))
        weights = torch.DoubleTensor(weights)
        sampler = torch.utils.data.WeightedRandomSampler(
            weights, num_samples=len(weights), replacement=True
        )
        print(f'  [Sampler] ACD1K weight={args.acd1k_w}, others=1.0')
        return sampler
    ds_module._build_weighted_sampler = _patched_sampler

    train_loader = build_dataloader(
        args.data_root, condition='joint', split='train',
        batch_size=args.batch_size, num_workers=args.num_workers,
        oversample_acd1k=True, seed=args.seed, splits_dir=args.splits_dir,
    )
    val_loader = build_dataloader(
        args.data_root, condition='acd1k', split='val',
        batch_size=args.batch_size, num_workers=args.num_workers,
        oversample_acd1k=False, seed=args.seed, splits_dir=args.splits_dir,
    )

    ds_module._build_weighted_sampler = _orig_sampler  # restore

    # ── Model ─────────────────────────────────────────────────────────────
    print('\n[Model] Loading SegFormer-B2 (ADE20K pretrained)...')
    model = build_model().to(device)

    # ── Optimizer & Scheduler ─────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=args.lr,
                      weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # ── Training loop ─────────────────────────────────────────────────────
    print(f'\n[Training] {args.epochs} epochs, lr={args.lr}, '
          f'acd1k_w={args.acd1k_w}')
    print('=' * 65)

    best_val_miou  = 0.0
    best_epoch     = 0
    patience_count = 0
    history        = []

    for epoch in range(1, args.epochs + 1):
        train_m = run_epoch(model, train_loader, optimizer, device,
                            train=True, accum_steps=args.accum_steps)
        val_m = run_epoch(model, val_loader, optimizer, device,
                          train=False, accum_steps=1)

        row = {
            'epoch':      epoch,
            'train_loss': round(train_m['loss'], 4),
            'train_mIoU': round(train_m['mIoU'], 4),
            'train_F1':   round(train_m['F1'],   4),
            'train_MAE':  round(train_m['MAE'],  4),
            'val_loss':   round(val_m['loss'],   4),
            'val_mIoU':   round(val_m['mIoU'],   4),
            'val_F1':     round(val_m['F1'],     4),
            'val_MAE':    round(val_m['MAE'],    4),
            'lr':         round(scheduler.get_last_lr()[0], 8),
        }
        history.append(row)

        improved = val_m['mIoU'] > best_val_miou
        marker   = ' ◀ best' if improved else ''
        print(
            f"Ep {epoch:3d}/{args.epochs} | "
            f"train loss={train_m['loss']:.4f} mIoU={train_m['mIoU']:.4f} | "
            f"val loss={val_m['loss']:.4f} mIoU={val_m['mIoU']:.4f} "
            f"F1={val_m['F1']:.4f} MAE={val_m['MAE']:.4f}{marker}"
        )

        if improved:
            best_val_miou  = val_m['mIoU']
            best_epoch     = epoch
            patience_count = 0
            torch.save({
                'epoch':      epoch,
                'state_dict': model.state_dict(),
                'val_mIoU':   best_val_miou,
                'config':     config,
            }, output_dir / 'best_model.pth')
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f'\n[Early stopping] No improvement for {args.patience} '
                      f'epochs. Best val mIoU={best_val_miou:.4f} at '
                      f'epoch {best_epoch}.')
                break

        # Save history every epoch (safe for Colab disconnects)
        with open(output_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)

    print('=' * 65)
    print(f'Training complete. Best val mIoU={best_val_miou:.4f} '
          f'at epoch {best_epoch}.')
    print(f'Best model saved → {output_dir}/best_model.pth')

    return best_val_miou


# ──────────────────────────────────────────────────────────────────────────────
# Argument parser
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Train Experiment 3 — Joint SegFormer-B2'
    )
    p.add_argument('--data_root',   default='data/',     help='Dataset root')
    p.add_argument('--splits_dir',  default='splits/',   help='Splits JSON dir')
    p.add_argument('--output_dir',  default='outputs/exp3/', help='Save dir')

    # ── Hyperparameters (tune these) ──────────────────────────────────────
    p.add_argument('--lr',          type=float, default=6e-5,
                   help='Learning rate. Try: 1e-4, 6e-5, 1e-5')
    p.add_argument('--acd1k_w',     type=float, default=8.0,
                   help='ACD1K oversample weight. Try: 4.0, 8.0, 16.0')
    p.add_argument('--weight_decay',type=float, default=1e-4)

    # ── Training settings ─────────────────────────────────────────────────
    p.add_argument('--epochs',      type=int,   default=50)
    p.add_argument('--batch_size',  type=int,   default=8)
    p.add_argument('--accum_steps', type=int, default=2,
                   help='Gradient accumulation steps (effective batch = batch_size * accum_steps)')
    p.add_argument('--patience',    type=int,   default=10,
                   help='Early stopping patience (epochs)')
    p.add_argument('--num_workers', type=int,   default=2)
    p.add_argument('--seed',        type=int,   default=42)

    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
