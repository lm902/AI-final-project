"""
evaluate.py — Final Hold-Out Evaluation for Experiment 3
COMP 9130 Capstone: Military Camouflage Object Detection
Author: Binger Yu (Data & Preprocessing Lead)

Runs the best saved model against the 200-image held-out final test set
(100 ACD1K + 50 COD10K + 50 noise) and reports mIoU, F1, MAE per subset
plus an overall summary table.

Usage (Colab):
    !python src/evaluate.py \
        --checkpoint outputs/exp3/best_model.pth \
        --data_root  data/ \
        --splits_dir splits/ \
        --output_dir outputs/exp3/eval/
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import build_holdout_dataset


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics_per_image(pred_prob, mask):
    """
    pred_prob : FloatTensor [1, H, W] — sigmoid probability
    mask      : FloatTensor [1, H, W] — binary GT {0, 1}
    Returns dict of per-image scalar metrics.
    """
    pred_bin = (pred_prob > 0.5).float()

    inter = (pred_bin * mask).sum().item()
    union = pred_bin.sum().item() + mask.sum().item() - inter

    iou_fg = (inter + 1e-6) / (union + 1e-6)

    inter_bg = ((1 - pred_bin) * (1 - mask)).sum().item()
    union_bg = ((1 - pred_bin).sum().item() + (1 - mask).sum().item()
                - inter_bg)
    iou_bg = (inter_bg + 1e-6) / (union_bg + 1e-6)

    miou = (iou_fg + iou_bg) / 2
    f1   = (2 * inter + 1e-6) / (pred_bin.sum().item() +
                                   mask.sum().item() + 1e-6)
    mae  = (pred_prob - mask).abs().mean().item()

    return {'mIoU': miou, 'F1': f1, 'MAE': mae}


# ──────────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path, device):
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model = SegformerForSemanticSegmentation.from_pretrained(
        'nvidia/segformer-b2-finetuned-ade-512-512',
        num_labels=1,
        ignore_mismatched_sizes=True,
    )
    model.load_state_dict(ckpt['state_dict'])
    model.to(device).eval()
    print(f'Loaded checkpoint: {checkpoint_path}')
    print(f"  Trained for {ckpt.get('epoch', '?')} epochs  |  "
          f"val mIoU={ckpt.get('val_mIoU', '?'):.4f}")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation loop
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_subset(model, dataset, device, input_size=512):
    """
    Evaluate model on a CamouflageDataset.
    Returns list of per-image result dicts.
    """
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=2, pin_memory=torch.cuda.is_available())
    results = []

    with torch.no_grad():
        for batch in loader:
            image = batch['image'].to(device)
            mask  = batch['mask'].to(device)

            outputs  = model(pixel_values=image)
            logits   = outputs.logits
            upsampled = F.interpolate(logits,
                                      size=(input_size, input_size),
                                      mode='bilinear', align_corners=False)
            prob = torch.sigmoid(upsampled)  # [1, 1, H, W]

            m = compute_metrics_per_image(
                prob[0].cpu(), mask[0].cpu()
            )
            m['filename'] = batch['filename'][0]
            m['dataset']  = batch['dataset'][0]
            results.append(m)

    return results


def summarise(results, label):
    mious = [r['mIoU'] for r in results]
    f1s   = [r['F1']   for r in results]
    maes  = [r['MAE']  for r in results]
    print(f"\n  {label} ({len(results)} images)")
    print(f"    mIoU : {np.mean(mious):.4f}  ±{np.std(mious):.4f}")
    print(f"    F1   : {np.mean(f1s):.4f}  ±{np.std(f1s):.4f}")
    print(f"    MAE  : {np.mean(maes):.4f}  ±{np.std(maes):.4f}")
    return {
        'label': label,
        'n': len(results),
        'mIoU_mean': round(np.mean(mious), 4),
        'mIoU_std':  round(np.std(mious),  4),
        'F1_mean':   round(np.mean(f1s),   4),
        'F1_std':    round(np.std(f1s),    4),
        'MAE_mean':  round(np.mean(maes),  4),
        'MAE_std':   round(np.std(maes),   4),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────
    model = load_model(args.checkpoint, device)

    # ── Hold-out subsets ──────────────────────────────────────────────────
    print('\n[Hold-out datasets]')
    subsets = {}
    for name in ['acd1k', 'cod10k', 'noise']:
        try:
            subsets[name] = build_holdout_dataset(
                args.data_root, name, splits_dir=args.splits_dir
            )
        except FileNotFoundError as e:
            print(f'  WARNING: {e}')

    if not subsets:
        print('ERROR: No hold-out sets found. Run generate_splits.py first.')
        return

    # ── Run evaluation ────────────────────────────────────────────────────
    print('\n[Evaluation results]')
    print('=' * 55)

    all_per_image = {}
    summary_rows  = []

    for name, ds in subsets.items():
        per_img = evaluate_subset(model, ds, device)
        all_per_image[name] = per_img
        row = summarise(per_img, name.upper())
        summary_rows.append(row)

    # ── Overall (ACD1K + COD10K only, excluding noise from avg) ──────────
    camouflage_results = (all_per_image.get('acd1k', []) +
                          all_per_image.get('cod10k', []))
    if camouflage_results:
        row = summarise(camouflage_results, 'ALL CAMOUFLAGE (ACD1K+COD10K)')
        summary_rows.append(row)

    # ── Full combined including noise ─────────────────────────────────────
    all_results = sum(all_per_image.values(), [])
    row = summarise(all_results, 'FULL HOLD-OUT (all 200 images)')
    summary_rows.append(row)

    print('\n' + '=' * 55)
    print('SUMMARY TABLE')
    print('=' * 55)
    print(f"{'Subset':<35} {'mIoU':>6} {'F1':>6} {'MAE':>6}")
    print('-' * 55)
    for r in summary_rows:
        print(f"{r['label']:<35} {r['mIoU_mean']:>6.4f} "
              f"{r['F1_mean']:>6.4f} {r['MAE_mean']:>6.4f}")
    print('=' * 55)

    # ── Success criteria check ────────────────────────────────────────────
    acd1k_row = next((r for r in summary_rows if r['label'] == 'ACD1K'), None)
    if acd1k_row:
        print('\n[Success Criteria Check]')
        miou_pass = acd1k_row['mIoU_mean'] >= 0.65
        f1_pass   = acd1k_row['F1_mean']   >= 0.75
        print(f"  mIoU >= 0.65 : {acd1k_row['mIoU_mean']:.4f}  "
              f"{'✅ PASS' if miou_pass else '❌ FAIL'}")
        print(f"  F1   >= 0.75 : {acd1k_row['F1_mean']:.4f}  "
              f"{'✅ PASS' if f1_pass else '❌ FAIL'}")

    # ── Save results ──────────────────────────────────────────────────────
    out = {
        'checkpoint':   str(args.checkpoint),
        'summary':      summary_rows,
        'per_image':    {k: v for k, v in all_per_image.items()},
    }
    results_path = output_dir / 'eval_results.json'
    with open(results_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nFull results saved → {results_path}')


# ──────────────────────────────────────────────────────────────────────────────
# Argument parser
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Final hold-out evaluation for Experiment 3'
    )
    p.add_argument('--checkpoint',  required=True,
                   help='Path to best_model.pth from train_exp3.py')
    p.add_argument('--data_root',   default='data/',   help='Dataset root')
    p.add_argument('--splits_dir',  default='splits/', help='Splits JSON dir')
    p.add_argument('--output_dir',  default='outputs/exp3/eval/',
                   help='Where to save eval_results.json')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    evaluate(args)
