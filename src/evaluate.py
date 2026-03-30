"""
evaluate.py — Final Hold-Out Evaluation for Experiment 3
COMP 9130 Capstone: Military Camouflage Object Detection
Author: Binger Yu (Data & Preprocessing Lead + Experiment 3 Lead)

Runs the best saved SegFormer-B2 checkpoint against the 200-image held-out
final test set (100 ACD1K + 50 COD10K + 50 noise distractors) and reports
mIoU, F1/Dice, and MAE per subset plus an overall summary table.

Architecture: SegFormer-B2 (Experiments 2 and 3 only).
Input resolution: 512×512 (matches nvidia/segformer-b2-finetuned-ade-512-512).
This script is not compatible with Experiment 1 (SINetV2, 352×352).

Success criteria (on ACD1K hold-out subset):
  mIoU >= 0.65
  F1   >= 0.75

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
    Compute mIoU, F1/Dice, and MAE for a single image prediction.

    mIoU is computed as the mean of foreground IoU and background IoU,
    consistent with binary segmentation evaluation practice.
    F1 (Dice) is computed over foreground pixels only.
    MAE is computed over the full probability map vs. binary GT mask.

    Args:
        pred_prob : FloatTensor [1, H, W] — sigmoid probability in [0, 1].
        mask      : FloatTensor [1, H, W] — binary GT mask {0, 1}.

    Returns:
        dict with keys 'mIoU', 'F1', 'MAE' (all scalar floats).
    """
    pred_bin = (pred_prob > 0.5).float()

    # Foreground IoU
    inter    = (pred_bin * mask).sum().item()
    union    = pred_bin.sum().item() + mask.sum().item() - inter
    iou_fg   = (inter + 1e-6) / (union + 1e-6)

    # Background IoU
    inter_bg = ((1 - pred_bin) * (1 - mask)).sum().item()
    union_bg = ((1 - pred_bin).sum().item() + (1 - mask).sum().item()
                - inter_bg)
    iou_bg   = (inter_bg + 1e-6) / (union_bg + 1e-6)

    miou = (iou_fg + iou_bg) / 2
    f1   = (2 * inter + 1e-6) / (pred_bin.sum().item() +
                                   mask.sum().item() + 1e-6)
    mae  = (pred_prob - mask).abs().mean().item()

    return {'mIoU': miou, 'F1': f1, 'MAE': mae}


# ──────────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path, device):
    """
    Load a SegFormer-B2 model from a saved checkpoint.

    The checkpoint is expected to be a dict with keys:
      'state_dict' : model weights
      'epoch'      : (optional) training epoch at which checkpoint was saved
      'val_mIoU'   : (optional) validation mIoU at checkpoint epoch

    Args:
        checkpoint_path (str): Path to best_model.pth.
        device          : torch.device to load the model onto.

    Returns:
        SegformerForSemanticSegmentation in eval mode.
    """
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model = SegformerForSemanticSegmentation.from_pretrained(
        'nvidia/segformer-b2-finetuned-ade-512-512',
        num_labels=1,
        ignore_mismatched_sizes=True,
    )
    model.load_state_dict(ckpt['state_dict'])
    model.to(device).eval()

    # Safely format optional checkpoint metadata for logging.
    # val_mIoU may be absent from older checkpoints — guard before formatting.
    epoch_str   = str(ckpt.get('epoch', '?'))
    val_miou    = ckpt.get('val_mIoU', None)
    val_miou_str = f"{val_miou:.4f}" if val_miou is not None else "?"

    print(f'Loaded checkpoint: {checkpoint_path}')
    print(f"  Trained for {epoch_str} epochs  |  val mIoU={val_miou_str}")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation loop
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_subset(model, dataset, device, input_size=512):
    """
    Run inference and compute per-image metrics on a CamouflageDataset.

    Logits from SegFormer-B2 are output at 1/4 input resolution and are
    bilinearly upsampled back to input_size before sigmoid thresholding.

    Args:
        model      : SegFormer-B2 in eval mode.
        dataset    : CamouflageDataset instance (val transforms applied).
        device     : torch.device.
        input_size (int): Target upsample resolution. Default 512 (SegFormer-B2).
                          This script is not intended for Experiment 1 (SINetV2,
                          which uses 352×352).

    Returns:
        list of per-image result dicts, each containing
        'mIoU', 'F1', 'MAE', 'filename', 'dataset'.
    """
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=2, pin_memory=torch.cuda.is_available())
    results = []

    with torch.no_grad():
        for batch in loader:
            image = batch['image'].to(device)
            mask  = batch['mask'].to(device)

            outputs   = model(pixel_values=image)
            logits    = outputs.logits

            # Upsample from SegFormer's 1/4-resolution output back to input_size
            upsampled = F.interpolate(logits,
                                      size=(input_size, input_size),
                                      mode='bilinear', align_corners=False)
            prob = torch.sigmoid(upsampled)  # [1, 1, H, W] in [0, 1]

            m = compute_metrics_per_image(
                prob[0].cpu(), mask[0].cpu()
            )
            m['filename'] = batch['filename'][0]
            m['dataset']  = batch['dataset'][0]
            results.append(m)

    return results


def summarise(results, label):
    """
    Compute and print mean ± std of mIoU, F1, MAE across a list of results.

    Args:
        results (list): Per-image result dicts from evaluate_subset().
        label   (str) : Display label for this subset.

    Returns:
        dict of aggregated statistics for this subset.
    """
    mious = [r['mIoU'] for r in results]
    f1s   = [r['F1']   for r in results]
    maes  = [r['MAE']  for r in results]
    print(f"\n  {label} ({len(results)} images)")
    print(f"    mIoU : {np.mean(mious):.4f}  ±{np.std(mious):.4f}")
    print(f"    F1   : {np.mean(f1s):.4f}  ±{np.std(f1s):.4f}")
    print(f"    MAE  : {np.mean(maes):.4f}  ±{np.std(maes):.4f}")
    return {
        'label':     label,
        'n':         len(results),
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

    # ── Build hold-out subsets ────────────────────────────────────────────
    # Three subsets together form the 200-image final test set:
    #   acd1k  — 100 military camouflage images (terrain-stratified)
    #   cod10k — 50  animal camouflage images   (super-class stratified)
    #   noise  — 50  ordinary outdoor scenes    (no camouflage target;
    #                                            GT masks are all-zero)
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

    # ── Camouflage-only aggregate (ACD1K + COD10K, noise excluded) ────────
    # Noise images have all-zero GT masks; excluding them from the camouflage
    # aggregate gives a cleaner picture of detection performance.
    camouflage_results = (all_per_image.get('acd1k', []) +
                          all_per_image.get('cod10k', []))
    if camouflage_results:
        row = summarise(camouflage_results, 'ALL CAMOUFLAGE (ACD1K+COD10K)')
        summary_rows.append(row)

    # ── Full combined (all 200 images including noise) ────────────────────
    all_results = sum(all_per_image.values(), [])
    row = summarise(all_results, 'FULL HOLD-OUT (all 200 images)')
    summary_rows.append(row)

    # ── Summary table ─────────────────────────────────────────────────────
    print('\n' + '=' * 55)
    print('SUMMARY TABLE')
    print('=' * 55)
    print(f"{'Subset':<35} {'mIoU':>6} {'F1':>6} {'MAE':>6}")
    print('-' * 55)
    for r in summary_rows:
        print(f"{r['label']:<35} {r['mIoU_mean']:>6.4f} "
              f"{r['F1_mean']:>6.4f} {r['MAE_mean']:>6.4f}")
    print('=' * 55)

    # ── Success criteria check (ACD1K subset only) ────────────────────────
    # Thresholds from proposal Section 1.5:
    #   mIoU >= 0.65 and F1 >= 0.75 on the ACD1K hold-out subset.
    acd1k_row = next((r for r in summary_rows if r['label'] == 'ACD1K'), None)
    if acd1k_row:
        print('\n[Success Criteria Check — ACD1K hold-out]')
        miou_pass = acd1k_row['mIoU_mean'] >= 0.65
        f1_pass   = acd1k_row['F1_mean']   >= 0.75
        print(f"  mIoU >= 0.65 : {acd1k_row['mIoU_mean']:.4f}  "
              f"{'✅ PASS' if miou_pass else '❌ FAIL'}")
        print(f"  F1   >= 0.75 : {acd1k_row['F1_mean']:.4f}  "
              f"{'✅ PASS' if f1_pass else '❌ FAIL'}")

    # ── Save full results to JSON ──────────────────────────────────────────
    out = {
        'checkpoint': str(args.checkpoint),
        'summary':    summary_rows,
        'per_image':  {k: v for k, v in all_per_image.items()},
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
        description='Final hold-out evaluation for Experiment 3 (SegFormer-B2)'
    )
    p.add_argument('--checkpoint',
                   required=True,
                   help='Path to best_model.pth saved by train_exp3.py')
    p.add_argument('--data_root',
                   default='data/',
                   help='Root directory containing COD10K, ACD1K, CAMO folders')
    p.add_argument('--splits_dir',
                   default='splits/',
                   help='Directory containing hold-out JSON index files')
    p.add_argument('--output_dir',
                   default='outputs/exp3/eval/',
                   help='Directory to save eval_results.json')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    evaluate(args)