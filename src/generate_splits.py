"""
generate_splits.py — Fixed split and hold-out index generator
COMP 9130 Capstone: Military Camouflage Object Detection
Author: Binger Yu (Data & Preprocessing Lead)

Generates and saves the following index files to splits/:
  hold_out_acd1k.json      — 100 ACD1K images held out permanently
  hold_out_cod10k.json     — 50 COD10K test images held out permanently
  hold_out_noise.json      — 50 noise image placeholders
  acd1k_splits.json        — 70/15/15 train/val/test for remaining ACD1K
  cod10k_splits.json       — 70/15/15 train/val/test for remaining COD10K

All splits use SEED=42. Run ONCE before any training begins and
commit all JSON files to the repository.

Usage:
    python src/generate_splits.py data/ [--splits_dir splits/]

Requirements:
    pip install numpy scikit-learn
"""

import os
import json
import random
import argparse
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# ── Fixed seed — never change this ──────────────────────────────────────────
SEED = 42

# ── Hold-out sizes (as specified in Section 2.3 of proposal) ────────────────
N_HOLDOUT_ACD1K  = 100
N_HOLDOUT_COD10K = 50
N_HOLDOUT_NOISE  = 50

# ── Train/Val/Test ratio for remaining images ────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# ── Image extensions ─────────────────────────────────────────────────────────
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def get_image_filenames(folder):
    """Return sorted list of image filenames (not full paths) in a folder."""
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    files = sorted([
        p.name for p in folder.iterdir()
        if p.suffix.lower() in IMG_EXTS
    ])
    if len(files) == 0:
        raise FileNotFoundError(f"No images found in: {folder}")
    return files


def stratified_holdout_acd1k(image_dir, n=N_HOLDOUT_ACD1K, seed=SEED):
    """
    Sample N images from ACD1K with rough terrain stratification.
    ACD1K terrain types are inferred from filenames where possible;
    if filenames don't contain terrain labels, falls back to random sampling.

    Returns: (holdout_list, remaining_list)
    """
    set_seed(seed)
    all_files = get_image_filenames(image_dir)

    # Try to stratify by terrain from filename keywords
    terrain_map = {'forest': [], 'desert': [], 'snow': [], 'other': []}
    for f in all_files:
        fl = f.lower()
        if 'forest' in fl or 'jungle' in fl or 'woodland' in fl:
            terrain_map['forest'].append(f)
        elif 'desert' in fl or 'rocky' in fl or 'sand' in fl or 'arid' in fl:
            terrain_map['desert'].append(f)
        elif 'snow' in fl or 'arctic' in fl or 'winter' in fl:
            terrain_map['snow'].append(f)
        else:
            terrain_map['other'].append(f)

    labeled = sum(len(v) for k, v in terrain_map.items() if k != 'other')

    # Check if stratification is meaningful (>10% of files have terrain labels)
    labeled = sum(len(v) for k, v in terrain_map.items() if k != 'other')
    if labeled > 0.1 * len(all_files):
        # Stratified sampling — proportional per terrain
        holdout = []
        for terrain, files in terrain_map.items():
            if len(files) == 0:
                continue
            proportion = len(files) / len(all_files)
            k = max(1, round(n * proportion))
            k = min(k, len(files))
            holdout.extend(random.sample(files, k))

        # Adjust to exactly N
        holdout = list(set(holdout))
        random.shuffle(holdout)
        if len(holdout) > n:
            holdout = holdout[:n]
        elif len(holdout) < n:
            remaining_pool = [f for f in all_files if f not in holdout]
            holdout.extend(random.sample(remaining_pool, n - len(holdout)))

        print(f"  [ACD1K hold-out] Stratified sampling used")
        for t, files in terrain_map.items():
            in_holdout = [f for f in holdout if f in files]
            print(f"    {t}: {len(in_holdout)}/{len(files)}")
    else:
        # Pure random sampling (filenames don't encode terrain)
        holdout = random.sample(all_files, n)
        print(f"  [ACD1K hold-out] Random sampling used "
              f"(terrain labels not found in filenames)")

    remaining = [f for f in all_files if f not in holdout]
    return sorted(holdout), sorted(remaining)


def random_holdout_cod10k(test_image_dir, n=N_HOLDOUT_COD10K, seed=SEED):
    """
    Sample N images from COD10K TEST partition for hold-out.
    These images come from the test partition to avoid contaminating
    the training split.

    Returns: (holdout_list, remaining_list)
    """
    set_seed(seed)
    all_files = get_image_filenames(test_image_dir)
    holdout   = sorted(random.sample(all_files, n))
    remaining = sorted([f for f in all_files if f not in holdout])
    return holdout, remaining


def generate_train_val_test(file_list, train_ratio=TRAIN_RATIO,
                            val_ratio=VAL_RATIO, seed=SEED):
    """
    Split a list of filenames into train/val/test using fixed ratios.
    Uses sklearn train_test_split with stratify=None (no class labels).

    Returns: {'train': [...], 'val': [...], 'test': [...]}
    """
    set_seed(seed)
    test_ratio = 1.0 - train_ratio - val_ratio

    # First split: train vs (val + test)
    train, val_test = train_test_split(
        file_list,
        train_size=train_ratio,
        random_state=seed,
        shuffle=True,
    )

    # Second split: val vs test (from the val+test pool)
    relative_val = val_ratio / (val_ratio + test_ratio)
    val, test = train_test_split(
        val_test,
        train_size=relative_val,
        random_state=seed,
        shuffle=True,
    )

    return {
        'train': sorted(train),
        'val':   sorted(val),
        'test':  sorted(test),
    }


def save_json(data, path):
    """Save data as formatted JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  Saved → {path}  ({_count_str(data)} images)")


def _count_str(data):
    """Return a readable count string for a splits dict or list."""
    if isinstance(data, list):
        return str(len(data))
    elif isinstance(data, dict):
        parts = []
        for k, v in data.items():
            if isinstance(v, list):
                parts.append(f"{k}={len(v)}")
        return '  '.join(parts)
    return '?'


def verify_no_overlap(splits_dir):
    """
    Verify that hold-out sets and train/val/test splits have no overlapping
    filenames. Prints a summary.
    """
    splits_dir = Path(splits_dir)
    print("\n[Verification] Checking for overlaps...")

    # Load all index files
    files = {}
    for json_file in sorted(splits_dir.glob('*.json')):
        with open(json_file) as f:
            data = json.load(f)
        if isinstance(data, list):
            files[json_file.stem] = set(data)
        elif isinstance(data, dict):
            for split_name, split_files in data.items():
                if isinstance(split_files, list) and len(split_files) > 0:
                    key = f"{json_file.stem}/{split_name}"
                    files[key] = set(split_files)

    # Check all pairs
    keys = list(files.keys())
    found_overlap = False
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            overlap = files[keys[i]] & files[keys[j]]
            if overlap:
                print(f"  ❌ OVERLAP: {keys[i]} ∩ {keys[j]} = "
                      f"{len(overlap)} files: {list(overlap)[:3]}...")
                found_overlap = True

    if not found_overlap:
        print("  ✅ No overlaps found — all splits are disjoint")

    # Print summary counts
    print("\n[Summary] Index file contents:")
    for key, fileset in sorted(files.items()):
        print(f"  {key:<45}: {len(fileset):>5} images")


def main():
    parser = argparse.ArgumentParser(
        description='Generate fixed train/val/test split index files.'
    )
    parser.add_argument('data_root', type=str,
                        help='Root directory containing all dataset folders')
    parser.add_argument('--splits_dir', type=str, default='splits',
                        help='Output directory for JSON index files '
                             '(default: splits/)')
    args = parser.parse_args()

    data_root  = Path(args.data_root)
    splits_dir = Path(args.splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Generating fixed split index files")
    print(f"  Data root  : {data_root}")
    print(f"  Output dir : {splits_dir}")
    print(f"  Seed       : {SEED}")
    print("=" * 60)

    # ── Dataset paths ──────────────────────────────────────────────────────
    ACD1K_TRAIN_IMAGES  = data_root / 'dataset-splitM/Training/images'
    ACD1K_TEST_IMAGES   = data_root / 'dataset-splitM/Testing/images'
    COD10K_TEST_IMAGES  = data_root / 'COD10K-v3/Test/Image'
    COD10K_TRAIN_IMAGES = data_root / 'COD10K-v3/Train/Image'

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 1 — ACD1K hold-out (100 images from training split)
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n[Step 1] ACD1K hold-out ({N_HOLDOUT_ACD1K} images)")
    acd1k_holdout, acd1k_remaining_train = stratified_holdout_acd1k(
        ACD1K_TRAIN_IMAGES, n=N_HOLDOUT_ACD1K
    )

    # Also get test filenames (kept separate, not split further)
    acd1k_test_files = get_image_filenames(ACD1K_TEST_IMAGES)

    save_json(
        {
            'description': 'ACD1K hold-out set — 100 images from training '
                           'partition. Held out permanently before any '
                           'training. Never used for training, validation, '
                           'or hyperparameter tuning.',
            'seed':        SEED,
            'source':      'ACD1K Training partition',
            'n':           len(acd1k_holdout),
            'files':       acd1k_holdout,
        },
        splits_dir / 'hold_out_acd1k.json'
    )

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 2 — COD10K hold-out (50 images from train partition)
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n[Step 2] COD10K hold-out ({N_HOLDOUT_COD10K} images)")
    cod10k_holdout, cod10k_remaining_train = random_holdout_cod10k(
        COD10K_TRAIN_IMAGES, n=N_HOLDOUT_COD10K
    )
    print(f"  Sampled {len(cod10k_holdout)} from COD10K train partition "
          f"({len(cod10k_remaining_train)} remaining in train)")

    save_json(
        {
            'description': 'COD10K hold-out set — 50 images from train '
                           'partition. Held out permanently for final '
                           'cross-domain evaluation.',
            'seed':        SEED,
            'source':      'COD10K Train/Image partition',
            'n':           len(cod10k_holdout),
            'files':       cod10k_holdout,
        },
        splits_dir / 'hold_out_cod10k.json'
    )

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 3 — Noise hold-out placeholder (50 images)
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n[Step 3] Noise hold-out (50 NonCAM images from COD10K)")

    all_cod10k_train = get_image_filenames(COD10K_TRAIN_IMAGES)
    noncam_files = [f for f in all_cod10k_train if 'NonCAM' in f]

    # Make sure these aren't already in cod10k_holdout
    noncam_available = [f for f in noncam_files if f not in cod10k_holdout]

    set_seed(SEED)
    noise_holdout = sorted(random.sample(noncam_available,
                                         min(N_HOLDOUT_NOISE,
                                             len(noncam_available))))

    print(f"  Found {len(noncam_files)} NonCAM images, "
          f"sampled {len(noise_holdout)} for noise hold-out")

    save_json(
        {
            'description': 'Noise/distractor hold-out — 50 non-camouflaged '
                           'outdoor scenes from COD10K NonCAM category. '
                           'Ground-truth masks are all-black.',
            'seed': SEED,
            'source': 'COD10K Train/Image — NonCAM files',
            'n': len(noise_holdout),
            'files': noise_holdout,
            'status': 'COMPLETE',
        },
        splits_dir / 'hold_out_noise.json'
    )

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 4 — ACD1K train/val/test splits (from remaining after hold-out)
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n[Step 4] ACD1K train/val/test splits "
          f"({len(acd1k_remaining_train)} remaining training images)")

    acd1k_splits = generate_train_val_test(acd1k_remaining_train)

    save_json(
        {
            'description': 'ACD1K fixed train/val/test splits. '
                           'Generated from remaining training images '
                           'after hold-out removal. '
                           'Shared identically across Experiments 1, 2, 3.',
            'seed':        SEED,
            'ratios':      {'train': TRAIN_RATIO,
                            'val':   VAL_RATIO,
                            'test':  TEST_RATIO},
            'counts':      {k: len(v) for k, v in acd1k_splits.items()},
            'train':       acd1k_splits['train'],
            'val':         acd1k_splits['val'],
            'test':        acd1k_splits['test'],
            'note':        'Official Testing/images partition (330 images) '
                           'is separate and not included here.',
        },
        splits_dir / 'acd1k_splits.json'
    )

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 5 — COD10K train/val/test splits (from remaining after hold-out)
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n[Step 5] COD10K train/val/test splits")

    # Remove noise hold-out files from the training pool
    noise_set = set(noise_holdout)
    cod10k_clean = [f for f in cod10k_remaining_train if f not in noise_set]
    print(f"  After removing noise hold-out: {len(cod10k_clean)} images "
          f"({len(cod10k_remaining_train) - len(cod10k_clean)} noise removed)")

    cod10k_train_files = cod10k_clean
    cod10k_splits = generate_train_val_test(cod10k_train_files)

    save_json(
        {
            'description': 'COD10K fixed train/val/test splits. '
                           'Generated from Train/Image partition after '
                           'removing 50 hold-out images. '
                           'Shared identically across Experiments 2 and 3.',
            'seed': SEED,
            'ratios': {'train': TRAIN_RATIO,
                       'val': VAL_RATIO,
                       'test': TEST_RATIO},
            'counts': {k: len(v) for k, v in cod10k_splits.items()},
            'train': cod10k_splits['train'],
            'val': cod10k_splits['val'],
            'test': cod10k_splits['test'],
            'note': 'COD10K hold-out (50 images) was taken from '
                    'Train/Image partition before this split was generated.',
        },
        splits_dir / 'cod10k_splits.json'
    )

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 6 — Verify no overlaps
    # ─────────────────────────────────────────────────────────────────────────
    verify_no_overlap(splits_dir)

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 7 — Print final summary
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Split generation complete.")
    print("=" * 60)
    print()
    print("Final validation set (200 images total):")
    print(f"  ACD1K hold-out  : {len(acd1k_holdout):>4} images")
    print(f"  COD10K hold-out : {len(cod10k_holdout):>4} images")
    print(f"  Noise hold-out  : {N_HOLDOUT_NOISE:>4} images (populate manually)")
    print()
    print("ACD1K working splits (after hold-out):")
    print(f"  Train : {len(acd1k_splits['train']):>4} images")
    print(f"  Val   : {len(acd1k_splits['val']):>4} images")
    print(f"  Test  : {len(acd1k_splits['test']):>4} images")
    print()
    print("COD10K working splits:")
    print(f"  Train : {len(cod10k_splits['train']):>4} images")
    print(f"  Val   : {len(cod10k_splits['val']):>4} images")
    print(f"  Test  : {len(cod10k_splits['test']):>4} images")
    print()
    print("  1. Noise hold-out uses NonCAM images from COD10K — no action needed")


if __name__ == '__main__':
    main()