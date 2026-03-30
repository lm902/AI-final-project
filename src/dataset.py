"""
dataset.py — Data loading and preprocessing pipeline
COMP 9130 Capstone: Military Camouflage Object Detection
Author: Binger Yu (Data & Preprocessing Lead)

Supports three dataset conditions used across three experiments:

  Condition 'acd1k'  : ACD1K only
      → Experiment 1 (SINetV2 CNN baseline trained on military data only)
      → Experiment 2 Stage 2 (ACD1K fine-tuning after COD10K pretraining)

  Condition 'cod10k' : COD10K only
      → Experiment 2 Stage 1 (COD10K pretraining for SegFormer-B2)

  Condition 'joint'  : COD10K + CAMO + ACD1K
      → Experiment 3 (SegFormer-B2 joint training across all three datasets)

Note on input resolution:
  INPUT_SIZE = 512 is the standard resolution for SegFormer-B2
  (matches nvidia/segformer-b2-finetuned-ade-512-512).
  Experiment 1 (SINetV2) uses 352×352 — override INPUT_SIZE when calling
  get_train_transforms() / get_val_transforms() for that experiment.

Split strategy (V3 — official splits):
  train → full official train partition
          (COD10K: minus 50 noise hold-out images, controlled via JSON index)
  val   → official test partition minus hold-out images
  final evaluation → use build_holdout_dataset() on 200-image hold-out set

Split index files (generated once by src/generate_splits.py, fixed seed):
  splits/acd1k_splits.json     — ACD1K train/val filenames
  splits/cod10k_splits.json    — COD10K train/val filenames (noise excluded)
  splits/hold_out_acd1k.json   — 100 ACD1K hold-out filenames
  splits/hold_out_cod10k.json  — 50 COD10K hold-out filenames
  splits/hold_out_noise.json   — 50 noise distractor hold-out filenames
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ── Per-dataset normalization constants ──────────────────────────────────────
# Computed from training splits in 01_EDA_Binger.ipynb.
# 'JOINT' stats are the weighted average across COD10K + CAMO + ACD1K
# and are used for all three datasets in the joint training condition.
DATASET_STATS = {
    'COD10K': {
        'mean': (0.407, 0.424, 0.340),
        'std':  (0.208, 0.198, 0.194),
    },
    'ACD1K': {
        'mean': (0.411, 0.405, 0.327),
        'std':  (0.196, 0.196, 0.184),
    },
    'CAMO': {
        'mean': (0.479, 0.461, 0.360),
        'std':  (0.197, 0.195, 0.185),
    },
    'JOINT': {
        'mean': (0.432, 0.430, 0.342),
        'std':  (0.200, 0.196, 0.188),
    },
}

# Standard input resolution for SegFormer-B2 (Experiments 2 and 3).
# For Experiment 1 (SINetV2), pass input_size=352 explicitly to the
# transform builders below.
INPUT_SIZE = 512


# ──────────────────────────────────────────────────────────────────────────────
# JSON split loaders
# ──────────────────────────────────────────────────────────────────────────────

def load_split_filenames(splits_dir, dataset, split):
    """
    Load filenames for a given dataset and split from a JSON index file.

    Index files are generated once by src/generate_splits.py with a fixed
    random seed to ensure identical splits are used across all experiments.

    Args:
        splits_dir (str): Path to splits/ folder.
        dataset    (str): 'acd1k' or 'cod10k'.
        split      (str): 'train' or 'val'.

    Returns:
        list of filename strings (not full paths).

    Raises:
        FileNotFoundError: If the JSON index file does not exist.
        KeyError: If the requested split key is absent from the JSON.
    """
    json_path = Path(splits_dir) / f'{dataset.lower()}_splits.json'

    if not json_path.exists():
        raise FileNotFoundError(
            f"Split file not found: {json_path}\n"
            f"Run src/generate_splits.py first."
        )

    with open(json_path) as f:
        data = json.load(f)

    if split not in data:
        raise KeyError(
            f"Split '{split}' not found in {json_path}. "
            f"Available keys: {list(data.keys())}"
        )

    filenames = data[split]
    print(f"  [Splits] {dataset.upper()} {split}: {len(filenames)} images "
          f"(from {json_path.name})")
    return filenames


def load_holdout_filenames(splits_dir, holdout_name):
    """
    Load filenames for a hold-out set from a JSON index file.

    Hold-out index files are committed to the repository before any training
    begins and are never modified, ensuring the final test set is fixed across
    all experiments.

    Args:
        splits_dir   (str): Path to splits/ folder.
        holdout_name (str): 'acd1k' (100 images), 'cod10k' (50 images),
                            or 'noise' (50 distractor images).

    Returns:
        list of filename strings (not full paths).
    """
    json_path = Path(splits_dir) / f'hold_out_{holdout_name.lower()}.json'

    if not json_path.exists():
        raise FileNotFoundError(
            f"Hold-out file not found: {json_path}\n"
            f"Run src/generate_splits.py first."
        )

    with open(json_path) as f:
        data = json.load(f)

    filenames = data.get('files', [])
    print(f"  [Hold-out] {holdout_name.upper()}: {len(filenames)} images")
    return filenames


def get_train_files(splits_dir, dataset_name):
    """Load train filenames from the JSON split index."""
    return load_split_filenames(splits_dir, dataset_name.lower(), 'train')


def get_val_files(splits_dir, dataset_name):
    """Load val filenames from the JSON split index."""
    return load_split_filenames(splits_dir, dataset_name.lower(), 'val')


# ──────────────────────────────────────────────────────────────────────────────
# Augmentation pipelines
# ──────────────────────────────────────────────────────────────────────────────

def get_train_transforms(mean, std, input_size=INPUT_SIZE):
    """
    Training augmentation pipeline using Albumentations.

    Geometric transforms (horizontal flip, rotation) are applied identically
    to both the image and its corresponding segmentation mask to preserve
    annotation alignment. Colour jitter is applied to the image only.

    Args:
        mean       (tuple): Per-channel mean for normalization.
        std        (tuple): Per-channel std for normalization.
        input_size (int)  : Resize target. Default 512 (SegFormer-B2).
                            Pass 352 for Experiment 1 (SINetV2).

    Returns:
        Albumentations Compose transform.
    """
    return A.Compose([
        A.Resize(input_size, input_size),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.4),
        A.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.1,
            hue=0.05,
            p=0.4
        ),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ], is_check_shapes=False)


def get_val_transforms(mean, std, input_size=INPUT_SIZE):
    """
    Validation and test transform pipeline.

    Resize and normalize only — no augmentation — to ensure fair and
    reproducible evaluation across all experiments.

    Args:
        mean       (tuple): Per-channel mean for normalization.
        std        (tuple): Per-channel std for normalization.
        input_size (int)  : Resize target. Default 512 (SegFormer-B2).
                            Pass 352 for Experiment 1 (SINetV2).

    Returns:
        Albumentations Compose transform.
    """
    return A.Compose([
        A.Resize(input_size, input_size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ], is_check_shapes=False)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset class
# ──────────────────────────────────────────────────────────────────────────────

class CamouflageDataset(Dataset):
    """
    Generic camouflage segmentation dataset for binary mask prediction.

    Supports two loading modes:
      1. Split-file mode  : loads only the filenames listed in a JSON index.
                            Used for train/val splits to ensure identical
                            image sets across all experiments.
      2. Folder-scan mode : loads all images found in image_dir.
                            Used for datasets with no excluded images
                            (e.g., CAMO full official train/test).

    All segmentation masks are binarized at threshold 127:
      pixel > 127 → 1 (foreground), otherwise → 0 (background).
    If no mask file is found for an image, an all-zero mask is returned,
    allowing noise distractor images (with no camouflage target) to be
    loaded without special-casing.

    Args:
        image_dir    (str)        : Path to folder containing RGB images.
        mask_dir     (str)        : Path to folder containing binary masks.
        transform                 : Albumentations transform pipeline.
        dataset_name (str)        : Label for logging, e.g. 'COD10K'.
        file_list    (list | None): Filenames to load. None = scan folder.
    """

    IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}

    def __init__(self, image_dir, mask_dir, transform=None,
                 dataset_name='', file_list=None):
        self.image_dir    = Path(image_dir)
        self.mask_dir     = Path(mask_dir)
        self.transform    = transform
        self.dataset_name = dataset_name

        if file_list is not None:
            # ── Split-file mode: load only the filenames in the index ──
            self.image_paths = []
            missing = []
            for fname in file_list:
                p = self.image_dir / fname
                if p.exists():
                    self.image_paths.append(p)
                else:
                    missing.append(fname)
            if missing:
                print(f"  ⚠️  [{dataset_name}] {len(missing)} listed files "
                      f"not found on disk: "
                      f"{missing[:3]}{'...' if len(missing) > 3 else ''}")
        else:
            # ── Folder-scan mode: load all images in image_dir ──
            self.image_paths = sorted([
                p for p in self.image_dir.iterdir()
                if p.suffix.lower() in self.IMG_EXTS
            ])

        if len(self.image_paths) == 0:
            raise FileNotFoundError(
                f"No images loaded for [{dataset_name}] from {image_dir}."
            )

        # Verify at least one corresponding mask exists on disk
        sample_mask = self._get_mask_path(self.image_paths[0])
        if not sample_mask.exists():
            raise FileNotFoundError(
                f"Mask not found for first image: {sample_mask}"
            )

        mode = "split-file" if file_list is not None else "folder-scan"
        print(f"  [{dataset_name}] {len(self.image_paths)} images "
              f"({mode} mode)")

    def _get_mask_path(self, img_path):
        """
        Return the mask path corresponding to an image path.
        Tries .png first, falls back to .jpg.
        """
        mask_path = self.mask_dir / (img_path.stem + '.png')
        if not mask_path.exists():
            mask_path = self.mask_dir / (img_path.stem + '.jpg')
        return mask_path

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path  = self.image_paths[idx]
        mask_path = self._get_mask_path(img_path)

        image = np.array(Image.open(img_path).convert('RGB'))

        if mask_path.exists():
            mask = np.array(Image.open(mask_path).convert('L'))
        else:
            # No mask file: return all-zero mask (background only).
            # Used for noise distractor images in the final hold-out set.
            mask = np.zeros(
                (image.shape[0], image.shape[1]), dtype=np.uint8
            )

        # Binarize: pixel > 127 → 1 (foreground), else → 0 (background)
        mask = (mask > 127).astype(np.uint8)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask  = augmented['mask']

        # Add channel dimension: [H, W] → [1, H, W] for BCEWithLogitsLoss
        mask = mask.float().unsqueeze(0)

        return {
            'image':    image,           # [3, H, W] float tensor
            'mask':     mask,            # [1, H, W] binary float tensor
            'filename': img_path.name,
            'dataset':  self.dataset_name,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Dataset factory
# ──────────────────────────────────────────────────────────────────────────────

def build_dataset(data_root, condition='joint', split='train',
                  splits_dir='splits'):
    """
    Build a dataset for a given experimental condition and split.

    Condition-to-experiment mapping:
      'acd1k'  → Experiment 1 (SINetV2 baseline, ACD1K only)
                 Experiment 2 Stage 2 (ACD1K fine-tuning)
      'cod10k' → Experiment 2 Stage 1 (COD10K pretraining for SegFormer-B2)
      'joint'  → Experiment 3 (SegFormer-B2 joint training,
                                COD10K + CAMO + ACD1K)

    Args:
        data_root  (str): Root directory containing all dataset folders.
        condition  (str): 'acd1k', 'cod10k', or 'joint'.
        split      (str): 'train' or 'val'.
        splits_dir (str): Path to splits/ folder with JSON index files.

    Returns:
        CamouflageDataset (single dataset) or ConcatDataset (joint condition).
    """
    data_root  = Path(data_root)
    splits_dir = Path(splits_dir)
    condition  = condition.lower()

    # ── Dataset path definitions ──────────────────────────────────────────
    PATHS = {
        'COD10K': {
            'train_images': data_root / 'COD10K-v3/Train/Image',
            'train_masks':  data_root / 'COD10K-v3/Train/GT_Object',
            'val_images':   data_root / 'COD10K-v3/Test/Image',
            'val_masks':    data_root / 'COD10K-v3/Test/GT_Object',
        },
        'ACD1K': {
            'train_images': data_root / 'dataset-splitM/Training/images',
            'train_masks':  data_root / 'dataset-splitM/Training/GT',
            'val_images':   data_root / 'dataset-splitM/Testing/images',
            'val_masks':    data_root / 'dataset-splitM/Testing/GT',
        },
        'CAMO': {
            'train_images': data_root / 'CAMO-V.1.0-CVIU2019/Images/Train',
            'train_masks':  data_root / 'CAMO-V.1.0-CVIU2019/GT',
            'val_images':   data_root / 'CAMO-V.1.0-CVIU2019/Images/Test',
            'val_masks':    data_root / 'CAMO-V.1.0-CVIU2019/GT',
        },
    }

    # ── Normalization stats ───────────────────────────────────────────────
    # Joint condition uses pooled stats across all three datasets.
    stats_key = 'JOINT' if condition == 'joint' else condition.upper()
    stats     = DATASET_STATS[stats_key]

    # ── Transforms ───────────────────────────────────────────────────────
    # INPUT_SIZE=512 (SegFormer-B2 default). For Experiment 1 (SINetV2),
    # call get_train_transforms() directly with input_size=352.
    transform = (get_train_transforms(stats['mean'], stats['std'])
                 if split == 'train'
                 else get_val_transforms(stats['mean'], stats['std']))

    # ── ACD1K only ────────────────────────────────────────────────────────
    # Used by: Experiment 1 (SINetV2 baseline)
    #          Experiment 2 Stage 2 (ACD1K fine-tuning)
    if condition == 'acd1k':
        if split == 'train':
            # Full official train partition; no images excluded.
            return CamouflageDataset(
                PATHS['ACD1K']['train_images'],
                PATHS['ACD1K']['train_masks'],
                transform=transform,
                dataset_name='ACD1K',
                file_list=None,
            )
        else:  # val
            # Official test partition minus 100 hold-out images (JSON index).
            return CamouflageDataset(
                PATHS['ACD1K']['val_images'],
                PATHS['ACD1K']['val_masks'],
                transform=transform,
                dataset_name='ACD1K',
                file_list=get_val_files(splits_dir, 'acd1k'),
            )

    # ── COD10K only ───────────────────────────────────────────────────────
    # Used by: Experiment 2 Stage 1 (COD10K pretraining for SegFormer-B2)
    elif condition == 'cod10k':
        if split == 'train':
            # JSON train index excludes the 50 noise distractor images
            # that are reserved for the final hold-out set.
            return CamouflageDataset(
                PATHS['COD10K']['train_images'],
                PATHS['COD10K']['train_masks'],
                transform=transform,
                dataset_name='COD10K',
                file_list=get_train_files(splits_dir, 'cod10k'),
            )
        else:  # val
            # Official test partition minus 50 hold-out images (JSON index).
            return CamouflageDataset(
                PATHS['COD10K']['val_images'],
                PATHS['COD10K']['val_masks'],
                transform=transform,
                dataset_name='COD10K',
                file_list=get_val_files(splits_dir, 'cod10k'),
            )

    # ── Joint training ────────────────────────────────────────────────────
    # Used by: Experiment 3 (SegFormer-B2, COD10K + CAMO + ACD1K)
    # Weighted oversampling (ACD1K weight=8.0) is applied at the DataLoader
    # level in build_dataloader() to compensate for the ~8:1 imbalance
    # between COD10K (6,000 train) and ACD1K (748 train).
    elif condition == 'joint':
        datasets = []

        if split == 'train':
            # COD10K: JSON train index (50 noise distractors excluded)
            datasets.append(CamouflageDataset(
                PATHS['COD10K']['train_images'],
                PATHS['COD10K']['train_masks'],
                transform=transform,
                dataset_name='COD10K',
                file_list=get_train_files(splits_dir, 'cod10k'),
            ))
            # CAMO: full official train partition, no exclusions needed
            datasets.append(CamouflageDataset(
                PATHS['CAMO']['train_images'],
                PATHS['CAMO']['train_masks'],
                transform=transform,
                dataset_name='CAMO',
                file_list=None,
            ))
            # ACD1K: full official train partition, no exclusions needed
            datasets.append(CamouflageDataset(
                PATHS['ACD1K']['train_images'],
                PATHS['ACD1K']['train_masks'],
                transform=transform,
                dataset_name='ACD1K',
                file_list=None,
            ))

        else:  # val
            # COD10K: official test minus 50 hold-out images (JSON index)
            datasets.append(CamouflageDataset(
                PATHS['COD10K']['val_images'],
                PATHS['COD10K']['val_masks'],
                transform=transform,
                dataset_name='COD10K',
                file_list=get_val_files(splits_dir, 'cod10k'),
            ))
            # CAMO: full official test partition, no exclusions needed
            datasets.append(CamouflageDataset(
                PATHS['CAMO']['val_images'],
                PATHS['CAMO']['val_masks'],
                transform=transform,
                dataset_name='CAMO',
                file_list=None,
            ))
            # ACD1K: official test minus 100 hold-out images (JSON index)
            datasets.append(CamouflageDataset(
                PATHS['ACD1K']['val_images'],
                PATHS['ACD1K']['val_masks'],
                transform=transform,
                dataset_name='ACD1K',
                file_list=get_val_files(splits_dir, 'acd1k'),
            ))

        return torch.utils.data.ConcatDataset(datasets)

    else:
        raise ValueError(
            f"Unknown condition '{condition}'. "
            f"Choose from: 'acd1k', 'cod10k', 'joint'."
        )


def build_holdout_dataset(data_root, holdout_name, splits_dir='splits'):
    """
    Build a dataset from a hold-out JSON index for final evaluation.

    The 200-image final test set consists of:
      - 100 ACD1K images  (military camouflage, terrain-stratified)
      - 50  COD10K images (animal camouflage, super-class stratified)
      - 50  noise images  (ordinary outdoor scenes, all-black GT masks)

    These indices are fixed before any training begins and never modified.
    No augmentation is applied — resize and normalize only.

    Args:
        data_root    (str): Root data directory.
        holdout_name (str): 'acd1k', 'cod10k', or 'noise'.
        splits_dir   (str): Path to splits/ folder.

    Returns:
        CamouflageDataset with val transforms applied.
    """
    data_root = Path(data_root)
    file_list = load_holdout_filenames(splits_dir, holdout_name)

    if holdout_name == 'acd1k':
        image_dir = data_root / 'dataset-splitM/Testing/images'
        mask_dir  = data_root / 'dataset-splitM/Testing/GT'
        stats     = DATASET_STATS['ACD1K']
    elif holdout_name == 'cod10k':
        image_dir = data_root / 'COD10K-v3/Test/Image'
        mask_dir  = data_root / 'COD10K-v3/Test/GT_Object'
        stats     = DATASET_STATS['COD10K']
    elif holdout_name == 'noise':
        # Noise distractors are sourced from the COD10K NonCAM (non-camouflaged)
        # subset of the train partition. GT masks are all-zero (no foreground).
        image_dir = data_root / 'COD10K-v3/Train/Image'
        mask_dir  = data_root / 'COD10K-v3/Train/GT_Object'
        stats     = DATASET_STATS['COD10K']
    else:
        raise ValueError(
            f"Unknown holdout_name '{holdout_name}'. "
            f"Choose from: 'acd1k', 'cod10k', 'noise'."
        )

    transform = get_val_transforms(stats['mean'], stats['std'])

    return CamouflageDataset(
        image_dir, mask_dir,
        transform=transform,
        dataset_name=f'HOLDOUT_{holdout_name.upper()}',
        file_list=file_list,
    )


# ──────────────────────────────────────────────────────────────────────────────
# DataLoader factory
# ──────────────────────────────────────────────────────────────────────────────

def build_dataloader(data_root, condition='joint', split='train',
                     batch_size=16, num_workers=4,
                     oversample_acd1k=True, seed=42,
                     splits_dir='splits'):
    """
    Build a DataLoader for a given condition and split.

    For the joint training condition (Experiment 3), WeightedRandomSampler
    is applied to the train split to oversample ACD1K images 8x, compensating
    for the ~8:1 size imbalance between COD10K (6,000 train) and ACD1K
    (748 train). This ensures ACD1K images appear approximately once per
    COD10K image on average within each epoch, preventing the model from
    ignoring the military target domain.

    Args:
        data_root       (str) : Root data directory.
        condition       (str) : 'acd1k', 'cod10k', or 'joint'.
        split           (str) : 'train' or 'val'.
        batch_size      (int) : Batch size.
        num_workers     (int) : DataLoader worker processes.
        oversample_acd1k(bool): Apply 8x ACD1K oversampling in joint train.
                                Has no effect for non-joint conditions.
        seed            (int) : Random seed for reproducibility.
        splits_dir      (str) : Path to splits/ folder.

    Returns:
        torch.utils.data.DataLoader
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    dataset = build_dataset(data_root, condition=condition,
                            split=split, splits_dir=splits_dir)
    shuffle = (split == 'train')
    sampler = None

    if condition == 'joint' and split == 'train' and oversample_acd1k:
        sampler = _build_weighted_sampler(dataset)
        shuffle = False  # shuffle and sampler are mutually exclusive

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=(split == 'train'),  # avoid incomplete batches during training
    )

    print(f"  [DataLoader] condition={condition} split={split} "
          f"samples={len(dataset)} batches={len(loader)}")
    return loader


def _build_weighted_sampler(concat_dataset):
    """
    Build a WeightedRandomSampler that upsamples ACD1K 8x.

    Weight assignment:
      ACD1K images → 8.0  (upsampled to match COD10K frequency)
      COD10K images → 1.0
      CAMO images  → 1.0

    Applied only to the joint training condition (Experiment 3).
    """
    weights = []
    for ds in concat_dataset.datasets:
        w = 8.0 if ds.dataset_name == 'ACD1K' else 1.0
        weights.extend([w] * len(ds))

    weights = torch.DoubleTensor(weights)
    sampler = WeightedRandomSampler(
        weights, num_samples=len(weights), replacement=True
    )
    print(f"  [Sampler] ACD1K weight=8.0, COD10K/CAMO weight=1.0")
    return sampler


# ──────────────────────────────────────────────────────────────────────────────
# Verification script
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python dataset.py <data_root> [splits_dir]")
        print("Example: python dataset.py data/ splits/")
        sys.exit(1)

    DATA_ROOT  = sys.argv[1]
    SPLITS_DIR = sys.argv[2] if len(sys.argv) > 2 else 'splits'

    print("=" * 60)
    print("Verifying all conditions and splits...")
    print("=" * 60)

    # ── Train/val splits ──────────────────────────────────────────────────
    for condition in ['acd1k', 'cod10k', 'joint']:
        for split in ['train', 'val']:
            print(f"\n--- {condition.upper()} / {split} ---")
            try:
                loader = build_dataloader(
                    DATA_ROOT,
                    condition=condition,
                    split=split,
                    batch_size=4,
                    num_workers=0,
                    oversample_acd1k=(condition == 'joint'
                                      and split == 'train'),
                    splits_dir=SPLITS_DIR,
                )
                batch = next(iter(loader))
                print(f"  image : {batch['image'].shape}  "
                      f"mask  : {batch['mask'].shape}  "
                      f"values: {batch['mask'].unique().tolist()}  "
                      f"datasets: {set(batch['dataset'])}")
                print(f"  ✅ OK")
            except Exception as e:
                print(f"  ❌ Error: {e}")

    # ── Hold-out sets ──────────────────────────────────────────────────────
    print("\n--- Hold-out sets (200-image final test set) ---")
    for name in ['acd1k', 'cod10k', 'noise']:
        try:
            ds = build_holdout_dataset(DATA_ROOT, name,
                                       splits_dir=SPLITS_DIR)
            print(f"  ✅ {name.upper()} hold-out: {len(ds)} images")
        except Exception as e:
            print(f"  ❌ {name.upper()} hold-out error: {e}")

    print("\n" + "=" * 60)
    print("Verification complete.")
    print("=" * 60)