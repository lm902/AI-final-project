"""
dataset.py — Data loading and preprocessing pipeline
COMP 9130 Capstone: Military Camouflage Object Detection
Author: Binger Yu (Member 1 — Data & Preprocessing Lead)

Supports three experimental conditions:
  Condition 1: COD10K only      (animal camouflage pretraining)
  Condition 2: ACD1K only       (military camouflage target domain)
  Condition 3: COD10K+CAMO+ACD1K (joint training)
"""

import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ── Per-dataset normalization constants (computed in 01_EDA_Binger.ipynb) ──
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
    # Combined mean/std for joint training
    'JOINT': {
        'mean': (0.432, 0.430, 0.342),
        'std':  (0.200, 0.196, 0.188),
    },
}

# ── Image size required by SINet-V2 ──
INPUT_SIZE = 352


# ──────────────────────────────────────────────────────────────────────────────
# Augmentation pipelines
# ──────────────────────────────────────────────────────────────────────────────

def get_train_transforms(mean, std, input_size=INPUT_SIZE):
    """
    Training augmentation pipeline.
    - Geometric: flip + rotation (masks transformed identically)
    - Colour: jitter applied to image only, NOT mask
    - Resize → Normalize → ToTensor
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
    ])


def get_val_transforms(mean, std, input_size=INPUT_SIZE):
    """
    Validation / test pipeline — no augmentation, only resize + normalize.
    """
    return A.Compose([
        A.Resize(input_size, input_size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


# ──────────────────────────────────────────────────────────────────────────────
# Dataset class
# ──────────────────────────────────────────────────────────────────────────────

class CamouflageDataset(Dataset):
    """
    Generic camouflage segmentation dataset.

    Expects a flat folder of images and a flat folder of binary masks.
    Mask filenames must match image filenames (same stem, .png extension).

    Args:
        image_dir  (str): Path to folder containing images.
        mask_dir   (str): Path to folder containing binary masks.
        transform       : Albumentations transform pipeline.
        dataset_name(str): Label for this dataset ('COD10K', 'ACD1K', 'CAMO').
    """

    IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}

    def __init__(self, image_dir, mask_dir, transform=None, dataset_name=''):
        self.image_dir    = Path(image_dir)
        self.mask_dir     = Path(mask_dir)
        self.transform    = transform
        self.dataset_name = dataset_name

        # Collect all image files
        self.image_paths = sorted([
            p for p in self.image_dir.iterdir()
            if p.suffix.lower() in self.IMG_EXTS
        ])

        if len(self.image_paths) == 0:
            raise FileNotFoundError(
                f"No images found in {image_dir}. "
                f"Check that the path is correct."
            )

        # Verify at least one matching mask exists
        sample_mask = self._get_mask_path(self.image_paths[0])
        if not sample_mask.exists():
            raise FileNotFoundError(
                f"Mask not found for {self.image_paths[0].name}. "
                f"Expected: {sample_mask}"
            )

        print(f"[{dataset_name}] Loaded {len(self.image_paths)} images "
              f"from {image_dir}")

    def _get_mask_path(self, img_path):
        """Return the corresponding mask path for a given image path."""
        mask_path = self.mask_dir / (img_path.stem + '.png')
        if not mask_path.exists():
            # Some datasets use .jpg masks
            mask_path = self.mask_dir / (img_path.stem + '.jpg')
        return mask_path

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path  = self.image_paths[idx]
        mask_path = self._get_mask_path(img_path)

        # Load image as RGB
        image = np.array(Image.open(img_path).convert('RGB'))

        # Load mask as grayscale and binarize (0 or 1)
        if mask_path.exists():
            mask = np.array(Image.open(mask_path).convert('L'))
        else:
            # Return empty mask if not found (for non-camouflaged images)
            mask = np.zeros(
                (image.shape[0], image.shape[1]), dtype=np.uint8
            )

        # Binarize: pixels > 127 → 1, else → 0
        mask = (mask > 127).astype(np.uint8)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']          # Tensor [3, H, W]
            mask  = augmented['mask']           # Tensor [H, W]

        # Ensure mask is float for loss computation
        mask = mask.float().unsqueeze(0)        # [1, H, W]

        return {
            'image':    image,
            'mask':     mask,
            'filename': img_path.name,
            'dataset':  self.dataset_name,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Dataset factory functions
# ──────────────────────────────────────────────────────────────────────────────

def build_dataset(data_root, condition='joint', split='train'):
    """
    Build a dataset (or concatenated dataset) for a given experimental condition.

    Args:
        data_root (str): Root directory containing all dataset folders.
        condition (str): One of 'cod10k', 'acd1k', 'joint'.
        split     (str): 'train' or 'test'.

    Returns:
        dataset: A CamouflageDataset or ConcatDataset.

    Directory structure expected under data_root:
        data_root/
        ├── COD10K-v3/
        │   ├── Train/Image/
        │   ├── Train/GT_Object/
        │   ├── Test/Image/
        │   └── Test/GT_Object/
        ├── dataset-splitM/
        │   ├── Training/images/
        │   ├── Training/GT/
        │   ├── Testing/images/
        │   └── Testing/GT/
        └── CAMO-V.1.0-CVIU2019/
            ├── Images/Train/
            ├── Images/Test/
            └── GT/
    """
    data_root = Path(data_root)
    condition = condition.lower()

    # ── Path definitions ──
    DATASET_PATHS = {
        'COD10K': {
            'train': {
                'images': data_root / 'COD10K-v3/Train/Image',
                'masks':  data_root / 'COD10K-v3/Train/GT_Object',
            },
            'test': {
                'images': data_root / 'COD10K-v3/Test/Image',
                'masks':  data_root / 'COD10K-v3/Test/GT_Object',
            },
        },
        'ACD1K': {
            'train': {
                'images': data_root / 'dataset-splitM/Training/images',
                'masks':  data_root / 'dataset-splitM/Training/GT',
            },
            'test': {
                'images': data_root / 'dataset-splitM/Testing/images',
                'masks':  data_root / 'dataset-splitM/Testing/GT',
            },
        },
        'CAMO': {
            'train': {
                'images': data_root / 'CAMO-V.1.0-CVIU2019/Images/Train',
                'masks':  data_root / 'CAMO-V.1.0-CVIU2019/GT',
            },
            'test': {
                'images': data_root / 'CAMO-V.1.0-CVIU2019/Images/Test',
                'masks':  data_root / 'CAMO-V.1.0-CVIU2019/GT',
            },
        },
    }

    # ── Select normalization stats ──
    if condition == 'joint':
        stats = DATASET_STATS['JOINT']
    elif condition == 'cod10k':
        stats = DATASET_STATS['COD10K']
    elif condition == 'acd1k':
        stats = DATASET_STATS['ACD1K']
    else:
        raise ValueError(
            f"Unknown condition '{condition}'. "
            f"Choose from: 'cod10k', 'acd1k', 'joint'."
        )

    # ── Select transforms ──
    if split == 'train':
        transform = get_train_transforms(stats['mean'], stats['std'])
    else:
        transform = get_val_transforms(stats['mean'], stats['std'])

    # ── Build dataset(s) ──
    if condition == 'cod10k':
        paths = DATASET_PATHS['COD10K'][split]
        return CamouflageDataset(
            paths['images'], paths['masks'],
            transform=transform, dataset_name='COD10K'
        )

    elif condition == 'acd1k':
        paths = DATASET_PATHS['ACD1K'][split]
        return CamouflageDataset(
            paths['images'], paths['masks'],
            transform=transform, dataset_name='ACD1K'
        )

    elif condition == 'joint':
        datasets = []
        for ds_name in ['COD10K', 'CAMO', 'ACD1K']:
            paths = DATASET_PATHS[ds_name][split]
            ds = CamouflageDataset(
                paths['images'], paths['masks'],
                transform=transform, dataset_name=ds_name
            )
            datasets.append(ds)
        return torch.utils.data.ConcatDataset(datasets)


def build_dataloader(data_root, condition='joint', split='train',
                     batch_size=16, num_workers=4,
                     oversample_acd1k=True, seed=42):
    """
    Build a DataLoader for a given condition and split.

    Args:
        data_root      (str):  Root data directory.
        condition      (str):  'cod10k', 'acd1k', or 'joint'.
        split          (str):  'train' or 'test'.
        batch_size     (int):  Batch size.
        num_workers    (int):  DataLoader worker threads.
        oversample_acd1k(bool): If True and condition=='joint', use
                                WeightedRandomSampler to balance
                                ACD1K against COD10K (~8:1 imbalance).
        seed           (int):  Random seed for reproducibility.

    Returns:
        DataLoader
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    dataset = build_dataset(data_root, condition=condition, split=split)
    shuffle = (split == 'train')

    # ── Oversampling for joint training ──
    sampler = None
    if condition == 'joint' and split == 'train' and oversample_acd1k:
        sampler = _build_weighted_sampler(dataset)
        shuffle = False  # shuffle=True is incompatible with sampler

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train'),
    )

    total = len(dataset)
    print(f"[DataLoader] condition={condition} split={split} "
          f"samples={total} batch_size={batch_size} "
          f"batches={len(loader)} oversample={oversample_acd1k}")

    return loader


def _build_weighted_sampler(concat_dataset):
    """
    Build a WeightedRandomSampler that upsamples ACD1K images 8x
    to compensate for the COD10K:ACD1K imbalance (~8:1 ratio).

    Weight assignment:
      COD10K images → weight 1.0
      CAMO   images → weight 1.0
      ACD1K  images → weight 8.0
    """
    weights = []
    for ds in concat_dataset.datasets:
        if ds.dataset_name == 'ACD1K':
            weights.extend([8.0] * len(ds))
        else:
            weights.extend([1.0] * len(ds))

    weights = torch.DoubleTensor(weights)
    sampler = WeightedRandomSampler(
        weights,
        num_samples=len(weights),
        replacement=True,
    )
    print(f"[Sampler] ACD1K weight=8.0, others=1.0 "
          f"(total samples per epoch: {len(weights)})")
    return sampler


# ──────────────────────────────────────────────────────────────────────────────
# Quick verification
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python dataset.py <data_root>")
        print("Example: python dataset.py /path/to/data")
        sys.exit(1)

    DATA_ROOT = sys.argv[1]

    print("=" * 60)
    print("Verifying all three dataset conditions...")
    print("=" * 60)

    for condition in ['cod10k', 'acd1k', 'joint']:
        print(f"\n--- Condition: {condition.upper()} ---")
        try:
            loader = build_dataloader(
                DATA_ROOT,
                condition=condition,
                split='train',
                batch_size=4,
                num_workers=0,
                oversample_acd1k=(condition == 'joint'),
            )
            batch = next(iter(loader))
            print(f"  Image tensor shape : {batch['image'].shape}")
            print(f"  Mask  tensor shape : {batch['mask'].shape}")
            print(f"  Image dtype        : {batch['image'].dtype}")
            print(f"  Mask  dtype        : {batch['mask'].dtype}")
            print(f"  Mask  unique values: {batch['mask'].unique().tolist()}")
            print(f"  Datasets in batch  : {set(batch['dataset'])}")
            print(f"  ✅ {condition.upper()} condition OK")
        except Exception as e:
            print(f"  ❌ Error: {e}")

    print("\n" + "=" * 60)
    print("Verification complete.")
    print("=" * 60)
