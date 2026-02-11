"""
Utility script to prepare data for TNBC segmentation training

Functions:
- Create annotation masks from pathologist annotations
- Split dataset into train/val
- Verify data integrity
- Generate dataset statistics
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import json
import argparse
from tqdm import tqdm
import shutil


def create_dummy_masks(image_dir, mask_dir, num_classes=4):
    """
    Create dummy segmentation masks for testing
    (Replace with actual annotation processing)
    
    Args:
        image_dir: Directory containing H&E images
        mask_dir: Directory to save masks
        num_classes: Number of segmentation classes
    """
    mask_dir.mkdir(parents=True, exist_ok=True)
    
    image_paths = list(Path(image_dir).glob('*.jpeg')) + \
                  list(Path(image_dir).glob('*.jpg'))
    
    print(f"Creating dummy masks for {len(image_paths)} images...")
    
    for img_path in tqdm(image_paths):
        # Read image to get dimensions
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        
        # Create random mask (replace with actual segmentation)
        mask = np.random.randint(0, num_classes, (h, w), dtype=np.uint8)
        
        # Save mask
        mask_path = mask_dir / f"{img_path.stem}.png"
        cv2.imwrite(str(mask_path), mask)
    
    print(f"Created {len(image_paths)} masks in {mask_dir}")


def verify_dataset(data_dir):
    """
    Verify that dataset is properly structured
    
    Expected structure:
    data_dir/
        images/
            *.jpeg
        masks/
            *.png
    """
    data_dir = Path(data_dir)
    image_dir = data_dir / 'images'
    mask_dir = data_dir / 'masks'
    
    print("Verifying dataset structure...")
    
    # Check directories exist
    assert image_dir.exists(), f"Images directory not found: {image_dir}"
    assert mask_dir.exists(), f"Masks directory not found: {mask_dir}"
    
    # Get images and masks
    image_paths = sorted(list(image_dir.glob('*.jpeg')) + list(image_dir.glob('*.jpg')))
    mask_paths = sorted(list(mask_dir.glob('*.png')))
    
    print(f"Found {len(image_paths)} images")
    print(f"Found {len(mask_paths)} masks")
    
    # Check matching
    missing_masks = []
    for img_path in image_paths:
        mask_path = mask_dir / f"{img_path.stem}.png"
        if not mask_path.exists():
            missing_masks.append(img_path.name)
    
    if missing_masks:
        print(f"WARNING: {len(missing_masks)} images missing masks:")
        for name in missing_masks[:10]:
            print(f"  - {name}")
        if len(missing_masks) > 10:
            print(f"  ... and {len(missing_masks) - 10} more")
    else:
        print("✓ All images have corresponding masks")
    
    # Check mask values
    print("\nChecking mask values...")
    unique_values = set()
    for mask_path in tqdm(mask_paths[:100], desc="Sampling masks"):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        unique_values.update(np.unique(mask).tolist())
    
    print(f"Unique mask values found: {sorted(unique_values)}")
    print(f"Expected values: [0, 1, 2, 3] (background, tumor, immune, stroma)")
    
    return len(image_paths), len(mask_paths)


def compute_dataset_statistics(data_dir):
    """
    Compute statistics for the dataset
    """
    data_dir = Path(data_dir)
    image_dir = data_dir / 'images'
    mask_dir = data_dir / 'masks'
    
    image_paths = list(image_dir.glob('*.jpeg')) + list(image_dir.glob('*.jpg'))
    
    print("Computing dataset statistics...")
    
    # Image statistics
    sizes = []
    for img_path in tqdm(image_paths[:100], desc="Analyzing images"):
        img = cv2.imread(str(img_path))
        sizes.append(img.shape[:2])
    
    sizes = np.array(sizes)
    print(f"\nImage dimensions:")
    print(f"  Mean: {sizes.mean(axis=0)}")
    print(f"  Std: {sizes.std(axis=0)}")
    print(f"  Min: {sizes.min(axis=0)}")
    print(f"  Max: {sizes.max(axis=0)}")
    
    # Class distribution
    class_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    mask_paths = list(mask_dir.glob('*.png'))
    
    for mask_path in tqdm(mask_paths[:100], desc="Analyzing masks"):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        for class_id in range(4):
            class_counts[class_id] += (mask == class_id).sum()
    
    total_pixels = sum(class_counts.values())
    print(f"\nClass distribution (sampled):")
    class_names = ['Background', 'Tumor', 'Immune', 'Stroma']
    for class_id, name in enumerate(class_names):
        count = class_counts[class_id]
        percentage = (count / total_pixels) * 100
        print(f"  {name}: {percentage:.2f}%")


def create_dataset_split(data_dir, output_file, train_ratio=0.8):
    """
    Create train/val split and save to JSON
    
    Args:
        data_dir: Root directory with images/ and masks/
        output_file: Path to save split information
        train_ratio: Ratio of training data
    """
    data_dir = Path(data_dir)
    image_dir = data_dir / 'images'
    
    image_paths = sorted(list(image_dir.glob('*.jpeg')) + list(image_dir.glob('*.jpg')))
    
    # Random split
    np.random.seed(42)
    indices = np.random.permutation(len(image_paths))
    split_idx = int(train_ratio * len(indices))
    
    train_images = [image_paths[i].name for i in indices[:split_idx]]
    val_images = [image_paths[i].name for i in indices[split_idx:]]
    
    split_info = {
        'train': train_images,
        'val': val_images,
        'train_count': len(train_images),
        'val_count': len(val_images),
        'train_ratio': train_ratio
    }
    
    with open(output_file, 'w') as f:
        json.dump(split_info, f, indent=4)
    
    print(f"Created dataset split:")
    print(f"  Train: {len(train_images)} images")
    print(f"  Val: {len(val_images)} images")
    print(f"  Saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare TNBC dataset")
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Root directory containing images/ and masks/')
    parser.add_argument('--action', type=str, required=True,
                       choices=['verify', 'stats', 'split', 'create_dummy'],
                       help='Action to perform')
    parser.add_argument('--output', type=str, default='dataset_split.json',
                       help='Output file for split (used with --action split)')
    
    args = parser.parse_args()
    
    if args.action == 'verify':
        verify_dataset(args.data_dir)
    elif args.action == 'stats':
        compute_dataset_statistics(args.data_dir)
    elif args.action == 'split':
        create_dataset_split(args.data_dir, args.output)
    elif args.action == 'create_dummy':
        image_dir = Path(args.data_dir) / 'images'
        mask_dir = Path(args.data_dir) / 'masks'
        create_dummy_masks(image_dir, mask_dir)
