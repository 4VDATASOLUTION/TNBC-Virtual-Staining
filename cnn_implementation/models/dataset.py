"""
Dataset classes for TNBC tissue segmentation and PD-L1 quantification

Handles:
- H&E histopathology images
- Segmentation masks (tumor, immune, stroma, background)
- PD-L1 staining images (for ground truth quantification)
- Annotations from Excel files
"""

import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import pandas as pd
import os
import glob
from pathlib import Path
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2


class TNBCSegmentationDataset(Dataset):
    """
    Dataset for tissue segmentation task
    
    Expected structure:
    root_dir/
        images/
            sample_001.jpeg
            sample_002.jpeg
        masks/
            sample_001.png  (0=background, 1=tumor, 2=immune, 3=stroma)
            sample_002.png
    """
    def __init__(self, root_dir, split='train', image_size=512, augment=True):
        self.root_dir = Path(root_dir)
        self.image_dir = self.root_dir / 'images'
        self.mask_dir = self.root_dir / 'masks'
        self.image_size = image_size
        self.split = split
        
        # Get all image files
        self.image_paths = sorted(glob.glob(str(self.image_dir / '*.jpeg')) + 
                                  glob.glob(str(self.image_dir / '*.jpg')) +
                                  glob.glob(str(self.image_dir / '*.png')))
        
        # Split into train/val (80/20)
        np.random.seed(42)
        indices = np.random.permutation(len(self.image_paths))
        split_idx = int(0.8 * len(indices))
        
        if split == 'train':
            self.image_paths = [self.image_paths[i] for i in indices[:split_idx]]
        else:
            self.image_paths = [self.image_paths[i] for i in indices[split_idx:]]
        
        # Define augmentations
        if augment and split == 'train':
            self.transform = A.Compose([
                A.RandomRotate90(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, 
                                   rotate_limit=45, p=0.5),
                A.OneOf([
                    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, 
                                        val_shift_limit=20, p=0.5),
                    A.RandomBrightnessContrast(brightness_limit=0.2, 
                                              contrast_limit=0.2, p=0.5),
                ], p=0.5),
                A.Blur(blur_limit=(3, 7), p=0.3),
                A.Resize(image_size, image_size),
                A.Normalize(mean=(0.9357, 0.8253, 0.8998), 
                           std=(0.0787, 0.1751, 0.1125)),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(image_size, image_size),
                A.Normalize(mean=(0.9357, 0.8253, 0.8998), 
                           std=(0.0787, 0.1751, 0.1125)),
                ToTensorV2()
            ])
            
        print(f"Loaded {len(self.image_paths)} images for {split} split")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load corresponding mask
        img_name = Path(img_path).stem
        mask_path = self.mask_dir / f"{img_name}.png"
        
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        else:
            # If no mask, create dummy mask (all background)
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        # Apply augmentations
        transformed = self.transform(image=image, mask=mask)
        image = transformed['image']
        mask = transformed['mask']
        
        return {
            'image': image,
            'mask': torch.tensor(mask, dtype=torch.long),
            'image_name': img_name
        }


class TNBCAnnotatedDataset(Dataset):
    """
    Dataset using annotation Excel file
    
    Loads matching H&E, PD-L1, and PD-1 images with labels
    """
    def __init__(self, root_dir, annotation_excel, split='train', image_size=512, augment=True):
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.split = split
        
        # Load annotations
        self.annotations = pd.read_excel(annotation_excel)
        
        # Split data
        np.random.seed(42)
        indices = np.random.permutation(len(self.annotations))
        split_idx = int(0.8 * len(indices))
        
        if split == 'train':
            self.annotations = self.annotations.iloc[indices[:split_idx]].reset_index(drop=True)
        else:
            self.annotations = self.annotations.iloc[indices[split_idx:]].reset_index(drop=True)
        
        # Augmentations
        if augment and split == 'train':
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomRotation(45),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.9357, 0.8253, 0.8998), 
                                   std=(0.0787, 0.1751, 0.1125))
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.9357, 0.8253, 0.8998), 
                                   std=(0.0787, 0.1751, 0.1125))
            ])
        
        print(f"Loaded {len(self.annotations)} annotated samples for {split} split")
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        
        # Load H&E image
        he_path = self.root_dir / row['HE_path']
        he_image = cv2.imread(str(he_path))
        he_image = cv2.cvtColor(he_image, cv2.COLOR_BGR2RGB)
        
        # Load PD-L1 image if available
        if 'PDL1_path' in row and pd.notna(row['PDL1_path']):
            pdl1_path = self.root_dir / row['PDL1_path']
            pdl1_image = cv2.imread(str(pdl1_path))
            pdl1_image = cv2.cvtColor(pdl1_image, cv2.COLOR_BGR2RGB)
        else:
            pdl1_image = np.zeros_like(he_image)
        
        # Get labels
        pdl1_label = int(row['PDL1_label']) if 'PDL1_label' in row and pd.notna(row['PDL1_label']) else 0
        pd1_label = int(row['PD1_label']) if 'PD1_label' in row and pd.notna(row['PD1_label']) else 0
        
        # Transform
        he_tensor = self.transform(he_image)
        pdl1_tensor = self.transform(pdl1_image)
        
        return {
            'he_image': he_tensor,
            'pdl1_image': pdl1_tensor,
            'pdl1_label': torch.tensor(pdl1_label, dtype=torch.long),
            'pd1_label': torch.tensor(pd1_label, dtype=torch.long),
            'image_name': row['HE_path']
        }


class TNBCInferenceDataset(Dataset):
    """
    Simple dataset for inference on a folder of images
    """
    def __init__(self, image_folder, image_size=512):
        self.image_folder = Path(image_folder)
        self.image_size = image_size
        
        # Get all images
        self.image_paths = sorted(
            glob.glob(str(self.image_folder / '*.jpeg')) +
            glob.glob(str(self.image_folder / '*.jpg')) +
            glob.glob(str(self.image_folder / '*.png'))
        )
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.9357, 0.8253, 0.8998), 
                               std=(0.0787, 0.1751, 0.1125))
        ])
        
        print(f"Loaded {len(self.image_paths)} images for inference")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        image_tensor = self.transform(image)
        
        return {
            'image': image_tensor,
            'image_path': img_path,
            'image_name': Path(img_path).name
        }


def create_dataloaders(root_dir, annotation_excel=None, batch_size=8, 
                       num_workers=4, image_size=512, dataset_type='segmentation'):
    """
    Create train and validation dataloaders
    
    Args:
        root_dir: Root directory containing data
        annotation_excel: Path to annotation Excel file (for annotated dataset)
        batch_size: Batch size
        num_workers: Number of worker processes
        image_size: Size to resize images
        dataset_type: 'segmentation' or 'annotated'
    """
    if dataset_type == 'segmentation':
        train_dataset = TNBCSegmentationDataset(root_dir, split='train', 
                                                image_size=image_size, augment=True)
        val_dataset = TNBCSegmentationDataset(root_dir, split='val', 
                                              image_size=image_size, augment=False)
    elif dataset_type == 'annotated':
        if annotation_excel is None:
            raise ValueError("annotation_excel must be provided for annotated dataset")
        train_dataset = TNBCAnnotatedDataset(root_dir, annotation_excel, split='train',
                                             image_size=image_size, augment=True)
        val_dataset = TNBCAnnotatedDataset(root_dir, annotation_excel, split='val',
                                           image_size=image_size, augment=False)
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                           shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test dataset
    print("Testing TNBCSegmentationDataset...")
    
    # Create dummy data structure
    test_dir = Path("test_data")
    test_dir.mkdir(exist_ok=True)
    (test_dir / "images").mkdir(exist_ok=True)
    (test_dir / "masks").mkdir(exist_ok=True)
    
    # Create dummy image and mask
    dummy_img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    dummy_mask = np.random.randint(0, 4, (512, 512), dtype=np.uint8)
    
    cv2.imwrite(str(test_dir / "images" / "test_001.jpeg"), dummy_img)
    cv2.imwrite(str(test_dir / "masks" / "test_001.png"), dummy_mask)
    
    # Test dataset
    dataset = TNBCSegmentationDataset(test_dir, split='train', image_size=512)
    sample = dataset[0]
    
    print(f"Image shape: {sample['image'].shape}")
    print(f"Mask shape: {sample['mask'].shape}")
    print(f"Unique mask values: {torch.unique(sample['mask'])}")
    
    # Cleanup
    import shutil
    shutil.rmtree(test_dir)
    
    print("\nDataset test successful!")
