"""
PyTorch Dataset for Virtual Staining (H&E → PD-L1)

Loads paired H&E images and DAB heatmap targets for training
a U-Net to predict PD-L1 expression from H&E images.
"""

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.model_selection import train_test_split

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False


class VirtualStainingDataset(Dataset):
    """
    Dataset for paired H&E → PD-L1 training.
    
    Loads H&E images and corresponding DAB heatmaps (extracted from PD-L1 IHC).
    Applies synchronized augmentations to both image and target.
    """
    
    def __init__(self, he_dir, target_dir, file_list=None, image_size=512, augment=False):
        """
        Args:
            he_dir: Directory containing H&E images (.jpeg)
            target_dir: Directory containing DAB heatmaps (.png, 16-bit)
            file_list: Optional list of filenames to use (for train/val split)
            image_size: Target image size
            augment: Whether to apply data augmentation
        """
        self.he_dir = Path(he_dir)
        self.target_dir = Path(target_dir)
        self.image_size = image_size
        self.augment = augment
        
        # Find matched pairs
        if file_list is not None:
            self.pairs = file_list
        else:
            self.pairs = self._find_pairs()
        
        # Normalization stats (from PD-L1 predictor)
        self.mean = (0.9357, 0.8253, 0.8998)
        self.std = (0.0787, 0.1751, 0.1125)
        
        # Setup augmentations
        if augment and HAS_ALBUMENTATIONS:
            self.transform = A.Compose([
                A.RandomRotate90(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                                   rotate_limit=30, p=0.3),
                A.OneOf([
                    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15,
                                         val_shift_limit=10, p=0.5),
                    A.RandomBrightnessContrast(brightness_limit=0.1,
                                              contrast_limit=0.1, p=0.5),
                ], p=0.3),
                A.Resize(image_size, image_size),
                A.Normalize(mean=self.mean, std=self.std),
                ToTensorV2()
            ], additional_targets={'target': 'mask'})
        else:
            self.transform = A.Compose([
                A.Resize(image_size, image_size),
                A.Normalize(mean=self.mean, std=self.std),
                ToTensorV2()
            ], additional_targets={'target': 'mask'}) if HAS_ALBUMENTATIONS else None
    
    def _find_pairs(self):
        """Find matched H&E-target pairs by filename stem."""
        he_stems = {f.stem: f.name for f in self.he_dir.glob('*.jpeg')}
        target_stems = {f.stem: f.name for f in self.target_dir.glob('*.png')}
        
        # Match by stem (tile_003_r1c3)
        common = sorted(set(he_stems.keys()) & set(target_stems.keys()))
        pairs = [(he_stems[s], target_stems[s]) for s in common]
        
        print(f"Found {len(pairs)} matched H&E-target pairs")
        return pairs
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        he_name, target_name = self.pairs[idx]
        
        # Load H&E image (BGR → RGB)
        he_img = cv2.imread(str(self.he_dir / he_name))
        he_img = cv2.cvtColor(he_img, cv2.COLOR_BGR2RGB)
        
        # Load DAB heatmap (16-bit PNG → float32 0-1)
        dab_raw = cv2.imread(str(self.target_dir / target_name), cv2.IMREAD_UNCHANGED)
        if dab_raw.dtype == np.uint16:
            dab_map = dab_raw.astype(np.float32) / 65535.0
        else:
            dab_map = dab_raw.astype(np.float32) / 255.0
        
        # Apply transforms
        if self.transform is not None:
            transformed = self.transform(image=he_img, target=dab_map)
            image_tensor = transformed['image']              # (3, H, W)
            target_raw = transformed['target']
            if isinstance(target_raw, torch.Tensor):
                target_tensor = target_raw.unsqueeze(0).float()
            else:
                target_tensor = torch.from_numpy(
                    np.ascontiguousarray(target_raw)
                ).unsqueeze(0).float()                           # (1, H, W)
        else:
            # Fallback without albumentations
            he_img = cv2.resize(he_img, (self.image_size, self.image_size))
            dab_map = cv2.resize(dab_map, (self.image_size, self.image_size))
            
            he_img = he_img.astype(np.float32) / 255.0
            for i in range(3):
                he_img[:, :, i] = (he_img[:, :, i] - self.mean[i]) / self.std[i]
            
            image_tensor = torch.from_numpy(he_img.transpose(2, 0, 1)).float()
            target_tensor = torch.from_numpy(dab_map).unsqueeze(0).float()
        
        return {
            'image': image_tensor,
            'target': target_tensor,
            'filename': he_name
        }


def create_virtual_staining_dataloaders(data_dir, batch_size=4, num_workers=4,
                                         image_size=512, val_split=0.2, seed=42):
    """
    Create train/val dataloaders for virtual staining.
    
    Args:
        data_dir: Root directory containing he_images/ and pdl1_targets/
        batch_size: Batch size
        num_workers: Number of data loading workers
        image_size: Target image size
        val_split: Fraction of data for validation
        seed: Random seed for reproducible splits
    
    Returns:
        train_loader, val_loader
    """
    data_dir = Path(data_dir)
    he_dir = data_dir / 'he_images'
    target_dir = data_dir / 'pdl1_targets'
    
    # Find all pairs
    temp_dataset = VirtualStainingDataset(he_dir, target_dir)
    all_pairs = temp_dataset.pairs
    
    # Split into train/val
    train_pairs, val_pairs = train_test_split(
        all_pairs, test_size=val_split, random_state=seed
    )
    
    print(f"Train: {len(train_pairs)} pairs, Val: {len(val_pairs)} pairs")
    
    # Create datasets
    train_dataset = VirtualStainingDataset(
        he_dir, target_dir, file_list=train_pairs,
        image_size=image_size, augment=True
    )
    val_dataset = VirtualStainingDataset(
        he_dir, target_dir, file_list=val_pairs,
        image_size=image_size, augment=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader
