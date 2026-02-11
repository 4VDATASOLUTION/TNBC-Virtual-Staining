"""
Training script for TNBC tissue segmentation CNN

Steps:
1. Load data (H&E images + segmentation masks)
2. Initialize model (U-Net with ResNet50 backbone)
3. Train with Dice loss + CrossEntropy
4. Validate and save best model
5. Log metrics to TensorBoard
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import json
from datetime import datetime

import sys
sys.path.append(str(Path(__file__).parent / 'models'))
sys.path.append(str(Path(__file__).parent))

from segmentation_cnn import TissueSegmentationCNN, IntegratedTNBCModel
from dataset import create_dataloaders


class DiceLoss(nn.Module):
    """Dice loss for segmentation"""
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, predictions, targets, num_classes=4):
        """
        Args:
            predictions: (B, C, H, W) - logits
            targets: (B, H, W) - class indices
        """
        # Convert to one-hot
        predictions = torch.softmax(predictions, dim=1)
        targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()
        
        # Compute Dice coefficient per class
        intersection = (predictions * targets_one_hot).sum(dim=(2, 3))
        union = predictions.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice.mean()
        
        return dice_loss


class CombinedLoss(nn.Module):
    """Combined Dice + CrossEntropy loss with class weights for imbalance"""
    def __init__(self, dice_weight=0.5, ce_weight=0.5, device='cpu'):
        super(CombinedLoss, self).__init__()
        self.dice_loss = DiceLoss()
        # Class weights inversely proportional to frequency
        # Distribution: Background ~7%, Tumor ~3%, Immune ~0.2%, Stroma ~90%
        class_weights = torch.tensor([1.0, 5.0, 15.0, 0.1], dtype=torch.float32).to(device)
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
    
    def forward(self, predictions, targets):
        dice = self.dice_loss(predictions, targets)
        ce = self.ce_loss(predictions, targets)
        return self.dice_weight * dice + self.ce_weight * ce


def calculate_iou(predictions, targets, num_classes=4):
    """Calculate Intersection over Union (IoU) for each class"""
    ious = []
    predictions = torch.argmax(predictions, dim=1)
    
    for cls in range(num_classes):
        pred_mask = (predictions == cls)
        target_mask = (targets == cls)
        
        intersection = (pred_mask & target_mask).sum().float()
        union = (pred_mask | target_mask).sum().float()
        
        if union == 0:
            iou = 1.0 if intersection == 0 else 0.0
        else:
            iou = intersection / union
        
        ious.append(iou.item())
    
    return ious


def calculate_pixel_accuracy(predictions, targets):
    """Calculate pixel-wise accuracy"""
    predictions = torch.argmax(predictions, dim=1)
    correct = (predictions == targets).sum().float()
    total = targets.numel()
    return (correct / total).item()


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    running_accuracy = 0.0
    running_ious = [0.0] * 4
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} - Train")
    
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        
        # Calculate loss
        loss = criterion(outputs, masks)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Metrics
        running_loss += loss.item()
        accuracy = calculate_pixel_accuracy(outputs, masks)
        running_accuracy += accuracy
        
        ious = calculate_iou(outputs, masks)
        for i, iou in enumerate(ious):
            running_ious[i] += iou
        
        # Update progress bar
        pbar.set_postfix({
            'loss': running_loss / (batch_idx + 1),
            'acc': running_accuracy / (batch_idx + 1),
            'mIoU': np.mean([iou / (batch_idx + 1) for iou in running_ious])
        })
    
    num_batches = len(dataloader)
    return {
        'loss': running_loss / num_batches,
        'accuracy': running_accuracy / num_batches,
        'ious': [iou / num_batches for iou in running_ious],
        'mean_iou': np.mean([iou / num_batches for iou in running_ious])
    }


def validate_epoch(model, dataloader, criterion, device, epoch):
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    running_accuracy = 0.0
    running_ious = [0.0] * 4
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} - Val")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Calculate loss
            loss = criterion(outputs, masks)
            
            # Metrics
            running_loss += loss.item()
            accuracy = calculate_pixel_accuracy(outputs, masks)
            running_accuracy += accuracy
            
            ious = calculate_iou(outputs, masks)
            for i, iou in enumerate(ious):
                running_ious[i] += iou
            
            # Update progress bar
            pbar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'acc': running_accuracy / (batch_idx + 1),
                'mIoU': np.mean([iou / (batch_idx + 1) for iou in running_ious])
            })
    
    num_batches = len(dataloader)
    return {
        'loss': running_loss / num_batches,
        'accuracy': running_accuracy / num_batches,
        'ious': [iou / num_batches for iou in running_ious],
        'mean_iou': np.mean([iou / num_batches for iou in running_ious])
    }


def train(args):
    """Main training function"""
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"segmentation_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Create dataloaders
    print("Loading data...")
    train_loader, val_loader = create_dataloaders(
        root_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        dataset_type='segmentation'
    )
    
    # Initialize model
    print("Initializing model...")
    model = TissueSegmentationCNN(num_classes=args.num_classes, pretrained=True)
    model = model.to(device)
    
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # Loss and optimizer
    criterion = CombinedLoss(dice_weight=0.5, ce_weight=0.5, device=device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # TensorBoard
    writer = SummaryWriter(output_dir / 'tensorboard')
    
    # Training loop
    best_val_loss = float('inf')
    best_val_iou = 0.0
    patience_counter = 0
    
    print(f"\nStarting training for {args.epochs} epochs...")
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Validate
        val_metrics = validate_epoch(model, val_loader, criterion, device, epoch)
        
        # Learning rate scheduling
        scheduler.step(val_metrics['loss'])
        
        # Log to TensorBoard
        writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
        writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
        writer.add_scalar('Accuracy/train', train_metrics['accuracy'], epoch)
        writer.add_scalar('Accuracy/val', val_metrics['accuracy'], epoch)
        writer.add_scalar('mIoU/train', train_metrics['mean_iou'], epoch)
        writer.add_scalar('mIoU/val', val_metrics['mean_iou'], epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        # Log per-class IoU
        class_names = ['Background', 'Tumor', 'Immune', 'Stroma']
        for i, name in enumerate(class_names):
            writer.add_scalar(f'IoU/{name}_train', train_metrics['ious'][i], epoch)
            writer.add_scalar(f'IoU/{name}_val', val_metrics['ious'][i], epoch)
        
        # Print summary
        print(f"\nEpoch {epoch}/{args.epochs} Summary:")
        print(f"  Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}, mIoU: {train_metrics['mean_iou']:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, mIoU: {val_metrics['mean_iou']:.4f}")
        print(f"  Val IoU per class: {[f'{name}: {iou:.4f}' for name, iou in zip(class_names, val_metrics['ious'])]}")
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'val_iou': val_metrics['mean_iou']
            }, output_dir / 'best_model_loss.pth')
            print(f"  → Saved best model (loss: {best_val_loss:.4f})")
        
        if val_metrics['mean_iou'] > best_val_iou:
            best_val_iou = val_metrics['mean_iou']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'val_iou': val_metrics['mean_iou']
            }, output_dir / 'best_model_iou.pth')
            print(f"  → Saved best model (mIoU: {best_val_iou:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            break
        
        # Save checkpoint
        if epoch % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'val_iou': val_metrics['mean_iou']
            }, output_dir / f'checkpoint_epoch_{epoch}.pth')
    
    writer.close()
    print(f"\nTraining completed! Models saved to {output_dir}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best validation mIoU: {best_val_iou:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TNBC tissue segmentation CNN")
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Root directory containing images/ and masks/ folders')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='Directory to save outputs')
    
    # Model arguments
    parser.add_argument('--num_classes', type=int, default=4,
                       help='Number of segmentation classes (default: 4)')
    parser.add_argument('--image_size', type=int, default=512,
                       help='Input image size (default: 512)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--patience', type=int, default=15,
                       help='Early stopping patience')
    parser.add_argument('--save_every', type=int, default=10,
                       help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    
    train(args)
