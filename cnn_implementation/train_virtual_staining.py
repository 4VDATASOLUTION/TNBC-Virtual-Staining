"""
Training Script for Virtual Staining CNN (H&E → PD-L1)

Trains a U-Net to predict PD-L1 DAB expression heatmaps from H&E images.

Usage:
    python cnn_implementation/train_virtual_staining.py --data_dir training_data --output_dir outputs --epochs 50
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import numpy as np
import argparse
import sys
import time
from tqdm import tqdm

# Add paths for imports
sys.path.append(str(Path(__file__).parent / 'models'))
sys.path.append(str(Path(__file__).parent))

from segmentation_cnn import VirtualStainingCNN
from virtual_staining_dataset import create_virtual_staining_dataloaders


# ============================================================================
# Loss Functions
# ============================================================================

class SSIMLoss(nn.Module):
    """Structural Similarity Index Loss (1 - SSIM)"""
    def __init__(self, window_size=11, channel=1):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.channel = channel
        
        # Gaussian window
        sigma = 1.5
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        window = g.unsqueeze(1) * g.unsqueeze(0)  # 2D gaussian
        self.register_buffer('window', window.unsqueeze(0).unsqueeze(0))  # (1, 1, H, W)
    
    def forward(self, pred, target):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        window = self.window.to(pred.device)
        
        mu_pred = nn.functional.conv2d(pred, window, padding=self.window_size // 2, groups=self.channel)
        mu_target = nn.functional.conv2d(target, window, padding=self.window_size // 2, groups=self.channel)
        
        mu_pred_sq = mu_pred ** 2
        mu_target_sq = mu_target ** 2
        mu_pred_target = mu_pred * mu_target
        
        sigma_pred_sq = nn.functional.conv2d(pred * pred, window, padding=self.window_size // 2, groups=self.channel) - mu_pred_sq
        sigma_target_sq = nn.functional.conv2d(target * target, window, padding=self.window_size // 2, groups=self.channel) - mu_target_sq
        sigma_pred_target = nn.functional.conv2d(pred * target, window, padding=self.window_size // 2, groups=self.channel) - mu_pred_target
        
        ssim = ((2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)) / \
               ((mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2))
        
        return 1 - ssim.mean()


class CombinedRegressionLoss(nn.Module):
    """Combined L1 + SSIM loss for virtual staining"""
    def __init__(self, l1_weight=0.5, ssim_weight=0.5):
        super(CombinedRegressionLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.ssim_loss = SSIMLoss()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
    
    def forward(self, pred, target):
        l1 = self.l1_loss(pred, target)
        ssim = self.ssim_loss(pred, target)
        return self.l1_weight * l1 + self.ssim_weight * ssim


# ============================================================================
# Metrics
# ============================================================================

def compute_metrics(pred, target):
    """Compute regression metrics between predicted and target heatmaps."""
    with torch.no_grad():
        # MSE
        mse = nn.functional.mse_loss(pred, target).item()
        
        # MAE
        mae = nn.functional.l1_loss(pred, target).item()
        
        # Pearson correlation (per batch, then averaged)
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        
        pred_mean = pred_flat.mean(dim=1, keepdim=True)
        target_mean = target_flat.mean(dim=1, keepdim=True)
        
        pred_centered = pred_flat - pred_mean
        target_centered = target_flat - target_mean
        
        numerator = (pred_centered * target_centered).sum(dim=1)
        denominator = torch.sqrt((pred_centered ** 2).sum(dim=1) * (target_centered ** 2).sum(dim=1))
        
        # Avoid division by zero
        pearson = (numerator / (denominator + 1e-8)).mean().item()
        
        # Mean predicted vs mean target (tile-level scores)
        pred_score = pred.mean(dim=(1, 2, 3)).cpu().numpy()
        target_score = target.mean(dim=(1, 2, 3)).cpu().numpy()
        
    return {
        'mse': mse,
        'mae': mae,
        'pearson': pearson,
        'pred_score': pred_score,
        'target_score': target_score,
    }


# ============================================================================
# Training & Validation
# ============================================================================

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    running_mae = 0.0
    running_pearson = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc="Train", leave=False)
    for batch in pbar:
        images = batch['image'].to(device)
        targets = batch['target'].to(device)
        
        optimizer.zero_grad()
        predictions = model(images)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()
        
        # Compute metrics
        metrics = compute_metrics(predictions, targets)
        running_loss += loss.item()
        running_mae += metrics['mae']
        running_pearson += metrics['pearson']
        num_batches += 1
        
        pbar.set_postfix(
            loss=f"{running_loss/num_batches:.4f}",
            mae=f"{running_mae/num_batches:.4f}",
            r=f"{running_pearson/num_batches:.3f}"
        )
    
    return {
        'loss': running_loss / num_batches,
        'mae': running_mae / num_batches,
        'pearson': running_pearson / num_batches,
    }


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    running_mae = 0.0
    running_mse = 0.0
    running_pearson = 0.0
    num_batches = 0
    all_pred_scores = []
    all_target_scores = []
    
    pbar = tqdm(dataloader, desc="Val", leave=False)
    with torch.no_grad():
        for batch in pbar:
            images = batch['image'].to(device)
            targets = batch['target'].to(device)
            
            predictions = model(images)
            loss = criterion(predictions, targets)
            
            metrics = compute_metrics(predictions, targets)
            running_loss += loss.item()
            running_mae += metrics['mae']
            running_mse += metrics['mse']
            running_pearson += metrics['pearson']
            num_batches += 1
            
            all_pred_scores.extend(metrics['pred_score'].tolist())
            all_target_scores.extend(metrics['target_score'].tolist())
            
            pbar.set_postfix(
                loss=f"{running_loss/num_batches:.4f}",
                mae=f"{running_mae/num_batches:.4f}",
                r=f"{running_pearson/num_batches:.3f}"
            )
    
    return {
        'loss': running_loss / num_batches,
        'mae': running_mae / num_batches,
        'mse': running_mse / num_batches,
        'pearson': running_pearson / num_batches,
        'pred_scores': all_pred_scores,
        'target_scores': all_target_scores,
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train Virtual Staining CNN")
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing he_images/ and pdl1_targets/')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Directory for outputs (models, logs)')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create dataloaders
    print("\n--- Loading Data ---")
    train_loader, val_loader = create_virtual_staining_dataloaders(
        args.data_dir, batch_size=args.batch_size,
        num_workers=args.num_workers, image_size=args.image_size
    )
    
    # Create model
    print("\n--- Creating Model ---")
    model = VirtualStainingCNN(pretrained=True).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {num_params:,}")
    print(f"Trainable parameters: {trainable:,}")
    
    # Loss, optimizer, scheduler
    criterion = CombinedRegressionLoss(l1_weight=0.5, ssim_weight=0.5)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # TensorBoard
    writer = SummaryWriter(output_dir / 'tensorboard_vs')
    
    # Training loop
    print(f"\n--- Training for {args.epochs} epochs ---\n")
    best_val_loss = float('inf')
    best_pearson = -1.0
    patience_counter = 0
    
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        
        # Learning rate scheduling
        scheduler.step(val_metrics['loss'])
        current_lr = optimizer.param_groups[0]['lr']
        
        elapsed = time.time() - start_time
        
        # Log to TensorBoard
        writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
        writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
        writer.add_scalar('MAE/train', train_metrics['mae'], epoch)
        writer.add_scalar('MAE/val', val_metrics['mae'], epoch)
        writer.add_scalar('Pearson/train', train_metrics['pearson'], epoch)
        writer.add_scalar('Pearson/val', val_metrics['pearson'], epoch)
        writer.add_scalar('LR', current_lr, epoch)
        
        # Print summary
        print(f"\nEpoch {epoch}/{args.epochs} ({elapsed:.1f}s):")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, MAE: {train_metrics['mae']:.4f}, "
              f"Pearson: {train_metrics['pearson']:.3f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, MAE: {val_metrics['mae']:.4f}, "
              f"Pearson: {val_metrics['pearson']:.3f}, MSE: {val_metrics['mse']:.6f}")
        print(f"  LR: {current_lr:.2e}")
        
        # Save best model by loss
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'val_pearson': val_metrics['pearson'],
            }, output_dir / 'best_virtual_staining_loss.pth')
            print(f"  → Saved best model (loss: {best_val_loss:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Save best model by Pearson correlation
        if val_metrics['pearson'] > best_pearson:
            best_pearson = val_metrics['pearson']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'val_pearson': val_metrics['pearson'],
            }, output_dir / 'best_virtual_staining_pearson.pth')
            print(f"  → Saved best model (Pearson: {best_pearson:.3f})")
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
            break
    
    writer.close()
    
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"{'='*60}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best Pearson correlation: {best_pearson:.3f}")
    print(f"Models saved to: {output_dir}")


if __name__ == '__main__':
    main()
