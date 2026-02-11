"""
Inference pipeline for TNBC tissue segmentation and CPS++ scoring

Performs:
1. Tissue segmentation (tumor, immune, stroma)
2. PD-L1 quantification per compartment
3. Feature extraction for CPS++ scoring
4. Visualization of results
"""

import torch
import torch.nn.functional as F
import cv2
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import json
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import sys
sys.path.append(str(Path(__file__).parent))

from models.segmentation_cnn import (
    TissueSegmentationCNN, 
    PDL1QuantificationCNN,
    IntegratedTNBCModel
)
from models.dataset import TNBCInferenceDataset


# Color map for visualization
CLASS_COLORS = {
    0: [0, 0, 0],        # Background - Black
    1: [255, 0, 0],      # Tumor - Red
    2: [0, 255, 0],      # Immune - Green
    3: [0, 0, 255]       # Stroma - Blue
}

CLASS_NAMES = ['Background', 'Tumor', 'Immune', 'Stroma']


def load_model(checkpoint_path, model_type='segmentation', device='cuda'):
    """Load trained model from checkpoint"""
    if model_type == 'segmentation':
        model = TissueSegmentationCNN(num_classes=4)
    elif model_type == 'integrated':
        model = IntegratedTNBCModel(num_classes=4)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print(f"Loaded model from {checkpoint_path}")
    if 'epoch' in checkpoint:
        print(f"  Epoch: {checkpoint['epoch']}")
    if 'val_iou' in checkpoint:
        print(f"  Val mIoU: {checkpoint['val_iou']:.4f}")
    
    return model


def segment_image(model, image_tensor, device):
    """
    Perform segmentation on a single image
    
    Args:
        model: Trained segmentation model
        image_tensor: (C, H, W) tensor
        device: torch device
    
    Returns:
        segmentation_map: (H, W) numpy array with class predictions
        class_probabilities: (C, H, W) numpy array with class probabilities
    """
    with torch.no_grad():
        image_batch = image_tensor.unsqueeze(0).to(device)
        logits = model(image_batch)
        probabilities = F.softmax(logits, dim=1)
        predictions = torch.argmax(logits, dim=1)
        
        seg_map = predictions[0].cpu().numpy()
        probs = probabilities[0].cpu().numpy()
    
    return seg_map, probs


def visualize_segmentation(image, segmentation, output_path, alpha=0.5):
    """
    Overlay segmentation on original image
    
    Args:
        image: Original image (H, W, 3) in RGB
        segmentation: Segmentation map (H, W) with class indices
        output_path: Where to save visualization
        alpha: Transparency of overlay
    """
    # Create colored segmentation mask
    h, w = segmentation.shape
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_id, color in CLASS_COLORS.items():
        mask = (segmentation == class_id)
        colored_mask[mask] = color
    
    # Overlay on original image
    if image.shape[:2] != (h, w):
        image = cv2.resize(image, (w, h))
    
    overlay = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)
    
    # Create figure with legend
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image', fontsize=14)
    axes[0].axis('off')
    
    # Segmentation mask
    axes[1].imshow(colored_mask)
    axes[1].set_title('Segmentation Mask', fontsize=14)
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay', fontsize=14)
    axes[2].axis('off')
    
    # Add legend
    legend_elements = [Patch(facecolor=np.array(color)/255, label=name) 
                      for name, color in zip(CLASS_NAMES, CLASS_COLORS.values())]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def compute_compartment_statistics(segmentation):
    """
    Compute statistics for each tissue compartment
    
    Returns:
        Dictionary with percentages and pixel counts
    """
    total_pixels = segmentation.size
    stats = {}
    
    for class_id, name in enumerate(CLASS_NAMES):
        mask = (segmentation == class_id)
        pixel_count = mask.sum()
        percentage = (pixel_count / total_pixels) * 100
        
        stats[name.lower()] = {
            'pixel_count': int(pixel_count),
            'percentage': float(percentage)
        }
    
    return stats


def compute_cps_score(segmentation, pdl1_quantification=None):
    """
    Compute CPS++ score from segmentation and PD-L1 quantification
    
    Simplified version - full implementation requires spatial analysis
    
    Args:
        segmentation: (H, W) segmentation map
        pdl1_quantification: Dictionary with PD-L1 values per compartment
    
    Returns:
        CPS++ score (0-100)
    """
    # Get compartment statistics
    stats = compute_compartment_statistics(segmentation)
    
    # Simplified CPS calculation (placeholder)
    # Real CPS++ would include:
    # - PD-L1 positive cell counts per compartment
    # - Spatial interaction indices
    # - Tumor proportion score
    
    tumor_pct = stats['tumor']['percentage']
    immune_pct = stats['immune']['percentage']
    
    # Placeholder formula (replace with actual CPS++ algorithm)
    cps_score = (tumor_pct * 0.6 + immune_pct * 0.4) * 1.5
    cps_score = min(100, max(0, cps_score))
    
    return cps_score


def process_image(model, image_path, output_dir, device, visualize=True):
    """
    Process a single image through the full pipeline
    
    Returns:
        Dictionary with results
    """
    # Load image
    image = cv2.imread(str(image_path))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Prepare for model
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.9357, 0.8253, 0.8998), 
                           std=(0.0787, 0.1751, 0.1125))
    ])
    
    image_tensor = transform(image_rgb)
    
    # Segment
    segmentation, probabilities = segment_image(model, image_tensor, device)
    
    # Compute statistics
    stats = compute_compartment_statistics(segmentation)
    
    # Compute CPS++ score
    cps_score = compute_cps_score(segmentation)
    
    # Visualize if requested
    if visualize:
        output_path = output_dir / f"{Path(image_path).stem}_segmentation.png"
        visualize_segmentation(image_rgb, segmentation, output_path)
    
    # Save segmentation mask
    mask_path = output_dir / f"{Path(image_path).stem}_mask.png"
    cv2.imwrite(str(mask_path), segmentation.astype(np.uint8))
    
    results = {
        'image_name': Path(image_path).name,
        'compartment_statistics': stats,
        'cps_score': cps_score,
        'segmentation_path': str(mask_path)
    }
    
    return results


def run_inference(args):
    """Main inference function"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running inference on {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print("Loading model...")
    model = load_model(args.model_path, model_type=args.model_type, device=device)
    
    # Get images
    image_folder = Path(args.image_folder)
    image_paths = list(image_folder.glob('*.jpeg')) + \
                  list(image_folder.glob('*.jpg')) + \
                  list(image_folder.glob('*.png'))
    
    print(f"Found {len(image_paths)} images to process")
    
    # Process images
    all_results = []
    
    for image_path in tqdm(image_paths, desc="Processing images"):
        try:
            results = process_image(
                model, 
                image_path, 
                output_dir, 
                device,
                visualize=args.visualize
            )
            all_results.append(results)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue
    
    # Save results to JSON
    results_json = output_dir / 'segmentation_results.json'
    with open(results_json, 'w') as f:
        json.dump(all_results, f, indent=4)
    
    # Create summary CSV
    summary_data = []
    for result in all_results:
        row = {
            'image_name': result['image_name'],
            'cps_score': result['cps_score'],
            'tumor_pct': result['compartment_statistics']['tumor']['percentage'],
            'immune_pct': result['compartment_statistics']['immune']['percentage'],
            'stroma_pct': result['compartment_statistics']['stroma']['percentage'],
            'background_pct': result['compartment_statistics']['background']['percentage']
        }
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    summary_csv = output_dir / 'segmentation_summary.csv'
    summary_df.to_csv(summary_csv, index=False)
    
    print(f"\nInference complete!")
    print(f"Results saved to {output_dir}")
    print(f"  - JSON: {results_json}")
    print(f"  - CSV: {summary_csv}")
    
    # Print summary statistics
    print(f"\nSummary Statistics:")
    print(f"  Mean CPS++ score: {summary_df['cps_score'].mean():.2f} ± {summary_df['cps_score'].std():.2f}")
    print(f"  Mean tumor %: {summary_df['tumor_pct'].mean():.2f}%")
    print(f"  Mean immune %: {summary_df['immune_pct'].mean():.2f}%")
    print(f"  Mean stroma %: {summary_df['stroma_pct'].mean():.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on TNBC images")
    
    parser.add_argument('--image_folder', type=str, required=True,
                       help='Folder containing images to process')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--output_dir', type=str, default='./inference_results',
                       help='Directory to save results')
    parser.add_argument('--model_type', type=str, default='segmentation',
                       choices=['segmentation', 'integrated'],
                       help='Type of model')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualization images')
    
    args = parser.parse_args()
    
    run_inference(args)
