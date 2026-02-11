"""
Inference Pipeline for Virtual Staining CNN

Takes H&E images and predicts PD-L1 expression heatmaps.
Generates overlay visualizations and tile-level scores.

Usage:
    python cnn_implementation/inference_virtual_staining.py --model_path outputs/best_virtual_staining_loss.pth --input_dir data_raw/02-008_HE_A12_v2_s13 --output_dir results
"""

import cv2
import numpy as np
import torch
from pathlib import Path
import argparse
import sys
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent / 'models'))
sys.path.append(str(Path(__file__).parent))

from segmentation_cnn import VirtualStainingCNN


def load_model(model_path, device):
    """Load trained virtual staining model."""
    model = VirtualStainingCNN(pretrained=False).to(device)
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from epoch {checkpoint.get('epoch', '?')}")
        print(f"  Val loss: {checkpoint.get('val_loss', '?')}")
        print(f"  Val Pearson: {checkpoint.get('val_pearson', '?')}")
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model


def preprocess_image(image_bgr, image_size=512):
    """Preprocess H&E image for model input."""
    # Resize
    img = cv2.resize(image_bgr, (image_size, image_size))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Normalize (same stats as training)
    mean = np.array([0.9357, 0.8253, 0.8998])
    std = np.array([0.0787, 0.1751, 0.1125])
    
    img_float = img_rgb.astype(np.float32) / 255.0
    img_norm = (img_float - mean) / std
    
    # To tensor: (H, W, 3) → (1, 3, H, W)
    tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).float()
    
    return tensor


def create_heatmap_overlay(image_bgr, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """Create colored heatmap overlay on original image."""
    # Resize heatmap to match image
    h, w = image_bgr.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    
    # Convert to colormap
    heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    
    # Overlay
    overlay = cv2.addWeighted(image_bgr, 1 - alpha, heatmap_colored, alpha, 0)
    
    return overlay


def predict_single(model, image_bgr, device, image_size=512):
    """Predict PD-L1 heatmap for a single image."""
    tensor = preprocess_image(image_bgr, image_size).to(device)
    
    with torch.no_grad():
        prediction = model(tensor)
    
    # (1, 1, H, W) → (H, W)
    heatmap = prediction.squeeze().cpu().numpy()
    
    return heatmap


def run_inference(model_path, input_dir, output_dir, image_size=512):
    """Run inference on a folder of H&E images."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load model
    model = load_model(model_path, device)
    
    # Setup output
    output_dir = Path(output_dir)
    heatmap_dir = output_dir / 'heatmaps'
    overlay_dir = output_dir / 'overlays'
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    
    # Find images
    input_dir = Path(input_dir)
    image_files = sorted(list(input_dir.glob('*.jpeg')) + list(input_dir.glob('*.jpg')))
    print(f"Found {len(image_files)} images")
    
    results = []
    
    for img_path in tqdm(image_files, desc="Processing"):
        # Load image
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            print(f"WARNING: Could not read {img_path.name}")
            continue
        
        # Predict
        heatmap = predict_single(model, image_bgr, device, image_size)
        
        # Compute tile-level score
        pdl1_score = float(heatmap.mean())
        
        results.append({
            'filename': img_path.name,
            'pdl1_score': pdl1_score,
            'max_intensity': float(heatmap.max()),
            'positive_ratio': float((heatmap > 0.5).mean()),
        })
        
        # Save heatmap
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        cv2.imwrite(str(heatmap_dir / f"{img_path.stem}_heatmap.png"), heatmap_uint8)
        
        # Save overlay
        overlay = create_heatmap_overlay(image_bgr, heatmap, alpha=0.4)
        cv2.imwrite(str(overlay_dir / f"{img_path.stem}_overlay.jpeg"), overlay)
    
    # Save results as CSV
    csv_path = output_dir / 'pdl1_predictions.csv'
    with open(csv_path, 'w') as f:
        f.write("filename,pdl1_score,max_intensity,positive_ratio\n")
        for r in results:
            f.write(f"{r['filename']},{r['pdl1_score']:.6f},{r['max_intensity']:.6f},{r['positive_ratio']:.6f}\n")
    
    print(f"\n{'='*60}")
    print(f"Inference Complete!")
    print(f"{'='*60}")
    print(f"Processed: {len(results)} images")
    print(f"Heatmaps: {heatmap_dir}")
    print(f"Overlays: {overlay_dir}")
    print(f"Results CSV: {csv_path}")
    
    if results:
        scores = [r['pdl1_score'] for r in results]
        print(f"\nPD-L1 Score Statistics:")
        print(f"  Mean:  {np.mean(scores):.4f}")
        print(f"  Std:   {np.std(scores):.4f}")
        print(f"  Min:   {np.min(scores):.4f}")
        print(f"  Max:   {np.max(scores):.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Virtual Staining Inference")
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing H&E images')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Output directory for predictions')
    parser.add_argument('--image_size', type=int, default=512)
    
    args = parser.parse_args()
    run_inference(args.model_path, args.input_dir, args.output_dir, args.image_size)
