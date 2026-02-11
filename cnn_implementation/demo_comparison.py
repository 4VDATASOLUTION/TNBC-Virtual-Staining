"""
Demo: Virtual Staining Verification

Picks random test images and generates a side-by-side comparison:
[ Original H&E ]  [ AI Predicted PD-L1 ]  [ Real Ground Truth PD-L1 ]

Usage:
    python cnn_implementation/demo_comparison.py --num_examples 5
"""

import cv2
import numpy as np
import torch
import argparse
from pathlib import Path
import random
import sys

# Add path for modules
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / 'models'))

from models.segmentation_cnn import VirtualStainingCNN
from prepare_virtual_staining_data import color_deconvolution

def load_model(model_path, device):
    model = VirtualStainingCNN(pretrained=False).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model

def predict(model, image_bgr, device):
    # Preprocess
    img = cv2.resize(image_bgr, (512, 512))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Normalize
    mean = np.array([0.9357, 0.8253, 0.8998])
    std = np.array([0.0787, 0.1751, 0.1125])
    img_float = img_rgb.astype(np.float32) / 255.0
    img_norm = (img_float - mean) / std
    
    # Tensor
    tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
    
    # Inference
    with torch.no_grad():
        pred = model(tensor)
    
    heatmap = pred.squeeze().cpu().numpy()
    return heatmap

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--he_dir', default='data_raw/02-008_HE_A12_v2_s13')
    parser.add_argument('--pdl1_dir', default='data_raw/02-008_PDL1(SP142)-Springbio_A12_v3_b3')
    parser.add_argument('--model_path', default='outputs/best_virtual_staining_loss.pth')
    parser.add_argument('--output_dir', default='test_examples')
    parser.add_argument('--num_examples', type=int, default=5)
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load model
    model = load_model(args.model_path, device)
    
    # Find paired files
    he_dir = Path(args.he_dir)
    pdl1_dir = Path(args.pdl1_dir)
    
    he_files = sorted(list(he_dir.glob('*.jpeg')) + list(he_dir.glob('*.jpg')))
    pdl1_files = sorted(list(pdl1_dir.glob('*.jpeg')) + list(pdl1_dir.glob('*.jpg')))
    
    # Match by grid position (e.g., r1c1)
    pairs = []
    for he_f in he_files:
        grid_pos = he_f.stem.split('_')[-1] # extracts r1c1
        # Find matching pdl1
        match = next((f for f in pdl1_files if f.stem.endswith(grid_pos)), None)
        if match:
            pairs.append((he_f, match))
            
    print(f"Found {len(pairs)} matched pairs.")
    
    # Select random examples
    selected = random.sample(pairs, min(len(pairs), args.num_examples))
    
    for i, (he_path, pdl1_path) in enumerate(selected):
        print(f"Processing pair {i+1}: {he_path.name}")
        
        # Load images
        he_img = cv2.imread(str(he_path))
        pdl1_img = cv2.imread(str(pdl1_path))
        
        if he_img is None or pdl1_img is None:
            continue
            
        # 1. Get Prediction
        pred_heatmap = predict(model, he_img, device)
        pred_colored = cv2.applyColorMap((pred_heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        # 2. Get Ground Truth (DAB channel)
        pdl1_rgb = cv2.cvtColor(pdl1_img, cv2.COLOR_BGR2RGB)
        deconv = color_deconvolution(pdl1_rgb)
        gt_dab = deconv['dab']
        gt_dab_resized = cv2.resize(gt_dab, (512, 512))
        gt_colored = cv2.applyColorMap((gt_dab_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        # 3. Resize base images for display
        he_resized = cv2.resize(he_img, (512, 512))
        pdl1_resized = cv2.resize(pdl1_img, (512, 512))
        
        # 4. Create Comparison Grid
        # Row 1: H&E | PD-L1 Image (Real)
        # Row 2: Prediction Heatmap | Ground Truth Heatmap
        
        # Let's do a side-by-side of Heatmaps overlaid on H&E
        # Overlay Prediction
        overlay_pred = cv2.addWeighted(he_resized, 0.6, pred_colored, 0.4, 0)
        cv2.putText(overlay_pred, "AI Prediction", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Overlay Ground Truth
        overlay_gt = cv2.addWeighted(he_resized, 0.6, gt_colored, 0.4, 0)
        cv2.putText(overlay_gt, "Ground Truth", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Combine
        combined = np.hstack([he_resized, overlay_pred, overlay_gt])
        
        # Save
        out_path = output_dir / f"comparison_{he_path.stem}.jpg"
        cv2.imwrite(str(out_path), combined)
        print(f"  Saved to {out_path}")

    print(f"\nDone! Check the '{args.output_dir}' folder for results.")

if __name__ == "__main__":
    main()
