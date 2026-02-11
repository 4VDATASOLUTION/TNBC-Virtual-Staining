"""
Prepare paired training data for Virtual Staining CNN

Matches H&E tiles with PD-L1 IHC tiles by grid position,
extracts DAB channel from PD-L1 images via color deconvolution,
and saves paired data for training.

Usage:
    python cnn_implementation/prepare_virtual_staining_data.py
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import re
import argparse


# Color deconvolution matrix for H-DAB staining
# Reference: Ruifrok & Johnston, "Quantification of histochemical staining"
# Rows: [Hematoxylin, DAB, Residual]
H_DAB_MATRIX = np.array([
    [0.650, 0.704, 0.286],   # Hematoxylin
    [0.268, 0.570, 0.776],   # DAB (brown)
    [0.711, 0.423, 0.562],   # Residual
])


def color_deconvolution(image_rgb, stain_matrix=None):
    """
    Perform color deconvolution to separate stain channels.
    
    Args:
        image_rgb: RGB image (H, W, 3), uint8
        stain_matrix: 3x3 stain matrix (rows = stain vectors)
    
    Returns:
        channels: dict with 'hematoxylin', 'dab', 'residual' channels (H, W) float32 0-1
    """
    if stain_matrix is None:
        stain_matrix = H_DAB_MATRIX
    
    # Normalize stain matrix rows
    stain_matrix = stain_matrix / np.linalg.norm(stain_matrix, axis=1, keepdims=True)
    
    # Compute inverse (deconvolution matrix)
    deconv_matrix = np.linalg.inv(stain_matrix)
    
    # Convert to optical density (OD) space
    image_float = image_rgb.astype(np.float64) / 255.0
    image_float = np.clip(image_float, 1e-6, 1.0)  # Avoid log(0)
    od = -np.log(image_float)
    
    # Reshape for matrix multiplication: (H*W, 3)
    h, w, _ = od.shape
    od_flat = od.reshape(-1, 3)
    
    # Deconvolve: multiply by inverse matrix
    channels_flat = od_flat @ deconv_matrix.T
    channels = channels_flat.reshape(h, w, 3)
    
    # Normalize each channel to 0-1
    result = {}
    names = ['hematoxylin', 'dab', 'residual']
    for i, name in enumerate(names):
        ch = channels[:, :, i]
        ch = np.clip(ch, 0, None)
        # Normalize using percentile to handle outliers
        p99 = np.percentile(ch, 99) if ch.max() > 0 else 1.0
        if p99 > 0:
            ch = ch / p99
        ch = np.clip(ch, 0, 1.0)
        result[name] = ch.astype(np.float32)
    
    return result


def extract_grid_position(filename):
    """Extract grid position (e.g., 'r1c3') from filename."""
    match = re.search(r'(r\d+c\d+)', filename)
    return match.group(1) if match else None


def extract_tile_number(filename):
    """Extract tile number (e.g., '003') from filename."""
    match = re.search(r'_(\d{3})_r\d+c\d+', filename)
    return match.group(1) if match else None


def find_matching_pairs(he_dir, pdl1_dir):
    """
    Find matching H&E ↔ PD-L1 tile pairs by grid position.
    
    Returns list of (he_path, pdl1_path, grid_pos) tuples
    """
    he_files = {extract_grid_position(f.name): f 
                for f in Path(he_dir).glob('*.jpeg') 
                if extract_grid_position(f.name)}
    
    pdl1_files = {extract_grid_position(f.name): f 
                  for f in Path(pdl1_dir).glob('*.jpeg') 
                  if extract_grid_position(f.name)}
    
    # Find common grid positions
    common_positions = sorted(set(he_files.keys()) & set(pdl1_files.keys()))
    
    pairs = [(he_files[pos], pdl1_files[pos], pos) for pos in common_positions]
    
    print(f"Found {len(he_files)} H&E tiles")
    print(f"Found {len(pdl1_files)} PD-L1 tiles")
    print(f"Matched {len(pairs)} tile pairs")
    
    return pairs


def prepare_data(he_dir, pdl1_dir, output_dir, image_size=512):
    """
    Prepare paired training data.
    
    1. Match tiles by grid position
    2. Extract DAB channel from PD-L1 images
    3. Save paired H&E images and DAB heatmaps
    """
    output_dir = Path(output_dir)
    he_output = output_dir / 'he_images'
    target_output = output_dir / 'pdl1_targets'
    
    he_output.mkdir(parents=True, exist_ok=True)
    target_output.mkdir(parents=True, exist_ok=True)
    
    # Find matching pairs
    pairs = find_matching_pairs(he_dir, pdl1_dir)
    
    if len(pairs) == 0:
        print("ERROR: No matching tile pairs found!")
        return
    
    print(f"\nProcessing {len(pairs)} tile pairs...")
    
    stats = {'dab_min': [], 'dab_max': [], 'dab_mean': []}
    
    for he_path, pdl1_path, grid_pos in tqdm(pairs, desc="Processing tiles"):
        # Read images
        he_img = cv2.imread(str(he_path))
        pdl1_img = cv2.imread(str(pdl1_path))
        
        if he_img is None or pdl1_img is None:
            print(f"WARNING: Could not read {grid_pos}, skipping")
            continue
        
        # Convert PD-L1 to RGB for color deconvolution
        pdl1_rgb = cv2.cvtColor(pdl1_img, cv2.COLOR_BGR2RGB)
        
        # Extract DAB channel
        channels = color_deconvolution(pdl1_rgb)
        dab_map = channels['dab']  # (H, W) float32 0-1
        
        # Resize both to target size
        he_resized = cv2.resize(he_img, (image_size, image_size), 
                                interpolation=cv2.INTER_LINEAR)
        dab_resized = cv2.resize(dab_map, (image_size, image_size), 
                                 interpolation=cv2.INTER_LINEAR)
        
        # Save H&E image
        tile_num = extract_tile_number(he_path.name) or grid_pos
        he_filename = f"tile_{tile_num}_{grid_pos}.jpeg"
        cv2.imwrite(str(he_output / he_filename), he_resized)
        
        # Save DAB heatmap as 16-bit PNG for precision
        dab_uint16 = (dab_resized * 65535).astype(np.uint16)
        dab_filename = f"tile_{tile_num}_{grid_pos}.png"
        cv2.imwrite(str(target_output / dab_filename), dab_uint16)
        
        # Track stats
        stats['dab_min'].append(dab_resized.min())
        stats['dab_max'].append(dab_resized.max())
        stats['dab_mean'].append(dab_resized.mean())
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Data Preparation Complete!")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"H&E images: {he_output} ({len(list(he_output.glob('*.jpeg')))} files)")
    print(f"DAB targets: {target_output} ({len(list(target_output.glob('*.png')))} files)")
    print(f"\nDAB channel statistics:")
    print(f"  Min:  {np.mean(stats['dab_min']):.4f}")
    print(f"  Max:  {np.mean(stats['dab_max']):.4f}")
    print(f"  Mean: {np.mean(stats['dab_mean']):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare virtual staining training data")
    parser.add_argument('--he_dir', type=str, 
                        default='data_raw/02-008_HE_A12_v2_s13',
                        help='Directory containing H&E images')
    parser.add_argument('--pdl1_dir', type=str,
                        default='data_raw/02-008_PDL1(SP142)-Springbio_A12_v3_b3',
                        help='Directory containing PD-L1 IHC images')
    parser.add_argument('--output_dir', type=str,
                        default='training_data',
                        help='Output directory for paired data')
    parser.add_argument('--image_size', type=int, default=512,
                        help='Target image size')
    
    args = parser.parse_args()
    prepare_data(args.he_dir, args.pdl1_dir, args.output_dir, args.image_size)
