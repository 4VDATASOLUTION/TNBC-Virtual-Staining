"""
Step-by-step guide to prepare your TNBC data for CNN training

This script helps you organize your data from data_raw/ into the format needed for training
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import shutil
from tqdm import tqdm

def check_existing_data():
    """Check what data we have in data_raw/"""
    print("="*60)
    print("STEP 1: Checking existing data")
    print("="*60)
    
    data_raw = Path("data_raw")
    
    # Check H&E images
    he_dir = data_raw / "02-008_HE_A12_v2_s13"
    if he_dir.exists():
        he_images = list(he_dir.glob("*.jpeg")) + list(he_dir.glob("*.jpg"))
        print(f"✓ Found {len(he_images)} H&E images in {he_dir.name}")
    else:
        print(f"✗ H&E directory not found")
        he_images = []
    
    # Check PD-L1 images
    pdl1_dir = data_raw / "02-008_PDL1(SP142)-Springbio_A12_v3_b3"
    if pdl1_dir.exists():
        pdl1_images = list(pdl1_dir.glob("*.jpeg")) + list(pdl1_dir.glob("*.jpg"))
        print(f"✓ Found {len(pdl1_images)} PD-L1 images in {pdl1_dir.name}")
    else:
        print(f"✗ PD-L1 directory not found")
        pdl1_images = []
    
    # Check annotations
    annotations = list(data_raw.glob("*.xlsx")) + list(data_raw.glob("*.csv"))
    if annotations:
        print(f"✓ Found {len(annotations)} annotation files:")
        for ann in annotations:
            print(f"  - {ann.name}")
    else:
        print("✗ No annotation files found")
    
    return he_images, pdl1_images, annotations

def create_training_dataset_structure():
    """Create the directory structure needed for training"""
    print("\n" + "="*60)
    print("STEP 2: Creating training dataset structure")
    print("="*60)
    
    training_dir = Path("training_data")
    
    # Create directories
    (training_dir / "images").mkdir(parents=True, exist_ok=True)
    (training_dir / "masks").mkdir(parents=True, exist_ok=True)
    
    print(f"✓ Created directory structure:")
    print(f"  {training_dir}/")
    print(f"    ├── images/  (for H&E images)")
    print(f"    └── masks/   (for segmentation masks)")
    
    return training_dir

def copy_he_images(he_images, training_dir):
    """Copy H&E images to training directory"""
    print("\n" + "="*60)
    print("STEP 3: Copying H&E images")
    print("="*60)
    
    if not he_images:
        print("⚠️  No H&E images to copy")
        return
    
    dest_dir = training_dir / "images"
    
    print(f"Copying {len(he_images)} images...")
    for img_path in tqdm(he_images[:100], desc="Copying"):  # Limit to 100 for demo
        shutil.copy2(img_path, dest_dir / img_path.name)
    
    print(f"✓ Copied {min(len(he_images), 100)} H&E images to {dest_dir}")

def create_masks_info():
    """Provide information about creating segmentation masks"""
    print("\n" + "="*60)
    print("STEP 4: Segmentation Masks - MANUAL ANNOTATION NEEDED")
    print("="*60)
    
    print("""
⚠️  IMPORTANT: Segmentation masks need to be created manually or semi-automatically.

Your masks should be grayscale PNG images where each pixel value represents:
  • 0 = Background
  • 1 = Tumor
  • 2 = Immune cells
  • 3 = Stroma

OPTIONS for creating masks:

1. Use the PD-L1_Annotator tool (for basic labeling):
   cd PD-L1_Annotator
   python annotation_tool.py --excel_file metadata/annotation_task.xlsx
   
2. Use annotation software:
   • QuPath (recommended for pathology): https://qupath.github.io/
   • CVAT: https://www.cvat.ai/
   • Label Studio: https://labelstud.io/
   
3. Use existing PD-L1 staining as weak supervision:
   • Convert PD-L1 intensity to mask approximations
   • Requires image processing pipeline
   
4. Use pre-trained models for initial segmentation:
   • Use PD-L1_predictor as starting point
   • Manually refine the predictions

MASK FORMAT:
  - Same dimensions as H&E image
  - Grayscale PNG (8-bit)
  - Filename: same as image but .png extension
    Example: 02-008_HE_A12_v2_s13_001_r1c1.png
""")

def create_dummy_masks_option(training_dir, num_images):
    """Option to create dummy masks for testing"""
    print("\n" + "="*60)
    print("STEP 5: Create dummy masks for testing?")
    print("="*60)
    
    response = input("\nCreate dummy masks to test training pipeline? (y/n): ").strip().lower()
    
    if response == 'y':
        mask_dir = training_dir / "masks"
        image_dir = training_dir / "images"
        
        image_files = list(image_dir.glob("*.jpeg")) + list(image_dir.glob("*.jpg"))
        
        print(f"\nCreating dummy masks for {len(image_files)} images...")
        
        for img_path in tqdm(image_files, desc="Creating masks"):
            # Read image to get dimensions
            img = cv2.imread(str(img_path))
            h, w = img.shape[:2]
            
            # Create random segmentation mask
            mask = np.zeros((h, w), dtype=np.uint8)
            
            # Random tumor regions
            for _ in range(np.random.randint(2, 5)):
                cx, cy = np.random.randint(100, w-100), np.random.randint(100, h-100)
                radius = np.random.randint(50, 150)
                cv2.circle(mask, (cx, cy), radius, 1, -1)
            
            # Random immune cells
            for _ in range(np.random.randint(10, 30)):
                cx, cy = np.random.randint(50, w-50), np.random.randint(50, h-50)
                radius = np.random.randint(5, 15)
                cv2.circle(mask, (cx, cy), radius, 2, -1)
            
            # Rest is stroma
            mask[mask == 0] = 3
            
            # Some background
            mask[:50, :] = 0
            mask[-50:, :] = 0
            
            # Save mask
            mask_name = img_path.stem + ".png"
            cv2.imwrite(str(mask_dir / mask_name), mask)
        
        print(f"✓ Created {len(image_files)} dummy masks in {mask_dir}")
        print("\n⚠️  Remember: These are DUMMY masks for testing only!")
        print("   For real training, you need properly annotated masks.")
        
        return True
    else:
        print("\nSkipped dummy mask creation.")
        print("You'll need to create real masks before training.")
        return False

def provide_next_steps(has_masks):
    """Show what to do next"""
    print("\n" + "="*60)
    print("NEXT STEPS SUMMARY")
    print("="*60)
    
    if has_masks:
        print("""
✓ Your training data is ready!

To start training:

1. Verify your data:
   python cnn_implementation/prepare_dataset.py --data_dir training_data --action verify

2. Check dataset statistics:
   python cnn_implementation/prepare_dataset.py --data_dir training_data --action stats

3. Start training (CPU mode):
   python cnn_implementation/train_segmentation.py \
       --data_dir training_data \
       --output_dir outputs \
       --epochs 20 \
       --batch_size 2 \
       --learning_rate 0.0001

4. Monitor training (if tensorboard installed):
   python -m tensorboard --logdir outputs

5. Run inference after training:
   python cnn_implementation/inference_pipeline.py \
       --image_folder training_data/images \
       --model_path outputs/segmentation_*/best_model_iou.pth \
       --visualize

TIPS:
• Start with few epochs (20) to test
• Use small batch size (2-4) for CPU
• Check outputs/ for trained models
• Visualizations saved in inference_results/
""")
    else:
        print("""
⚠️  Segmentation masks are required before training!

Your options:

OPTION 1 - Quick Test (Recommended for first-time users):
   Rerun this script and choose 'y' to create dummy masks
   This lets you test the training pipeline
   
OPTION 2 - Manual Annotation:
   Use QuPath or other annotation tools to create masks:
   1. Download QuPath: https://qupath.github.io/
   2. Open H&E images in QuPath
   3. Draw annotations for tumor/immune/stroma
   4. Export as labeled PNG masks
   5. Place masks in training_data/masks/
   
OPTION 3 - Use PD-L1 Staining:
   If you have paired PD-L1 images, you can:
   1. Use them as weak supervision
   2. Threshold intensity to create initial masks
   3. Manually refine the masks
   
OPTION 4 - Use Existing Predictor:
   1. Run PD-L1_predictor on your images
   2. Convert predictions to segmentation masks
   3. Manually verify and correct masks

After creating masks:
   python cnn_implementation/prepare_dataset.py --data_dir training_data --action verify
""")

def main():
    """Main workflow"""
    print("="*60)
    print("TNBC CNN TRAINING - DATA PREPARATION GUIDE")
    print("="*60)
    
    # Check existing data
    he_images, pdl1_images, annotations = check_existing_data()
    
    if not he_images:
        print("\n⚠️  No H&E images found in data_raw/")
        print("Please ensure your images are in the correct directory.")
        return
    
    # Create training structure
    training_dir = create_training_dataset_structure()
    
    # Copy H&E images
    copy_he_images(he_images, training_dir)
    
    # Explain masks
    create_masks_info()
    
    # Option to create dummy masks
    has_masks = create_dummy_masks_option(training_dir, len(he_images))
    
    # Final instructions
    provide_next_steps(has_masks)
    
    print("\n" + "="*60)
    print("Data preparation guide completed!")
    print("="*60)

if __name__ == "__main__":
    main()
