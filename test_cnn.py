"""
Quick test script for CNN model
Creates dummy data and runs a mini training test
"""

import torch
import numpy as np
import cv2
from pathlib import Path
import sys

# Add cnn_implementation to path
sys.path.insert(0, str(Path(__file__).parent / 'cnn_implementation'))

from models.segmentation_cnn import TissueSegmentationCNN, IntegratedTNBCModel
from models.dataset import TNBCSegmentationDataset

def create_dummy_data(data_dir, num_samples=10):
    """Create dummy H&E images and segmentation masks for testing"""
    data_dir = Path(data_dir)
    image_dir = data_dir / 'images'
    mask_dir = data_dir / 'masks'
    
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating {num_samples} dummy samples...")
    
    for i in range(num_samples):
        # Create realistic-looking H&E image (pinkish/purplish)
        img = np.random.randint(180, 255, (512, 512, 3), dtype=np.uint8)
        img[:, :, 0] = np.random.randint(200, 255, (512, 512))  # More red
        img[:, :, 1] = np.random.randint(150, 220, (512, 512))  # Less green
        img[:, :, 2] = np.random.randint(200, 255, (512, 512))  # More blue
        
        # Create segmentation mask with realistic patterns
        mask = np.zeros((512, 512), dtype=np.uint8)
        
        # Tumor regions (class 1) - central areas
        cv2.circle(mask, (256, 256), 100, 1, -1)
        cv2.ellipse(mask, (150, 150), (60, 40), 45, 0, 360, 1, -1)
        
        # Immune regions (class 2) - scattered around tumor
        for _ in range(15):
            x, y = np.random.randint(50, 462, 2)
            cv2.circle(mask, (x, y), np.random.randint(10, 25), 2, -1)
        
        # Stroma regions (class 3) - background tissue
        mask[mask == 0] = 3
        mask[:50, :] = 0  # Some background
        mask[-50:, :] = 0
        mask[:, :50] = 0
        mask[:, -50:] = 0
        
        # Save
        cv2.imwrite(str(image_dir / f'sample_{i:03d}.jpeg'), img)
        cv2.imwrite(str(mask_dir / f'sample_{i:03d}.png'), mask)
    
    print(f"✓ Created dummy data in {data_dir}")
    return data_dir

def test_model_creation():
    """Test if models can be created"""
    print("\n" + "="*60)
    print("TEST 1: Model Creation")
    print("="*60)
    
    try:
        # Test segmentation model
        model = TissueSegmentationCNN(num_classes=4, pretrained=False)
        num_params = sum(p.numel() for p in model.parameters())
        print(f"✓ TissueSegmentationCNN created")
        print(f"  Parameters: {num_params:,}")
        
        # Test integrated model
        integrated = IntegratedTNBCModel(num_classes=4)
        print(f"✓ IntegratedTNBCModel created")
        
        return True
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False

def test_forward_pass():
    """Test forward pass through model"""
    print("\n" + "="*60)
    print("TEST 2: Forward Pass")
    print("="*60)
    
    try:
        model = TissueSegmentationCNN(num_classes=4, pretrained=False)
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(2, 3, 512, 512)
        print(f"Input shape: {dummy_input.shape}")
        
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"Output shape: {output.shape}")
        print(f"Expected: torch.Size([2, 4, 512, 512])")
        
        if output.shape == torch.Size([2, 4, 512, 512]):
            print("✓ Forward pass successful!")
            return True
        else:
            print("✗ Output shape mismatch")
            return False
            
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dataset_loading(data_dir):
    """Test dataset loading"""
    print("\n" + "="*60)
    print("TEST 3: Dataset Loading")
    print("="*60)
    
    try:
        dataset = TNBCSegmentationDataset(data_dir, split='train', 
                                         image_size=512, augment=False)
        
        print(f"✓ Dataset created with {len(dataset)} samples")
        
        # Load one sample
        sample = dataset[0]
        print(f"  Image shape: {sample['image'].shape}")
        print(f"  Mask shape: {sample['mask'].shape}")
        print(f"  Unique mask values: {torch.unique(sample['mask']).tolist()}")
        
        if sample['image'].shape == torch.Size([3, 512, 512]):
            print("✓ Dataset loading successful!")
            return True
        else:
            print("✗ Unexpected data shape")
            return False
            
    except Exception as e:
        print(f"✗ Dataset loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_mini_training(data_dir):
    """Run a mini training loop to verify everything works"""
    print("\n" + "="*60)
    print("TEST 4: Mini Training Loop (3 iterations)")
    print("="*60)
    
    try:
        from torch.utils.data import DataLoader
        import torch.nn as nn
        import torch.optim as optim
        
        # Create dataset and loader
        dataset = TNBCSegmentationDataset(data_dir, split='train', 
                                         image_size=512, augment=False)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
        
        # Create model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = TissueSegmentationCNN(num_classes=4, pretrained=False).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        print(f"Device: {device}")
        print(f"Starting mini training...")
        
        model.train()
        for i, batch in enumerate(dataloader):
            if i >= 3:  # Only 3 iterations
                break
                
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            print(f"  Iteration {i+1}: Loss = {loss.item():.4f}")
        
        print("✓ Mini training successful!")
        return True
        
    except Exception as e:
        print(f"✗ Mini training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_inference():
    """Test inference mode"""
    print("\n" + "="*60)
    print("TEST 5: Inference Mode")
    print("="*60)
    
    try:
        model = TissueSegmentationCNN(num_classes=4, pretrained=False)
        model.eval()
        
        # Create dummy image
        dummy_input = torch.randn(1, 3, 512, 512)
        
        with torch.no_grad():
            output = model(dummy_input)
            predictions = torch.argmax(output, dim=1)
        
        print(f"Input: {dummy_input.shape}")
        print(f"Logits: {output.shape}")
        print(f"Predictions: {predictions.shape}")
        print(f"Unique predictions: {torch.unique(predictions).tolist()}")
        
        print("✓ Inference mode successful!")
        return True
        
    except Exception as e:
        print(f"✗ Inference failed: {e}")
        return False

def main():
    """Run all tests"""
    print("="*60)
    print("CNN MODEL COMPREHENSIVE TEST")
    print("="*60)
    
    results = []
    
    # Test 1: Model Creation
    results.append(("Model Creation", test_model_creation()))
    
    # Test 2: Forward Pass
    results.append(("Forward Pass", test_forward_pass()))
    
    # Create dummy data
    test_data_dir = Path("test_data_cnn")
    create_dummy_data(test_data_dir, num_samples=10)
    
    # Test 3: Dataset Loading
    results.append(("Dataset Loading", test_dataset_loading(test_data_dir)))
    
    # Test 4: Mini Training
    results.append(("Mini Training", test_mini_training(test_data_dir)))
    
    # Test 5: Inference
    results.append(("Inference", test_inference()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:20s}: {status}")
    
    all_passed = all(result[1] for result in results)
    
    print("="*60)
    if all_passed:
        print("🎉 ALL TESTS PASSED! CNN is ready to use.")
        print("\nNext steps:")
        print("1. Prepare your real data (images/ and masks/)")
        print("2. Run: python cnn_implementation/train_segmentation.py --data_dir ./your_data")
    else:
        print("⚠️ Some tests failed. Check errors above.")
    print("="*60)
    
    # Cleanup
    import shutil
    if test_data_dir.exists():
        shutil.rmtree(test_data_dir)
        print(f"\nCleaned up test data: {test_data_dir}")

if __name__ == "__main__":
    main()
