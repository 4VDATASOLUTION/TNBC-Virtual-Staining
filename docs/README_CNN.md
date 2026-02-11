# TNBC CNN Implementation - Complete Guide

## Project Overview

This implementation provides a complete CNN-based pipeline for **Triple-Negative Breast Cancer (TNBC)** tissue analysis, based on the SMRIST project proposal.

### CNN Purpose (from Proposal)

**"CNNs are used for tumor, immune, and stromal segmentation and PD-L1 quantification from histopathology and multiplex images, generating features required for CPS++ scoring."**

## Architecture Components

### 1. Tissue Segmentation CNN (`models/segmentation_cnn.py`)

**Architecture**: U-Net with ResNet50 backbone

**Tasks**:
- Segment histopathology images into 4 classes:
  - Background (0)
  - Tumor (1)
  - Immune (2)
  - Stroma (3)

**Key Features**:
- Pre-trained ResNet50 encoder for robust feature extraction
- Skip connections for preserving spatial information
- Multi-scale feature fusion

### 2. PD-L1 Quantification CNN

**Tasks**:
- Quantify PD-L1 expression per tissue compartment
- Output: Intensity and percentage of PD-L1+ cells

### 3. CPS++ Feature Extractor

**Tasks**:
- Extract CNN-derived features for CPS++ scoring
- Combines segmentation with PD-L1 signals
- Generates unified biomarker score

## Project Structure

```
TNBC/
├── models/
│   ├── segmentation_cnn.py       # CNN model definitions
│   └── dataset.py                # Dataset classes
├── train_segmentation.py         # Training script
├── inference_pipeline.py         # Inference and CPS++ scoring
├── prepare_dataset.py            # Data preparation utilities
├── requirements_cnn.txt          # Python dependencies
└── README_CNN.md                 # This file
```

## Installation

### Step 1: Install Dependencies

```bash
pip install -r requirements_cnn.txt
```

For CUDA support (GPU acceleration):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Step 2: Prepare Your Data

**Required data structure**:
```
your_data/
├── images/
│   ├── sample_001.jpeg
│   ├── sample_002.jpeg
│   └── ...
└── masks/
    ├── sample_001.png    # 0=bg, 1=tumor, 2=immune, 3=stroma
    ├── sample_002.png
    └── ...
```

**Verify dataset**:
```bash
python prepare_dataset.py --data_dir ./your_data --action verify
```

**Compute statistics**:
```bash
python prepare_dataset.py --data_dir ./your_data --action stats
```

**Create train/val split**:
```bash
python prepare_dataset.py --data_dir ./your_data --action split --output dataset_split.json
```

## Training

### Basic Training Command

```bash
python train_segmentation.py \
    --data_dir ./your_data \
    --output_dir ./outputs \
    --epochs 100 \
    --batch_size 8 \
    --learning_rate 0.0001
```

### Training Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--data_dir` | Directory with images/ and masks/ | Required |
| `--output_dir` | Where to save outputs | `./outputs` |
| `--epochs` | Number of training epochs | 100 |
| `--batch_size` | Batch size | 8 |
| `--learning_rate` | Learning rate | 0.0001 |
| `--image_size` | Input image size | 512 |
| `--num_classes` | Number of classes | 4 |
| `--patience` | Early stopping patience | 15 |
| `--num_workers` | Data loading workers | 4 |

### Training Features

✓ **Data Augmentation**: Rotations, flips, color jitter, Gaussian blur  
✓ **Loss Function**: Combined Dice Loss + CrossEntropy  
✓ **Metrics**: Pixel accuracy, per-class IoU, mean IoU  
✓ **Learning Rate Scheduling**: ReduceLROnPlateau  
✓ **Early Stopping**: Patience-based  
✓ **Checkpointing**: Best model by loss and IoU  
✓ **TensorBoard Logging**: Real-time training visualization  

### Monitor Training

```bash
tensorboard --logdir outputs/segmentation_YYYYMMDD_HHMMSS/tensorboard
```

## Inference

### Run Inference on New Images

```bash
python inference_pipeline.py \
    --image_folder ./test_images \
    --model_path ./outputs/segmentation_YYYYMMDD_HHMMSS/best_model_iou.pth \
    --output_dir ./inference_results \
    --visualize
```

### Inference Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--image_folder` | Folder with images to process | Required |
| `--model_path` | Path to trained model | Required |
| `--output_dir` | Where to save results | `./inference_results` |
| `--model_type` | Model type: segmentation/integrated | `segmentation` |
| `--visualize` | Generate visualization images | Flag |

### Output Files

After inference, you'll get:

1. **`segmentation_summary.csv`**: 
   - CPS++ scores
   - Compartment percentages per image

2. **`segmentation_results.json`**: 
   - Detailed results in JSON format

3. **`*_segmentation.png`**: 
   - Visualization overlays (if `--visualize` flag used)

4. **`*_mask.png`**: 
   - Raw segmentation masks

## CNN Model Details

### TissueSegmentationCNN

```python
from models.segmentation_cnn import TissueSegmentationCNN

model = TissueSegmentationCNN(num_classes=4, pretrained=True)

# Input: (B, 3, 512, 512) - H&E images
# Output: (B, 4, 512, 512) - segmentation logits
```

### IntegratedTNBCModel

Full end-to-end model combining:
- Tissue segmentation
- PD-L1 quantification  
- CPS++ feature extraction

```python
from models.segmentation_cnn import IntegratedTNBCModel

model = IntegratedTNBCModel(num_classes=4)

# Input: (B, 3, 512, 512) - H&E images
# Output: Dictionary with segmentation, PD-L1 values, and CPS++ score
```

## CNN Tasks Summary (from Proposal)

### Layer: Processing & Analytics

**Segmentation & Phenotyping (CNNs)**:
1. ✅ Tumor/immune/stroma segmentation
2. ✅ Cell detection
3. ✅ PD-L1 quantification per compartment

### Integration with CPS++

The CNN provides inputs to the CPS++ algorithm:
- Tissue compartment proportions
- PD-L1 expression levels per compartment
- Spatial interaction features
- Morphological characteristics

**Note**: Full CPS++ implementation requires spatial analysis algorithms (not included in basic CNN training).

## Performance Metrics

### Segmentation Metrics
- **Pixel Accuracy**: Overall correct pixel classification
- **IoU per class**: Intersection over Union for each tissue type
- **Mean IoU (mIoU)**: Average IoU across all classes
- **Dice Coefficient**: Alternative overlap metric

### Expected Performance
With proper annotations:
- mIoU: 0.75-0.85
- Tumor IoU: 0.80-0.90
- Immune IoU: 0.70-0.80
- Stroma IoU: 0.65-0.75

## Troubleshooting

### Common Issues

**1. CUDA out of memory**
```bash
# Reduce batch size
python train_segmentation.py --batch_size 4
```

**2. Slow training**
```bash
# Reduce image size
python train_segmentation.py --image_size 256

# Use fewer workers if system is slow
python train_segmentation.py --num_workers 2
```

**3. Missing albumentations**
```bash
pip install albumentations
```

## Next Steps

### 1. Prepare Annotated Data
- Use the PD-L1_Annotator tool to create annotations
- Convert annotations to segmentation masks
- Organize into images/ and masks/ folders

### 2. Train Segmentation Model
- Start with small learning rate (1e-4)
- Monitor validation IoU
- Stop when validation plateaus

### 3. Run Inference
- Test on held-out images
- Visualize segmentations
- Validate CPS++ scores with clinical data

### 4. Integrate with AI-Agent
- Implement adaptive retraining across cohorts
- Add spatial interaction analysis
- Complete CPS++ scoring algorithm

## Citation

If you use this implementation, please cite the SMRIST project proposal:

```
"CNN Image Classification for Triple-Negative Breast Cancer"
Project Proposal SMRIST, 2026
```

## License

See individual LICENSE files in subdirectories.

## Contact

For questions about the CNN implementation, refer to the project proposal documentation.
