# TNBC - Triple-Negative Breast Cancer Analysis

CNN-based image analysis for tumor, immune, and stromal segmentation with PD-L1 quantification and CPS++ scoring.

## 📁 Project Structure

```
TNBC/
├── cnn_implementation/          # Main CNN implementation (NEW)
│   ├── models/                  # CNN model architectures
│   │   ├── segmentation_cnn.py  # U-Net with ResNet backbone
│   │   └── dataset.py           # Data loading & augmentation
│   ├── train_segmentation.py    # Training script
│   ├── inference_pipeline.py    # Inference & CPS++ scoring
│   ├── prepare_dataset.py       # Data preparation utilities
│   └── requirements_cnn.txt     # Python dependencies
│
├── PD-L1_Annotator/             # Annotation tool for pathologists
│   └── annotation_tool.py       # Interactive labeling interface
│
├── PD-L1_predictor/             # Original PD-L1 predictor
│   ├── predict_on_folder.py     # Inference script
│   └── models/                  # Pre-trained models
│
├── data_raw/                    # Raw data and annotations
│   ├── 02-008_HE_A12_v2_s13/           # H&E images
│   ├── 02-008_PD1(NAT105)-CellMarque/  # PD-1 staining
│   ├── 02-008_PDL1(SP142)-Springbio/   # PD-L1 staining
│   └── *.csv, *.txt, *.xlsx            # Ground truth & results
│
├── legacy_scripts/              # Previous implementations
│   ├── train_pd1_model.py       # Old PD-1 training script
│   ├── merge_predictions*.py    # Prediction merging utilities
│   └── generate_*.py            # Data generation scripts
│
├── docs/                        # Documentation
│   ├── README_CNN.md            # Complete CNN guide
│   └── Project Proposal SMRIST.pdf  # Original proposal
│
└── outputs/                     # Training outputs (created during training)
    └── segmentation_YYYYMMDD_HHMMSS/
        ├── best_model_iou.pth
        ├── best_model_loss.pth
        └── tensorboard/
```

## 🚀 Quick Start

### 1. Install Dependencies

```powershell
cd cnn_implementation
python -m pip install -r requirements_cnn.txt
```

### 2. Prepare Data

Organize your data:
```
your_data/
├── images/   # H&E histopathology images
└── masks/    # Segmentation masks (0=bg, 1=tumor, 2=immune, 3=stroma)
```

### 3. Train Model

```powershell
python train_segmentation.py `
    --data_dir ../your_data `
    --output_dir ../outputs `
    --epochs 50 `
    --batch_size 4
```

### 4. Run Inference

```powershell
python inference_pipeline.py `
    --image_folder ../test_images `
    --model_path ../outputs/best_model_iou.pth `
    --visualize
```

## 📖 Documentation

- **[Complete CNN Guide](docs/README_CNN.md)** - Detailed implementation documentation
- **[Project Proposal](docs/Project%20Proposal%20SMRIST.pdf)** - Original research proposal

## 🏗️ CNN Architecture

### Tasks (from Proposal)
1. **Tissue Segmentation**: Tumor, immune, stroma classification
2. **PD-L1 Quantification**: Expression levels per compartment
3. **Feature Extraction**: CNN-derived features for CPS++ scoring

### Model
- **Architecture**: U-Net with ResNet50 backbone
- **Input**: 512×512 RGB H&E images
- **Output**: 4-class segmentation (background, tumor, immune, stroma)

## 🔧 Tools Included

### CNN Implementation (New)
State-of-the-art segmentation with:
- Advanced data augmentation
- Dice + CrossEntropy loss
- TensorBoard logging
- Automatic checkpointing

### Annotation Tool
Interactive GUI for pathologists to label images

### Virtual Staining Pipeline (Newest)
Predicts PD-L1 expression heatmaps directly from H&E images using deep learning (CycleGAN/Regression U-Net concept).
- **Structure-to-Stain Translation**: Infers protein expression from morphology.
- **Metrics**: L1 Loss, SSIM, Pearson Correlation.
- **Output**: continuous 0-1 probability maps for CPS++ scoring.

### PD-L1 Predictor (Legacy)
Pre-trained model for PD-L1 status prediction from H&E images

## 📊 Expected Performance

- **Mean IoU**: 0.75-0.85
- **Tumor IoU**: 0.80-0.90
- **Immune IoU**: 0.70-0.80
- **Stroma IoU**: 0.65-0.75

## 🔬 Research Context

**Project**: CNN Image Classification for Triple-Negative Breast Cancer (SMRIST)

**Purpose**: CNNs perform tumor, immune, and stromal segmentation and PD-L1 quantification from histopathology images, generating features for CPS++ scoring algorithm.

## 📝 Citation

```
"CNN Image Classification for Triple-Negative Breast Cancer"
Project Proposal SMRIST, 2026
```

## 🛠️ System Requirements

- **Python**: 3.8+
- **PyTorch**: 1.11.0+
- **RAM**: 8GB+ (16GB recommended)
- **GPU**: Optional (CUDA-compatible for faster training)

## 📞 Getting Help

See detailed guides:
- Installation: `docs/README_CNN.md` (Setup section)
- Training: `docs/README_CNN.md` (Training section)
- Inference: `docs/README_CNN.md` (Inference section)
