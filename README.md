# Virtual Staining for PD-L1 Predictive Analysis (TNBC)

This repository hosts a deep learning pipeline designed to predict **PD-L1 expression** directly from **H&E histopathology images** via "Virtual Staining". 

By predicting the PD-L1 stain (DAB channel) from H&E morphology, we enable **CPS++ scoring** without requiring expensive immunohistochemistry (IHC) staining for every slide.

---

## 🌟 Key Features

*   **Virtual Staining CNN**: A custom **U-Net** with **ResNet50 Encoder** that translates H&E structure into PD-L1 protein expression maps.
*   **Regression Approach**: Instead of simple classification, the model predicts continuous stain intensity (0.0 to 1.0) for every pixel.
*   **Advanced Loss Function**: Uses **L1 + SSIM (Structural Similarity)** loss to ensure the predicted stain respects cell boundaries and tissue texture.
*   **Tile-Level & Slide-Level Scoring**: Automatically aggregates predictions to calculate PD-L1 metrics.

---

## 🛠️ Installation

```bash
cd cnn_implementation
pip install -r requirements_cnn.txt
```

---

## 🚀 Usage Guide

### 1. Data Preparation
Match H&E tiles with their corresponding PD-L1 IHC tiles and perform color deconvolution to extract the Ground Truth (DAB channel).

```bash
python cnn_implementation/prepare_virtual_staining_data.py \
    --he_dir path/to/he_images \
    --pdl1_dir path/to/pdl1_images \
    --output_dir training_data
```

### 2. Training
Train the Virtual Staining U-Net. Logs are saved to `runs/` for TensorBoard.

```bash
python cnn_implementation/train_virtual_staining.py \
    --data_dir training_data \
    --epochs 50 \
    --batch_size 16
```

### 3. Inference (Virtual Staining)
Run the trained model on new H&E images to generate "virtual" PD-L1 heatmaps and score CSVs.

```bash
python cnn_implementation/inference_virtual_staining.py \
    --model_path outputs/best_virtual_staining_loss.pth \
    --input_dir data_raw/test_images \
    --output_dir results
```

---

## 📂 Repository Structure

```
TNBC/
├── cnn_implementation/          # NEW: Virtual Staining Pipeline
│   ├── prepare_virtual_staining_data.py  # Data matching & deconvolution
│   ├── train_virtual_staining.py         # Main training script
│   ├── inference_virtual_staining.py     # Inference & Visualization
│   ├── models/
│   │   ├── segmentation_cnn.py           # VirtualStainingCNN (U-Net) class
│   │   └── virtual_staining_dataset.py   # Dataset class
│
├── legacy_scripts/              # Old scripts (SegFormer, simple classification)
├── PD-L1_Annotator/             # Manual annotation tool
└── data_raw/                    # Input images (not tracked by git)
```

---

## 🧠 Model Architecture

The model is an **Image-to-Image Translation** network:
1.  **Input**: 512x512 RGB H&E Image.
2.  **Encoder**: ResNet50 (ImageNet pretrained) extracts morphological features (e.g., nuclear irregularity, immune infiltration).
3.  **Decoder**: U-Net decoder upsamples features to restore spatial resolution.
4.  **Output**: Single-channel Sigmoid map representing PD-L1 stain probability.

---

## 📜 Legacy Modules
*   **PD-L1 Predictor**: Old classification-based approach.
*   **Annotation Tool**: Helper for manual pathologist annotations.
