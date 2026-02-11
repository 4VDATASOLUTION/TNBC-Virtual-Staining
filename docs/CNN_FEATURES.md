# Features Used in TNBC CNN Implementation

## Overview

The CNN extracts multiple levels of features from H&E histopathology images for tissue segmentation and PD-L1 quantification, which feed into CPS++ scoring.

---

## 1. 🔬 IMAGE-LEVEL FEATURES (Input)

### H&E Stained Images
- **Input Size**: 512×512 pixels, 3 channels (RGB)
- **Normalization**: Mean=(0.9357, 0.8253, 0.8998), Std=(0.0787, 0.1751, 0.1125)
- **Source**: Hematoxylin and Eosin stained tissue sections

**What H&E reveals:**
- Hematoxylin (blue/purple) → Cell nuclei
- Eosin (pink) → Cytoplasm and extracellular matrix
- Color patterns indicate tissue types

---

## 2. 🧠 DEEP FEATURES (CNN Encoder - ResNet50)

The ResNet50 backbone extracts hierarchical features at multiple scales:

### Layer 1 (Low-level features)
- **Channels**: 64
- **Resolution**: 1/2 of input (256×256)
- **Captures**: Edges, textures, basic patterns

### Layer 2 (Mid-level features)  
- **Channels**: 256
- **Resolution**: 1/4 of input (128×128)
- **Captures**: Cell shapes, tissue structures

### Layer 3 (High-level features)
- **Channels**: 512
- **Resolution**: 1/8 of input (64×64)
- **Captures**: Tissue organization patterns

### Layer 4 (Abstract features)
- **Channels**: 1024
- **Resolution**: 1/16 of input (32×32)
- **Captures**: Complex tissue architectures

### Layer 5 (Semantic features)
- **Channels**: 2048
- **Resolution**: 1/32 of input (16×16)
- **Captures**: Tumor vs. immune vs. stroma characteristics

**These features are learned during training and automatically encode:**
- Cell density patterns
- Nuclear morphology
- Tissue architecture
- Spatial organization
- Color/staining characteristics

---

## 3. 📊 SEGMENTATION OUTPUT FEATURES

The U-Net decoder produces per-pixel segmentation:

### Tissue Compartment Masks
**4 Classes** - Each pixel classified as:

#### Class 0: Background
- Non-tissue areas
- Artifacts, slide edges, empty space

#### Class 1: Tumor Cells
- **Features detected**:
  - High nuclear density
  - Abnormal cell morphology
  - Irregular arrangement
  - Epithelial characteristics

#### Class 2: Immune Cells
- **Features detected**:
  - Lymphocyte morphology
  - Small, round nuclei
  - Infiltrating patterns
  - Immune cell markers (in staining)

#### Class 3: Stromal Tissue
- **Features detected**:
  - Connective tissue patterns
  - Fibroblast organization
  - Extracellular matrix
  - Lower cell density

**Output**: (Batch, 4, 512, 512) - probability map for each class

---

## 4. 🎯 QUANTITATIVE FEATURES (From Segmentation)

### Compartment Statistics
Computed from segmentation masks:

```python
# Per image:
- tumor_percentage: % of pixels classified as tumor
- immune_percentage: % of pixels classified as immune  
- stroma_percentage: % of pixels classified as stroma
- background_percentage: % of pixels classified as background

# Spatial features:
- tumor_area: Total tumor region area
- immune_infiltration: Immune cells within/near tumor
- stroma_distribution: Stromal tissue organization
```

---

## 5. 💊 PD-L1 QUANTIFICATION FEATURES

From the PDL1QuantificationCNN model:

### Per-Compartment PD-L1 Metrics

#### Tumor Compartment
- **PD-L1 intensity**: Average expression level (0-1)
- **PD-L1 percentage**: % of PD-L1+ tumor cells

#### Immune Compartment  
- **PD-L1 intensity**: Average expression on immune cells
- **PD-L1 percentage**: % of PD-L1+ immune cells

#### Stroma Compartment
- **PD-L1 intensity**: Average expression in stroma
- **PD-L1 percentage**: % of PD-L1+ stromal cells

**Total**: 6 PD-L1 features (2 per compartment)

---

## 6. 🔢 CPS++ INPUT FEATURES

The final feature vector for CPS++ scoring contains **14 features**:

### Segmentation-derived (3 features)
1. `tumor_percentage` - % tumor tissue
2. `immune_percentage` - % immune cells
3. `stroma_percentage` - % stromal tissue

### PD-L1 quantification (6 features)
4. `tumor_pdl1_intensity` - PD-L1 expression in tumor
5. `tumor_pdl1_percentage` - % PD-L1+ tumor cells
6. `immune_pdl1_intensity` - PD-L1 expression in immune cells
7. `immune_pdl1_percentage` - % PD-L1+ immune cells
8. `stroma_pdl1_intensity` - PD-L1 expression in stroma
9. `stroma_pdl1_percentage` - % PD-L1+ stromal cells

### Morphological features (5 features - placeholder)
10-14. Additional spatial/morphological characteristics:
   - Tumor-immune interface length
   - Immune cell clustering index
   - Spatial heterogeneity
   - Cell density gradients
   - Architectural patterns

**Feature Vector Shape**: (Batch, 14)

---

## 7. 📈 FINAL OUTPUT: CPS++ SCORE

From the CPSPlusPlusFeatureExtractor:

```python
Input: 14 features
↓
Dense Layer (14 → 64) + ReLU + BatchNorm + Dropout
↓
Dense Layer (64 → 32) + ReLU + BatchNorm + Dropout  
↓
Dense Layer (32 → 1) + Sigmoid
↓
Output: CPS++ score (0-1 scale)
```

**CPS++ Score**: Single value (0-100) representing:
- Combined PD-L1 expression score
- Accounts for all tissue compartments
- Weighted by spatial interactions
- Predicts immunotherapy response

---

## 🔍 FEATURE EXTRACTION PIPELINE

```
H&E Image (512×512×3)
        ↓
ResNet50 Encoder (Multi-scale features)
        ↓
U-Net Decoder (Skip connections)
        ↓
Segmentation Map (512×512, 4 classes)
        ├─→ Compartment Statistics (3 features)
        └─→ PD-L1 Quantification Module
                ↓
            PD-L1 Features (6 features)
                ↓
        Feature Aggregation (14 features total)
                ↓
        CPS++ Scoring Network
                ↓
        Final CPS++ Score (0-100)
```

---

## 📋 FEATURE IMPORTANCE (from Proposal)

### Primary Features (CNN-derived)
✅ **Tumor/immune/stroma segmentation** - Most critical  
✅ **PD-L1 expression per compartment** - Key biomarker  
✅ **Spatial organization** - Tumor-immune interactions  

### Supporting Features
- Cell density patterns
- Nuclear morphology
- Tissue architecture
- Immune infiltration patterns

---

## 💻 Accessing Features in Code

### During Training:
```python
# In train_segmentation.py
outputs = model(images)  # (B, 4, H, W) segmentation logits
predictions = torch.argmax(outputs, dim=1)  # (B, H, W) class predictions
```

### During Inference:
```python
# In inference_pipeline.py
results = model(image, return_intermediates=True)

# Access features:
segmentation = results['segmentation_pred']  # Tissue compartments
pdl1_values = results['pdl1_quantification']  # PD-L1 per compartment
cnn_features = results['cnn_features']  # 14-feature vector
cps_score = results['cps_score']  # Final CPS++ score
```

---

## 🎓 Summary

**The CNN automatically learns to extract:**
1. Low-level image features (edges, textures)
2. Mid-level features (cell shapes, patterns)  
3. High-level features (tissue organization)
4. Semantic features (tumor vs. immune vs. stroma)

**These are combined into:**
- Segmentation masks (pixel-level classification)
- Quantitative statistics (percentages, areas)
- PD-L1 expression metrics (per compartment)
- Spatial features (tissue interactions)

**Final output:**
- CPS++ score for immunotherapy prediction
- Interpretable compartment statistics
- Visualization-ready segmentation maps

All features are **learned end-to-end** from annotated training data, requiring no manual feature engineering!
