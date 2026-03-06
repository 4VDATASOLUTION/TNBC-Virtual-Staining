# Kandu's Method — Computational Pathology Pipeline
"""
kandus_method
=============
Weakly-supervised CNN/MIL pipeline for PD-L1 biomarker prediction
in Triple-Negative Breast Cancer (TNBC) histopathology images.

Modules
-------
cnn_model        : MIL classifier (ResNet50 / EfficientNet-B0 backbone)
dataset_kandu    : Dataset loader and H&E tiling
train_cnn        : Training entry-point
infer_cnn        : Inference / feature extraction entry-point
"""

__version__ = "0.1.0"
__author__  = "Kandu's Method"
