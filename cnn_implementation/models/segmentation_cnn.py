"""
CNN Model for TNBC Tissue Segmentation and PD-L1 Quantification
Based on Project Proposal SMRIST

Purpose:
- Segment histopathology images into: Tumor, Immune, Stroma, Background
- Quantify PD-L1 expression per compartment
- Extract features for CPS++ scoring algorithm
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ResNetEncoder(nn.Module):
    """ResNet encoder for feature extraction"""
    def __init__(self, pretrained=True):
        super(ResNetEncoder, self).__init__()
        resnet = models.resnet50(pretrained=pretrained)
        
        # Remove the final FC and AvgPool layers
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1  # 256 channels
        self.layer2 = resnet.layer2  # 512 channels
        self.layer3 = resnet.layer3  # 1024 channels
        self.layer4 = resnet.layer4  # 2048 channels
        
    def forward(self, x):
        # Store intermediate features for skip connections
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        
        x2 = self.maxpool(x1)
        x2 = self.layer1(x2)  # 1/4 resolution
        
        x3 = self.layer2(x2)  # 1/8 resolution
        x4 = self.layer3(x3)  # 1/16 resolution
        x5 = self.layer4(x4)  # 1/32 resolution
        
        return x1, x2, x3, x4, x5


class DecoderBlock(nn.Module):
    """Decoder block with upsampling and skip connections"""
    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels // 2, 
                                           kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels // 2 + skip_channels, out_channels, 
                               kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 
                               kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x, skip):
        x = self.upsample(x)
        
        # Handle size mismatch
        if x.size() != skip.size():
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class TissueSegmentationCNN(nn.Module):
    """
    U-Net style CNN with ResNet50 backbone for tissue segmentation
    
    Segments into 4 classes:
    0: Background
    1: Tumor
    2: Immune
    3: Stroma
    """
    def __init__(self, num_classes=4, pretrained=True):
        super(TissueSegmentationCNN, self).__init__()
        self.num_classes = num_classes
        
        # Encoder
        self.encoder = ResNetEncoder(pretrained=pretrained)
        
        # Decoder with skip connections
        self.decoder4 = DecoderBlock(2048, 1024, 512)
        self.decoder3 = DecoderBlock(512, 512, 256)
        self.decoder2 = DecoderBlock(256, 256, 128)
        self.decoder1 = DecoderBlock(128, 64, 64)
        
        # Final upsampling to original resolution
        self.final_upsample = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)
        
    def forward(self, x):
        # Encoder
        x1, x2, x3, x4, x5 = self.encoder(x)
        
        # Decoder with skip connections
        d4 = self.decoder4(x5, x4)
        d3 = self.decoder3(d4, x3)
        d2 = self.decoder2(d3, x2)
        d1 = self.decoder1(d2, x1)
        
        # Final segmentation map
        out = self.final_upsample(d1)
        out = self.final_conv(out)
        
        return out


class VirtualStainingCNN(nn.Module):
    """
    U-Net for Virtual Staining: H&E → PD-L1 expression heatmap
    
    Same architecture as TissueSegmentationCNN but with:
    - 1-channel output (regression, not classification)
    - Sigmoid activation for 0-1 bounded output
    """
    def __init__(self, pretrained=True):
        super(VirtualStainingCNN, self).__init__()
        
        # Encoder (shared ResNet50 backbone)
        self.encoder = ResNetEncoder(pretrained=pretrained)
        
        # Decoder with skip connections
        self.decoder4 = DecoderBlock(2048, 1024, 512)
        self.decoder3 = DecoderBlock(512, 512, 256)
        self.decoder2 = DecoderBlock(256, 256, 128)
        self.decoder1 = DecoderBlock(128, 64, 64)
        
        # Final output: single channel (PD-L1 expression intensity)
        self.final_upsample = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Encoder
        x1, x2, x3, x4, x5 = self.encoder(x)
        
        # Decoder with skip connections
        d4 = self.decoder4(x5, x4)
        d3 = self.decoder3(d4, x3)
        d2 = self.decoder2(d3, x2)
        d1 = self.decoder1(d2, x1)
        
        # Output: PD-L1 expression heatmap (0-1)
        out = self.final_upsample(d1)
        out = self.final_conv(out)
        out = self.sigmoid(out)
        
        return out


class PDL1QuantificationCNN(nn.Module):
    """
    CNN for PD-L1 expression quantification per tissue compartment
    
    Takes:
    - H&E image
    - Segmentation mask (tumor/immune/stroma)
    
    Outputs:
    - PD-L1 intensity per compartment
    - Percentage of PD-L1+ cells per compartment
    """
    def __init__(self):
        super(PDL1QuantificationCNN, self).__init__()
        
        # Shared feature extractor
        resnet = models.resnet34(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        
        # Compartment-specific branches
        self.tumor_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)  # [intensity, percentage]
        )
        
        self.immune_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )
        
        self.stroma_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )
        
    def forward(self, image, segmentation_mask):
        """
        Args:
            image: (B, 3, H, W) - H&E stained image
            segmentation_mask: (B, H, W) - tissue segmentation (0=bg, 1=tumor, 2=immune, 3=stroma)
        
        Returns:
            Dictionary with PD-L1 quantification per compartment
        """
        # Extract features
        features = self.features(image)
        
        # Apply compartment-specific analysis
        tumor_features = self.tumor_branch(features)
        immune_features = self.immune_branch(features)
        stroma_features = self.stroma_branch(features)
        
        return {
            'tumor_pdl1': tumor_features,      # [intensity, percentage]
            'immune_pdl1': immune_features,
            'stroma_pdl1': stroma_features
        }


class CPSPlusPlusFeatureExtractor(nn.Module):
    """
    Feature extraction module for CPS++ scoring
    
    Combines:
    - CNN-derived segmentation
    - CNN-derived PD-L1 signals
    - Spatial interaction indices (computed externally)
    """
    def __init__(self):
        super(CPSPlusPlusFeatureExtractor, self).__init__()
        
        # Feature aggregation network
        self.feature_aggregator = nn.Sequential(
            nn.Linear(14, 64),  # 14 features from CNN outputs
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),
            nn.Linear(32, 1),  # CPS++ score
            nn.Sigmoid()
        )
        
    def forward(self, cnn_features):
        """
        Args:
            cnn_features: Tensor of shape (B, 14) containing:
                - Tumor percentage
                - Immune percentage
                - Stroma percentage
                - Tumor PD-L1 intensity
                - Tumor PD-L1 percentage
                - Immune PD-L1 intensity
                - Immune PD-L1 percentage
                - Stroma PD-L1 intensity
                - Stroma PD-L1 percentage
                - Additional morphological features (5)
        
        Returns:
            CPS++ score (0-1)
        """
        score = self.feature_aggregator(cnn_features)
        return score


class IntegratedTNBCModel(nn.Module):
    """
    Integrated model combining all components:
    1. Tissue segmentation
    2. PD-L1 quantification
    3. Feature extraction for CPS++
    """
    def __init__(self, num_classes=4):
        super(IntegratedTNBCModel, self).__init__()
        
        self.segmentation_model = TissueSegmentationCNN(num_classes=num_classes)
        self.pdl1_quantification = PDL1QuantificationCNN()
        self.cps_feature_extractor = CPSPlusPlusFeatureExtractor()
        
    def forward(self, image, return_intermediates=False):
        """
        End-to-end forward pass
        
        Args:
            image: (B, 3, H, W) - H&E stained histopathology image
            return_intermediates: If True, return all intermediate outputs
        
        Returns:
            Dictionary containing segmentation, PD-L1 quantification, and CPS++ score
        """
        # Step 1: Tissue segmentation
        segmentation_logits = self.segmentation_model(image)
        segmentation_pred = torch.argmax(segmentation_logits, dim=1)
        
        # Step 2: PD-L1 quantification per compartment
        pdl1_results = self.pdl1_quantification(image, segmentation_pred)
        
        # Step 3: Extract features for CPS++
        # Compute compartment statistics
        batch_size = image.size(0)
        cnn_features = []
        
        for b in range(batch_size):
            seg = segmentation_pred[b]
            
            # Compute percentages
            total_pixels = seg.numel()
            tumor_pct = (seg == 1).sum().float() / total_pixels
            immune_pct = (seg == 2).sum().float() / total_pixels
            stroma_pct = (seg == 3).sum().float() / total_pixels
            
            # Get PD-L1 values
            tumor_pdl1 = pdl1_results['tumor_pdl1'][b]
            immune_pdl1 = pdl1_results['immune_pdl1'][b]
            stroma_pdl1 = pdl1_results['stroma_pdl1'][b]
            
            # Placeholder morphological features (to be computed from image analysis)
            morph_features = torch.zeros(5, device=image.device)
            
            # Concatenate all features
            feature_vec = torch.cat([
                tumor_pct.unsqueeze(0),
                immune_pct.unsqueeze(0),
                stroma_pct.unsqueeze(0),
                tumor_pdl1,
                immune_pdl1,
                stroma_pdl1,
                morph_features
            ])
            cnn_features.append(feature_vec)
        
        cnn_features = torch.stack(cnn_features)
        
        # Compute CPS++ score
        cps_score = self.cps_feature_extractor(cnn_features)
        
        if return_intermediates:
            return {
                'segmentation_logits': segmentation_logits,
                'segmentation_pred': segmentation_pred,
                'pdl1_quantification': pdl1_results,
                'cnn_features': cnn_features,
                'cps_score': cps_score
            }
        else:
            return cps_score


def get_model(model_type='integrated', **kwargs):
    """
    Factory function to get specific model
    
    Args:
        model_type: 'segmentation', 'pdl1', 'integrated'
    """
    if model_type == 'segmentation':
        return TissueSegmentationCNN(**kwargs)
    elif model_type == 'pdl1':
        return PDL1QuantificationCNN(**kwargs)
    elif model_type == 'integrated':
        return IntegratedTNBCModel(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test the models
    print("Testing Tissue Segmentation CNN...")
    seg_model = TissueSegmentationCNN(num_classes=4)
    test_input = torch.randn(2, 3, 512, 512)
    seg_output = seg_model(test_input)
    print(f"Segmentation output shape: {seg_output.shape}")  # Should be (2, 4, 512, 512)
    
    print("\nTesting PD-L1 Quantification CNN...")
    pdl1_model = PDL1QuantificationCNN()
    test_mask = torch.randint(0, 4, (2, 512, 512))
    pdl1_output = pdl1_model(test_input, test_mask)
    print(f"PD-L1 quantification output keys: {pdl1_output.keys()}")
    
    print("\nTesting Integrated Model...")
    integrated_model = IntegratedTNBCModel()
    integrated_output = integrated_model(test_input, return_intermediates=True)
    print(f"Integrated model output keys: {integrated_output.keys()}")
    print(f"CPS++ score shape: {integrated_output['cps_score'].shape}")
    
    print("\nModel test successful!")
