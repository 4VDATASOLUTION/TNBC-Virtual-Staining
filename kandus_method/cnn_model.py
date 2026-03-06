"""
cnn_model.py — Kandu's Method CNN + MIL Classifier
=====================================================
Implements an attention-based Multiple Instance Learning (MIL) classifier
for weakly-supervised PD-L1 biomarker prediction from H&E tiles.

Architecture
------------
  Input : Bag of N tiles  [N, 3, 512, 512]
  Stage 1 — Feature Extraction
      ResNet50 / EfficientNet-B0 backbone (ImageNet pretrained)
      → strips global average pool + classifier head
      → feature map per tile [N, feature_dim]
  Stage 2 — MIL Attention Aggregation (Ilse et al., 2018)
      Two-layer attention network → attention weights [N, 1]
      Weighted sum → aggregated POI embedding [1, feature_dim]
  Stage 3 — Classification Head
      Linear → sigmoid → POI-level probability scalar

Outputs
-------
  poi_prob      : float [0, 1] — POI-level PD-L1 positivity probability
  tile_probs    : [N]           — per-tile probabilities
  attn_weights  : [N, 1]       — attention weights (interpretability)

Usage
-----
  from kandus_method.cnn_model import MILClassifier
  model = MILClassifier(backbone='resnet50')
  poi_prob, tile_probs, attn = model(bag_tensor)   # bag_tensor: [N, 3, 512, 512]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
    _TIMM_AVAILABLE = True
except ImportError:
    _TIMM_AVAILABLE = False


# ---------------------------------------------------------------------------
# Backbone factory
# ---------------------------------------------------------------------------

def _build_backbone(name: str) -> tuple[nn.Module, int]:
    """
    Load a pretrained CNN backbone, strip the classification head,
    and return (backbone_module, feature_dim).

    Supported names
    ---------------
      'resnet50'         → 2048-d features  (standard)
      'resnet50d'        → 2048-d features  (improved stem + avg-pool downsample)
      'resnet101'        → 2048-d features  (deeper, better accuracy — RECOMMENDED)
      'resnet152'        → 2048-d features  (deepest ResNet)
      'efficientnet_b0'  → 1280-d features  (lightweight)
      'efficientnet_b2'  → 1408-d features  (better than b0)
    """
    if not _TIMM_AVAILABLE:
        raise ImportError(
            "timm is required for backbone loading. "
            "Install with: pip install timm"
        )

    _TIMM_MAP = {
        'resnet50':         ('resnet50',         2048),
        'resnet50d':        ('resnet50d',         2048),   # avg-pool downsampling, better accuracy
        'resnet101':        ('resnet101',         2048),   # RECOMMENDED: deep + strong features
        'resnet152':        ('resnet152',         2048),
        'efficientnet':     ('efficientnet_b0',   1280),
        'efficientnet_b0':  ('efficientnet_b0',   1280),
        'efficientnet_b2':  ('efficientnet_b2',   1408),
    }

    if name not in _TIMM_MAP:
        raise ValueError(
            f"Unsupported backbone '{name}'. "
            f"Choose from: {list(_TIMM_MAP.keys())}"
        )

    timm_name, feature_dim = _TIMM_MAP[name]
    backbone = timm.create_model(timm_name, pretrained=True, num_classes=0)
    actual_dim = backbone.num_features
    # Use actual dim in case timm internal changes
    return backbone, actual_dim


# ---------------------------------------------------------------------------
# Attention-based MIL aggregation (Ilse et al., 2018)
# ---------------------------------------------------------------------------

class AttentionMIL(nn.Module):
    """
    Gated attention mechanism for MIL.

    For a bag of N tile features [N, D]:
      - two-layer attention network computes raw scores [N, 1]
      - softmax → normalized attention weights
      - weighted sum of tile features → bag embedding [1, D]

    Parameters
    ----------
    feature_dim  : int   — input feature dimension (D)
    hidden_dim   : int   — hidden projection dimension (L)
    dropout      : float — dropout on attention weights
    """

    def __init__(self, feature_dim: int, hidden_dim: int = 256, dropout: float = 0.25):
        super().__init__()

        # Attention gate V (tanh branch)
        self.attention_V = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
        )
        # Attention gate U (sigmoid branch — gating)
        self.attention_U = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Sigmoid(),
        )
        # Final scoring layer
        self.attention_w = nn.Linear(hidden_dim, 1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        h : [N, D] — tile-level feature vectors

        Returns
        -------
        z       : [1, D] — attention-weighted bag embedding
        weights : [N, 1] — normalized attention weights (softmax)
        """
        # Gated attention
        a_v = self.attention_V(h)            # [N, L]
        a_u = self.attention_U(h)            # [N, L]
        a   = self.attention_w(a_v * a_u)    # [N, 1]  (element-wise gating)

        weights = F.softmax(a, dim=0)        # [N, 1]  (normalize across tiles)
        weights = self.dropout(weights)

        # Weighted aggregation: z = h^T * weights
        z = (weights.T @ h)                  # [1, D]
        return z, weights


# ---------------------------------------------------------------------------
# Full MIL Classifier
# ---------------------------------------------------------------------------

class MILClassifier(nn.Module):
    """
    Attention-based MIL classifier for H&E tile bags.

    Parameters
    ----------
    backbone    : str   — 'resnet50' (default) or 'efficientnet_b0'
    hidden_dim  : int   — attention hidden dim
    dropout     : float — attention dropout
    freeze_bn   : bool  — freeze backbone BatchNorm layers (useful for small bags)

    Forward
    -------
    Input:  bag [N, 3, 512, 512]  — N tiles from one POI
    Output: (poi_prob, tile_probs, attn_weights)
      poi_prob     : scalar float [0, 1]  — POI-level PD-L1 probability
      tile_probs   : [N]                  — per-tile probabilities (from tile head)
      attn_weights : [N, 1]              — MIL attention weights
    """

    def __init__(
        self,
        backbone:       str   = 'resnet101',
        hidden_dim:     int   = 256,
        dropout:        float = 0.25,
        freeze_bn:      bool  = False,
        use_checkpoint: bool  = True,    # gradient checkpointing: saves ~50% VRAM
    ):
        super().__init__()

        self.backbone_name    = backbone
        self.use_checkpoint   = use_checkpoint
        self.feature_extractor, self.feature_dim = _build_backbone(backbone)

        if freeze_bn:
            self._freeze_backbone_bn()

        # Tile-level head: lightweight projection → single probability
        self.tile_head = nn.Sequential(
            nn.Linear(self.feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        # MIL attention aggregation
        self.attention_mil = AttentionMIL(
            feature_dim=self.feature_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        # POI-level head
        self.poi_head = nn.Sequential(
            nn.Linear(self.feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    # ------------------------------------------------------------------
    def _freeze_backbone_bn(self):
        """Freeze all BatchNorm layers in the backbone (eval mode)."""
        for m in self.feature_extractor.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    # ------------------------------------------------------------------
    def extract_features(
        self,
        tiles: torch.Tensor,
        sub_batch: int = 8,
    ) -> torch.Tensor:
        """
        Run backbone in small sub-batches to avoid CUDA OOM.
        Gradient checkpointing recomputes activations during backward
        instead of storing them, cutting VRAM ~50%.

        Parameters
        ----------
        tiles     : [N, 3, 512, 512]
        sub_batch : int  tiles per GPU forward call (default 8)

        Returns
        -------
        h : [N, feature_dim]
        """
        from torch.utils.checkpoint import checkpoint as grad_ckpt

        chunks = []
        for i in range(0, tiles.size(0), sub_batch):
            chunk = tiles[i : i + sub_batch]
            if self.use_checkpoint and chunk.requires_grad:
                # Recompute activations during backward — saves VRAM
                feat = grad_ckpt(self.feature_extractor, chunk, use_reentrant=False)
            else:
                feat = self.feature_extractor(chunk)
            chunks.append(feat)
        return torch.cat(chunks, dim=0)    # [N, D]

    # ------------------------------------------------------------------
    def forward(
        self,
        bag:       torch.Tensor,
        sub_batch: int = 8,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        bag       : [N, 3, 512, 512]  tiles from one POI
        sub_batch : int  tiles per GPU forward chunk (reduce if OOM)

        Returns
        -------
        poi_prob     : scalar [0, 1]
        tile_probs   : [N]
        attn_weights : [N, 1]
        """
        # Stage 1: Sub-batch feature extraction (memory-efficient)
        h = self.extract_features(bag, sub_batch=sub_batch)   # [N, D]

        # Stage 2: Per-tile probability
        tile_probs = self.tile_head(h).squeeze(-1)             # [N]

        # Stage 3: Attention MIL aggregation
        z, attn_weights = self.attention_mil(h)                # [1,D], [N,1]

        # Stage 4: POI-level prediction
        poi_prob = self.poi_head(z).squeeze()                  # scalar

        return poi_prob, tile_probs, attn_weights

    # ------------------------------------------------------------------
    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def get_num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Convenience: build model from config dict
# ---------------------------------------------------------------------------

def build_model(config: dict) -> MILClassifier:
    """
    Build a MILClassifier from a config dictionary.

    Example config
    --------------
    {
        "backbone":   "resnet50",
        "hidden_dim": 256,
        "dropout":    0.25,
        "freeze_bn":  False,
    }
    """
    return MILClassifier(
        backbone   = config.get("backbone",   "resnet50"),
        hidden_dim = config.get("hidden_dim", 256),
        dropout    = config.get("dropout",    0.25),
        freeze_bn  = config.get("freeze_bn",  False),
    )
