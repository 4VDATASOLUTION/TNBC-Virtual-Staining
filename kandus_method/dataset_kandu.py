"""
dataset_kandu.py — Kandu's Method Dataset & Tiling Module
==========================================================
Provides two main classes:

  KanduDataset   — high-level dataset iterating patient → POI structure
  HETileDataset  — PyTorch Dataset for H&E tiles extracted from one POI image

Dataset directory structure expected by KanduDataset
------------------------------------------------------
  dataset_root/
    patient_001/
      poi_01/
        HE_image.png
        PDL1_image.png
        PD1_image.png        (optional)
        labels.json
      poi_02/ ...
      ...
    patient_002/ ...

labels.json format (per POI)
-----------------------------
  {
    "PDL1_label": 1,          # 1 = positive, 0 = negative  (weak supervision)
    "PD1_label": 0,           # optional
    "stain_score": 2.5,       # optional quantitative score
    "patient_id": "patient_001",
    "poi_id": "poi_01"
  }

Tiling
------
  Tile size : 512 × 512 px
  Stride    : 256 px  (50 % overlap)
  Tiles that fall outside the image boundary are discarded (no padding).
  Each tile is returned with its (x, y) top-left pixel coordinate.

Usage
-----
  # 1. High-level dataset
  from kandus_method.dataset_kandu import KanduDataset
  ds = KanduDataset("path/to/dataset")
  poi_meta = ds[0]           # dict with metadata + tile dataset

  # 2. Low-level tiling
  from kandus_method.dataset_kandu import HETileDataset
  tile_ds = HETileDataset("path/to/HE_image.png")
  tile_tensor, (x, y) = tile_ds[0]
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


# ---------------------------------------------------------------------------
# Image normalisation statistics (H&E histopathology)
# Mean / std computed over large H&E cohorts (ImageNet stats also work well)
# ---------------------------------------------------------------------------

_HE_MEAN = (0.485, 0.456, 0.406)
_HE_STD  = (0.229, 0.224, 0.225)

_he_transform = T.Compose([
    T.ToTensor(),                               # [H, W, C] uint8 → [C, H, W] float [0,1]
    T.Normalize(mean=_HE_MEAN, std=_HE_STD),
])


# ---------------------------------------------------------------------------
# Utility: load image safely
# ---------------------------------------------------------------------------

def _load_image_rgb(path: str | Path) -> np.ndarray:
    """Load an image from disk; convert BGR→RGB; return uint8 ndarray [H, W, 3]."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# HETileDataset
# ---------------------------------------------------------------------------

class HETileDataset(Dataset):
    """
    Tile a single H&E image into 512×512 patches with a configurable stride.

    Parameters
    ----------
    he_image_path : str | Path  — path to HE_image.png (or any raster image)
    tile_size     : int         — tile height = tile width (default 512)
    stride        : int         — step between tile top-left corners (default 256)
    transform     : callable    — optional additional transform; if None, uses
                                  standard H&E normalisation.

    Returns (per index)
    -------------------
    tile  : torch.Tensor [3, tile_size, tile_size]  — normalised RGB tile
    coord : tuple (x, y)                           — top-left pixel coordinate
    """

    def __init__(
        self,
        he_image_path: str | Path,
        tile_size: int = 512,
        stride:    int = 256,
        transform  = None,
    ):
        super().__init__()
        self.he_image_path = Path(he_image_path)
        self.tile_size = tile_size
        self.stride    = stride
        self.transform = transform if transform is not None else _he_transform

        # Load image once; store as numpy array
        self.image = _load_image_rgb(self.he_image_path)   # [H, W, 3] uint8
        self.H, self.W = self.image.shape[:2]

        # Pre-compute all valid (x, y) top-left coordinates
        self.coords = self._compute_tile_coords()

    # ------------------------------------------------------------------
    def _compute_tile_coords(self) -> list[tuple[int, int]]:
        """
        Return list of (x, y) top-left pixel coords for all valid tiles.
        A tile is valid if it fits entirely within the image.
        """
        coords = []
        y = 0
        while y + self.tile_size <= self.H:
            x = 0
            while x + self.tile_size <= self.W:
                coords.append((x, y))
                x += self.stride
            y += self.stride
        return coords

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.coords)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, tuple[int, int]]:
        x, y = self.coords[idx]
        tile = self.image[y : y + self.tile_size, x : x + self.tile_size]  # [H, W, 3]
        tile_tensor = self.transform(tile)
        return tile_tensor, (x, y)

    # ------------------------------------------------------------------
    @property
    def image_size(self) -> tuple[int, int]:
        """(width, height) of the source image."""
        return (self.W, self.H)

    def __repr__(self) -> str:
        return (
            f"HETileDataset(image={self.he_image_path.name}, "
            f"size={self.W}×{self.H}, tiles={len(self)}, "
            f"tile_size={self.tile_size}, stride={self.stride})"
        )


# ---------------------------------------------------------------------------
# POI-level metadata container
# ---------------------------------------------------------------------------

class POIRecord:
    """
    Lightweight data-class holding all file paths and labels for one POI.

    Attributes
    ----------
    patient_id    : str
    poi_id        : str
    poi_dir       : Path
    he_path       : Path
    pdl1_path     : Path  (may be None if file missing)
    pd1_path      : Path  (may be None if file missing)
    pdl1_label    : int   (0 or 1; -1 if unknown / not in labels.json)
    pd1_label     : int
    stain_score   : float
    meta          : dict  (full labels.json content)
    """

    _HE_NAMES   = ("HE_image.png",   "HE_image.jpg")
    _PDL1_NAMES = ("PDL1_image.png", "PDL1_image.jpg")
    _PD1_NAMES  = ("PD1_image.png",  "PD1_image.jpg")

    def __init__(self, patient_id: str, poi_id: str, poi_dir: Path):
        self.patient_id = patient_id
        self.poi_id     = poi_id
        self.poi_dir    = poi_dir

        # Locate image files
        self.he_path   = self._find_file(self._HE_NAMES)
        self.pdl1_path = self._find_file(self._PDL1_NAMES)
        self.pd1_path  = self._find_file(self._PD1_NAMES)

        # Read labels.json if present
        label_path = poi_dir / "labels.json"
        if label_path.exists():
            with open(label_path, "r") as f:
                self.meta = json.load(f)
        else:
            self.meta = {}

        self.pdl1_label  = int(self.meta.get("PDL1_label",  -1))
        self.pd1_label   = int(self.meta.get("PD1_label",   -1))
        self.stain_score = float(self.meta.get("stain_score", -1.0))

    def _find_file(self, candidates: tuple) -> Optional[Path]:
        for name in candidates:
            p = self.poi_dir / name
            if p.exists():
                return p
        return None

    def has_he(self)   -> bool: return self.he_path   is not None
    def has_pdl1(self) -> bool: return self.pdl1_path is not None
    def has_pd1(self)  -> bool: return self.pd1_path  is not None

    def get_tile_dataset(
        self,
        tile_size: int = 512,
        stride:    int = 256,
        transform = None,
    ) -> HETileDataset:
        """Return an HETileDataset for this POI's H&E image."""
        if not self.has_he():
            raise FileNotFoundError(
                f"No H&E image found in POI dir: {self.poi_dir}"
            )
        return HETileDataset(self.he_path, tile_size=tile_size,
                             stride=stride, transform=transform)

    def to_dict(self) -> dict:
        return {
            "patient_id":   self.patient_id,
            "poi_id":       self.poi_id,
            "poi_dir":      str(self.poi_dir),
            "he_path":      str(self.he_path)   if self.he_path   else None,
            "pdl1_path":    str(self.pdl1_path) if self.pdl1_path else None,
            "pd1_path":     str(self.pd1_path)  if self.pd1_path  else None,
            "pdl1_label":   self.pdl1_label,
            "pd1_label":    self.pd1_label,
            "stain_score":  self.stain_score,
        }

    def __repr__(self) -> str:
        return (
            f"POIRecord(patient={self.patient_id}, poi={self.poi_id}, "
            f"pdl1_label={self.pdl1_label})"
        )


# ---------------------------------------------------------------------------
# KanduDataset — high-level patient / POI dataset
# ---------------------------------------------------------------------------

class KanduDataset(Dataset):
    """
    High-level dataset that discovers all Patient → POI directories under
    `dataset_root` and returns POIRecord objects.

    Parameters
    ----------
    dataset_root  : str | Path — root directory; see expected structure above
    require_he    : bool       — skip POIs without an H&E image (default True)
    require_label : bool       — skip POIs without a valid PDL1_label (default False)
    tile_size     : int        — tile size for get_tile_dataset()  (default 512)
    stride        : int        — tile stride for get_tile_dataset() (default 256)

    Usage
    -----
      ds = KanduDataset("path/to/dataset")
      for poi in ds:
          tiles_ds = poi.get_tile_dataset()
          ...

    Or indexed:
      poi = ds[0]
    """

    def __init__(
        self,
        dataset_root:  str | Path,
        require_he:    bool = True,
        require_label: bool = False,
        tile_size:     int  = 512,
        stride:        int  = 256,
    ):
        self.dataset_root  = Path(dataset_root)
        self.require_he    = require_he
        self.require_label = require_label
        self.tile_size     = tile_size
        self.stride        = stride

        self.records: list[POIRecord] = self._discover()

        if not self.records:
            raise RuntimeError(
                f"No valid POI records found under: {self.dataset_root}\n"
                "Expected structure: dataset_root/patient_id/poi_id/"
                "  with HE_image.png, PDL1_image.png, labels.json"
            )

    # ------------------------------------------------------------------
    def _discover(self) -> list[POIRecord]:
        """
        Walk the dataset root two levels deep:
          level 1 = patient directories
          level 2 = POI directories (must contain at least HE_image.png)
        """
        records = []
        if not self.dataset_root.is_dir():
            return records

        for patient_dir in sorted(self.dataset_root.iterdir()):
            if not patient_dir.is_dir():
                continue
            patient_id = patient_dir.name

            for poi_dir in sorted(patient_dir.iterdir()):
                if not poi_dir.is_dir():
                    continue
                poi_id = poi_dir.name

                rec = POIRecord(patient_id, poi_id, poi_dir)

                if self.require_he and not rec.has_he():
                    continue
                if self.require_label and rec.pdl1_label < 0:
                    continue

                records.append(rec)

        return records

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> POIRecord:
        return self.records[idx]

    # ------------------------------------------------------------------
    def patient_ids(self) -> list[str]:
        return sorted({r.patient_id for r in self.records})

    def get_by_patient(self, patient_id: str) -> list[POIRecord]:
        return [r for r in self.records if r.patient_id == patient_id]

    def summary(self) -> str:
        n_patients = len(self.patient_ids())
        n_poi      = len(self.records)
        n_pos      = sum(1 for r in self.records if r.pdl1_label == 1)
        n_neg      = sum(1 for r in self.records if r.pdl1_label == 0)
        n_unk      = n_poi - n_pos - n_neg
        return (
            f"KanduDataset: {n_patients} patients, {n_poi} POIs "
            f"(PDL1+={n_pos}, PDL1-={n_neg}, unknown={n_unk})"
        )

    def __repr__(self) -> str:
        return self.summary()


# ---------------------------------------------------------------------------
# Collate function for DataLoader (bags of variable tile counts)
# ---------------------------------------------------------------------------

def collate_poi_bag(batch: list) -> dict:
    """
    Custom collate_fn for DataLoader when returning (tile_tensor, coord) pairs.

    Returns a dict with:
      tiles  : list of [N_i, 3, H, W] tensors  (one per POI in the batch)
      coords : list of list of (x, y) tuples
    """
    tiles  = [item[0] for item in batch]
    coords = [item[1] for item in batch]
    return {"tiles": tiles, "coords": coords}
