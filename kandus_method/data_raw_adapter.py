"""
data_raw_adapter.py — Kandu's Method: Adapter for the actual data_raw/ structure
==================================================================================
Your data_raw/ folder has 168 paired TMA (Tissue Microarray) core images across
three stain folders, plus a results.txt with per-image PD-L1 scores.

Actual structure detected
--------------------------
  data_raw/
    02-008_HE_A12_v2_s13/                    ← 168 H&E images
      02-008_HE_A12_v2_s13_001_r1c1.jpg.jpeg
      02-008_HE_A12_v2_s13_002_r1c2.jpg.jpeg
      ...
    02-008_PDL1(SP142)-Springbio_A12_v3_b3/  ← 168 matched PDL1 images
    02-008_PD1(NAT105)-CellMarque_A12_v3_b3/ ← 168 matched PD1 images
    results.txt                              ← "HE_path: score" per line
    annotation_task_automated.xlsx           ← optional extended metadata

Each TMA core (numbered _001_ to _168_) is treated as one POI.
All 168 cores are from patient "02-008".

Label strategy
---------------
  results.txt contains a continuous score per core.
  We binarize at `label_threshold` (default=0.5):
    score >= threshold → PDL1_label = 1 (positive)
    score <  threshold → PDL1_label = 0 (negative)
  The raw score is also stored as `stain_score`.

Usage
-----
  from kandus_method.data_raw_adapter import DataRawAdapter, DataRawTileDataset

  # High-level: get list of POI-like records
  adapter = DataRawAdapter("data_raw")
  print(adapter.summary())

  rec = adapter.records[0]
  print(rec.he_path, rec.pdl1_label, rec.stain_score)

  # Get tiles for one record
  tile_ds = rec.get_tile_dataset(tile_size=512, stride=256)
  tile, coord = tile_ds[0]

  # Training-ready: DataRawTileDataset (bag per TMA core)
  from kandus_method.data_raw_adapter import DataRawBagDataset
  bag_ds = DataRawBagDataset(adapter.records, max_tiles=32)
  bag, label = bag_ds[0]
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset

from kandus_method.dataset_kandu import HETileDataset


# ---------------------------------------------------------------------------
# TMA core record
# ---------------------------------------------------------------------------

class TMACoreRecord:
    """
    Represents one TMA core image — the equivalent of one POI in this dataset.

    Attributes
    ----------
    patient_id   : str   — e.g. "02-008"
    core_id      : str   — e.g. "001_r1c1"
    he_path      : Path
    pdl1_path    : Path  (or None)
    pd1_path     : Path  (or None)
    stain_score  : float — continuous score from results.txt [0, 1]
    pdl1_label   : int   — binarised label (0 or 1)
    """

    def __init__(
        self,
        patient_id:  str,
        core_id:     str,
        he_path:     Path,
        pdl1_path:   Optional[Path],
        pd1_path:    Optional[Path],
        stain_score: float,
        label_threshold: float = 0.5,
    ):
        self.patient_id  = patient_id
        self.poi_id      = core_id          # alias so it matches POIRecord API
        self.core_id     = core_id
        self.he_path     = he_path
        self.pdl1_path   = pdl1_path
        self.pd1_path    = pd1_path
        self.stain_score = stain_score
        self.pdl1_label  = 1 if stain_score >= label_threshold else 0

    def has_he(self)   -> bool: return self.he_path   is not None and self.he_path.exists()
    def has_pdl1(self) -> bool: return self.pdl1_path is not None and self.pdl1_path.exists()
    def has_pd1(self)  -> bool: return self.pd1_path  is not None and self.pd1_path.exists()

    def get_tile_dataset(
        self,
        tile_size: int = 512,
        stride:    int = 256,
        transform=None,
    ) -> HETileDataset:
        if not self.has_he():
            raise FileNotFoundError(f"HE image not found: {self.he_path}")
        return HETileDataset(self.he_path, tile_size=tile_size,
                             stride=stride, transform=transform)

    def to_dict(self) -> dict:
        return {
            "patient_id":  self.patient_id,
            "core_id":     self.core_id,
            "he_path":     str(self.he_path),
            "pdl1_path":   str(self.pdl1_path) if self.pdl1_path else None,
            "pd1_path":    str(self.pd1_path)  if self.pd1_path  else None,
            "stain_score": self.stain_score,
            "pdl1_label":  self.pdl1_label,
        }

    def __repr__(self) -> str:
        return (
            f"TMACoreRecord(patient={self.patient_id}, core={self.core_id}, "
            f"score={self.stain_score:.2f}, label={self.pdl1_label})"
        )


# ---------------------------------------------------------------------------
# DataRawAdapter
# ---------------------------------------------------------------------------

class DataRawAdapter:
    """
    Reads the actual data_raw/ folder and produces a list of TMACoreRecord
    objects — compatible with the kandus_method training pipeline.

    Parameters
    ----------
    data_raw_root    : str | Path — path to data_raw/ folder
    label_threshold  : float      — binarisation threshold (default 0.5)
    results_filename : str        — name of the scores file (default 'results.txt')
    """

    def __init__(
        self,
        data_raw_root:    str | Path,
        label_threshold:  float = 0.5,
        results_filename: str   = "results.txt",
    ):
        self.root            = Path(data_raw_root)
        self.label_threshold = label_threshold
        self.results_file    = self.root / results_filename

        # Locate stain directories
        self.he_dir   = self._find_stain_dir("HE")
        self.pdl1_dir = self._find_stain_dir("PDL1")
        self.pd1_dir  = self._find_stain_dir("PD1")

        # Parse scores
        self._scores = self._parse_results()

        # Build records
        self.records: list[TMACoreRecord] = self._build_records()

        if not self.records:
            raise RuntimeError(
                f"No TMA core records found in: {self.root}\n"
                "Expected folders containing 'HE', 'PDL1', 'PD1' in their names,\n"
                "and a results.txt with 'path: score' per line."
            )

    # ------------------------------------------------------------------
    def _find_stain_dir(self, stain_key: str) -> Optional[Path]:
        """Find the subdirectory whose name contains the stain_key."""
        for d in self.root.iterdir():
            if d.is_dir() and stain_key in d.name.upper():
                return d
        return None

    # ------------------------------------------------------------------
    def _parse_results(self) -> dict[str, float]:
        """
        Parse results.txt.  Each line: "relative\\path\\image.jpeg: 0.55"
        Returns dict mapping filename stem (e.g. "_003_r1c3") → score.
        """
        scores = {}
        if not self.results_file.exists():
            print(f"[DataRawAdapter] Warning: results file not found: {self.results_file}")
            return scores

        with open(self.results_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line or ":" not in line:
                    continue
                # Format: path: score
                parts = line.rsplit(":", 1)
                if len(parts) != 2:
                    continue
                path_part  = parts[0].strip()
                score_part = parts[1].strip()
                try:
                    score = float(score_part)
                except ValueError:
                    continue

                # Extract the NNN_rRcC identifier from the filename
                fname = Path(path_part).name
                match = re.search(r'_(\d{3}_r\d+c\d+)', fname)
                if match:
                    core_id = match.group(1)
                    scores[core_id] = score
                else:
                    # Fallback: use full filename stem
                    scores[fname] = score

        return scores

    # ------------------------------------------------------------------
    def _build_records(self) -> list[TMACoreRecord]:
        """
        Match HE images with their PDL1/PD1 counterparts by core_id (NNN_rRcC).
        """
        if self.he_dir is None:
            return []

        records = []
        # Infer patient_id from HE folder name prefix (e.g. "02-008")
        patient_id = self.he_dir.name.split("_")[0]

        # Collect all HE files
        he_files = sorted(
            f for f in self.he_dir.iterdir()
            if f.is_file() and f.name.lower().endswith((".jpg", ".jpeg", ".png"))
        )

        for he_file in he_files:
            # Extract core_id
            match = re.search(r'_(\d{3}_r\d+c\d+)', he_file.name)
            if not match:
                continue
            core_id = match.group(1)

            # Find matching PDL1 file
            pdl1_file = self._find_matching(self.pdl1_dir, core_id) if self.pdl1_dir else None
            pd1_file  = self._find_matching(self.pd1_dir,  core_id) if self.pd1_dir  else None

            # Score lookup
            score = self._scores.get(core_id, -1.0)
            if score < 0:
                # Try by full file stem
                score = self._scores.get(he_file.name, -1.0)

            rec = TMACoreRecord(
                patient_id      = patient_id,
                core_id         = core_id,
                he_path         = he_file,
                pdl1_path       = pdl1_file,
                pd1_path        = pd1_file,
                stain_score     = max(score, 0.0),  # treat unknown as 0
                label_threshold = self.label_threshold,
            )
            records.append(rec)

        return records

    # ------------------------------------------------------------------
    def _find_matching(self, stain_dir: Path, core_id: str) -> Optional[Path]:
        """Find the file in stain_dir whose name contains core_id."""
        for f in stain_dir.iterdir():
            if core_id in f.name:
                return f
        return None

    # ------------------------------------------------------------------
    @property
    def patient_id(self) -> str:
        return self.records[0].patient_id if self.records else "unknown"

    def summary(self) -> str:
        n   = len(self.records)
        pos = sum(1 for r in self.records if r.pdl1_label == 1)
        neg = n - pos
        scored = sum(1 for r in self.records if r.stain_score > 0)
        return (
            f"DataRawAdapter: patient={self.patient_id}  "
            f"TMA cores={n}  (PDL1+={pos}, PDL1-={neg})  "
            f"scored={scored}  threshold={self.label_threshold}"
        )

    def __repr__(self) -> str:
        return self.summary()

    def get_by_label(self, label: int) -> list[TMACoreRecord]:
        return [r for r in self.records if r.pdl1_label == label]


# ---------------------------------------------------------------------------
# DataRawBagDataset (drop-in replacement for POIBagDataset in train_cnn.py)
# ---------------------------------------------------------------------------

class DataRawBagDataset(Dataset):
    """
    Wraps a list of TMACoreRecords for training.
    Each item: (bag_tensor [N, 3, H, W], label float)

    Parameters
    ----------
    records   : list[TMACoreRecord]
    tile_size : int (default 512)
    stride    : int (default 256)
    max_tiles : int — random subsample if bag is larger (default 32)
                      Use smaller value on CPU / limited RAM.
    """

    def __init__(
        self,
        records:   list[TMACoreRecord],
        tile_size: int = 512,
        stride:    int = 256,
        max_tiles: int = 32,
    ):
        import random
        self.records   = [r for r in records if r.has_he()]
        self.tile_size = tile_size
        self.stride    = stride
        self.max_tiles = max_tiles
        self._random   = random

        if not self.records:
            raise RuntimeError("No records with valid HE images found.")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        import random
        rec     = self.records[idx]
        tile_ds = rec.get_tile_dataset(self.tile_size, self.stride)

        indices = list(range(len(tile_ds)))
        if len(indices) > self.max_tiles:
            indices = random.sample(indices, self.max_tiles)
        if len(indices) == 0:
            # Fallback: return a single blank tile
            return torch.zeros(1, 3, self.tile_size, self.tile_size), \
                   torch.tensor(float(rec.pdl1_label))

        tiles = [tile_ds[i][0] for i in indices]
        bag   = torch.stack(tiles, dim=0)  # [N, 3, H, W]
        label = torch.tensor(float(rec.pdl1_label))
        return bag, label
