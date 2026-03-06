"""
train_cnn.py — Kandu's Method CNN/MIL Training Script
======================================================
Weakly-supervised training of the MIL classifier on H&E tiles.

Supervision signal
------------------
  Each POI provides a single binary label (PDL1_label: 0 or 1) from  
  labels.json.  The model learns tile-level morphological features  
  without any per-tile annotation (Multiple Instance Learning).

Loss
----
  A combined loss is used:
    L = BCE(poi_prob, poi_label)                     — MIL-level loss
      + λ_tile * BCE(mean(tile_probs), poi_label)    — tile-level consistency
  
  The tile consistency term encourages tile heads to also learn
  discriminative features, improving interpretability.

Training setup
--------------
  Optimizer : AdamW
  Scheduler : CosineAnnealingLR (cosine decay to lr_min over all epochs)
  Metric    : ROC-AUC (validation set)
  Checkpoint: best validation AUC saved to checkpoints/

Usage
-----
  cd c:/Users/kandu/Downloads/TNBC
  python -m kandus_method.train_cnn --dataset_root ./dataset --epochs 30

  # All arguments
  python -m kandus_method.train_cnn \\
      --dataset_root  ./dataset    \\
      --backbone      resnet50     \\
      --epochs        30           \\
      --lr            1e-4         \\
      --batch_size    8            \\
      --tile_size     512          \\
      --stride        256          \\
      --val_split     0.2          \\
      --checkpoint_dir ./kandus_method/checkpoints \\
      --device        cuda
"""

from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

try:
    from sklearn.metrics import roc_auc_score
    _SKLEARN = True
except ImportError:
    _SKLEARN = False

from kandus_method.cnn_model import MILClassifier
from kandus_method.dataset_kandu import KanduDataset, HETileDataset, POIRecord


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# POI-bag Dataset (wraps KanduDataset for training loop)
# ---------------------------------------------------------------------------

class POIBagDataset(Dataset):
    """
    Wraps a list of POIRecords.
    Each item:  (bag_tensor [N, 3, H, W], label [float])
    Tiles are loaded fresh each epoch (no caching) for memory efficiency.

    Parameters
    ----------
    records   : list of POIRecord
    tile_size : int
    stride    : int
    max_tiles : int — randomly sub-sample if a POI has more than this many
                      tiles (avoids GPU OOM with very large images).
    """

    def __init__(
        self,
        records:   list[POIRecord],
        tile_size: int = 512,
        stride:    int = 256,
        max_tiles: int = 64,
    ):
        super().__init__()
        # Keep only POIs with a valid label and HE image
        self.records   = [r for r in records if r.pdl1_label >= 0 and r.has_he()]
        self.tile_size = tile_size
        self.stride    = stride
        self.max_tiles = max_tiles

        if not self.records:
            raise RuntimeError(
                "No usable POI records: make sure POIs have HE images + PDL1_label in labels.json."
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        tile_ds = rec.get_tile_dataset(self.tile_size, self.stride)

        # Randomly sub-sample tiles if the bag is too large
        indices = list(range(len(tile_ds)))
        if len(indices) > self.max_tiles:
            indices = random.sample(indices, self.max_tiles)

        tiles = [tile_ds[i][0] for i in indices]       # list of [3, H, W]
        bag   = torch.stack(tiles, dim=0)               # [N, 3, H, W]
        label = torch.tensor(rec.pdl1_label, dtype=torch.float32)
        return bag, label


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def mil_loss(
    poi_prob:   torch.Tensor,
    tile_probs: torch.Tensor,
    label:      torch.Tensor,
    lambda_tile: float = 0.3,
) -> torch.Tensor:
    """
    Combined MIL loss:
      L = BCE(poi_prob, label)  +  λ * BCE(mean(tile_probs), label)
    """
    bce = nn.BCELoss()
    poi_loss  = bce(poi_prob.unsqueeze(0),  label.unsqueeze(0))
    tile_loss = bce(tile_probs.mean().unsqueeze(0), label.unsqueeze(0))
    return poi_prob.new_tensor(poi_loss + lambda_tile * tile_loss)


# ---------------------------------------------------------------------------
# Train / eval utilities
# ---------------------------------------------------------------------------

def train_one_epoch(
    model:       MILClassifier,
    loader:      DataLoader,
    optimizer:   torch.optim.Optimizer,
    device:      torch.device,
    lambda_tile: float = 0.3,
) -> dict:
    model.train()
    total_loss = 0.0
    all_probs, all_labels = [], []

    for bags, labels in loader:
        # bags: list of [N, 3, H, W]  (variable N per POI)
        # We process each POI in the batch individually (variable bag size)
        batch_loss = torch.tensor(0.0, device=device, requires_grad=True)

        for bag, label in zip(bags, labels):
            bag   = bag.to(device)
            label = label.to(device)

            poi_prob, tile_probs, _ = model(bag)
            loss = mil_loss(poi_prob, tile_probs, label, lambda_tile)
            batch_loss = batch_loss + loss

            all_probs.append(poi_prob.detach().cpu().item())
            all_labels.append(label.cpu().item())

        optimizer.zero_grad()
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += batch_loss.item()

    auc = _compute_auc(all_labels, all_probs)
    return {"loss": total_loss / max(len(loader), 1), "auc": auc}


@torch.no_grad()
def eval_one_epoch(
    model:       MILClassifier,
    loader:      DataLoader,
    device:      torch.device,
    lambda_tile: float = 0.3,
) -> dict:
    model.eval()
    total_loss = 0.0
    all_probs, all_labels = [], []

    for bags, labels in loader:
        for bag, label in zip(bags, labels):
            bag   = bag.to(device)
            label = label.to(device)

            poi_prob, tile_probs, _ = model(bag)
            loss = mil_loss(poi_prob, tile_probs, label, lambda_tile)
            total_loss += loss.item()

            all_probs.append(poi_prob.cpu().item())
            all_labels.append(label.cpu().item())

    auc = _compute_auc(all_labels, all_probs)
    return {"loss": total_loss / max(len(loader), 1), "auc": auc}


def _compute_auc(labels: list, probs: list) -> float:
    if not _SKLEARN:
        return -1.0
    try:
        return roc_auc_score(labels, probs)
    except ValueError:
        return -1.0                 # only one class present in split


# ---------------------------------------------------------------------------
# Split records into train / val
# ---------------------------------------------------------------------------

def train_val_split(
    records: list[POIRecord],
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[POIRecord], list[POIRecord]]:
    """
    Patient-level split (keeps all POIs of a patient together).
    """
    patient_ids = sorted({r.patient_id for r in records})
    rng = random.Random(seed)
    rng.shuffle(patient_ids)

    n_val = max(1, int(len(patient_ids) * val_fraction))
    val_patients = set(patient_ids[:n_val])

    train_recs = [r for r in records if r.patient_id not in val_patients]
    val_recs   = [r for r in records if r.patient_id in val_patients]
    return train_recs, val_recs


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(args):
    set_seed(42)

    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu"
        else "cpu"
    )
    print(f"[Kandu's Method] Device: {device}")

    # ---- Dataset ----
    print(f"[Kandu's Method] Loading dataset from: {args.dataset_root}")
    full_ds = KanduDataset(
        dataset_root  = args.dataset_root,
        require_he    = True,
        require_label = True,
        tile_size     = args.tile_size,
        stride        = args.stride,
    )
    print(f"[Kandu's Method] {full_ds.summary()}")

    train_recs, val_recs = train_val_split(full_ds.records, args.val_split)
    print(f"[Kandu's Method] Train POIs: {len(train_recs)}, Val POIs: {len(val_recs)}")

    train_bag_ds = POIBagDataset(train_recs, args.tile_size, args.stride, args.max_tiles)
    val_bag_ds   = POIBagDataset(val_recs,   args.tile_size, args.stride, args.max_tiles)

    # DataLoader — batch_size=1 because bags have variable tile counts.
    # Effective batch accumulation is handled inside train_one_epoch.
    train_loader = DataLoader(train_bag_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=0, collate_fn=_bag_collate)
    val_loader   = DataLoader(val_bag_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0, collate_fn=_bag_collate)

    # ---- Model ----
    model = MILClassifier(
        backbone   = args.backbone,
        hidden_dim = 256,
        dropout    = 0.25,
    ).to(device)
    print(f"[Kandu's Method] Model: {args.backbone.upper()}, "
          f"params={model.get_num_params():,}")

    # ---- Optimiser + Scheduler ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    # ---- Checkpoint directory ----
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ---- Training loop ----
    best_val_auc = -1.0
    best_epoch   = -1

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_metrics = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics   = eval_one_epoch(model, val_loader, device)
        scheduler.step()

        elapsed = time.time() - t0
        lr_now  = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:03d}/{args.epochs}  "
            f"train_loss={train_metrics['loss']:.4f}  train_auc={train_metrics['auc']:.4f}  "
            f"val_loss={val_metrics['loss']:.4f}  val_auc={val_metrics['auc']:.4f}  "
            f"lr={lr_now:.2e}  t={elapsed:.1f}s"
        )

        # Save best model
        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            best_epoch   = epoch
            ckpt_path    = ckpt_dir / f"best_model_{args.backbone}.pt"
            torch.save({
                "epoch":        epoch,
                "model_state":  model.state_dict(),
                "val_auc":      best_val_auc,
                "config": {
                    "backbone":   args.backbone,
                    "hidden_dim": 256,
                    "dropout":    0.25,
                    "tile_size":  args.tile_size,
                    "stride":     args.stride,
                },
            }, ckpt_path)
            print(f"  ✓ Saved best checkpoint → {ckpt_path}  (AUC={best_val_auc:.4f})")

    print(f"\n[Kandu's Method] Training complete. Best val AUC: {best_val_auc:.4f} at epoch {best_epoch}")


def _bag_collate(batch):
    """Collate bags (variable tile count) into a list."""
    bags   = [item[0] for item in batch]
    labels = torch.stack([item[1] for item in batch])
    return bags, labels


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Kandu's Method — CNN/MIL Training")
    p.add_argument("--dataset_root",   default="./dataset",
                   help="Root dir: dataset_root/patient_id/poi_id/")
    p.add_argument("--backbone",       default="resnet50",
                   choices=["resnet50", "efficientnet_b0"])
    p.add_argument("--epochs",         type=int,   default=30)
    p.add_argument("--lr",             type=float, default=1e-4)
    p.add_argument("--batch_size",     type=int,   default=4,
                   help="Number of POIs per gradient step (bags are variable size)")
    p.add_argument("--tile_size",      type=int,   default=512)
    p.add_argument("--stride",         type=int,   default=256)
    p.add_argument("--max_tiles",      type=int,   default=64,
                   help="Max tiles per bag (random subsample if exceeded)")
    p.add_argument("--val_split",      type=float, default=0.2)
    p.add_argument("--checkpoint_dir", default="./kandus_method/checkpoints")
    p.add_argument("--device",         default="cuda",
                   help="'cuda' or 'cpu'")
    return p.parse_args()


if __name__ == "__main__":
    train(_parse_args())
