"""
train_data_raw.py — Train directly on data_raw/ (your actual data structure)
=============================================================================
This is the correct entry-point for your dataset. It reads from:

  data_raw/
    02-008_HE_A12_v2_s13/              (H&E images)
    02-008_PDL1(SP142)-..._b3/         (PDL1 images)
    02-008_PD1(NAT105)-..._b3/         (PD1 images)
    results.txt                        (per-image PD-L1 scores)

Each numbered image (001_r1c1 ... 168_r14c1) is treated as one POI.
Labels are binarised from results.txt scores at threshold 0.5.

Usage
-----
  cd c:/Users/kandu/Downloads/TNBC
  python -m kandus_method.train_data_raw

  # With options
  python -m kandus_method.train_data_raw
      --data_raw        ./data_raw
      --epochs          30
      --backbone        resnet50
      --label_threshold 0.5
      --max_tiles       32
      --device          cuda

Notes
-----
  - With only 168 cores from 1 patient, train/val split is done at the
    CORE level (80/20 random split) rather than patient level.
  - Use small --max_tiles (16-32) on CPU to keep memory manageable.
  - For GPU training, increase to --max_tiles 64.
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

try:
    from sklearn.metrics import roc_auc_score
    _SKLEARN = True
except ImportError:
    _SKLEARN = False

from kandus_method.cnn_model import MILClassifier
from kandus_method.data_raw_adapter import DataRawAdapter, DataRawBagDataset


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def mil_loss(poi_prob, tile_probs, label, lambda_tile=0.3):
    bce = torch.nn.BCELoss()
    poi_loss  = bce(poi_prob.unsqueeze(0),            label.unsqueeze(0))
    tile_loss = bce(tile_probs.mean().unsqueeze(0),   label.unsqueeze(0))
    return poi_loss + lambda_tile * tile_loss


# ---------------------------------------------------------------------------
# Core-level train/val split (since only 1 patient)
# ---------------------------------------------------------------------------

def core_split(records, val_fraction=0.2, seed=42):
    rng = random.Random(seed)
    idxs = list(range(len(records)))
    rng.shuffle(idxs)
    n_val = max(1, int(len(idxs) * val_fraction))
    val_idxs   = set(idxs[:n_val])
    train_recs = [records[i] for i in idxs if i not in val_idxs]
    val_recs   = [records[i] for i in idxs if i in val_idxs]
    return train_recs, val_recs


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------

def _bag_collate(batch):
    bags   = [item[0] for item in batch]
    labels = torch.stack([item[1] for item in batch])
    return bags, labels


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    all_probs, all_labels = [], []

    for bags, labels in loader:
        batch_loss = torch.tensor(0.0, device=device, requires_grad=True)
        for bag, label in zip(bags, labels):
            bag, label = bag.to(device), label.to(device)
            poi_prob, tile_probs, _ = model(bag)
            loss = mil_loss(poi_prob, tile_probs, label)
            batch_loss = batch_loss + loss
            all_probs.append(poi_prob.detach().cpu().item())
            all_labels.append(label.cpu().item())

        optimizer.zero_grad()
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += batch_loss.item()

    auc = _auc(all_labels, all_probs)
    return {"loss": total_loss / max(len(loader), 1), "auc": auc}


@torch.no_grad()
def eval_one_epoch(model, loader, device):
    model.eval()
    total_loss = 0.0
    all_probs, all_labels = [], []

    for bags, labels in loader:
        for bag, label in zip(bags, labels):
            bag, label = bag.to(device), label.to(device)
            poi_prob, tile_probs, _ = model(bag)
            loss = mil_loss(poi_prob, tile_probs, label)
            total_loss += loss.item()
            all_probs.append(poi_prob.cpu().item())
            all_labels.append(label.cpu().item())

    auc = _auc(all_labels, all_probs)
    return {"loss": total_loss / max(len(loader), 1), "auc": auc}


def _auc(labels, probs):
    if not _SKLEARN:
        return -1.0
    try:
        return roc_auc_score(labels, probs)
    except ValueError:
        return -1.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train(args):
    set_seed(42)

    # Determine device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[train_data_raw] CUDA not available, falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"[train_data_raw] Device: {device}")

    # ---- Load adapter ----
    print(f"[train_data_raw] Loading data from: {args.data_raw}")
    adapter = DataRawAdapter(
        data_raw_root   = args.data_raw,
        label_threshold = args.label_threshold,
    )
    print(f"[train_data_raw] {adapter.summary()}")

    # ---- Split ----
    train_recs, val_recs = core_split(adapter.records, val_fraction=0.2)
    print(f"[train_data_raw] Train cores: {len(train_recs)}, Val cores: {len(val_recs)}")

    train_ds = DataRawBagDataset(train_recs, args.tile_size, args.stride, args.max_tiles)
    val_ds   = DataRawBagDataset(val_recs,   args.tile_size, args.stride, args.max_tiles)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=0, collate_fn=_bag_collate)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0, collate_fn=_bag_collate)

    # ---- Model ----
    model = MILClassifier(backbone=args.backbone).to(device)
    print(f"[train_data_raw] Model: {args.backbone.upper()},  "
          f"params={model.get_num_params():,}")

    # ---- Optimizer + Scheduler ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    # ---- Checkpoint dir ----
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ---- Training loop ----
    best_val_auc = -1.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        tr = train_one_epoch(model, train_loader, optimizer, device)
        va = eval_one_epoch(model, val_loader,   device)
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:03d}/{args.epochs}  "
            f"train_loss={tr['loss']:.4f} train_auc={tr['auc']:.4f}  "
            f"val_loss={va['loss']:.4f} val_auc={va['auc']:.4f}  "
            f"lr={lr:.2e}  t={time.time()-t0:.1f}s"
        )

        if va["auc"] > best_val_auc:
            best_val_auc = va["auc"]
            ckpt_path = ckpt_dir / f"best_model_{args.backbone}.pt"
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_auc":     best_val_auc,
                "config": {
                    "backbone":   args.backbone,
                    "hidden_dim": 256,
                    "dropout":    0.25,
                    "tile_size":  args.tile_size,
                    "stride":     args.stride,
                },
            }, ckpt_path)
            print(f"  ✓ Saved → {ckpt_path}  (AUC={best_val_auc:.4f})")

    print(f"\n[train_data_raw] Done. Best val AUC: {best_val_auc:.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Kandu's Method — Train on data_raw/")
    p.add_argument("--data_raw",        default="./data_raw",
                   help="Path to data_raw/ folder (default: ./data_raw)")
    p.add_argument("--backbone",        default="resnet101",
                   choices=["resnet50", "resnet50d", "resnet101", "resnet152",
                            "efficientnet_b0", "efficientnet_b2"])
    p.add_argument("--epochs",          type=int,   default=30)
    p.add_argument("--lr",              type=float, default=1e-4)
    p.add_argument("--batch_size",      type=int,   default=4)
    p.add_argument("--tile_size",       type=int,   default=512)
    p.add_argument("--stride",          type=int,   default=256)
    p.add_argument("--max_tiles",       type=int,   default=32,
                   help="Max tiles per bag (lower for CPU: 16, GPU: 64)")
    p.add_argument("--label_threshold", type=float, default=0.5,
                   help="Binarise results.txt score at this threshold")
    p.add_argument("--checkpoint_dir",  default="./kandus_method/checkpoints")
    p.add_argument("--device",          default="cuda")
    return p.parse_args()


if __name__ == "__main__":
    train(_parse_args())
