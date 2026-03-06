"""
infer_cnn.py — Kandu's Method CNN/MIL Inference & Feature Extraction
=====================================================================
Loads a trained MILClassifier checkpoint and runs inference on a given
POI directory, producing:

  1. POI-level PD-L1 morphology probability (from H&E)
  2. Per-tile probabilities + attention weights (interpretability)
  3. Feature dict ready for downstream CPS/CPS++ computation

Output feature dict
-------------------
  {
    "patient_id"   : str,
    "poi_id"       : str,
    "he_path"      : str,
    "pdl1_prob_he" : float,        # POI-level PD-L1 probability from H&E CNN
    "n_tiles"      : int,
    "tile_probs"   : list[float],  # per-tile probabilities
    "attn_weights" : list[float],  # MIL attention weights per tile
    "tile_coords"  : list[tuple],  # (x, y) pixel coordinates
    "top_tiles"    : list[dict],   # top-K tiles by attention weight
  }

Usage — command line
---------------------
  cd c:/Users/kandu/Downloads/TNBC
  python -m kandus_method.infer_cnn \\
      --poi_dir       ./dataset/patient_001/poi_01 \\
      --checkpoint    ./kandus_method/checkpoints/best_model_resnet50.pt \\
      --top_k         5

Usage — programmatic
---------------------
  from kandus_method.infer_cnn import run_inference
  result = run_inference(poi_dir="...", checkpoint="...", device="cuda")
  print(result["pdl1_prob_he"])
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from kandus_method.cnn_model import MILClassifier, build_model
from kandus_method.dataset_kandu import HETileDataset, POIRecord


# ---------------------------------------------------------------------------
# Core inference function
# ---------------------------------------------------------------------------

def run_inference(
    poi_dir:    str | Path,
    checkpoint: str | Path,
    device:     str   = "cuda",
    tile_size:  int   = 512,
    stride:     int   = 256,
    batch_size: int   = 16,
    top_k:      int   = 5,
) -> dict:
    """
    Run CNN/MIL inference on a single POI folder and return its feature dict.

    Parameters
    ----------
    poi_dir    : path to a POI directory containing HE_image.png
    checkpoint : path to a saved checkpoint (.pt file)
    device     : 'cuda' or 'cpu'
    tile_size  : tile size (must match training config)
    stride     : tile stride (must match training config)
    batch_size : number of tiles to process per GPU batch (for feature extraction)
    top_k      : number of top-attention tiles to include in output

    Returns
    -------
    dict — see module docstring for field descriptions
    """
    poi_dir    = Path(poi_dir)
    checkpoint = Path(checkpoint)
    device_obj = torch.device(
        device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
    )

    # ---- Load checkpoint ----
    ckpt = torch.load(checkpoint, map_location=device_obj, weights_only=False)
    config = ckpt.get("config", {})

    model = MILClassifier(
        backbone   = config.get("backbone",   "resnet50"),
        hidden_dim = config.get("hidden_dim", 256),
        dropout    = config.get("dropout",    0.25),
    ).to(device_obj)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[infer] Loaded checkpoint: {checkpoint.name}  "
          f"(trained {ckpt.get('epoch','?')} epochs, "
          f"val_auc={ckpt.get('val_auc', '?'):.4f})")

    # ---- Locate H&E image ----
    rec = POIRecord(
        patient_id = poi_dir.parent.name,
        poi_id     = poi_dir.name,
        poi_dir    = poi_dir,
    )
    if not rec.has_he():
        raise FileNotFoundError(f"No HE_image.png found in: {poi_dir}")

    print(f"[infer] POI: {rec.patient_id}/{rec.poi_id}")

    # ---- Tile the H&E image ----
    tile_ds = HETileDataset(rec.he_path, tile_size=tile_size, stride=stride)
    print(f"[infer] Tiles: {len(tile_ds)}  "
          f"(image {tile_ds.W}×{tile_ds.H}px, "
          f"tile={tile_size}, stride={stride})")

    tile_loader = DataLoader(tile_ds, batch_size=batch_size,
                             shuffle=False, num_workers=0)

    # ---- Extract features in batches ----
    all_features = []
    all_coords   = []

    with torch.no_grad():
        for tiles_batch, coords_batch in tile_loader:
            tiles_batch = tiles_batch.to(device_obj)
            feats = model.feature_extractor(tiles_batch)    # [B, D]
            all_features.append(feats.cpu())
            # coords_batch is a tuple of two tensors (xs, ys) from DataLoader collation
            xs, ys = coords_batch
            for x, y in zip(xs.tolist(), ys.tolist()):
                all_coords.append((x, y))

    all_features = torch.cat(all_features, dim=0)           # [N, D]
    N = all_features.size(0)

    # ---- MIL aggregation + prediction (full bag, on device) ----
    with torch.no_grad():
        h = all_features.to(device_obj)

        # Tile-level probabilities
        tile_probs_t  = model.tile_head(h).squeeze(-1)      # [N]

        # Attention aggregation
        z, attn_t     = model.attention_mil(h)              # [1,D], [N,1]

        # POI-level prediction
        poi_logit     = model.poi_head(z)
        poi_prob      = poi_logit.squeeze().item()

    tile_probs   = tile_probs_t.cpu().tolist()
    attn_weights = attn_t.squeeze(-1).cpu().tolist()

    # ---- Top-K attended tiles ----
    sorted_idx = sorted(range(N), key=lambda i: attn_weights[i], reverse=True)
    top_tiles  = [
        {
            "rank":        rank + 1,
            "tile_idx":    idx,
            "coord":       all_coords[idx],
            "attn_weight": round(attn_weights[idx], 6),
            "tile_prob":   round(tile_probs[idx],   6),
        }
        for rank, idx in enumerate(sorted_idx[:top_k])
    ]

    # ---- Assemble feature dict ----
    result = {
        "patient_id":   rec.patient_id,
        "poi_id":       rec.poi_id,
        "he_path":      str(rec.he_path),
        "pdl1_prob_he": round(poi_prob, 6),     # key field for downstream CPS
        "n_tiles":      N,
        "tile_probs":   [round(p, 6) for p in tile_probs],
        "attn_weights": [round(w, 6) for w in attn_weights],
        "tile_coords":  all_coords,
        "top_tiles":    top_tiles,
    }

    print(f"[infer] PDL1 probability (H&E CNN): {poi_prob:.4f}")
    print(f"[infer] Top-{top_k} tiles by attention:")
    for t in top_tiles:
        print(f"  rank={t['rank']}  coord={t['coord']}  "
              f"attn={t['attn_weight']:.4f}  prob={t['tile_prob']:.4f}")

    return result


# ---------------------------------------------------------------------------
# Batch inference over all POIs in a patient folder
# ---------------------------------------------------------------------------

def run_patient_inference(
    patient_dir: str | Path,
    checkpoint:  str | Path,
    output_json: str | Path | None = None,
    **kwargs,
) -> list[dict]:
    """
    Run inference on all POIs in a patient directory.

    Parameters
    ----------
    patient_dir : path containing poi_01/, poi_02/, ...
    checkpoint  : path to saved model checkpoint
    output_json : if provided, saves results to this JSON file
    **kwargs    : passed to run_inference (device, tile_size, stride, ...)

    Returns
    -------
    list of feature dicts (one per POI)
    """
    patient_dir = Path(patient_dir)
    results = []

    poi_dirs = sorted([d for d in patient_dir.iterdir() if d.is_dir()])
    print(f"[infer] Patient: {patient_dir.name}  ({len(poi_dirs)} POIs)")

    for poi_dir in poi_dirs:
        try:
            res = run_inference(poi_dir, checkpoint, **kwargs)
            results.append(res)
        except FileNotFoundError as e:
            print(f"  [skip] {poi_dir.name}: {e}")

    if output_json and results:
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[infer] Saved results → {output_json}")

    # Print patient-level summary
    if results:
        avg_prob = sum(r["pdl1_prob_he"] for r in results) / len(results)
        print(f"\n[infer] Patient-level avg PD-L1 probability: {avg_prob:.4f}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Kandu's Method — CNN Inference")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--poi_dir",     help="Single POI directory to infer on")
    mode.add_argument("--patient_dir", help="Patient directory (runs all POIs)")

    p.add_argument("--checkpoint",  required=True,
                   help="Path to trained checkpoint .pt file")
    p.add_argument("--device",      default="cuda")
    p.add_argument("--tile_size",   type=int, default=512)
    p.add_argument("--stride",      type=int, default=256)
    p.add_argument("--batch_size",  type=int, default=16,
                   help="Tile batch size during feature extraction")
    p.add_argument("--top_k",       type=int, default=5,
                   help="Number of top-attended tiles to report")
    p.add_argument("--output_json", default=None,
                   help="Save results to this JSON file (patient mode only)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    kwargs = dict(
        device     = args.device,
        tile_size  = args.tile_size,
        stride     = args.stride,
        batch_size = args.batch_size,
        top_k      = args.top_k,
    )

    if args.poi_dir:
        result = run_inference(args.poi_dir, args.checkpoint, **kwargs)
        print(json.dumps(result, indent=2))
    else:
        results = run_patient_inference(
            args.patient_dir, args.checkpoint,
            output_json=args.output_json,
            **kwargs,
        )
