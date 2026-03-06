"""
visualization_debug.py - Kandu's Method: Debug Overlay Visualizations
=======================================================================
Generates colour-coded overlay images for visual inspection of the pipeline:

  1. Tissue mask overlay
  2. Compartment segmentation overlay (TC=red, LC=green, ST=none)
  3. DAB positive mask overlay (PD-L1 brown staining = yellow)
  4. Nuclei / watershed overlay
  5. Combined summary overlay

Each overlay is saved as a PNG in the specified output directory.

Usage
-----
  from kandus_method.visualization_debug import save_debug_overlays

  save_debug_overlays(
      image_path  = "data_raw/02-008_HE_.../core_005.jpeg",
      stain_result = stain_result,    # from analyze_pdl1_image(debug=True)
      output_dir  = "./results/debug/core_005",
  )
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Colour constants for overlays (BGR for cv2)
# ---------------------------------------------------------------------------
_RED    = (0,   0,   220)     # Tumor cells
_GREEN  = (0,   200, 0)       # Immune / lymphocytes
_BLUE   = (200, 0,   0)       # Stroma (subtle)
_YELLOW = (0,   220, 220)     # PD-L1 DAB+ mask
_CYAN   = (220, 200, 0)       # Nuclei
_WHITE  = (255, 255, 255)
_ALPHA  = 0.40                # Overlay transparency


def _overlay(base_bgr: np.ndarray, mask: np.ndarray, color: tuple, alpha: float = _ALPHA) -> np.ndarray:
    """Paint color onto base image wherever mask is True."""
    out = base_bgr.copy()
    m   = mask.astype(bool)
    overlay_layer        = np.zeros_like(out)
    overlay_layer[m]     = color
    out = cv2.addWeighted(out, 1.0, overlay_layer, alpha, 0)
    return out


def _add_legend(img: np.ndarray, entries: list[tuple[str, tuple]]) -> np.ndarray:
    """Add a colour legend box to the top-right corner."""
    h, w = img.shape[:2]
    x0, y0, pad = w - 220, 10, 5
    bar_h, bar_w = 18, 25
    font = cv2.FONT_HERSHEY_SIMPLEX

    for i, (label, color) in enumerate(entries):
        y = y0 + i * (bar_h + pad)
        cv2.rectangle(img, (x0, y), (x0 + bar_w, y + bar_h), color, -1)
        cv2.putText(img, label, (x0 + bar_w + 6, y + bar_h - 4),
                    font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, label, (x0 + bar_w + 6, y + bar_h - 4),
                    font, 0.45, (30, 30, 30), 1, cv2.LINE_AA)
    return img


def save_debug_overlays(
    image_path:   str | Path,
    stain_result: dict,
    output_dir:   str | Path,
    core_id:      str = "core",
    pdl1_image_path: Optional[str | Path] = None,
) -> list[str]:
    """
    Generate and save debug overlay images.

    Parameters
    ----------
    image_path    : H&E image path (used as base for overlays)
    stain_result  : dict from analyze_pdl1_image(debug=True)
    output_dir    : directory to save PNGs
    core_id       : used in filenames
    pdl1_image_path : PDL1 image for DAB overlay (optional)

    Returns
    -------
    list of saved file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = []

    # Load base H&E image
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        print(f"[debug] Cannot load image: {image_path}")
        return saved

    # Resize large images for faster saving (max 1024px wide)
    h, w = img_bgr.shape[:2]
    if w > 1024:
        scale = 1024.0 / w
        img_bgr = cv2.resize(img_bgr, (1024, int(h * scale)))

    def _resize_mask(mask: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if mask is None:
            return None
        if mask.shape[:2] != img_bgr.shape[:2]:
            return cv2.resize(mask.astype(np.uint8),
                              (img_bgr.shape[1], img_bgr.shape[0]),
                              interpolation=cv2.INTER_NEAREST).astype(bool)
        return mask.astype(bool)

    def _save(img: np.ndarray, suffix: str):
        path = output_dir / f"{core_id}_{suffix}.png"
        cv2.imwrite(str(path), img)
        saved.append(str(path))
        return path

    # ---------------------------------------------------------------
    # 1. Base image (original)
    # ---------------------------------------------------------------
    _save(img_bgr, "00_original")

    # ---------------------------------------------------------------
    # 2. Tissue mask overlay
    # ---------------------------------------------------------------
    tissue_mask = _resize_mask(stain_result.get("_tissue_mask"))
    if tissue_mask is not None:
        vis = _overlay(img_bgr, ~tissue_mask, _WHITE, alpha=0.7)   # background = white
        cv2.putText(vis, f"Tissue Mask  area={stain_result.get('tissue_area',0):,}px",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        _save(vis, "01_tissue_mask")

    # ---------------------------------------------------------------
    # 3. DAB positive overlay (PD-L1 staining)
    # ---------------------------------------------------------------
    dab_mask = _resize_mask(stain_result.get("_dab_mask"))
    if dab_mask is not None:
        base = img_bgr.copy()
        if pdl1_image_path:
            pdl1_bgr = cv2.imread(str(pdl1_image_path))
            if pdl1_bgr is not None:
                pdl1_bgr = cv2.resize(pdl1_bgr, (img_bgr.shape[1], img_bgr.shape[0]))
                base = pdl1_bgr
        vis = _overlay(base, dab_mask, _YELLOW, alpha=0.5)
        txt = (f"DAB+ (PD-L1)  "
               f"PDL1%={stain_result.get('PDL1_percent',0):.3f}  "
               f"area={stain_result.get('dab_area',0):,}px")
        cv2.putText(vis, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 220), 2)
        _save(vis, "02_dab_mask")

    # ---------------------------------------------------------------
    # 4. Compartment segmentation
    # ---------------------------------------------------------------
    compartments = stain_result.get("_compartments", {})
    tc_mask = _resize_mask(compartments.get("tc_mask"))
    lc_mask = _resize_mask(compartments.get("lc_mask"))
    nuclei  = _resize_mask(compartments.get("nuclei_mask"))

    if tc_mask is not None and lc_mask is not None:
        vis = img_bgr.copy()
        vis = _overlay(vis, tc_mask, _RED,   alpha=0.45)   # Tumor = red
        vis = _overlay(vis, lc_mask, _GREEN, alpha=0.55)   # Immune = green
        if nuclei is not None:
            vis = _overlay(vis, nuclei, _CYAN, alpha=0.25) # Nuclei = cyan edge

        _add_legend(vis, [
            ("Tumor cells",   _RED),
            ("Immune/Lympho", _GREEN),
            ("Nuclei",        _CYAN),
        ])

        tc_pct = stain_result.get("_tc_count", compartments.get("tc_count", "?"))
        lc_pct = stain_result.get("_lc_count", compartments.get("lc_count", "?"))
        cv2.putText(vis, f"TC nuclei={tc_pct}  LC nuclei={lc_pct}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        _save(vis, "03_compartments")

    # ---------------------------------------------------------------
    # 5. Combined summary overlay
    # ---------------------------------------------------------------
    if tc_mask is not None and lc_mask is not None and dab_mask is not None:
        vis = img_bgr.copy()
        vis = _overlay(vis, tc_mask,               _RED,    alpha=0.30)
        vis = _overlay(vis, lc_mask,               _GREEN,  alpha=0.35)
        vis = _overlay(vis, dab_mask & tc_mask,    _YELLOW, alpha=0.60)   # PDL1+ TC
        vis = _overlay(vis, dab_mask & lc_mask,    _YELLOW, alpha=0.60)   # PDL1+ LC

        _add_legend(vis, [
            ("Tumor (TC)",     _RED),
            ("Immune (LC)",    _GREEN),
            ("PD-L1+ cells",   _YELLOW),
        ])

        cps     = stain_result.get("CPS",         "?")
        cpp     = stain_result.get("CPS_plus_plus","?")
        pdl1pct = stain_result.get("PDL1_percent", 0)
        cv2.putText(vis,
                    f"PDL1%={pdl1pct:.3f}  CPS={cps}  CPS++={cpp}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 220), 2)
        _save(vis, "04_summary")

    print(f"[debug] Saved {len(saved)} overlays to: {output_dir}")
    return saved


def create_debug_grid(image_dir: str | Path, output_path: str | Path, n_cols: int = 4) -> str:
    """
    Stitch all *_04_summary.png overlays from a debug directory into a grid image.
    Useful for visual QC of all 168 cores at once.
    """
    import glob
    image_dir  = Path(image_dir)
    output_path = Path(output_path)

    summaries = sorted(glob.glob(str(image_dir / "**/*_04_summary.png"), recursive=True))
    if not summaries:
        print("[debug] No summary images found.")
        return ""

    thumbs = []
    thumb_size = (256, 256)
    for p in summaries:
        img = cv2.imread(p)
        if img is not None:
            thumbs.append(cv2.resize(img, thumb_size))

    n_rows = (len(thumbs) + n_cols - 1) // n_cols
    rows   = []
    for r in range(n_rows):
        row_imgs = thumbs[r*n_cols : (r+1)*n_cols]
        while len(row_imgs) < n_cols:
            row_imgs.append(np.zeros((*thumb_size, 3), dtype=np.uint8))
        rows.append(np.hstack(row_imgs))

    grid = np.vstack(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), grid)
    print(f"[debug] Grid saved: {output_path}  ({len(thumbs)} images)")
    return str(output_path)
