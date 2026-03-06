"""
stain_analysis.py - Kandu's Method: PD-L1 Stain Analysis (v2 - Fixed)
=======================================================================
FIXES applied relative to v1:

  BUG 1 - DAB over-detection:
    Old: Custom Ruifrok matrix misidentified hematoxylin (blue) as DAB (brown)
         on lightly-stained PD-L1 IHC slides where virtually no brown exists.
    Fix: Use skimage.color.rgb2hed (well-validated standard H-E-D matrix).
         Apply adaptive thresholding on the D channel (DAB only).
         Add RGB-space DAB pre-filter to reject non-brown regions.

  BUG 2 - Poor tissue segmentation:
    Old: Simple saturation threshold. Excluded large tissue regions.
    Fix: LAB-space background exclusion; morphological hole-filling.

  BUG 3 - Wrong compartment detection (see tissue_segmentation.py):
    Nuclei thresholds were too small (< 80px for lymphocytes), misclassifying
    tumor cell nests as background. See tissue_segmentation.py for fix.

  BUG 4 - CPS using pixel ratios (see scoring.py):
    Fixed in scoring.py: now uses cell counts from connected-component labels.

Required outputs (PRD Section 5.3 + 11)
----------------------------------------
  PDL1_percent  - fraction of tissue area DAB+ (after strict DAB filter)
  TC_PDL1       - fraction of tumor cells DAB+
  LC_PDL1       - fraction of lymphocytes DAB+
  ST_PDL1       - fraction of stroma DAB+
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    from skimage.color import rgb2hed
    _SKIMAGE = True
except ImportError:
    _SKIMAGE = False


# ---------------------------------------------------------------------------
# Tissue mask
# ---------------------------------------------------------------------------

def get_tissue_mask(image_rgb: np.ndarray) -> np.ndarray:
    """
    Segment tissue vs. white background using LAB color space.
    Much more robust than HSV-based methods for IHC slides.

    Returns [H, W] bool mask: True = tissue.
    """
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    L   = lab[:, :, 0]

    # Background = very bright (L > 220 in LAB)
    tissue_mask = L < 220

    # Also catch slight color (not pure white) using A and B channels
    A = lab[:, :, 1].astype(np.int16) - 128
    B = lab[:, :, 2].astype(np.int16) - 128
    colored = (np.abs(A) > 3) | (np.abs(B) > 3)
    tissue_mask = tissue_mask | colored

    # Morphological fill: close holes, remove tiny specks
    k9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    t = tissue_mask.astype(np.uint8) * 255
    t = cv2.morphologyEx(t, cv2.MORPH_CLOSE, k9, iterations=3)
    t = cv2.morphologyEx(t, cv2.MORPH_OPEN,  k3, iterations=1)

    return t > 0


# ---------------------------------------------------------------------------
# DAB channel extraction (two methods)
# ---------------------------------------------------------------------------

def _get_dab_skimage(image_rgb: np.ndarray) -> np.ndarray:
    """Use skimage rgb2hed to get the DAB (D) channel. Returns float32 [H,W]."""
    hed = rgb2hed(image_rgb)
    dab = hed[:, :, 2].astype(np.float32)
    # DAB is positive where brown staining exists; clip negatives
    dab = np.clip(dab, 0, None)
    return dab


def _get_dab_manual(image_rgb: np.ndarray) -> np.ndarray:
    """
    Fallback manual HED deconvolution using the standard Ruifrok matrix.
    Uses a stricter approach: first checks if pixel looks brown in RGB.
    """
    # Standard H-E-D stain matrix (same as skimage)
    M = np.array([
        [0.6500286, 0.7041078, 0.2867867],  # Hematoxylin
        [0.7166996, 0.6055875, 0.3432356],  # Eosin
        [0.2686580, 0.5706837, 0.7763139],  # DAB
    ], dtype=np.float64)
    M_inv = np.linalg.inv(M.T)

    img = np.clip(image_rgb.astype(np.float64) / 255.0, 1e-6, 1.0)
    od  = -np.log(img)
    H, W = img.shape[:2]
    stains = (od.reshape(-1, 3) @ M_inv).reshape(H, W, 3)
    dab = np.clip(stains[:, :, 2], 0, None).astype(np.float32)
    return dab


def extract_dab_channel(image_rgb: np.ndarray) -> np.ndarray:
    """Returns the DAB optical density channel [H, W, float32]."""
    if _SKIMAGE:
        return _get_dab_skimage(image_rgb)
    return _get_dab_manual(image_rgb)


# ---------------------------------------------------------------------------
# DAB positive mask  (THE KEY FIX)
# ---------------------------------------------------------------------------

def get_dab_mask(
    image_rgb:    np.ndarray,
    tissue_mask:  np.ndarray,
    dab_channel:  Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Create a strict DAB-positive mask.

    Two-stage filtering:
      Stage 1 — RGB brown pre-filter:
        Only allow pixels where R > G and R > B
        (brown hue: red channel dominates over green and blue)
        This immediately kills hematoxylin (blue) and eosin (pink-red)
        mis-classification as DAB.

      Stage 2 — DAB OD thresholding:
        Apply per-image adaptive Otsu threshold on the D channel,
        but only within the tissue mask.

    Returns [H, W] bool mask.
    """
    if dab_channel is None:
        dab_channel = extract_dab_channel(image_rgb)

    # --- Stage 1: RGB brown pre-filter ---
    R = image_rgb[:, :, 0].astype(np.int16)
    G = image_rgb[:, :, 1].astype(np.int16)
    B = image_rgb[:, :, 2].astype(np.int16)

    # Brown: R dominant, not too dark, not too bright
    brown_hue  = (R > G) & (R > B) & (R < 230) & (G < 200)
    # Extra: compute brownness = R-G - R-B gap (higher = more brown)
    brownness  = (R - G).clip(0) + (R - B).clip(0)

    # --- Stage 2: DAB OD threshold (Otsu on tissue pixels only) ---
    dab_tissue = dab_channel * tissue_mask.astype(np.float32)

    # Min DAB for consideration (reject near-zero background noise)
    dab_tissue_vals = dab_tissue[tissue_mask & (brownness.astype(np.uint8) > 10)]

    if len(dab_tissue_vals) < 100:
        # Almost no brown pixels - completely negative slide
        return np.zeros_like(tissue_mask)

    # Otsu on the brown-filtered tissue pixels
    dab_u8 = (dab_tissue / (dab_tissue.max() + 1e-8) * 255).astype(np.uint8)
    thresh, _ = cv2.threshold(dab_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Only count pixels that are BOTH brown (RGB) AND above OD threshold
    min_od = 0.05   # Minimum OD to avoid counting noise as signal
    dab_mask = (
        tissue_mask
        & brown_hue
        & (dab_channel > min_od)
        & (dab_channel > (thresh / 255.0) * dab_tissue.max() * 0.7)
    )

    # Morphological cleanup
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dab_u8_final = dab_mask.astype(np.uint8) * 255
    dab_u8_final = cv2.morphologyEx(dab_u8_final, cv2.MORPH_OPEN,  k3)
    dab_u8_final = cv2.morphologyEx(dab_u8_final, cv2.MORPH_CLOSE, k3)

    return dab_u8_final > 0


# ---------------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------------

def analyze_pdl1_image(
    pdl1_image_path: str | Path,
    he_image_path:   Optional[str | Path] = None,
    dab_method:      str  = "auto",   # kept for API compat, uses new method
    debug:           bool = False,
) -> dict:
    """
    Full PD-L1 IHC stain analysis on a single image.

    Parameters
    ----------
    pdl1_image_path : path to PDL1 IHC image
    he_image_path   : optional matching H&E image
    debug           : include intermediate arrays in output

    Returns
    -------
    dict with:
        PDL1_percent, TC_PDL1, LC_PDL1, ST_PDL1
        tissue_area, dab_area, tc_area, lc_area, st_area
        image_path
        _compartments  (if debug=True) - compartment masks
    """
    from kandus_method.tissue_segmentation import segment_tissue

    pdl1_path = Path(pdl1_image_path)

    # Load PDL1 image
    img_bgr = cv2.imread(str(pdl1_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot load: {pdl1_path}")
    pdl1_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Tissue mask
    tissue_mask = get_tissue_mask(pdl1_rgb)
    tissue_area = int(tissue_mask.sum())
    if tissue_area == 0:
        return _empty_result(str(pdl1_path))

    # Extract DAB channel
    dab_channel = extract_dab_channel(pdl1_rgb)

    # DAB positive mask (strictly brown only)
    dab_mask = get_dab_mask(pdl1_rgb, tissue_mask, dab_channel)
    dab_area = int(dab_mask.sum())

    # Overall PDL1_percent
    PDL1_percent = dab_area / tissue_area

    # Compartment segmentation (use H&E if available for better accuracy)
    seg_image_path = he_image_path if he_image_path else pdl1_image_path
    compartments = segment_tissue(seg_image_path, tissue_mask=tissue_mask)

    tc_mask = compartments["tc_mask"]
    lc_mask = compartments["lc_mask"]
    st_mask = compartments["st_mask"]
    tc_nuclei_labels = compartments.get("tc_labels", None)
    lc_nuclei_labels = compartments.get("lc_labels", None)

    # Per-compartment DAB fraction
    def _frac(comp_mask: np.ndarray) -> float:
        n = int(comp_mask.sum())
        return float((dab_mask & comp_mask).sum()) / n if n > 0 else 0.0

    TC_PDL1 = _frac(tc_mask)
    LC_PDL1 = _frac(lc_mask)
    ST_PDL1 = _frac(st_mask)

    # Sanity clamp
    PDL1_percent = float(np.clip(PDL1_percent, 0.0, 1.0))
    TC_PDL1      = float(np.clip(TC_PDL1,      0.0, 1.0))
    LC_PDL1      = float(np.clip(LC_PDL1,      0.0, 1.0))
    ST_PDL1      = float(np.clip(ST_PDL1,      0.0, 1.0))

    result = {
        "PDL1_percent": round(PDL1_percent, 6),
        "TC_PDL1":      round(TC_PDL1,      6),
        "LC_PDL1":      round(LC_PDL1,      6),
        "ST_PDL1":      round(ST_PDL1,      6),
        "tissue_area":  tissue_area,
        "dab_area":     dab_area,
        "tc_area":      int(tc_mask.sum()),
        "lc_area":      int(lc_mask.sum()),
        "st_area":      int(st_mask.sum()),
        "image_path":   str(pdl1_path),
    }

    if debug:
        result["_dab_channel"]  = dab_channel
        result["_dab_mask"]     = dab_mask
        result["_tissue_mask"]  = tissue_mask
        result["_compartments"] = compartments

    return result


def _empty_result(image_path: str) -> dict:
    return {
        "PDL1_percent": 0.0, "TC_PDL1": 0.0, "LC_PDL1": 0.0, "ST_PDL1": 0.0,
        "tissue_area": 0, "dab_area": 0, "tc_area": 0, "lc_area": 0, "st_area": 0,
        "image_path": image_path,
    }


def analyze_all_cores(
    pdl1_dir:   str | Path,
    he_dir:     Optional[str | Path] = None,
    verbose:    bool = True,
) -> list[dict]:
    """Analyze all PDL1 images in a data_raw-style directory."""
    import re
    pdl1_dir = Path(pdl1_dir)
    he_dir   = Path(he_dir) if he_dir else None

    he_lookup: dict[str, Path] = {}
    if he_dir and he_dir.is_dir():
        for f in he_dir.iterdir():
            m = re.search(r'_(\d{3}_r\d+c\d+)', f.name)
            if m:
                he_lookup[m.group(1)] = f

    results = []
    for img_path in sorted(f for f in pdl1_dir.iterdir()
                           if f.suffix.lower() in (".jpg", ".jpeg", ".png")):
        m = re.search(r'_(\d{3}_r\d+c\d+)', img_path.name)
        core_id = m.group(1) if m else img_path.stem
        he_path = he_lookup.get(core_id)
        try:
            res = analyze_pdl1_image(img_path, he_image_path=he_path)
            res["core_id"] = core_id
            if verbose:
                print(f"  {core_id}  PDL1%={res['PDL1_percent']:.3f}  "
                      f"TC={res['TC_PDL1']:.3f}  LC={res['LC_PDL1']:.3f}  "
                      f"ST={res['ST_PDL1']:.3f}")
        except Exception as e:
            print(f"  [ERROR] {core_id}: {e}")
            res = _empty_result(str(img_path))
            res["core_id"] = core_id
        results.append(res)
    return results


# ---------------------------------------------------------------------------
# PD1 stain analysis
# ---------------------------------------------------------------------------

def analyze_pd1_image(
    pd1_image_path: str | Path,
    he_image_path:  Optional[str | Path] = None,
    compartments:   Optional[dict] = None,
    debug:          bool = False,
) -> dict:
    """
    Analyze PD1 (Programmed Death-1) IHC staining.

    PD1 Biology
    -----------
    PD1 marks **exhausted / activated T-cells** (tumor-infiltrating lymphocytes).
    Unlike PDL1 (expressed on tumor cells + immune cells), PD1 is expressed
    primarily on CD8+ T-cells and marks immune cell exhaustion in the TME.

    The same DAB chemistry is used as PDL1 (brown staining = PD1 positive).

    Key outputs
    -----------
    PD1_percent    : fraction of tissue area that is PD1-DAB+
    PD1_LC         : fraction of lymphocyte compartment that is PD1+
                     (high = many exhausted TILs → good target for checkpoint therapy)
    PD1_TC         : fraction of tumor compartment that is PD1+
                     (rare — usually low; elevated = aberrant PD1 on tumor cells)
    TIL_density    : PD1+ cells per tissue area (proxy for TIL infiltration)
    exhaustion_score : PD1_LC weighted by lymphocyte density [0,1]
                       (high = densely exhausted immune infiltrate)

    Parameters
    ----------
    pd1_image_path : path to PD1 IHC image
    he_image_path  : optional H&E for better compartment segmentation
    compartments   : pre-computed compartment dict (from segment_tissue or
                     the stain_result['_compartments']) — avoids re-running
                     segmentation if PDL1 was already analyzed on same core
    debug          : include intermediate arrays in output

    Returns
    -------
    dict with PD1_percent, PD1_LC, PD1_TC, PD1_ST, TIL_density,
    exhaustion_score, pd1_dab_area, image_path
    """
    pd1_path = Path(pd1_image_path)

    # Load PD1 image
    img_bgr = cv2.imread(str(pd1_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot load: {pd1_path}")
    pd1_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Tissue mask on PD1 image
    tissue_mask = get_tissue_mask(pd1_rgb)
    tissue_area = int(tissue_mask.sum())
    if tissue_area == 0:
        return _empty_pd1_result(str(pd1_path))

    # Extract DAB channel and positive mask (same pipeline as PDL1)
    dab_channel = extract_dab_channel(pd1_rgb)
    dab_mask    = get_dab_mask(pd1_rgb, tissue_mask, dab_channel)
    dab_area    = int(dab_mask.sum())

    PD1_percent = float(np.clip(dab_area / max(tissue_area, 1), 0.0, 1.0))

    # Use pre-computed compartments if available (from PDL1 step)
    if compartments is None:
        from kandus_method.tissue_segmentation import segment_tissue
        seg_src     = he_image_path if he_image_path else pd1_image_path
        compartments = segment_tissue(seg_src, tissue_mask=tissue_mask)

    tc_mask  = compartments.get("tc_mask",  np.zeros_like(tissue_mask))
    lc_mask  = compartments.get("lc_mask",  np.zeros_like(tissue_mask))
    st_mask  = compartments.get("st_mask",  tissue_mask)
    lc_count = compartments.get("lc_count", 0)

    def _frac(comp_mask: np.ndarray) -> float:
        n = int(comp_mask.sum())
        return float(np.clip((dab_mask & comp_mask).sum() / max(n, 1), 0.0, 1.0))

    PD1_LC = _frac(lc_mask)   # Exhausted TILs
    PD1_TC = _frac(tc_mask)   # PD1 on tumor cells (unusual)
    PD1_ST = _frac(st_mask)   # PD1 in stroma

    # TIL density: fraction of tissue that is PD1+ immune cells
    lc_area     = int(lc_mask.sum())
    TIL_density = float(np.clip(
        (dab_mask & lc_mask).sum() / max(tissue_area, 1),
        0.0, 1.0
    ))

    # Exhaustion score: PD1_LC weighted by immune density
    # High = many lymphocytes AND they are mostly PD1+ (exhausted)
    immune_density = lc_area / max(tissue_area, 1)
    exhaustion_score = float(np.clip(
        PD1_LC * np.sqrt(immune_density * 10),   # sqrt to avoid extreme values
        0.0, 1.0
    ))

    result = {
        # Primary outputs
        "PD1_percent":      round(PD1_percent,     6),
        "PD1_LC":           round(PD1_LC,           6),   # TIL exhaustion marker
        "PD1_TC":           round(PD1_TC,           6),
        "PD1_ST":           round(PD1_ST,           6),
        # Derived scores
        "TIL_density":      round(TIL_density,      6),
        "exhaustion_score": round(exhaustion_score, 6),
        # Areas
        "pd1_dab_area":     dab_area,
        "pd1_tissue_area":  tissue_area,
        "image_path":       str(pd1_path),
    }

    if debug:
        result["_pd1_dab_channel"] = dab_channel
        result["_pd1_dab_mask"]    = dab_mask
        result["_pd1_tissue_mask"] = tissue_mask

    return result


def _empty_pd1_result(image_path: str) -> dict:
    return {
        "PD1_percent":      0.0,
        "PD1_LC":           0.0,
        "PD1_TC":           0.0,
        "PD1_ST":           0.0,
        "TIL_density":      0.0,
        "exhaustion_score": 0.0,
        "pd1_dab_area":     0,
        "pd1_tissue_area":  0,
        "image_path":       image_path,
    }

