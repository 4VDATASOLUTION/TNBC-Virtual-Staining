"""
scoring.py - Kandu's Method: CPS and CPS++ Clinical Score Computation (v2)
===========================================================================
FIXES vs v1:
  BUG: CPS used pixel-area ratios instead of cell counts.
       tc_area was badly estimated (~0.6% of tissue), so with LC_PDL1=0.33
       and lc_area >> tc_area, CPS exploded to 160+.
  FIX: Uses tc_count / lc_count (number of connected-component nuclei)
       instead of pixel areas.  CPS is now clamped to [0, 100].

CPS formula (FDA-approved, TNBC):
  CPS = (PDL1+ tumor cells + PDL1+ immune cells) / total tumor cells * 100
  (Numerator uses PDL1+ cells, denominator uses ALL tumor cells)

CPS++ (Kandu's Method extension):
  CPS++ = alpha * cps_norm + (1 - alpha) * spatial_modifier
  where cps_norm = CPS / 100 and spatial_modifier in [0,1]

All 8 PRD Section 11 outputs are assembled in compute_scores().
"""

from __future__ import annotations

import numpy as np
from typing import Optional


# ---------------------------------------------------------------------------
# CPS using cell counts
# ---------------------------------------------------------------------------

def compute_cps(
    TC_PDL1:   float,
    LC_PDL1:   float,
    tc_count:  int,
    lc_count:  int,
) -> float:
    """
    CPS = (PDL1+ tumor cells + PDL1+ immune cells) / total tumor cells * 100

    Parameters
    ----------
    TC_PDL1   : fraction of tumor cells that are PD-L1+ [0,1]
    LC_PDL1   : fraction of lymphocytes that are PD-L1+ [0,1]
    tc_count  : total count of tumor cell nuclei
    lc_count  : total count of lymphocyte nuclei

    Returns
    -------
    CPS : float, clamped to [0, 100]
    """
    if tc_count == 0:
        return 0.0

    pdl1_pos_tc = TC_PDL1 * tc_count
    pdl1_pos_lc = LC_PDL1 * lc_count

    cps = (pdl1_pos_tc + pdl1_pos_lc) / tc_count * 100.0
    return float(np.clip(cps, 0.0, 100.0))   # CPS clamped [0, 100]


# ---------------------------------------------------------------------------
# Spatial interaction for CPS++
# ---------------------------------------------------------------------------

def compute_spatial_interaction(
    lc_area:     int,
    tc_area:     int,
    lc_mask:     Optional[np.ndarray] = None,
    tc_mask:     Optional[np.ndarray] = None,
    tissue_area: int = 1,
) -> dict:
    """
    Spatial interaction modifier for CPS++.

    Returns
    -------
    dict:
        immune_density             : lc_area / tissue_area
        tumor_boundary_interaction : fraction of immune cells near tumor
        spatial_modifier           : combined score [0, 1]
    """
    import cv2

    immune_density = float(np.clip(lc_area / max(tissue_area, 1), 0.0, 1.0))
    tumor_boundary_interaction = 0.0

    if lc_mask is not None and tc_mask is not None and tc_area > 0 and lc_area > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        tc_dil   = cv2.dilate(tc_mask.astype(np.uint8), kernel) > 0
        border   = tc_dil & ~tc_mask
        lc_border = int((lc_mask & border).sum())
        tumor_boundary_interaction = float(np.clip(lc_border / max(lc_area, 1), 0.0, 1.0))

    # Saturate immune_density: >20% immune cells is already very high
    spatial_modifier = float(np.clip(
        0.6 * np.clip(immune_density * 5.0, 0.0, 1.0)
        + 0.4 * tumor_boundary_interaction,
        0.0, 1.0
    ))

    return {
        "immune_density":             round(immune_density, 6),
        "tumor_boundary_interaction": round(tumor_boundary_interaction, 6),
        "spatial_modifier":           round(spatial_modifier, 6),
    }


# ---------------------------------------------------------------------------
# Main compute_scores (assembles all 8 PRD outputs)
# ---------------------------------------------------------------------------

def compute_scores(
    stain_result: dict,
    cnn_prob:     float = 0.0,
    lc_mask:      Optional[np.ndarray] = None,
    tc_mask:      Optional[np.ndarray] = None,
    alpha:        float = 0.7,
) -> dict:
    """
    Compute all clinical scores for one POI/core.

    Assembles the complete 8-output feature vector per PRD Section 11:
      tumor_percent, immune_percent, PDL1_percent,
      TC_PDL1, LC_PDL1, ST_PDL1, CPS, CPS++

    Parameters
    ----------
    stain_result : output dict from stain_analysis.analyze_pdl1_image()
    cnn_prob     : pdl1_prob_he from CNN inference [0,1]
    lc_mask      : lymphocyte mask [H,W] bool (for spatial interaction)
    tc_mask      : tumor mask [H,W] bool
    alpha        : CPS++ weighting (0.7 = 70% CPS, 30% spatial)
    """
    PDL1_percent = float(np.clip(stain_result.get("PDL1_percent", 0.0), 0.0, 1.0))
    TC_PDL1      = float(np.clip(stain_result.get("TC_PDL1",      0.0), 0.0, 1.0))
    LC_PDL1      = float(np.clip(stain_result.get("LC_PDL1",      0.0), 0.0, 1.0))
    ST_PDL1      = float(np.clip(stain_result.get("ST_PDL1",      0.0), 0.0, 1.0))

    tc_area     = stain_result.get("tc_area",     0)
    lc_area     = stain_result.get("lc_area",     0)
    st_area     = stain_result.get("st_area",     0)
    tissue_area = stain_result.get("tissue_area", 1)

    # Tissue composition percentages (from pixel areas)
    tumor_percent  = float(np.clip(tc_area / max(tissue_area, 1), 0.0, 1.0))
    immune_percent = float(np.clip(lc_area / max(tissue_area, 1), 0.0, 1.0))
    stroma_percent = float(np.clip(st_area / max(tissue_area, 1), 0.0, 1.0))

    # Cell counts (from tissue_segmentation via stain_result or stain compartments)
    # If segmentation provided counts, use them; fallback to area-based estimate
    tc_count = stain_result.get("_tc_count", max(tc_area // 500, 1))
    lc_count = stain_result.get("_lc_count", max(lc_area // 100, 0))

    # CPS (cell-count based, clamped to [0, 100])
    cps = compute_cps(TC_PDL1, LC_PDL1, tc_count, lc_count)

    # Spatial interaction
    spatial = compute_spatial_interaction(
        lc_area=lc_area, tc_area=tc_area,
        lc_mask=lc_mask, tc_mask=tc_mask,
        tissue_area=tissue_area,
    )

    # CPS++ = alpha * (CPS/100) + (1-alpha) * spatial_modifier
    cps_norm      = cps / 100.0
    cps_plus_plus = float(np.clip(
        alpha * cps_norm + (1.0 - alpha) * spatial["spatial_modifier"],
        0.0, 1.0
    ))

    # === Sanity checks ===
    assert 0.0 <= tumor_percent  <= 1.0, f"tumor_percent out of range: {tumor_percent}"
    assert 0.0 <= immune_percent <= 1.0, f"immune_percent out of range: {immune_percent}"
    assert 0.0 <= PDL1_percent   <= 1.0, f"PDL1_percent out of range: {PDL1_percent}"
    assert 0.0 <= cps            <= 100.0, f"CPS out of range: {cps}"

    return {
        # === 8 Required PRD Outputs ===
        "tumor_percent":  round(tumor_percent,  6),
        "immune_percent": round(immune_percent, 6),
        "PDL1_percent":   round(PDL1_percent,   6),
        "TC_PDL1":        round(TC_PDL1,        6),
        "LC_PDL1":        round(LC_PDL1,        6),
        "ST_PDL1":        round(ST_PDL1,        6),
        "CPS":            round(cps,             4),
        "CPS_plus_plus":  round(cps_plus_plus,  6),
        # === Additional ===
        "stroma_percent":             round(stroma_percent,  6),
        "immune_density":             spatial["immune_density"],
        "tumor_boundary_interaction": spatial["tumor_boundary_interaction"],
        "spatial_modifier":           spatial["spatial_modifier"],
        "pdl1_prob_he":               round(float(cnn_prob), 6),
        "tc_count":                   tc_count,
        "lc_count":                   lc_count,
        "alpha":                      alpha,
    }


# ---------------------------------------------------------------------------
# Patient-level aggregation
# ---------------------------------------------------------------------------

def aggregate_patient(
    poi_scores: list[dict],
    method:     str = "mean",
    weights:    Optional[list[float]] = None,
) -> dict:
    """Aggregate per-POI scores to patient level."""
    if not poi_scores:
        return {}

    _score_keys = [
        "tumor_percent", "immune_percent", "stroma_percent",
        "PDL1_percent", "TC_PDL1", "LC_PDL1", "ST_PDL1",
        "CPS", "CPS_plus_plus",
        "immune_density", "tumor_boundary_interaction",
        "spatial_modifier", "pdl1_prob_he",
    ]

    result = {"n_pois": len(poi_scores), "method": method}

    for key in _score_keys:
        vals = [s.get(key, 0.0) for s in poi_scores]
        if method == "mean":
            agg = float(np.mean(vals))
        elif method == "max":
            agg = float(np.max(vals))
        elif method == "weighted" and weights:
            w = np.array(weights, dtype=np.float64)
            w /= w.sum() + 1e-8
            agg = float(np.dot(w, vals))
        else:
            agg = float(np.mean(vals))
        result[key] = round(agg, 6)

    result["CPS_category"] = _cps_category(result.get("CPS", 0.0))
    return result


def _cps_category(cps: float) -> str:
    if cps >= 10:
        return "High (CPS >= 10): Likely immunotherapy responder"
    elif cps >= 1:
        return "Low-positive (1 <= CPS < 10): Possible responder"
    return "Negative (CPS < 1): Unlikely to respond"
