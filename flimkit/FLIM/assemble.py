"""
assemble.py — Canvas assembly and global tau derivation for per-tile FLIM fitting.

Takes a list of per-tile fit results (from fit_flim_tiles) and assembles them
into full-ROI maps.

Overlap regions use intensity-weighted averaging: each pixel value is the
photon-count-weighted mean across all tiles covering it.  This is equivalent
to how stitch_flim_tiles normalises the raw histogram cube by the weight map,
and eliminates tile boundary seams in the assembled lifetime maps.
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
import tifffile


# ── Canvas assembly ────────────────────────────────────────────────────────────

def assemble_tile_maps(
    tile_results: List[Dict[str, Any]],
    canvas_height: int,
    canvas_width: int,
    n_exp: int,
) -> Dict[str, np.ndarray]:
    """
    Assemble per-tile pixel maps onto a canvas using intensity-weighted averaging.

    For each canvas pixel covered by N tiles:
        value[y,x] = Σ(value_k[y,x] × intensity_k[y,x]) / Σ(intensity_k[y,x])

    Single-coverage pixels are unaffected (weighted average of one = that value).
    Overlap pixels are the photon-count weighted mean across contributing tiles,
    which eliminates tile boundary seams in lifetime and amplitude maps.

    Args:
        tile_results: list of dicts, each with keys:
            'pixel_maps'  — output of fit_per_pixel with keys:
                            intensity, tau_mean_amp, chi2,
                            tau1..N, a1..N
            'pixel_y'     — top-left y coordinate on canvas
            'pixel_x'     — top-left x coordinate on canvas
            'tile_h'      — tile height in pixels
            'tile_w'      — tile width in pixels
        canvas_height, canvas_width: full ROI dimensions in pixels
        n_exp: number of exponential components

    Returns:
        Dict of assembled 2D float32 arrays:
            intensity, tau_mean_amp, chi2, tau1..[n_exp], a1..[n_exp], coverage
    """
    H, W = canvas_height, canvas_width

    keys_scalar = ['tau_mean_amp', 'chi2']
    keys_tau    = [f'tau{k}' for k in range(1, n_exp + 1)]
    keys_amp    = [f'a{k}'   for k in range(1, n_exp + 1)]
    keys_all    = keys_scalar + keys_tau + keys_amp

    # Accumulate weighted numerator and denominator separately
    wsum     = {k: np.zeros((H, W), dtype=np.float64) for k in keys_all}
    wt       = np.zeros((H, W), dtype=np.float64)    # Σ intensity (weight)
    intensity_sum = np.zeros((H, W), dtype=np.float32)
    coverage = np.zeros((H, W), dtype=np.uint16)

    for tr in tile_results:
        pm = tr.get('pixel_maps')
        if pm is None:
            continue

        y0, x0 = tr['pixel_y'], tr['pixel_x']
        th, tw  = tr['tile_h'],  tr['tile_w']
        y1 = min(y0 + th, H)
        x1 = min(x0 + tw, W)
        dy, dx = y1 - y0, x1 - x0

        tile_int = np.asarray(
            pm.get('intensity', np.zeros((th, tw), dtype=np.float32)),
            dtype=np.float64)[:dy, :dx]

        wt[y0:y1, x0:x1]            += tile_int
        intensity_sum[y0:y1, x0:x1] += tile_int.astype(np.float32)
        coverage[y0:y1, x0:x1]      += 1

        for key in keys_all:
            src = pm.get(key)
            if src is None:
                continue
            src = np.asarray(src, dtype=np.float64)[:dy, :dx]
            # NaN pixels (unfitted) contribute zero weight so they don't
            # corrupt neighbours — only fitted pixels are averaged
            valid = np.isfinite(src)
            wsum[key][y0:y1, x0:x1] += np.where(valid, src * tile_int, 0.0)

    # Normalise: divide weighted sum by total weight
    safe_wt = np.where(wt > 0, wt, np.nan)

    canvas = {
        'intensity': (intensity_sum / np.where(coverage > 0, coverage, 1)
                      ).astype(np.float32),
        'coverage':  coverage.astype(np.float32),
    }

    for key in keys_all:
        result = (wsum[key] / safe_wt).astype(np.float32)
        # Pixels with no coverage stay NaN (float32 NaN propagates correctly)
        canvas[key] = result

    n_overlap = int((coverage > 1).sum())
    pct = n_overlap / (H * W) * 100
    print(f"  Canvas {H}×{W}  |  overlap: {n_overlap:,} px ({pct:.1f}%)  "
          f"|  max coverage: {int(coverage.max())}×")

    return canvas


# ── Global tau derivation ──────────────────────────────────────────────────────

def derive_global_tau(
    canvas: Dict[str, np.ndarray],
    n_exp: int,
    tissue_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Derive ROI-level lifetime summary statistics from assembled canvas maps.

    The amplitude-weighted mean tau is computed per pixel first, then summarised
    across the ROI.  This is more physically meaningful than fitting the pooled
    summed decay because it reflects the distribution of lifetimes across tissue
    rather than the photon-count-weighted ensemble.

    Args:
        canvas:       output of assemble_tile_maps
        n_exp:        number of exponential components
        tissue_mask:  optional boolean mask (H, W); if None, all non-NaN pixels used

    Returns:
        Dict with scalar summary statistics
    """
    tau_arrays = [canvas[f'tau{k}'] for k in range(1, n_exp + 1)]
    a_arrays   = [canvas[f'a{k}']   for k in range(1, n_exp + 1)]

    total_amp = np.zeros_like(tau_arrays[0])
    tau_mean  = np.zeros_like(tau_arrays[0])
    for tau_k, a_k in zip(tau_arrays, a_arrays):
        valid = np.isfinite(tau_k) & np.isfinite(a_k)
        tau_mean  = np.where(valid, tau_mean  + tau_k * a_k, tau_mean)
        total_amp = np.where(valid, total_amp + a_k,          total_amp)

    with np.errstate(invalid='ignore', divide='ignore'):
        tau_mean_amp = np.where(total_amp > 0, tau_mean / total_amp, np.nan)

    valid_px = np.isfinite(tau_mean_amp)
    if tissue_mask is not None:
        valid_px = valid_px & tissue_mask

    if not valid_px.any():
        return {'error': 'No valid fitted pixels found'}

    summary = {
        'n_pixels_fitted':          int(valid_px.sum()),
        'tau_mean_amp_global_ns':   float(np.nanmean(tau_mean_amp[valid_px])),
        'tau_std_amp_global_ns':    float(np.nanstd(tau_mean_amp[valid_px])),
        'tau_median_amp_global_ns': float(np.nanmedian(tau_mean_amp[valid_px])),
    }

    for k, (tau_k, a_k) in enumerate(zip(tau_arrays, a_arrays), start=1):
        valid_k = np.isfinite(tau_k) & np.isfinite(a_k) & valid_px
        if valid_k.any():
            summary[f'tau{k}_mean_ns'] = float(
                np.average(tau_k[valid_k], weights=a_k[valid_k]))
            summary[f'tau{k}_std_ns']  = float(np.nanstd(tau_k[valid_k]))
            summary[f'a{k}_mean_frac'] = float(
                np.nanmean(a_k[valid_k] / total_amp[valid_k]))

    return summary


# ── Save assembled outputs ─────────────────────────────────────────────────────

def save_assembled_maps(
    canvas: Dict[str, np.ndarray],
    global_summary: Dict[str, Any],
    output_dir: Path,
    roi_name: str,
    n_exp: int,
    tau_display_min: Optional[float] = None,
    tau_display_max: Optional[float] = None,
    intensity_display_min: Optional[float] = None,
    intensity_display_max: Optional[float] = None,
):
    """Save assembled canvas maps and global summary to output_dir."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save raw numpy maps for downstream use
    for key, arr in canvas.items():
        np.save(str(output_dir / f"{roi_name}_{key}.npy"), arr)

    # ── Intensity TIFF ─────────────────────────────────────────────────────────
    intensity = canvas['intensity']
    max_val   = intensity.max()
    if max_val > 0:
        i_min = intensity_display_min or 0.0
        i_max = intensity_display_max or max_val
        intensity_scaled = np.clip(
            (intensity - i_min) / (i_max - i_min) * 65535, 0, 65535
        ).astype(np.uint16)
    else:
        intensity_scaled = np.zeros_like(intensity, dtype=np.uint16)
    tifffile.imwrite(str(output_dir / f"{roi_name}_intensity.tif"), intensity_scaled)

    # ── tau_mean_amp TIFF (float32, ns) ────────────────────────────────────────
    # Save as float32 so downstream tools see actual lifetime values in ns.
    # Also save a uint16 scaled version for quick viewing.
    tau_map = canvas['tau_mean_amp']
    finite  = np.isfinite(tau_map)

    # float32 TIFF — raw ns values, NaN → 0
    tau_f32 = np.where(finite, tau_map, 0.0).astype(np.float32)
    tifffile.imwrite(str(output_dir / f"{roi_name}_tau_mean_amp.tif"), tau_f32)

    # uint16 scaled TIFF — for display, NaN replaced before cast to avoid warning
    t_min = tau_display_min if tau_display_min is not None else float(np.nanmin(tau_map))
    t_max = tau_display_max if tau_display_max is not None else float(np.nanmax(tau_map))
    if t_max > t_min:
        tau_norm   = np.where(finite,
                              (tau_map - t_min) / (t_max - t_min) * 65535,
                              0.0)
        tau_scaled = np.clip(tau_norm, 0, 65535).astype(np.uint16)
    else:
        tau_scaled = np.zeros_like(tau_map, dtype=np.uint16)
    tifffile.imwrite(
        str(output_dir / f"{roi_name}_tau_mean_amp_display.tif"), tau_scaled)

    # ── Global summary text ────────────────────────────────────────────────────
    summary_path = output_dir / f"{roi_name}_global_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("PER-TILE FIT — GLOBAL TAU SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"ROI: {roi_name}\n")
        n_pixels_fitted = global_summary.get('n_pixels_fitted', 'N/A')
        if isinstance(n_pixels_fitted, int):
            f.write(f"Pixels fitted: {n_pixels_fitted:,}\n\n")
        else:
            f.write(f"Pixels fitted: {n_pixels_fitted}\n\n")
        f.write("Amplitude-weighted mean lifetime (global):\n")
        f.write(f"  tau_mean  = {global_summary.get('tau_mean_amp_global_ns', float('nan')):.4f} ns\n")
        f.write(f"  tau_std   = {global_summary.get('tau_std_amp_global_ns',  float('nan')):.4f} ns\n")
        f.write(f"  tau_median= {global_summary.get('tau_median_amp_global_ns', float('nan')):.4f} ns\n\n")
        f.write("Per-component (amplitude-weighted across ROI):\n")
        for k in range(1, n_exp + 1):
            tau_k = global_summary.get(f'tau{k}_mean_ns', None)
            a_k   = global_summary.get(f'a{k}_mean_frac', None)
            if tau_k is not None:
                f.write(f"  tau{k} = {tau_k:.4f} ns   "
                        f"a{k} = {a_k:.3f} (mean amplitude fraction)\n")
        f.write("\n" + "=" * 60 + "\n")

    print(f"  ✓ Assembled maps saved to {output_dir}")
    print(f"  ✓ Global summary: {summary_path.name}")

    return summary_path