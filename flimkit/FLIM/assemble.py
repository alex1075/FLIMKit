"""
assemble.py — Canvas assembly and global tau derivation for per-tile FLIM fitting.

Takes a list of per-tile fit results (from fit_flim_tiles) and assembles them
into full-ROI maps.  Overlap regions use the tile with the higher total intensity
(more photons → more reliable fit).
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
    Place per-tile pixel_maps onto a canvas.

    Args:
        tile_results: list of dicts, each with keys:
            'pixel_maps'  — output of fit_per_pixel (tau1, a1, tau2, a2, …, intensity, chi2)
            'pixel_y'     — top-left y coordinate on canvas
            'pixel_x'     — top-left x coordinate on canvas
            'tile_h'      — tile height in pixels
            'tile_w'      — tile width in pixels
        canvas_height, canvas_width: full ROI dimensions
        n_exp: number of exponential components

    Returns:
        Dict of assembled 2D float32 arrays:
            tau1, [tau2, tau3], a1, [a2, a3], tau_mean_amp, intensity, chi2
    """
    H, W = canvas_height, canvas_width

    # Initialise with NaN so unfitted pixels are distinguishable from tau=0
    canvas = {
        'intensity': np.zeros((H, W), dtype=np.float32),
        'chi2':      np.full((H, W), np.nan, dtype=np.float32),
        'tau_mean_amp': np.full((H, W), np.nan, dtype=np.float32),
    }
    for k in range(1, n_exp + 1):
        canvas[f'tau{k}'] = np.full((H, W), np.nan, dtype=np.float32)
        canvas[f'a{k}']   = np.zeros((H, W), dtype=np.float32)

    # Track which canvas pixels have been filled and by how many photons,
    # so overlap regions keep the tile with the higher intensity.
    best_intensity = np.zeros((H, W), dtype=np.float32)

    for tr in tile_results:
        pm   = tr.get('pixel_maps')
        if pm is None:
            continue

        y0   = tr['pixel_y']
        x0   = tr['pixel_x']
        th   = tr['tile_h']
        tw   = tr['tile_w']
        y1   = min(y0 + th, H)
        x1   = min(x0 + tw, W)
        dy   = y1 - y0
        dx   = x1 - x0

        tile_intensity = pm.get('intensity', np.zeros((th, tw), dtype=np.float32))
        tile_intensity = tile_intensity[:dy, :dx]

        # For each pixel in the overlap region, keep whichever tile has more photons
        replace_mask = tile_intensity > best_intensity[y0:y1, x0:x1]

        best_intensity[y0:y1, x0:x1] = np.where(
            replace_mask, tile_intensity, best_intensity[y0:y1, x0:x1]
        )
        canvas['intensity'][y0:y1, x0:x1] = np.where(
            replace_mask, tile_intensity, canvas['intensity'][y0:y1, x0:x1]
        )

        for key in ['chi2', 'tau_mean_amp'] + \
                   [f'tau{k}' for k in range(1, n_exp + 1)] + \
                   [f'a{k}'   for k in range(1, n_exp + 1)]:
            src = pm.get(key)
            if src is None:
                continue
            src = np.asarray(src, dtype=np.float32)[:dy, :dx]
            canvas[key][y0:y1, x0:x1] = np.where(
                replace_mask, src, canvas[key][y0:y1, x0:x1]
            )

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
    # Per-pixel amplitude-weighted mean tau
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

    # Build valid pixel mask
    valid_px = np.isfinite(tau_mean_amp)
    if tissue_mask is not None:
        valid_px = valid_px & tissue_mask

    if not valid_px.any():
        return {'error': 'No valid fitted pixels found'}

    summary = {
        'n_pixels_fitted': int(valid_px.sum()),
        'tau_mean_amp_global_ns':  float(np.nanmean(tau_mean_amp[valid_px])),
        'tau_std_amp_global_ns':   float(np.nanstd(tau_mean_amp[valid_px])),
        'tau_median_amp_global_ns': float(np.nanmedian(tau_mean_amp[valid_px])),
    }

    # Per-component amplitude-weighted means
    for k, (tau_k, a_k) in enumerate(zip(tau_arrays, a_arrays), start=1):
        valid_k = np.isfinite(tau_k) & np.isfinite(a_k) & valid_px
        if valid_k.any():
            summary[f'tau{k}_mean_ns'] = float(
                np.average(tau_k[valid_k], weights=a_k[valid_k])
            )
            summary[f'tau{k}_std_ns']  = float(np.nanstd(tau_k[valid_k]))
            summary[f'a{k}_mean_frac'] = float(
                np.nanmean(a_k[valid_k] / total_amp[valid_k])
            )

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

    # Save intensity TIFF
    intensity = canvas['intensity']
    max_val = intensity.max()
    if max_val > 0:
        i_min = intensity_display_min or 0.0
        i_max = intensity_display_max or max_val
        intensity_scaled = np.clip(
            (intensity - i_min) / (i_max - i_min) * 65535, 0, 65535
        ).astype(np.uint16)
    else:
        intensity_scaled = np.zeros_like(intensity, dtype=np.uint16)
    tifffile.imwrite(str(output_dir / f"{roi_name}_intensity.tif"), intensity_scaled)

    # Save tau_mean_amp as colour-scaled TIFF
    tau_map = canvas['tau_mean_amp']
    t_min = tau_display_min if tau_display_min is not None else float(np.nanmin(tau_map))
    t_max = tau_display_max if tau_display_max is not None else float(np.nanmax(tau_map))
    if t_max > t_min:
        tau_scaled = np.clip(
            (tau_map - t_min) / (t_max - t_min) * 65535, 0, 65535
        ).astype(np.uint16)
        tau_scaled[~np.isfinite(tau_map)] = 0
    else:
        tau_scaled = np.zeros_like(tau_map, dtype=np.uint16)
    tifffile.imwrite(str(output_dir / f"{roi_name}_tau_mean_amp.tif"), tau_scaled)

    # Save global summary text
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