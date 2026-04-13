import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
import tifffile


# Canvas assembly

def assemble_tile_maps(
    tile_results: List[Dict[str, Any]],
    canvas_height: int,
    canvas_width: int,
    n_exp: int,
) -> Dict[str, np.ndarray]:
    """
    Assemble per-tile pixel maps onto a canvas using nearest-centre selection.

    For each canvas pixel, only the tile whose centre is closest contributes.
    This avoids blurring in overlap regions that occurs when averaging two
    slightly misaligned tiles — even sub-pixel misalignment is enough to blur
    sharp histological features when averaged 50/50.

    Single-coverage pixels are unaffected (only one tile anyway).

    Args
    ----
    tile_results : list of dicts from fit_flim_tiles, each with:
                   pixel_maps, pixel_y, pixel_x, tile_h, tile_w
    canvas_height, canvas_width : full ROI canvas size in pixels
    n_exp : number of exponential components

    Returns
    -------
    Dict of assembled 2-D float32 arrays:
        intensity, tau_mean_amp, chi2, tau1..N, a1..N, coverage
    """
    H, W = canvas_height, canvas_width

    keys_scalar = ['tau_mean_amp', 'chi2']
    keys_tau    = [f'tau{k}' for k in range(1, n_exp + 1)]
    keys_amp    = [f'a{k}'   for k in range(1, n_exp + 1)]
    keys_all    = keys_scalar + keys_tau + keys_amp

    # Build nearest-centre ownership map
    # For every canvas pixel, record the index of the tile whose centre is
    # closest.  Ties broken by later tiles (last write wins, doesn't matter).
    # Use squared Euclidean distance — no sqrt needed for comparison.
    min_dist2 = np.full((H, W), np.inf, dtype=np.float64)
    owner     = np.full((H, W), -1,     dtype=np.int32)

    for ti, tr in enumerate(tile_results):
        if tr.get('pixel_maps') is None:
            continue
        y0, x0 = tr['pixel_y'], tr['pixel_x']
        th, tw  = tr['tile_h'],  tr['tile_w']
        y1      = min(y0 + th, H)
        x1      = min(x0 + tw, W)

        cy = y0 + th / 2.0   # tile centre row
        cx = x0 + tw / 2.0   # tile centre col

        # Distance from tile centre for every pixel in this tile's footprint
        rows = np.arange(y0, y1, dtype=np.float64)
        cols = np.arange(x0, x1, dtype=np.float64)
        dy2  = (rows - cy) ** 2               # (dy,)
        dx2  = (cols - cx) ** 2               # (dx,)
        dist2 = dy2[:, np.newaxis] + dx2      # (dy, dx) broadcast

        # Only take ownership where this tile is strictly closer
        region = min_dist2[y0:y1, x0:x1]
        closer = dist2 < region
        min_dist2[y0:y1, x0:x1] = np.where(closer, dist2, region)
        owner[y0:y1, x0:x1]     = np.where(closer, ti, owner[y0:y1, x0:x1])

    # Place tile data using ownership map
    canvas = {k: np.full((H, W), np.nan, dtype=np.float32) for k in keys_all}
    intensity_canvas = np.zeros((H, W), dtype=np.float32)
    coverage         = np.zeros((H, W), dtype=np.uint16)

    for ti, tr in enumerate(tile_results):
        pm = tr.get('pixel_maps')
        if pm is None:
            continue
        y0, x0 = tr['pixel_y'], tr['pixel_x']
        th, tw  = tr['tile_h'],  tr['tile_w']
        y1 = min(y0 + th, H)
        x1 = min(x0 + tw, W)
        dy, dx = y1 - y0, x1 - x0

        # Mask: pixels owned by this tile
        owned = (owner[y0:y1, x0:x1] == ti)   # (dy, dx) bool

        tile_int = np.asarray(
            pm.get('intensity', np.zeros((th, tw), dtype=np.float32)),
            dtype=np.float32)[:dy, :dx]

        intensity_canvas[y0:y1, x0:x1] = np.where(
            owned, tile_int, intensity_canvas[y0:y1, x0:x1])
        coverage[y0:y1, x0:x1] = np.where(
            owned, coverage[y0:y1, x0:x1] + 1, coverage[y0:y1, x0:x1])

        for key in keys_all:
            raw = pm.get(key)
            if raw is None:
                continue
            raw = np.asarray(raw, dtype=np.float32)[:dy, :dx]
            # Only place fitted (finite) pixels that this tile owns
            place = owned & np.isfinite(raw)
            canvas[key][y0:y1, x0:x1] = np.where(
                place, raw, canvas[key][y0:y1, x0:x1])

    canvas['intensity'] = intensity_canvas
    canvas['coverage']  = coverage.astype(np.float32)

    n_covered = int((owner >= 0).sum())
    print(
        f"  Canvas {H}×{W}  |  tiles={len(tile_results)}  "
        f"|  covered={n_covered:,} px  "
        f"|  ownership map built (nearest-centre, no blending)"
    )

    return canvas


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


# Save assembled outputs

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

    # Intensity TIFF (uint16)
    # Use percentile-based clipping so a handful of very bright pixels (cell
    # clusters, tile-overlap edges) do not compress the bulk of tissue to
    # near-zero.  Display range overrides take priority when provided.
    intensity = canvas['intensity']
    pos       = intensity[intensity > 0]
    if pos.size > 0:
        # lo=0 so dim tissue is never clipped to black.
        # hi=99.5th percentile clips only the top 0.5% of bright outliers.
        i_min = float(intensity_display_min) if intensity_display_min is not None else 0.0
        i_max = float(intensity_display_max) if intensity_display_max is not None                 else float(np.percentile(pos, 99.0))
        i_max = max(i_max, i_min + 1e-6)
        intensity_scaled = np.clip(
            np.clip(intensity, 0.0, i_max) / i_max * 65535,
            0, 65535
        ).astype(np.uint16)
    else:
        intensity_scaled = np.zeros_like(intensity, dtype=np.uint16)
        i_min, i_max = 0.0, 0.0
    tifffile.imwrite(str(output_dir / f"{roi_name}_intensity.tif"), intensity_scaled)
    print(f"  intensity display range: {i_min:.1f} – {i_max:.1f} counts")

    #  tau_mean_amp TIFF (uint16) 
    # 0     → tau_min (or 0 if auto)
    # 65535 → tau_max (or nanmax if auto)
    # 0     → unfitted pixels (NaN → 0 before cast)
    # This matches the convention used by Leica LAS X FLIM exports.
    tau_map = canvas['tau_mean_amp']
    finite  = np.isfinite(tau_map)

    t_min = float(tau_display_min) if tau_display_min is not None \
            else float(np.nanpercentile(tau_map[finite], 0.5) if finite.any() else 0.0)
    t_max = float(tau_display_max) if tau_display_max is not None \
            else float(np.nanpercentile(tau_map[finite], 99.5) if finite.any() else 1.0)
    t_max = max(t_max, t_min + 1e-6)

    tau_u16 = np.where(
        finite,
        np.clip((tau_map - t_min) / (t_max - t_min) * 65535, 0, 65535),
        0                            # unfitted → 0 (black), never NaN cast
    ).astype(np.uint16)
    tifffile.imwrite(str(output_dir / f"{roi_name}_tau_mean_amp.tif"), tau_u16)
    print(f"  tau_mean_amp display range: {t_min:.3f} – {t_max:.3f} ns  "
          f"→  0 – 65535")

    # Global summary text 
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

    # Component RGB TIFF
    try:
        from ..utils.lifetime_image import make_component_rgb_tiff
        make_component_rgb_tiff(canvas, output_dir, roi_name, n_exp, verbose=True)
    except Exception as _e:
        print(f"  ⚠ component RGB skipped: {_e}")

    print(f"  ✓ Assembled maps saved to {output_dir}")
    print(f"  ✓ Global summary: {summary_path.name}")

    return summary_path