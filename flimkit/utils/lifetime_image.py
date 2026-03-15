"""
flimkit/utils/lifetime_image.py
================================
Generates the intensity-weighted lifetime colour image from an assembled canvas.

Called by _run_tile_fit() in interactive.py after save_assembled_maps().
Can also be called standalone on any saved canvas .npy files.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Optional


def make_lifetime_image(
    canvas: dict,
    output_dir: Path,
    roi_name: str,
    tau_min_ns: float = 0.0,
    tau_max_ns: float = 5.0,
    smooth_sigma_px: float = 2.0,
    intensity_percentile_lo: float = 1.0,
    intensity_percentile_hi: float = 99.0,
    gamma: float = 0.5,
    dpi: int = 200,
    verbose: bool = True,
) -> Path:
    """
    Generate an intensity-weighted lifetime colour PNG from an assembled canvas.

    Hue      = τ_amp smoothed (NaN-aware Gaussian, removes tile seams)
    Brightness = photon intensity (percentile-clipped + gamma)
    Black    = unfitted / zero-intensity pixels

    Also saves a float32 TIFF of the smoothed τ values in ns.

    Parameters
    ----------
    canvas           : output of assemble_tile_maps (contains tau_mean_amp, intensity)
    output_dir       : directory to write outputs
    roi_name         : filename prefix
    tau_min_ns       : colour scale minimum (ns)
    tau_max_ns       : colour scale maximum (ns)
    smooth_sigma_px  : Gaussian σ in pixels for NaN-aware smoothing
    intensity_percentile_lo/hi : clip low/high intensity outliers before scaling
    gamma            : intensity gamma (<1 boosts dim pixels)
    dpi              : output PNG resolution
    verbose          : print saved paths

    Returns
    -------
    Path to the saved PNG
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from scipy.ndimage import gaussian_filter

    try:
        import tifffile as _tifffile
        _has_tifffile = True
    except ImportError:
        _has_tifffile = False

    from flimkit.configs import FLIM_CMAP

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tau_map = np.asarray(canvas['tau_mean_amp'], dtype=float)
    int_map = np.asarray(canvas['intensity'],    dtype=float)
    valid   = np.isfinite(tau_map) & (int_map > 0)

    # ── NaN-aware Gaussian smooth ──────────────────────────────────────────
    tau_filled = np.where(valid, tau_map, 0.0)
    w          = valid.astype(float)
    denom      = gaussian_filter(w,          sigma=smooth_sigma_px)
    tau_smooth = np.where(denom > 0.01,
                          gaussian_filter(tau_filled, sigma=smooth_sigma_px) / denom,
                          np.nan)

    # ── Save uint16 TIFF (τ scaled over tau_min–tau_max range) ───────────────
    if _has_tifffile:
        tau_clipped = np.clip(
            np.where(np.isfinite(tau_smooth), tau_smooth, tau_min_ns),
            tau_min_ns, tau_max_ns)
        tau_u16 = ((tau_clipped - tau_min_ns) /
                   (tau_max_ns - tau_min_ns + 1e-12) * 65535
                   ).astype(np.uint16)
        tiff_path = output_dir / f"{roi_name}_tau_intensity_weighted.tif"
        _tifffile.imwrite(str(tiff_path), tau_u16)
        if verbose:
            print(f"  ✓ τ TIFF → {tiff_path}  "
                  f"(uint16, {tau_min_ns}–{tau_max_ns} ns → 0–65535)")

    # ── Colour mapping ─────────────────────────────────────────────────────
    tau_norm = np.clip(
        (tau_smooth - tau_min_ns) / (tau_max_ns - tau_min_ns + 1e-12),
        0.0, 1.0)

    int_vals = int_map[valid]
    lo = np.percentile(int_vals, intensity_percentile_lo)
    hi = np.percentile(int_vals, intensity_percentile_hi)
    int_norm = np.power(
        np.clip((int_map - lo) / (hi - lo + 1e-12), 0.0, 1.0), gamma)
    int_norm[~valid] = 0.0

    rgb = FLIM_CMAP(tau_norm)[..., :3]
    rgb *= int_norm[..., np.newaxis]
    rgb[~valid] = 0.0
    rgb = np.clip(rgb, 0.0, 1.0)

    # ── Save pure-image PNG (no axes) ──────────────────────────────────────
    png_path = output_dir / f"{roi_name}_tau_intensity_weighted.png"
    plt.imsave(str(png_path), rgb, dpi=dpi)
    if verbose:
        print(f"  ✓ lifetime PNG → {png_path}")

    # ── Save annotated preview with colourbar ─────────────────────────────
    fig, (ax, cax) = plt.subplots(
        1, 2, figsize=(14, 6),
        gridspec_kw={'width_ratios': [1, 0.03]})
    ax.imshow(rgb, interpolation='nearest', aspect='equal')
    ax.set_title(
        f"{roi_name}  τ_amp {tau_min_ns}–{tau_max_ns} ns  "
        f"σ={smooth_sigma_px} px  γ={gamma}",
        fontsize=10, color='white')
    ax.axis('off')
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')

    sm = plt.cm.ScalarMappable(
        cmap=FLIM_CMAP,
        norm=mcolors.Normalize(tau_min_ns, tau_max_ns))
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("τ_amp (ns)", fontsize=10, color='white')
    plt.setp(cb.ax.yaxis.get_ticklabels(), color='white')
    plt.tight_layout(pad=0.3)

    preview_path = output_dir / f"{roi_name}_tau_intensity_weighted_preview.png"
    plt.savefig(str(preview_path), dpi=150,
                bbox_inches='tight', facecolor='black')
    plt.close(fig)
    if verbose:
        print(f"  ✓ preview PNG → {preview_path}")

    # ── Stats ──────────────────────────────────────────────────────────────
    if verbose and valid.any():
        tau_v = tau_map[valid]
        print(f"  τ_amp  median={np.nanmedian(tau_v):.3f}  "
              f"mean={np.nanmean(tau_v):.3f}  "
              f"p5={np.nanpercentile(tau_v, 5):.3f}  "
              f"p95={np.nanpercentile(tau_v, 95):.3f} ns  "
              f"n={valid.sum():,}")

    return png_path