import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, PowerNorm
from typing import Dict, Tuple, Optional, Union


# Colormap presets
COLORMAPS = {
    'hsv': 'hsv',
    'viridis': 'viridis',
    'cool': 'cool',
    'hot': 'hot',
    'twilight': 'twilight',
}


def compute_intensity_weighted_lifetime(
    pixel_maps: Dict[str, np.ndarray],
    intensity: np.ndarray,
    n_exp: int = 2,
) -> np.ndarray:
    """τ_int = Σ(τᵢ × aᵢ) / total_intensity. Returns (Y,X) float32; unfitted=NaN."""
    # If pre-computed tau_mean_int exists in pixel_maps, use it directly
    if 'tau_mean_int' in pixel_maps:
        return np.asarray(pixel_maps['tau_mean_int'], dtype=np.float32)
    
    shape = intensity.shape
    tau_int_weighted = np.zeros(shape, dtype=np.float32)

    # Sum weighted lifetimes
    for i in range(1, n_exp + 1):
        tau_key = f'tau_{i}'
        amp_key = f'a{i}'

        if tau_key in pixel_maps and amp_key in pixel_maps:
            tau = pixel_maps[tau_key]
            amp = pixel_maps[amp_key]

            # Weight by amplitude
            tau_int_weighted += tau * amp

    # Normalize by total intensity
    mask = intensity > 0
    tau_int_weighted[mask] /= intensity[mask]
    tau_int_weighted[~mask] = np.nan

    return tau_int_weighted


def apply_color_scale(
    image: np.ndarray,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    gamma: float = 1.0,
    percentile_auto: Tuple[float, float] = (2, 98),
) -> np.ndarray:
    # Create working copy, preserve NaN
    valid_mask = ~np.isnan(image)
    valid_pixels = image[valid_mask]

    # Auto-detect range if not provided
    if vmin is None:
        vmin = np.percentile(valid_pixels, percentile_auto[0]) if valid_pixels.size > 0 else 0
    if vmax is None:
        vmax = np.percentile(valid_pixels, percentile_auto[1]) if valid_pixels.size > 0 else 1

    # Clip to range
    clipped = np.clip(image, vmin, vmax)

    # Normalize to [0, 1]
    if vmax > vmin:
        normalized = (clipped - vmin) / (vmax - vmin)
    else:
        normalized = np.zeros_like(clipped)

    # Apply gamma correction
    if gamma != 1.0:
        normalized = np.power(normalized, 1.0 / gamma)

    # Restore NaN
    normalized[~valid_mask] = np.nan

    return normalized


def get_colormap(name: str = 'viridis') -> plt.cm.ScalarMappable:
    cmap_name = COLORMAPS.get(name, name)
    return plt.cm.get_cmap(cmap_name)


def compute_region_stats(
    lifetime_map: np.ndarray,
    intensity_map: np.ndarray,
    region_mask: np.ndarray,
    full_stats: bool = False,
) -> Dict[str, Union[float, dict]]:
    """Extract region mask, compute median_tau, mean_amplitude, photon_count, n_pixels. Excludes NaN."""
    # Extract pixels in region
    region_lifetime = lifetime_map[region_mask]
    region_intensity = intensity_map[region_mask]

    # Filter valid (non-NaN) pixels
    valid_mask = ~np.isnan(region_lifetime)
    valid_lifetime = region_lifetime[valid_mask]
    valid_intensity = region_intensity[valid_mask]

    stats = {
        'median_tau': float(np.nanmedian(valid_lifetime)) if valid_lifetime.size > 0 else np.nan,
        'mean_amplitude': float(np.mean(valid_intensity)) if valid_intensity.size > 0 else np.nan,
        'photon_count': int(np.sum(valid_intensity)),
        'n_pixels': int(np.sum(valid_mask)),
    }

    if full_stats and valid_lifetime.size > 0:
        stats.update({
            'min_tau': float(np.nanmin(valid_lifetime)),
            'max_tau': float(np.nanmax(valid_lifetime)),
            'std_tau': float(np.nanstd(valid_lifetime)),
            'mean_tau': float(np.nanmean(valid_lifetime)),
            'percentiles': {
                'p25': float(np.nanpercentile(valid_lifetime, 25)),
                'p50': float(np.nanpercentile(valid_lifetime, 50)),
                'p75': float(np.nanpercentile(valid_lifetime, 75)),
                'p90': float(np.nanpercentile(valid_lifetime, 90)),
                'p95': float(np.nanpercentile(valid_lifetime, 95)),
            },
        })

    return stats


def mask_to_rgba(
    mask: np.ndarray,
    color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    alpha: float = 0.3,
) -> np.ndarray:
    rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
    rgba[mask, 0] = color[0]  # R
    rgba[mask, 1] = color[1]  # G
    rgba[mask, 2] = color[2]  # B
    rgba[mask, 3] = alpha     # A
    return rgba
