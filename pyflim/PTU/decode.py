import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any
from .reader import normalise_flim, PTUFile

def get_flim_histogram_from_ptufile(
    ptu_path: Path,
    rotate_cw: bool = True,
    binning: int = 1,
    channel: int = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    from .reader import PTUFile
    ptu = PTUFile(str(ptu_path))
    # Use raw_pixel_stack to get integer counts
    stack = ptu.raw_pixel_stack(binning=binning, channel=channel)
    if rotate_cw:
        stack = np.rot90(stack, k=-1, axes=(0, 1))
    # frequency (if needed)
    freq = getattr(ptu, 'sync_rate', 1.0 / ptu.tcspc_res if hasattr(ptu, 'tcspc_res') else None)
    metadata = {
        'tcspc_resolution': ptu.tcspc_res,
        'n_time_bins': ptu.n_bins,
        'tile_shape': (ptu.n_y // binning, ptu.n_x // binning),
        'frequency': freq,
        'binning': binning,
        'channel': channel,
    }
    return stack, metadata


def create_time_axis(n_bins: int, tcspc_resolution: float) -> np.ndarray:
    """
    Create time axis in nanoseconds.
    
    Args:
        n_bins: Number of time bins
        tcspc_resolution: Time per bin in seconds
    
    Returns:
        time_axis_ns: Array in nanoseconds
    
    Example:
        >>> t = create_time_axis(256, 97e-12)
        >>> print(f"Range: 0 - {t[-1]:.2f} ns")
    """
    return np.arange(n_bins) * tcspc_resolution * 1e9


def get_intensity_from_ptufile(
    ptu_path: Path,
    rotate_cw: bool = True,
    binning: int = 1
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Get 2D intensity image from PTU file.
    
    Args:
        ptu_path: Path to PTU file
        rotate_cw: Rotate 90° clockwise
        binning: Spatial binning factor
    
    Returns:
        Tuple of (intensity_image, metadata)
    """
    stack, metadata = get_flim_histogram_from_ptufile(
        ptu_path, rotate_cw, binning, channel=None
    )
    
    # Sum over time axis
    intensity = stack.sum(axis=2)
    
    return intensity, metadata

def get_raw_flim_histogram(ptu_path, rotate_cw: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
    ptu = PTUFile(str(ptu_path), verbose=False)  # or True for debugging
    stack = ptu.raw_pixel_stack(channel=None, binning=1)
    if rotate_cw:
        stack = np.rot90(stack, k=-1, axes=(0, 1))
    metadata = {
        'tcspc_resolution': ptu.tcspc_res,
        'n_time_bins': ptu.n_bins,
        'tile_shape': (ptu.n_y, ptu.n_x),
        'frequency': ptu.sync_rate,
    }
    return stack, metadata