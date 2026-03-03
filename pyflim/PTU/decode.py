"""PTU FLIM Data Extraction - Simplified for Existing PTUFile Class

Uses your existing reader.PTUFile class - no external dependencies.
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any


def get_flim_histogram_from_ptufile(
    ptu_path: Path,
    rotate_cw: bool = True,
    binning: int = 1,
    channel: int = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Extract FLIM histogram using your existing PTUFile class.
    
    Args:
        ptu_path: Path to PTU file
        rotate_cw: If True, rotate tile 90° clockwise (Leica convention)
        binning: Spatial binning factor
        channel: Detection channel (None = auto-detect)
    
    Returns:
        Tuple of:
            - hist: (Y, X, H) FLIM histogram
            - metadata: dict with tcspc_resolution, n_time_bins, etc.
    
    Example:
        >>> from code.PTU.reader import PTUFile
        >>> hist, meta = get_flim_histogram_from_ptufile("tile_s1.ptu")
        >>> print(f"Shape: {hist.shape}, TCSPC: {meta['tcspc_resolution']*1e12:.2f} ps")
    """
    from .reader import PTUFile
    
    ptu = PTUFile(str(ptu_path), verbose=False)
    
    # Get FLIM histogram (Y, X, H)
    # Your PTUFile.pixel_stack() already handles rotation internally if needed
    stack = ptu.pixel_stack(binning=binning, channel=channel)
    
    # Additional rotation if Leica data needs it
    # (Check if your PTUFile already rotates - if so, set rotate_cw=False)
    if rotate_cw:
        stack = np.rot90(stack, k=-1, axes=(0, 1))
    
    # Extract metadata from your PTUFile class
    metadata = {
        'tcspc_resolution': ptu.tcspc_res,  # seconds
        'n_time_bins': ptu.n_bins,
        'tile_shape': (ptu.n_y // binning, ptu.n_x // binning),
        'frequency': ptu.frequency,  # Hz
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
