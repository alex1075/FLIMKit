import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any
from .reader import normalise_flim, PTUFile

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

def get_flim_histogram_from_ptufile(
    ptu_path: Path,
    rotate_cw: bool = True,
    binning: int = 1,
    channel: int = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load raw FLIM histogram (uint32 counts) using custom PTUFile if it returns
    valid data; otherwise fall back to ptufile loader.
    """
    # Attempt with custom PTUFile

    from .reader import PTUFile
    ptu = PTUFile(str(ptu_path), verbose=False)
    # Use raw_pixel_stack if available; else pixel_stack (but we prefer raw)
    if hasattr(ptu, 'raw_pixel_stack'):
        stack = ptu.raw_pixel_stack(channel=channel, binning=binning)
    else:
        stack = ptu.pixel_stack(channel=channel, binning=binning)
        # If pixel_stack returns normalized floats, treat as failure
        if stack.max() <= 1.0 and stack.sum() > 0:
            raise ValueError("Custom class returned normalized data")

    # Check if stack has any photons
    if stack.sum() == 0:
        raise ValueError("Custom class returned zero photons")

    # Success: rotate if needed and build metadata
    if rotate_cw:
        stack = np.rot90(stack, k=-1, axes=(0, 1))
    metadata = {
        'tcspc_resolution': ptu.tcspc_res,
        'n_time_bins': ptu.n_bins,
        'tile_shape': (ptu.n_y // binning, ptu.n_x // binning),
        'frequency': ptu.sync_rate,
        'binning': binning,
        'channel': channel,
    }
    return stack, metadata

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

def get_raw_flim_histogram2(ptu_path, rotate_cw: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load raw FLIM histogram using ptufile (reference implementation).
    Returns uint32 array (Y, X, H) and metadata.
    """
    import ptufile
    ptu = ptufile.PtuFile(str(ptu_path))
    data = ptu[:].squeeze()          # (Y, X, H) or (T, Y, X, C, H)
    if data.ndim != 3:
        # If there are extra dims (e.g., T, C), sum over them
        # For simplicity, assume H is last and others are singletons or to be summed
        # This matches typical Leica data: (T, Y, X, C, H) with T=1, C=1
        data = data.reshape((data.shape[0], data.shape[1], -1))
    if rotate_cw:
        data = np.rot90(data, k=-1, axes=(0, 1))
    metadata = {
        'tcspc_resolution': ptu.tcspc_resolution,
        'n_time_bins': data.shape[2],
        'tile_shape': (data.shape[0], data.shape[1]),
        'frequency': ptu.frequency,
    }
    return data.astype(np.uint32), metadata