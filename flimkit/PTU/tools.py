import time
import numpy as np
from numpy.typing import DTypeLike
from .reader import PTUFile, PTUArray5D
from os import PathLike
from typing import Any, Literal, Sequence
from types import EllipsisType

def filter_photons_with_mask(ptu_path, mask, channel=None, binning=1, verbose=False):
    """
    Filter photons from a PTU file using a cell mask.
    
    Keeps photons where mask[y, x] != 0 (non-zero / white).
    Discards photons where mask[y, x] == 0 (black).
    
    Args:
        ptu_path: Path to PTU file
        mask: 2D numpy array (Y, X) with 0 = discard, non-zero = keep
        channel: Detection channel (None = auto-detect)
        binning: Spatial binning factor (default: 1)
        verbose: Print progress messages
    
    Returns:
        stack: 3D array (Y, X, H) with filtered photon histograms
    """
    ptu = PTUFile(str(ptu_path), verbose=False)
    
    if channel is None:
        if ptu.photon_channel is None:
            ptu.summed_decay(channel=None)
        channel = ptu.photon_channel
    
    if verbose:
        print(f"Filtering photons with mask (channel={channel}, binning={binning})")
        print(f"Mask shape: {mask.shape}")
        print(f"Mask: keep {np.sum(mask != 0):,} pixels, discard {np.sum(mask == 0):,} pixels")
    
    t0 = time.time()
    
    records = ptu._load_records()
    ch, dtime, _ = ptu._decode_picoharp_t3(records)
    
    special = ch == 0xF
    ph_mask = (~special) & (ch == channel)
    ph_idx = np.where(ph_mask)[0]
    ph_dtime = dtime[ph_mask].astype(np.int32)
    
    marker_mask = special & (dtime != 0)
    marker_idx = np.where(marker_mask)[0]
    marker_dtime = dtime[marker_mask]
    
    line_start_abs = marker_idx[marker_dtime & 1 != 0]
    line_stop_abs = marker_idx[marker_dtime & 2 != 0]
    
    n_lines = min(len(line_start_abs), len(line_stop_abs))
    ny_out = ptu.n_y // binning
    nx_out = ptu.n_x // binning
    
    # Resize mask and create a boolean array for "keep" (True = keep)
    if mask.shape != (ny_out, nx_out):
        from scipy.ndimage import zoom
        zoom_factors = (ny_out / mask.shape[0], nx_out / mask.shape[1])
        mask_resized = zoom(mask, zoom_factors, order=0) != 0   # True = keep
    else:
        mask_resized = mask != 0   # True = keep
    
    stack = np.zeros((ny_out, nx_out, ptu.n_bins), dtype=np.uint32)
    
    photons_kept = 0
    photons_discarded = 0
    
    for line_num in range(n_lines):
        ls = line_start_abs[line_num]
        le = line_stop_abs[line_num]
        if le <= ls:
            continue
        
        row = (line_num % ptu.n_y) // binning
        if row >= ny_out:
            continue
        
        lo = np.searchsorted(ph_idx, ls, side="right")
        hi = np.searchsorted(ph_idx, le, side="left")
        if hi <= lo:
            continue
        
        ph_in = ph_idx[lo:hi]
        dt_in = ph_dtime[lo:hi]
        line_len = le - ls
        rel_pos = ph_in - ls
        px = np.clip((rel_pos * ptu.n_x) // line_len, 0, ptu.n_x - 1)
        px_bin = px // binning
        
        for i in range(len(dt_in)):
            col = px_bin[i]
            if col >= nx_out:
                continue
            
            # Keep photon if mask says keep (True)
            if not mask_resized[row, col]:
                photons_discarded += 1
                continue
            
            tb = dt_in[i]
            if tb < ptu.n_bins:
                stack[row, col, tb] += 1
                photons_kept += 1
    
    elapsed = time.time() - t0
    
    if verbose:
        print(f"Filtering complete:")
        print(f"  Photons kept: {photons_kept:,}")
        print(f"  Photons discarded: {photons_discarded:,}")
        print(f"  Elapsed: {elapsed:.1f}s")
        print(f"  Output shape: {stack.shape}")
    
    return stack.astype(np.float32)


import time
import numpy as np
from .reader import PTUFile

def filter_photons_with_mask_optimized(ptu_path, mask, channel=None, binning=1, verbose=False):
    """
    Filter photons from a PTU file using a cell mask (optimized version).
    
    Keeps photons where mask[y, x] != 0 (non-zero / white).
    Discards photons where mask[y, x] == 0 (black).
    
    Args:
        ptu_path: Path to PTU file
        mask: 2D numpy array (Y, X) with 0 = discard, non-zero = keep
        channel: Detection channel (None = auto-detect)
        binning: Spatial binning factor (default: 1)
        verbose: Print progress messages
    
    Returns:
        stack: 3D array (Y, X, H) with filtered photon histograms
    """
    ptu = PTUFile(str(ptu_path), verbose=False)

    # Auto-detect channel
    if channel is None:
        if ptu.photon_channel is None:
            ptu.summed_decay(channel=None)
        channel = ptu.photon_channel

    t0 = time.time()

    # Load and decode records
    records = ptu._load_records()
    ch, dtime, _ = ptu._decode_picoharp_t3(records)

    # Photon and marker indices
    special = ch == 0xF
    ph_mask = (~special) & (ch == channel)
    ph_idx = np.where(ph_mask)[0]
    ph_dtime = dtime[ph_mask].astype(np.int32)

    marker_mask = special & (dtime != 0)
    marker_idx = np.where(marker_mask)[0]
    marker_dtime = dtime[marker_mask]

    # Line start/stop markers (assumes standard scanning pattern)
    line_start_abs = marker_idx[marker_dtime & 1 != 0]
    line_stop_abs = marker_idx[marker_dtime & 2 != 0]

    n_lines = min(len(line_start_abs), len(line_stop_abs))
    ny_out = ptu.n_y // binning
    nx_out = ptu.n_x // binning

    # Resize mask and create a boolean array for "keep" (True = keep)
    if mask.shape != (ny_out, nx_out):
        from scipy.ndimage import zoom
        zoom_factors = (ny_out / mask.shape[0], nx_out / mask.shape[1])
        # order=0 (nearest) preserves integer mask values
        mask_resized = zoom(mask, zoom_factors, order=0)
        mask_keep = mask_resized != 0
    else:
        mask_keep = mask != 0

    if verbose:
        print(f"Filtering photons with mask (channel={channel}, binning={binning})")
        print(f"Mask shape: {mask.shape} -> resized to {mask_keep.shape}")
        print(f"Mask: keep {np.sum(mask_keep):,} pixels, discard {np.sum(~mask_keep):,} pixels")

    # Output stack
    stack = np.zeros((ny_out, nx_out, ptu.n_bins), dtype=np.uint32)

    # Process each line
    for line_num in range(n_lines):
        ls = line_start_abs[line_num]
        le = line_stop_abs[line_num]
        if le <= ls:
            continue

        row = (line_num % ptu.n_y) // binning
        if row >= ny_out:
            continue

        # Photons in this line
        lo = np.searchsorted(ph_idx, ls, side="right")
        hi = np.searchsorted(ph_idx, le, side="left")
        if hi <= lo:
            continue

        ph_in = ph_idx[lo:hi]
        dt_in = ph_dtime[lo:hi]
        line_len = le - ls
        rel_pos = ph_in - ls
        px = np.clip((rel_pos * ptu.n_x) // line_len, 0, ptu.n_x - 1)
        px_bin = px // binning

        # Clip to valid column indices (avoid index error)
        px_bin = np.clip(px_bin, 0, nx_out - 1)

        # Vectorized mask check: keep only where mask_keep is True and dt_in < n_bins
        keep_mask = mask_keep[row, px_bin] & (dt_in < ptu.n_bins)
        kept_indices = np.where(keep_mask)[0]

        # Accumulate photons
        for i in kept_indices:
            stack[row, px_bin[i], dt_in[i]] += 1

    elapsed = time.time() - t0

    if verbose:
        total_photons = stack.sum()
        print(f"Filtering complete:")
        print(f"  Photons kept: {total_photons:,}")
        print(f"  Elapsed: {elapsed:.1f}s")
        print(f"  Output shape: {stack.shape}")

    return stack.astype(np.float32)

def signal_from_PTUFile(
    filename: str | PathLike[Any],
    /,
    *,
    dtype: DTypeLike | None = None,
    frame: int | None = None,
    channel: int | None = 0,
    dtime: int | None = 0,
    binning: int = 1,
    keepdims: bool = False,
):
    """Return TCSPC histogram and metadata from a PTU T3 mode file.

    Uses flimkit's PTUFile / PTUArray5D for decoding instead of ptufile.

    Parameters
    
    filename : str or Path
        Path to a PicoQuant PTU file.
    dtype : dtype_like, optional, default: uint16
        Unsigned integer type for the histogram array.
    frame : int, optional
        If < 0, integrate (sum) over the time/frame axis.
        If >= 0, select that single frame.
        If None, keep all frames.
    channel : int, optional, default: 0
        Detection channel index to return.
        If < 0, integrate (sum) over the channel axis.
        If None, keep all channels.
    dtime : int, optional, default: 0
        Number of histogram bins to keep.
        0  -> use all bins in one period (default).
        >0 -> keep the first *dtime* bins.
        <0 -> integrate (sum) over the histogram axis.
    binning : int, optional, default: 1
        Spatial binning factor applied when building the pixel stack.
    keepdims : bool, optional, default: False
        If True, reduced axes are kept as length-1 dimensions.

    Returns
    -
    xarray.DataArray
        TCSPC histogram with axes ``'TYXCH'``.

        - ``coords['H']``: delay-time bin centres in nanoseconds.
        - ``attrs['frequency']``: laser repetition frequency in MHz.
        - ``attrs['ptu_tags']``: raw tag dictionary from the PTU header.
    """
    from xarray import DataArray

    ptu = PTUFile(str(filename), verbose=False)
    is_image = ptu.n_x > 0 and ptu.n_y > 0

    if is_image:
        arr5d = PTUArray5D(ptu, binning=binning)
        # arr5d.array has shape (T, Y, X, C, H)  dtype uint32
        data = arr5d.array.copy()
        _active_channels = arr5d.active_channels  # hardware IDs in array-index order
    else:
        # Point-mode: build (1, 1, 1, C, H) from summed decays per channel
        records = ptu._load_records()
        ch_raw, dtime_raw, _ = ptu._decode_picoharp_t3(records)
        active_chs = np.unique(ch_raw[ch_raw != 0xF])
        hists = []
        for c in active_chs:
            hists.append(ptu.summed_decay(channel=int(c)))
        # stack to (C, H), then expand to (T=1, Y=1, X=1, C, H)
        data = np.stack(hists, axis=0)[np.newaxis, np.newaxis, np.newaxis, :, :]
        _active_channels = active_chs  # hardware IDs in array-index order

    #dtype 
    if dtype is None:
        dtype = np.uint16
    data = data.astype(dtype)

    n_bins_full = data.shape[-1]  # H axis length before any trimming

    #dtime selection (H axis = -1) 
    if dtime is not None:
        if dtime < 0:
            data = data.sum(axis=-1, keepdims=keepdims)
        elif dtime > 0:
            data = data[..., :dtime]
        # dtime == 0 → keep all bins (no-op)

    #channel selection (C axis = 3) 
    if channel is not None:
        if channel < 0:
            data = data.sum(axis=3, keepdims=keepdims)
        else:
            # _active_channels[i] is the hardware ID stored at array index i.
            # Translate the hardware channel ID to its 0-based array index.
            ch_positions = np.where(_active_channels == channel)[0]
            if len(ch_positions) == 0:
                raise ValueError(
                    f"Channel {channel} not found in PTU file. "
                    f"Available hardware channels: {_active_channels.tolist()}"
                )
            channel_idx = int(ch_positions[0])
            if keepdims:
                data = data[:, :, :, channel_idx : channel_idx + 1]
            else:
                data = data[:, :, :, channel_idx]
            # after removing C, H was axis 4 → now axis 3

    #frame selection (T axis = 0) 
    if frame is not None:
        if frame < 0:
            data = data.sum(axis=0, keepdims=keepdims)
        else:
            if keepdims:
                data = data[frame : frame + 1]
            else:
                data = data[frame]

    #Build dimension labels 
    all_dims = ['T', 'Y', 'X', 'C', 'H']
    # Figure out which dims survived
    removed = []
    if dtime is not None and dtime < 0 and not keepdims:
        removed.append('H')
    if channel is not None and channel >= 0 and not keepdims:
        removed.append('C')
    if frame is not None and not keepdims:
        removed.append('T')
    dims = [d for d in all_dims if d not in removed]

    #Build coordinates 
    coords = {}
    h_bins = dtime if (dtime is not None and dtime > 0) else n_bins_full
    time_ns = ptu.time_ns[:h_bins]
    if 'H' in dims:
        coords['H'] = ('H', time_ns)

    #Frequency in MHz 
    frequency_mhz = ptu.sync_rate * 1e-6

    da = DataArray(
        data,
        dims=dims,
        coords=coords,
        attrs={
            'frequency': frequency_mhz,
            'ptu_tags': ptu.tags,
        },
    )
    return da