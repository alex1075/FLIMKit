import time
import numpy as np
from .reader import PTUFile

def filter_photons_with_mask(ptu_path, mask, channel=None, binning=1, verbose=False):
    """
    Filter photons from a PTU file using a cell mask.
    
    Keeps photons where mask[y, x] == 0 (black).
    Discards photons where mask[y, x] != 0 (white/non-zero).
    
    Args:
        ptu_path: Path to PTU file
        mask: 2D numpy array (Y, X) with 0 = keep, non-zero = discard
        channel: Detection channel (None = auto-detect)
        binning: Spatial binning factor (default: 1)
        verbose: Print progress messages
    
    Returns:
        stack: 3D array (Y, X, H) with filtered photon histograms
        
    Note:
        This is a masked version of PTUFile.pixel_stack() that only
        accumulates photons where the mask is black (0).
    """
    ptu = PTUFile(str(ptu_path), verbose=False)
    
    # Auto-detect channel if needed
    if channel is None:
        if ptu.photon_channel is None:
            ptu.summed_decay(channel=None)
        channel = ptu.photon_channel
    
    if verbose:
        print(f"Filtering photons with mask (channel={channel}, binning={binning})")
        print(f"Mask shape: {mask.shape}")
        print(f"Mask: keep {np.sum(mask == 0):,} pixels, discard {np.sum(mask != 0):,} pixels")
    
    t0 = time.time()
    
    # Load and decode records
    records = ptu._load_records()
    ch, dtime, _ = ptu._decode_picoharp_t3(records)
    
    # Identify photons and markers
    special = ch == 0xF
    ph_mask = (~special) & (ch == channel)
    ph_idx = np.where(ph_mask)[0]
    ph_dtime = dtime[ph_mask].astype(np.int32)
    
    marker_mask = special & (dtime != 0)
    marker_idx = np.where(marker_mask)[0]
    marker_dtime = dtime[marker_mask]
    
    # Line start/stop markers
    line_start_abs = marker_idx[marker_dtime & 1 != 0]
    line_stop_abs = marker_idx[marker_dtime & 2 != 0]
    
    n_lines = min(len(line_start_abs), len(line_stop_abs))
    ny_out = ptu.n_y // binning
    nx_out = ptu.n_x // binning
    
    # Resize mask if needed
    if mask.shape != (ny_out, nx_out):
        from scipy.ndimage import zoom
        zoom_factors = (ny_out / mask.shape[0], nx_out / mask.shape[1])
        mask_resized = zoom(mask, zoom_factors, order=0) > 0  # Convert to boolean
        if verbose:
            print(f"Resized mask from {mask.shape} to {mask_resized.shape}")
    else:
        mask_resized = mask > 0  # Convert to boolean (True = discard)
    
    # Allocate output
    stack = np.zeros((ny_out, nx_out, ptu.n_bins), dtype=np.uint32)
    
    # Process each line
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
        
        # Find photons in this line
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
        
        # Filter by mask and accumulate
        for i in range(len(dt_in)):
            col = px_bin[i]
            if col >= nx_out:
                continue
            
            # Check mask: if True (non-zero), discard photon
            if mask_resized[row, col]:
                photons_discarded += 1
                continue
            
            # Mask is False (zero), keep photon
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


def filter_photons_with_mask_optimized(ptu_path, mask, channel=None, binning=1, verbose=False):
    """
    Optimized version that pre-computes which pixels to keep.
    
    Same interface as filter_photons_with_mask but faster for large masks.
    """
    ptu = PTUFile(str(ptu_path), verbose=False)
    
    if channel is None:
        if ptu.photon_channel is None:
            ptu.summed_decay(channel=None)
        channel = ptu.photon_channel
    
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
    
    # Resize mask
    if mask.shape != (ny_out, nx_out):
        from scipy.ndimage import zoom
        zoom_factors = (ny_out / mask.shape[0], nx_out / mask.shape[1])
        mask_bool = zoom(mask, zoom_factors, order=0) == 0  # True = KEEP
    else:
        mask_bool = mask == 0  # True = KEEP
    
    stack = np.zeros((ny_out, nx_out, ptu.n_bins), dtype=np.uint32)
    
    # Process lines
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
        
        # Vectorized mask check
        keep_mask = mask_bool[row, px_bin] & (dt_in < ptu.n_bins)
        kept_indices = np.where(keep_mask)[0]
        
        for i in kept_indices:
            stack[row, px_bin[i], dt_in[i]] += 1
    
    elapsed = time.time() - t0
    
    if verbose:
        total_photons = stack.sum()
        print(f"Filtered {total_photons:,} photons in {elapsed:.1f}s")
    
    return stack.astype(np.float32)