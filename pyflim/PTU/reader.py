import struct
import time
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any, Optional


_TAG_TYPES = {
    0xFFFF0008: ("Empty8",      None),
    0x00000008: ("Bool8",       "q"),
    0x10000008: ("Int8",        "q"),
    0x11000008: ("BitSet64",    "Q"),
    0x12000008: ("Color8",      "Q"),
    0x20000008: ("Float8",      "d"),
    0x21000008: ("TDateTime",   "d"),
    0x2001FFFF: ("Float8Array", "arr"),
    0x4001FFFF: ("AnsiString",  "str"),
    0x4002FFFF: ("WideString",  "str"),
    0xFFFFFFFF: ("BinaryBlob",  "blob"),
}


def _read_ptu_header(path: str) -> tuple[dict, int]:
    tags: dict = {}
    data_offset: int = 0

    with open(path, "rb") as fh:
        magic = fh.read(8)
        if b"PTU" not in magic and b"PQTTTR" not in magic:
            raise ValueError(f"Not a PTU/PQTTTR file: magic={magic!r}")
        fh.read(8)

        buf = bytearray()
        while True:
            chunk = fh.read(65536)
            if not chunk:
                break
            buf.extend(chunk)
            if b"Header_End" in buf:
                break
        buf.extend(fh.read())

    pos = 0
    while pos + 48 <= len(buf):
        ident  = buf[pos:pos+32].decode("ascii", errors="replace").rstrip("\x00")
        tagidx = struct.unpack_from("<i", buf, pos+32)[0]
        tagtyp = struct.unpack_from("<I", buf, pos+36)[0]
        tagval = buf[pos+40:pos+48]
        pos   += 48

        if ident == "Header_End":
            data_offset = 16 + pos
            break

        info = _TAG_TYPES.get(tagtyp)
        if info is None:
            continue
        name, fmt = info

        if fmt in ("arr", "str", "blob"):
            blen = struct.unpack("<q", tagval)[0]
            blob = bytes(buf[pos:pos+blen])
            pos += blen
            val: object = blob.decode("utf-8", errors="replace").rstrip("\x00") \
                          if fmt == "str" else blob
        elif fmt:
            val = struct.unpack(f"<{fmt}", tagval)[0]
            if tagtyp == 0x00000008:
                val = bool(val)
        else:
            val = None

        key = f"{ident}[{tagidx}]" if tagidx >= 0 else ident
        tags[key] = val

    return tags, data_offset


class PTUFile:
    """
    PTU file reader - DO NOT MODIFY - used by flim9.py
    
    This class reads PicoQuant PTU files and provides methods to extract
    summed decays and pixel stacks.
    """
    def __init__(self, path: str, verbose: bool = True):
        self.path    = str(path)
        self.verbose = verbose
        self.tags, self._data_offset = _read_ptu_header(path)
        self._parse_meta()

    def _parse_meta(self):
        t = self.tags
        self.tcspc_res  = float(t.get("MeasDesc_Resolution", 9.697e-11))
        global_res      = float(t.get("MeasDesc_GlobalResolution",
                                       1 / t.get("TTResult_SyncRate", 20e6)))
        self.sync_rate  = 1.0 / global_res
        self.period_ns  = global_res * 1e9
        self.n_bins     = int(round(self.period_ns / (self.tcspc_res * 1e9)))
        self.n_x        = int(t.get("ImgHdr_PixX", t.get("$ReqHdr_PixelNumber_X", 256)))
        self.n_y        = int(t.get("ImgHdr_PixY", t.get("$ReqHdr_PixelNumber_Y", 256)))
        self.rec_type   = int(t.get("TTResultFormat_TTTRRecType", 0x00010303))
        self.n_records  = int(t.get("TTResult_NumberOfRecords", 0))
        self.time_ns    = (np.arange(self.n_bins) + 0.5) * self.tcspc_res * 1e9
        self.photon_channel = None

        if self.verbose:
            print(f"  HW type  : {t.get('HW_Type','?')}")
            print(f"  RecType  : 0x{self.rec_type:08X}  "
                  f"({'PicoHarpT3' if self.rec_type==0x00010303 else 'other'})")
            print(f"  TCSPC    : {self.n_bins} bins × {self.tcspc_res*1e12:.2f} ps")
            print(f"  Laser    : {self.sync_rate/1e6:.3f} MHz  ({self.period_ns:.3f} ns)")
            print(f"  Image    : {self.n_x} × {self.n_y} px")
            print(f"  Records  : {self.n_records:,}")

    def _load_records(self) -> np.ndarray:
        size = Path(self.path).stat().st_size - self._data_offset
        n    = size // 4
        with open(self.path, "rb") as fh:
            fh.seek(self._data_offset)
            return np.frombuffer(fh.read(n * 4), dtype="<u4")

    def _decode_picoharp_t3(self, records: np.ndarray):
        ch    = (records >> 28) & 0xF
        dtime = (records >> 16) & 0xFFF
        nsync = records & 0xFFFF
        return ch, dtime, nsync

    def summed_decay(self, channel: int | None = None) -> np.ndarray:
        records = self._load_records()
        ch, dtime, _ = self._decode_picoharp_t3(records)
        special = ch == 0xF
        photon  = ~special

        if channel is None:
            ch_counts = np.bincount(ch[photon], minlength=16)
            channel   = int(np.argmax(ch_counts))
            self.photon_channel = channel
            if self.verbose:
                print(f"  Auto-detected photon channel: {channel} "
                      f"({ch_counts[channel]:,} photons)")

        ph_mask = photon & (ch == channel)
        dt_ph   = dtime[ph_mask].astype(np.int32)
        decay   = np.bincount(dt_ph, minlength=self.n_bins).astype(float)
        return decay[:self.n_bins]

    def pixel_stack(self, channel: int | None = None,
                    binning: int = 1) -> np.ndarray:
        if self.photon_channel is None:
            self.summed_decay(channel=channel)
        ch_use = channel if channel is not None else self.photon_channel

        if self.verbose:
            print(f"  Building pixel stack (channel={ch_use}, binning={binning}) …")
        t0 = time.time()

        records  = self._load_records()
        ch, dtime, _ = self._decode_picoharp_t3(records)

        special  = ch == 0xF
        ph_mask  = (~special) & (ch == ch_use)
        ph_idx   = np.where(ph_mask)[0]
        ph_dtime = dtime[ph_mask].astype(np.int32)

        marker_mask  = special & (dtime != 0)
        marker_idx   = np.where(marker_mask)[0]
        marker_dtime = dtime[marker_mask]

        line_start_abs = marker_idx[marker_dtime & 1 != 0]
        line_stop_abs  = marker_idx[marker_dtime & 2 != 0]

        n_lines = min(len(line_start_abs), len(line_stop_abs))
        ny_out  = self.n_y  // binning
        nx_out  = self.n_x  // binning
        stack   = np.zeros((ny_out, nx_out, self.n_bins), dtype=np.uint32)

        for line_num in range(n_lines):
            ls = line_start_abs[line_num]
            le = line_stop_abs[line_num]
            if le <= ls:
                continue
            row = (line_num % self.n_y) // binning

            lo = np.searchsorted(ph_idx, ls, side="right")
            hi = np.searchsorted(ph_idx, le, side="left")
            if hi <= lo:
                continue

            ph_in    = ph_idx[lo:hi]
            dt_in    = ph_dtime[lo:hi]
            line_len = le - ls
            rel_pos  = ph_in - ls
            px       = np.clip((rel_pos * self.n_x) // line_len, 0, self.n_x - 1)
            px_bin   = px // binning

            for i in range(len(dt_in)):
                tb = dt_in[i]
                if tb < self.n_bins:
                    stack[row, px_bin[i], tb] += 1

        elapsed = time.time() - t0
        total   = stack.sum()
        if self.verbose:
            print(f"  Stack built: {ny_out}×{nx_out}×{self.n_bins}  "
                  f"({total:,} photons, {elapsed:.1f}s)")
        return stack.astype(float)


class PTUArray5D:
    """
    5D array builder for multi-frame, multi-channel PTU files.
    
    Produces shape (T, Y, X, C, H) where:
        T = frames/time points
        Y = rows
        X = columns
        C = detection channels
        H = histogram bins (TCSPC)
    
    Note: Currently assumes single-frame data. Frame detection needs to be
    implemented based on your specific PTU file structure.
    """
    def __init__(self, ptu_file: PTUFile, binning: int = 1):
        self.ptu = ptu_file
        self.binning = binning
        self.n_y_out = self.ptu.n_y // binning
        self.n_x_out = self.ptu.n_x // binning
        self._load_and_process()

    def _find_frames(self, records: np.ndarray, ch: np.ndarray, 
                     dtime: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Detect frame boundaries from markers.
        
        IMPORTANT: This is a placeholder implementation that treats the entire
        acquisition as a single frame. For actual multi-frame data, you need to:
        
        1. Identify frame start/stop markers in your PTU file
        2. Look for specific marker patterns (e.g., channel 0xF with specific dtime values)
        3. PTU files may use different marker schemes:
           - Frame markers (common in Z-stacks or time series)
           - Special event records with specific marker values
        
        Current implementation: Single frame containing all data
        """
        frame_starts = np.array([0], dtype=np.int64)
        frame_ends = np.array([len(records)], dtype=np.int64)
        
        # TODO: Implement actual frame detection for multi-frame acquisitions
        
        return frame_starts, frame_ends

    def _load_and_process(self):
        records = self.ptu._load_records()
        ch, dtime, nsync = self.ptu._decode_picoharp_t3(records)

        # 1. Identify frames
        frame_starts, frame_ends = self._find_frames(records, ch, dtime)
        self.n_frames = len(frame_starts)

        # 2. Identify active channels (excluding marker 0xF)
        active_channels = np.unique(ch[ch != 0xF])
        self.n_channels = len(active_channels)
        self.active_channels = active_channels

        # 3. Allocate 5D array
        ny = self.n_y_out
        nx = self.n_x_out
        nb = self.ptu.n_bins
        self.array = np.zeros((self.n_frames, ny, nx, self.n_channels, nb), 
                              dtype=np.uint32)

        # 4. Fill array
        for frame_idx, (fs, fe) in enumerate(zip(frame_starts, frame_ends)):
            self._fill_frame(frame_idx, fs, fe, ch, dtime, active_channels)

    def _fill_frame(self, frame_idx, start, end, ch, dtime, active_channels):
        frame_mask = np.zeros(len(ch), dtype=bool)
        frame_mask[start:end] = True
        
        marker_mask = frame_mask & (ch == 0xF) & (dtime != 0)
        marker_idx = np.where(marker_mask)[0]
        marker_dtime = dtime[marker_mask]
        
        line_start_mask = (marker_dtime & 1) != 0
        line_start_abs = marker_idx[line_start_mask]
        
        line_stop_mask = (marker_dtime & 2) != 0
        line_stop_abs = marker_idx[line_stop_mask]
        
        n_lines = min(len(line_start_abs), len(line_stop_abs))
        
        photon_mask = frame_mask & (ch != 0xF) & np.isin(ch, active_channels)
        photon_idx = np.where(photon_mask)[0]
        photon_ch = ch[photon_idx]
        photon_dtime = dtime[photon_idx].astype(np.int32)
        
        ch_to_idx = {ch_val: idx for idx, ch_val in enumerate(active_channels)}
        
        ny_out = self.n_y_out
        nx_out = self.n_x_out
        binning = self.binning
        n_bins = self.ptu.n_bins
        
        for line_num in range(n_lines):
            ls = line_start_abs[line_num]
            le = line_stop_abs[line_num]
            if le <= ls:
                continue
            
            global_row = line_num % self.ptu.n_y
            row_bin = global_row // binning
            if row_bin >= ny_out:
                continue
            
            lo = np.searchsorted(photon_idx, ls, side='right')
            hi = np.searchsorted(photon_idx, le, side='left')
            if hi <= lo:
                continue
            
            line_ph_idx = photon_idx[lo:hi]
            line_ph_ch = photon_ch[lo:hi]
            line_ph_dt = photon_dtime[lo:hi]
            
            line_len = le - ls
            rel_pos = line_ph_idx - ls
            col_full = (rel_pos * self.ptu.n_x) // line_len
            col_bin = col_full // binning
            col_bin = np.clip(col_bin, 0, nx_out - 1)
            
            for i in range(len(line_ph_idx)):
                tb = line_ph_dt[i]
                if tb >= n_bins:
                    continue
                ch_val = line_ph_ch[i]
                if ch_val not in ch_to_idx:
                    continue
                ch_idx = ch_to_idx[ch_val]
                self.array[frame_idx, row_bin, col_bin[i], ch_idx, tb] += 1

    def __getitem__(self, key):
        return self.array[key]

    @property
    def shape(self):
        return self.array.shape

    @property
    def dims(self):
        return ('T', 'Y', 'X', 'C', 'H')


# ══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS - Compatible with custom PTUFile
# ══════════════════════════════════════════════════════════════════════════════

def read_ptu(path: str, binning: int = 1, channel: Optional[int] = None,
             verbose: bool = False) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Read a PTU file and return the FLIM data array and metadata.
    
    This is the custom PTUFile equivalent of ptufile.PtuFile.
    
    Args:
        path: Path to PTU file
        binning: Spatial binning factor (default: 1)
        channel: Detection channel to use (None = auto-detect)
        verbose: Print file information
        
    Returns:
        data: numpy array with shape (Y, X, H) - pixel stack with histogram per pixel
        metadata: dict with file metadata including frequency, resolution, etc.
    """
    ptu = PTUFile(path, verbose=verbose)
    
    # Get pixel stack (Y, X, H)
    data = ptu.pixel_stack(channel=channel, binning=binning)
    
    metadata = {
        'frequency': ptu.sync_rate,  # Laser repetition rate in Hz
        'tcspc_resolution': ptu.tcspc_res,  # TCSPC resolution in seconds
        'shape': data.shape,
        'dims': ('Y', 'X', 'H'),
        'tags': ptu.tags,
        'x_pixel_size': ptu.tags.get('ImgHdr_PixRes', 0),
        'y_pixel_size': ptu.tags.get('ImgHdr_PixRes', 0),
        'n_bins': ptu.n_bins,
        'time_ns': ptu.time_ns,
        'photon_channel': ptu.photon_channel,
    }
    
    return data, metadata


def read_ptu_5d(path: str, binning: int = 1, verbose: bool = False) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Read a PTU file and return 5D FLIM data array and metadata.
    
    Parameters
    ----------
    path : str
        Path to PTU file
    binning : int, optional
        Spatial binning factor (default: 1)
    verbose : bool, optional
        Print file information (default: False)
        
    Returns
    -------
    data : np.ndarray
        5D array with shape (T, Y, X, C, H)
    metadata : dict
        File metadata
    """
    ptu = PTUFile(path, verbose=verbose)
    builder = PTUArray5D(ptu, binning=binning)
    data = builder.array
    
    metadata = {
        'frequency': ptu.sync_rate,
        'tcspc_resolution': ptu.tcspc_res,
        'shape': data.shape,
        'dims': builder.dims,
        'tags': ptu.tags,
        'x_pixel_size': ptu.tags.get('ImgHdr_PixRes', 0),
        'y_pixel_size': ptu.tags.get('ImgHdr_PixRes', 0),
        'n_channels': builder.n_channels,
        'channel_list': list(builder.active_channels),
    }
    
    if verbose:
        print(f"\n5D Array Summary:")
        print(f"  Shape: {data.shape}")
        print(f"  Dims:  {builder.dims}")
        print(f"  Channels: {list(builder.active_channels)}")
        print(f"  Total photons: {data.sum():,.0f}")
    
    return data, metadata


def get_intensity_image(path: str, binning: int = 1, channel: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Get a 2D intensity image from a PTU file by summing histogram bins.
    
    Args:
        path: Path to PTU file
        binning: Spatial binning factor (default: 1)
        channel: Detection channel to use (None = auto-detect)
    
    Returns:
        img: 2D numpy array (Y, X) with photon counts per pixel
        metadata: dict with file metadata
    """
    data, metadata = read_ptu(path, binning=binning, channel=channel, verbose=False)
    
    # Sum over histogram dimension (axis 2)
    img = data.sum(axis=2)
    
    return img, metadata


def get_flim_data(path: str, binning: int = 1, channel: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Get the full FLIM data cube from a PTU file.
    
    Args:
        path: Path to PTU file
        binning: Spatial binning factor (default: 1)
        channel: Detection channel to use (None = auto-detect)
    
    Returns:
        data: 3D numpy array (Y, X, H) with histogram per pixel
        metadata: dict with file metadata
    """
    # This is identical to read_ptu, just provided for API compatibility
    return read_ptu(path, binning=binning, channel=channel, verbose=False)


def normalise_flim(flim: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """
    Ensure a FLIM cube has shape (Y, X, T).
    
    Parameters:
        flim : np.ndarray or None
            Raw or fitted FLIM data.
    
    Returns:
        np.ndarray with shape (Y, X, T) or None if spatial info is missing.
    """
    if flim is None:
        return None

    # If the array has 5 dims (T, Y, X, C, H), take first frame and first channel
    if flim.ndim == 5:
        return flim[0, :, :, 0, :]
    
    # If the array has 4 dims, assume (frame, Y, X, H) or (Y, X, C, H)
    if flim.ndim == 4:
        # Check if first dimension is 1 (likely frame)
        if flim.shape[0] == 1:
            return flim[0]
        # Otherwise assume (Y, X, C, H) and take first channel
        else:
            return flim[:, :, 0, :]

    # If already (Y, X, H), return as-is
    if flim.ndim == 3:
        return flim

    # If 2D or 1D (no spatial info), cannot normalise
    return None


def create_time_axis(n_bins: int, tcspc_resolution: float) -> np.ndarray:
    """
    Create time axis in nanoseconds for FLIM fitting.
    
    Args:
        n_bins: Number of time bins
        tcspc_resolution: Time per bin in seconds
    
    Returns:
        time_axis_ns: Array of time values in nanoseconds
    """
    return np.arange(n_bins) * tcspc_resolution * 1e9


def decode_ptu(ptu_file: str) -> Tuple[Dict[str, Any], np.ndarray, Dict[str, Any]]:
    """
    Legacy function for backwards compatibility.
    
    Args:
        ptu_file: Path to PTU file
    
    Returns:
        tuple: (tags, intensity_image, metadata)
    """
    if not Path(ptu_file).exists():
        raise FileNotFoundError(f"File not found: {ptu_file}")
    
    img, metadata = get_intensity_image(ptu_file)
    
    return metadata['tags'], img, metadata


def decode_t3_record(ptu_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Decode T3 records from a PTU file.

    This function reads all time-tagged records from a PicoHarp T3 PTU file and
    returns the raw components: channel, dtime, and nsync.

    Parameters
    ----------
    ptu_path : str
        Path to the PTU file.

    Returns
    -------
    channels : np.ndarray (uint8)
        Channel numbers (0–15). Special marker events have channel = 15.
    dtimes : np.ndarray (uint16)
        Dtime values (0–4095) representing the time between sync and photon.
    nsyncs : np.ndarray (uint32)
        Sync counter values (0–65535) that overflow every 65536 sync pulses.

    Raises
    ------
    ValueError
        If the file does not contain PicoHarp T3 records (record type 0x00010303).
    FileNotFoundError
        If the file does not exist.
    """
    if not Path(ptu_path).exists():
        raise FileNotFoundError(f"PTU file not found: {ptu_path}")

    # Use PTUFile to read header and records (verbose=False to suppress output)
    ptu = PTUFile(ptu_path, verbose=False)

    # Check that the file contains T3 records (PicoHarp format)
    if ptu.rec_type != 0x00010303:
        raise ValueError(
            f"Unsupported record type: 0x{ptu.rec_type:08X}. "
            "Expected PicoHarp T3 (0x00010303)."
        )

    # Load all raw 4‑byte records
    records = ptu._load_records()

    # Decode into channel, dtime, nsync
    ch, dtime, nsync = ptu._decode_picoharp_t3(records)

    return ch, dtime, nsync


def decode_ptu_raw_cube(ptu_path: str, n_bins: Optional[int] = None,
                        tile_shape: Optional[Tuple[int, int]] = None,
                        channel: Optional[int] = None) -> np.ndarray:
    """
    Return a raw FLIM cube (Y, X, H) from a PTU file.

    Parameters
    ----------
    ptu_path : str
        Path to PTU file.
    n_bins : int, optional
        Number of time bins to use (if None, use the full range from file).
    tile_shape : tuple, optional
        Ignored in this basic implementation.
    channel : int, optional
        Detection channel to extract (None = auto-detect).

    Returns
    -------
    cube : np.ndarray
        3D array of shape (Y, X, H) with photon counts (float).
    """
    # Read the file with binning=1 to preserve full resolution
    cube, metadata = read_ptu(ptu_path, binning=1, channel=channel, verbose=False)

    # Optionally truncate to n_bins
    if n_bins is not None:
        cube = cube[..., :n_bins]

    return cube

def get_flim_cube(path):
    return read_ptu(path)[0]

def get_raw_flim_cube(path, n_bins=None, tile_shape=None, channel=None):
    return decode_ptu_raw_cube(path, n_bins, tile_shape, channel)

def get_flim_histogram(ptu_path: str, rotate_cw: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load FLIM histogram data from a PTU file with proper shape handling.
    
    Args:
        ptu_path: Path to PTU file
        rotate_cw: If True, rotate tile 90° clockwise (for Leica data)
    
    Returns:
        hist: numpy array with shape (Y, X, H)
        metadata: dict with tcspc_resolution, n_time_bins, tile_shape, etc.
    """
    # Get histogram data using read_ptu
    hist, metadata = read_ptu(ptu_path, binning=1, channel=None, verbose=False)
    
    # Rotate 90 degrees clockwise for Leica data if requested
    if rotate_cw:
        hist = np.rot90(hist, k=-1, axes=(0, 1))
    
    # Add additional metadata fields expected by this function
    metadata['n_time_bins'] = hist.shape[2]
    metadata['tile_shape'] = (hist.shape[0], hist.shape[1])
    
    return hist, metadata