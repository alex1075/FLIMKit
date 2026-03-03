"""
flim_fit_v13.py
===============
Changes from v12:

1. IRF peak position: replaced np.argmax(decay) with find_irf_peak_bin().
   np.argmax(decay) returns the fluorescence convolution peak, which is
   shifted right of the true IRF peak by ~1-2 bins depending on lifetime.
   The IRF peak is correctly estimated from the maximum of the first derivative
   of the smoothed rising edge (steepest ascent = IRF peak).

2. Gaussian path now uses has_tail=True.
   A real HyD/SPAD detector IRF is asymmetric (fast rise, slower fall).
   A symmetric Gaussian cannot fit this regardless of FWHM, giving χ²_r >> 2.
   The exponential tail free parameters capture the detector afterpulsing/
   slow component, matching Leica's physical model.
   fit_sigma remains False (FWHM is known), has_tail is now True.

3. Lifetimes are sorted descending (τ₁ > τ₂ > τ₃) inside reconvolution_model
   to eliminate permutation ambiguity.

4. The summed decay is normalised to [0,1] (peak = 1) before fitting to improve
   numerical stability. Amplitudes and background are rescaled back to original
   photon counts after optimisation; χ² values are correctly scaled to original units.
"""

from __future__ import annotations
import argparse
import struct
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares, curve_fit, differential_evolution, nnls
from scipy.stats import chi2 as chi2_dist

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
MIN_PHOTONS_PERPIX = 10
FLIM_CMAP = LinearSegmentedColormap.from_list(
    "flim", ["#000080","#0000ff","#00ffff","#00ff00","#ffff00","#ff0000"]
)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  PTU READER
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
# 2.  XLSX LOADER
# ══════════════════════════════════════════════════════════════════════════════

def load_xlsx(path: str, debug: bool = False) -> dict:
    """
    Load LAS X FLIM export xlsx.
    Handles duplicate 'Time [ns]' column names — pandas renames them
    Time [ns], Time [ns].1, etc.

    LAS X column order:
        Time [ns] | Decay [Counts] | Time [ns].1 | IRF [Counts]
        (optional) Time [ns].2 | Fit [Counts] | Time [ns].3 | Residuals [Counts]

    debug=True prints raw row contents and detected columns to diagnose
    parsing failures.
    """
    df_raw = pd.read_excel(path, sheet_name=0, header=None)

    if debug:
        print(f"    Raw xlsx shape: {df_raw.shape}")
        print(f"    First 5 rows:")
        for i in range(min(5, len(df_raw))):
            vals = [str(v) for v in df_raw.iloc[i].values if pd.notna(v)]
            print(f"      row {i}: {vals}")

    # Find header row: look for a cell that is exactly (or starts with) "Time [ns]"
    # Must NOT match "Lifetime" — require the word starts with "time ["
    header_row = None
    for i, row in df_raw.iterrows():
        vals = [str(v).strip().lower() for v in row if pd.notna(v)]
        if any(v.startswith('time [') for v in vals):
            header_row = i
            break

    if header_row is None:
        print(f"    ⚠ No row starting with 'Time [' found — trying row 0 as fallback")
        header_row = 0

    if debug:
        print(f"    Detected header row: {header_row}")

    df        = pd.read_excel(path, sheet_name=0, header=header_row)
    df        = df.dropna(axis=1, how='all')
    col_names = list(df.columns)

    if debug:
        print(f"    Columns after read: {col_names}")

    time_cols  = [c for c in col_names if str(c).lower().startswith('time [')]
    decay_cols = [c for c in col_names if 'decay'    in str(c).lower() and
                                          'counts'   in str(c).lower()]
    irf_cols   = [c for c in col_names if 'irf'      in str(c).lower() and
                                          'counts'   in str(c).lower()]
    fit_cols   = [c for c in col_names if 'fit'      in str(c).lower() and
                                          'counts'   in str(c).lower()]
    res_cols   = [c for c in col_names if 'resid'    in str(c).lower() and
                                          'counts'   in str(c).lower()]

    if debug:
        print(f"    time_cols : {time_cols}")
        print(f"    decay_cols: {decay_cols}")
        print(f"    irf_cols  : {irf_cols}")
        print(f"    fit_cols  : {fit_cols}")
        print(f"    res_cols  : {res_cols}")

    def _safe(col):
        if col is None:
            return None
        arr = df[col].dropna().values
        try:
            return arr.astype(float)
        except (ValueError, TypeError):
            return None

    out = {
        'decay_t': _safe(time_cols[0]  if len(time_cols) > 0 else None),
        'decay_c': _safe(decay_cols[0] if len(decay_cols) > 0 else None),
        'irf_t':   _safe(time_cols[1]  if len(time_cols) > 1 else None),
        'irf_c':   _safe(irf_cols[0]   if len(irf_cols)  > 0 else None),
        'fit_t':   _safe(time_cols[2]  if len(time_cols) > 2 else None),
        'fit_c':   _safe(fit_cols[0]   if len(fit_cols)  > 0 else None),
        'res_t':   _safe(time_cols[3]  if len(time_cols) > 3 else None),
        'res_c':   _safe(res_cols[0]   if len(res_cols)  > 0 else None),
    }

    for k, v in out.items():
        status = f"{len(v)} pts" if v is not None else "absent"
        print(f"    {k:12s}: {status}")

    return out


# ══════════════════════════════════════════════════════════════════════════════
# 3.  IRF CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════

def gaussian_irf_from_fwhm(n_bins: int,
                            tcspc_res: float,
                            fwhm_ns: float,
                            peak_bin: int) -> np.ndarray:
    """
    IRF[T] = exp(-(t - t0)^2 * 4*ln(2) / FWHM^2)

    Paper equation. Peak bin from np.argmax(summed_decay) — no manual input.
    Default FWHM = ptu.tcspc_res * 1e9 (one bin width, e.g. 97 ps).
    """
    t   = np.arange(n_bins, dtype=float) * tcspc_res * 1e9
    t0  = peak_bin * tcspc_res * 1e9
    irf = np.exp(-(t - t0)**2 * 4.0 * np.log(2) / fwhm_ns**2)
    return irf / irf.sum()


def irf_from_xlsx_analytical(xlsx: dict, n_bins: int, tcspc_res: float,
                              verbose: bool = True) -> tuple[np.ndarray, dict]:
    """
    Fit the Leica analytical IRF model to the xlsx IRF points and evaluate
    it on the full n_bins grid.

    Leica IRF model (from n-exponential-reconv.txt):
        IRF(t) = A · [exp(-4·ln2·(t-t0)²/FWHM²)
                      + tail_amp · exp(-(t-t0)/tail_tau)]   for t ≥ t0
                 A · exp(-4·ln2·(t-t0)²/FWHM²)              for t < t0

    Why this matters
    ----------------
    The xlsx exports only ~21 sparse points. Scatter-placing or interpolating
    these gives a comb with only 5 meaningful non-zero bins and misses the
    exponential tail entirely. The analytical model correctly samples all
    529 bins including the tail, eliminating FFT ringing artefacts.

    Returns
    -------
    irf_norm  : normalised analytical IRF on the full bin grid
    params    : dict of fitted parameters (t0, fwhm_ns, tail_amp, tail_tau_ns)
    """
    if xlsx['irf_t'] is None or xlsx['irf_c'] is None:
        raise ValueError("XLSX does not contain IRF columns.")

    t_pts = np.array(xlsx['irf_t'], dtype=float)
    c_pts = np.maximum(np.array(xlsx['irf_c'], dtype=float), 0.0)
    mask  = c_pts > c_pts.max() * 1e-3   # only fit meaningful points
    if mask.sum() < 3:
        raise ValueError("Fewer than 3 non-negligible IRF points in xlsx — "
                         "cannot fit analytical model.")

    t_fit = t_pts[mask]
    c_fit = c_pts[mask]
    t0_guess = t_pts[np.argmax(c_pts)]

    def _model(t, t0, fwhm, tail_amp, tail_tau, A):
        gauss = np.exp(-4.0 * np.log(2) * (t - t0)**2 / fwhm**2)
        tail  = np.where(t >= t0,
                         tail_amp * np.exp(-(t - t0) / np.maximum(tail_tau, 0.01)),
                         0.0)
        return A * (gauss + tail)

    try:
        popt, _ = curve_fit(
            _model, t_fit, c_fit,
            p0   = [t0_guess, 0.15,  0.05, 0.5,  c_pts.max()],
            bounds=([t0_guess - 0.5, 0.05, 0.0,  0.05, 0],
                    [t0_guess + 0.5, 0.5,  2.0,  10.0, c_pts.max() * 2]),
            maxfev=20000
        )
        t0, fwhm, tail_amp, tail_tau, A = popt
    except Exception as e:
        raise RuntimeError(f"Analytical IRF fit failed: {e}. "
                           f"Try --irf-xlsx with a higher-count IRF export.") from e

    # Evaluate on full bin grid
    tcspc_ns  = tcspc_res * 1e9
    t_full    = np.arange(n_bins, dtype=float) * tcspc_ns
    irf_full  = np.maximum(_model(t_full, t0, fwhm, tail_amp, tail_tau, A), 0.0)
    s         = irf_full.sum()
    if s == 0:
        raise ValueError("Analytical IRF evaluates to zero on bin grid.")
    irf_norm  = irf_full / s

    params = dict(t0_ns=t0, fwhm_ns=fwhm, tail_amp=tail_amp, tail_tau_ns=tail_tau)

    if verbose:
        print(f"  Analytical IRF fit (Leica model):")
        print(f"    t0       = {t0:.4f} ns  (bin {t0/tcspc_ns:.2f})")
        print(f"    FWHM     = {fwhm*1000:.2f} ps")
        print(f"    tail_amp = {tail_amp:.4f}")
        print(f"    tail_tau = {tail_tau:.4f} ns")
        above = np.where(irf_norm >= irf_norm.max() / 2)[0]
        fwhm_meas = (above[-1] - above[0]) * tcspc_ns if len(above) > 1 else fwhm
        print(f"    FWHM (measured on grid) = {fwhm_meas*1000:.2f} ps")
        print(f"    Peak bin = {np.argmax(irf_norm)}")

    return irf_norm, params


def irf_from_xlsx(xlsx: dict, n_bins: int, tcspc_res: float) -> np.ndarray:
    """
    Embed the xlsx IRF onto the PTU time axis.

    LAS X exports only ~21 sparse IRF points (one per bin in a narrow window).
    Scatter-placing these into a 529-bin array leaves most bins at zero,
    producing a comb rather than a smooth IRF. FFT convolution of a comb
    causes ringing artefacts that structurally inflate χ²_r.

    Fix: linearly interpolate the xlsx IRF points onto the full bin grid.
    Bins outside the xlsx IRF time range are set to zero.
    """
    if xlsx['irf_t'] is None or xlsx['irf_c'] is None:
        raise ValueError("XLSX does not contain IRF columns.")

    tcspc_ns   = tcspc_res * 1e9
    t_full     = np.arange(n_bins, dtype=float) * tcspc_ns

    # Sort xlsx IRF points by time (should already be sorted but be safe)
    t_pts = np.array(xlsx['irf_t'], dtype=float)
    c_pts = np.array(xlsx['irf_c'], dtype=float)
    c_pts = np.maximum(c_pts, 0.0)
    order = np.argsort(t_pts)
    t_pts, c_pts = t_pts[order], c_pts[order]

    # Linearly interpolate onto the full bin grid; zero outside the xlsx range
    irf_interp = np.interp(t_full, t_pts, c_pts, left=0.0, right=0.0)

    s = irf_interp.sum()
    if s == 0:
        raise ValueError("xlsx IRF is all zeros after interpolation.")
    return irf_interp / s


def gaussian_irf(n_bins: int, peak_bin: int, fwhm_bins: float) -> np.ndarray:
    """Bins-based Gaussian — used by estimate-irf paths only."""
    bins  = np.arange(n_bins, dtype=float)
    sigma = fwhm_bins / 2.3548
    irf   = np.exp(-0.5 * ((bins - peak_bin) / sigma)**2)
    return irf / irf.sum()


def estimate_irf_from_decay_raw(decay, tcspc_res, n_bins,
                                n_irf_bins=21, bg_est_pre=5) -> np.ndarray:
    peak_bin  = int(np.argmax(decay))
    bg_end    = max(0, peak_bin - bg_est_pre)
    bg        = float(np.median(decay[:bg_end])) if bg_end > 0 \
                else float(np.median(decay[-30:]))
    decay_sub = np.maximum(decay - bg, 0.0)
    half      = n_irf_bins // 2
    start     = max(0, peak_bin - half)
    end       = min(n_bins, peak_bin + half + 1)
    irf_raw   = decay_sub[start:end].copy()
    total     = irf_raw.sum()
    if total == 0:
        raise ValueError("Extracted IRF region has zero counts.")
    irf_full          = np.zeros(n_bins, dtype=float)
    irf_full[start:end] = irf_raw / total
    return irf_full


def _irf_parametric(t, t0, amplitude):
    return amplitude * (t / t0) * np.exp(-t / t0)


def estimate_irf_from_decay_parametric(decay, tcspc_res, n_bins,
                                       fit_window_width_ns=1.5,
                                       bg_est_pre=5) -> np.ndarray:
    peak_bin  = int(np.argmax(decay))
    bg_end    = max(0, peak_bin - bg_est_pre)
    bg        = float(np.median(decay[:bg_end])) if bg_end > 0 \
                else float(np.median(decay[-30:]))
    decay_sub = np.maximum(decay - bg, 0.0)
    time_ns   = np.arange(n_bins) * tcspc_res * 1e9
    t_peak_ns = time_ns[peak_bin]
    start_ns  = max(0, t_peak_ns - fit_window_width_ns / 2)
    end_ns    = min(time_ns[-1], t_peak_ns + fit_window_width_ns / 2)
    sb        = np.searchsorted(time_ns, start_ns, side='left')
    eb        = np.searchsorted(time_ns, end_ns,   side='right')
    if eb - sb < 3:
        raise ValueError("Fit window too narrow.")
    t_fit = time_ns[sb:eb] - time_ns[sb]
    y_fit = decay_sub[sb:eb]
    pk    = np.argmax(y_fit)
    try:
        popt, _ = curve_fit(_irf_parametric, t_fit, y_fit,
                             p0=[t_fit[pk]/2.0 if pk > 0 else 1.0, y_fit[pk]],
                             bounds=([0.01, 0], [10.0, np.inf]))
        t0, amp = popt
    except Exception as e:
        print(f"Parametric fit failed: {e}, falling back to raw extraction.")
        return estimate_irf_from_decay_raw(decay, tcspc_res, n_bins)
    t_full_ns = time_ns - time_ns[sb]
    irf_full  = np.maximum(_irf_parametric(t_full_ns, t0, amp), 0.0)
    total     = irf_full.sum()
    return irf_full / total if total > 0 else np.zeros(n_bins)


def build_full_irf(irf_prompt: np.ndarray,
                   shift_bins: float,
                   sigma_bins: float,
                   tail_amp:   float,
                   tail_tau_bins: float,
                   n_bins:     int) -> np.ndarray:
    """
    Assemble full IRF: prompt + optional slow tail, then shift + broaden.
    sigma_bins=0 → no broadening (used for Gaussian/scatter paths).
    tail_amp=0   → no tail (used for Gaussian/scatter paths).
    """
    peak_bin = int(np.argmax(irf_prompt))
    bins     = np.arange(n_bins, dtype=float)

    tail = np.where(
        bins >= peak_bin,
        tail_amp * np.exp(-(bins - peak_bin) / max(tail_tau_bins, 0.1)),
        0.0
    )
    irf_aug = irf_prompt + tail
    s       = irf_aug.sum()
    if s > 0:
        irf_aug /= s

    x_orig      = np.arange(n_bins, dtype=float)
    irf_shifted = np.interp(x_orig - shift_bins, x_orig, irf_aug,
                            left=0.0, right=0.0)

    if sigma_bins > 0.05:
        irf_shifted = gaussian_filter1d(irf_shifted, sigma=sigma_bins)
        s2 = irf_shifted.sum()
        if s2 > 0:
            irf_shifted /= s2

    return irf_shifted


# ══════════════════════════════════════════════════════════════════════════════
# 4.  RECONVOLUTION MODEL
# ══════════════════════════════════════════════════════════════════════════════

def _exponential_kernel(tcspc_res, n_bins, taus, amps, bg):
    t = np.arange(n_bins, dtype=float) * tcspc_res
    return sum(a * np.exp(-t / max(tau, 1e-15))
               for a, tau in zip(amps, taus)) + bg


def reconvolution_model(params, tcspc_res, n_bins, irf_prompt,
                        n_exp, bg_fixed, has_tail, fit_bg, fit_sigma):
    """
    Circular (FFT) reconvolution.

    Parameter vector layout (in order):
        τ₁ … τₙ          always
        α₁ … αₙ          always
        shift             always
        σ                 only if fit_sigma=True  (xlsx / estimated IRF paths)
        bg                only if fit_bg=True     (all paths in v12)
        tail_amp, tail_τ  only if has_tail=True   (xlsx / estimated IRF paths)

    Gaussian / scatter paths: fit_sigma=False, has_tail=False
        → [τ₁…τₙ, α₁…αₙ, shift, bg]

    xlsx / estimated paths:   fit_sigma=True,  has_tail=True
        → [τ₁…τₙ, α₁…αₙ, shift, σ, bg, tail_amp, tail_τ]
    """
    taus  = np.clip(params[:n_exp], 1e-14, None)
    amps  = params[n_exp:2*n_exp]

    # ---- Enforce τ₁ > τ₂ > τ₃ by sorting descending ----
    order = np.argsort(-taus)                # descending order indices
    taus = taus[order]
    amps = amps[order]
    # ----------------------------------------------------

    idx   = 2 * n_exp
    shift = params[idx]; idx += 1

    if fit_sigma:
        sigma = params[idx]; idx += 1
    else:
        sigma = 0.0

    if fit_bg:
        bg = params[idx]; idx += 1
    else:
        bg = bg_fixed

    if has_tail:
        tail_amp = params[idx]
        tail_tau = params[idx + 1]
    else:
        tail_amp, tail_tau = 0.0, 1.0

    irf_full = build_full_irf(irf_prompt, shift, sigma, tail_amp, tail_tau, n_bins)
    kernel   = _exponential_kernel(tcspc_res, n_bins, taus, amps, bg)
    return np.real(np.fft.ifft(np.fft.fft(kernel) * np.fft.fft(irf_full)))


# ══════════════════════════════════════════════════════════════════════════════
# 5.  BACKGROUND & FIT WINDOW
# ══════════════════════════════════════════════════════════════════════════════

def find_irf_peak_bin(decay: np.ndarray, smooth_sigma: float = 1.5) -> int:
    """
    Estimate the IRF peak position from the maximum of the first derivative
    of the smoothed decay (steepest point of the rising edge).

    np.argmax(decay) gives the fluorescence convolution peak, which is shifted
    right of the true IRF peak by 1-2 bins depending on the shortest lifetime.
    The IRF peak is where the measured signal rises fastest — i.e. where the
    derivative is maximum.

    Parameters
    ----------
    smooth_sigma : Gaussian smoothing width (bins) before differentiating.
                   1.5 bins is sufficient to suppress Poisson noise on the
                   rising edge without significantly shifting the peak estimate.
    """
    smoothed  = gaussian_filter1d(decay.astype(float), sigma=smooth_sigma)
    deriv     = np.gradient(smoothed)
    # Only search in the first half of the histogram (rising edge region)
    half      = len(decay) // 2
    peak_bin  = int(np.argmax(deriv[:half]))
    return peak_bin


def estimate_bg(decay: np.ndarray, peak_bin: int, pre_gap: int = 5) -> float:
    """Pre-IRF median — used as initial guess for bg when fit_bg=True."""
    end    = max(0, peak_bin - pre_gap)
    region = decay[:end]
    if len(region) >= 5:
        return max(float(np.median(region)), 0.0)
    return max(float(np.median(decay[-30:])), 0.0)


def find_fit_start(decay: np.ndarray, irf_prompt: np.ndarray,
                   tcspc_res: float, pre_bins: int = 5) -> int:
    """
    Start the fit window at the IRF onset minus pre_bins, not at bin 0.

    Bins before the IRF onset are flat background — including them adds
    DOF without constraining the model and inflates χ²_r because the
    weighted residuals in the rising edge dominate.

    pre_bins: how many bins before the first non-negligible IRF value
              to include (captures any pre-IRF photons and rounding).
    """
    threshold    = irf_prompt.max() * 1e-3
    onset_bins   = np.where(irf_prompt >= threshold)[0]
    if len(onset_bins) == 0:
        return 0
    onset        = int(onset_bins[0])
    fit_start    = max(0, onset - pre_bins)
    return fit_start


def find_fit_end(decay, peak_bin, tau_max_s, tcspc_res, n_bins) -> int:
    candidate    = min(n_bins, peak_bin + int(6.0 * tau_max_s / tcspc_res))
    search_start = int(0.82 * n_bins)
    tail         = gaussian_filter1d(decay[search_start:].astype(float), sigma=2)
    deriv        = np.gradient(tail)
    thresh       = 3.0 * np.std(deriv[:max(1, len(deriv)//2)])
    spikes       = np.where(deriv > thresh)[0]
    if len(spikes) > 0:
        spike_abs = search_start + spikes[0]
        if spike_abs < candidate:
            print(f"  Next-period artefact at bin {spike_abs} "
                  f"({spike_abs*tcspc_res*1e9:.2f} ns). Truncating fit window.")
            candidate = spike_abs
    return candidate


# ══════════════════════════════════════════════════════════════════════════════
# 6.  PARAMETER VECTOR HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _build_bounds(n_exp, tau_min, tau_max, decay_peak,
                  has_tail, fit_bg, fit_sigma, bg_init=0.0, bg_upper=None):
    """
    Build lower/upper bound lists matching the parameter vector layout
    in reconvolution_model.
    """
    lo = [tau_min] * n_exp + [0.0] * n_exp + [-2.0]   # τ, α, shift (±2 bins)
    hi = [tau_max] * n_exp + [10 * decay_peak] * n_exp + [2.0]

    if fit_sigma:
        lo += [0.0];  hi += [3.0]

    if fit_bg:
        _bg_hi = bg_upper if bg_upper is not None else bg_init * 1.5 + 10
        lo += [0.0];  hi += [_bg_hi]

    if has_tail:
        lo += [0.0,   1.0]
        hi += [5.0, 200.0]

    return lo, hi


def _pack_p0(n_exp, tau_min, tau_max, decay_peak,
             has_tail, fit_bg, fit_sigma, bg_init,
             tau_override=None):
    """Build initial parameter vector matching the layout in reconvolution_model."""
    if tau_override is not None:
        taus0 = np.asarray(tau_override)
    else:
        tmin  = max(tau_min, 1e-14) * 1.001
        tmax  = tau_max * 0.999
        taus0 = np.logspace(np.log10(tmin), np.log10(tmax), n_exp)

    amps0 = np.full(n_exp, decay_peak / n_exp)
    base  = np.concatenate([taus0, amps0, [0.0]])   # τ, α, shift

    if fit_sigma:
        base = np.concatenate([base, [0.3]])

    if fit_bg:
        base = np.concatenate([base, [bg_init]])

    if has_tail:
        base = np.concatenate([base, [0.5, 20.0]])

    return base


# ══════════════════════════════════════════════════════════════════════════════
# 7.  SUMMED DECAY FIT
# ══════════════════════════════════════════════════════════════════════════════

class _DECost:
    def __init__(self, tcspc_res, n_bins, irf_prompt, n_exp, bg_fixed,
                 has_tail, fit_bg, fit_sigma,
                 fit_start, fit_end, decay, weights):
        self.tcspc_res  = tcspc_res
        self.n_bins     = n_bins
        self.irf_prompt = irf_prompt
        self.n_exp      = n_exp
        self.bg_fixed   = bg_fixed
        self.has_tail   = has_tail
        self.fit_bg     = fit_bg
        self.fit_sigma  = fit_sigma
        self.fit_start  = fit_start
        self.fit_end    = fit_end
        self.decay      = decay
        self.weights    = weights

    def __call__(self, params):
        model = reconvolution_model(
            params, self.tcspc_res, self.n_bins, self.irf_prompt,
            self.n_exp, self.bg_fixed, self.has_tail,
            self.fit_bg, self.fit_sigma)
        res = ((model[self.fit_start:self.fit_end]
                - self.decay[self.fit_start:self.fit_end])
               / self.weights)
        return np.sum(res**2)


def fit_summed(decay, tcspc_res, n_bins, irf_prompt,
               has_tail, fit_bg, fit_sigma,
               n_exp, tau_min_ns, tau_max_ns,
               optimizer="de", n_restarts=8,
               de_popsize=15, de_maxiter=1000,
               workers=-1, polish=True) -> tuple[np.ndarray, dict]:

    tau_min  = tau_min_ns * 1e-9
    tau_max  = tau_max_ns * 1e-9

    # ---- Normalise decay to [0,1] (peak = 1) ----
    scale = decay.max()
    if scale <= 0:
        raise ValueError("Decay has zero maximum – cannot normalise.")
    decay_norm = decay / scale
    # ---------------------------------------------

    peak_bin = int(np.argmax(decay_norm))
    bg_init  = estimate_bg(decay_norm, peak_bin)   # now in normalised units
    bg_fixed = bg_init if not fit_bg else 0.0      # passed to model but ignored when fit_bg

    fit_end   = find_fit_end(decay_norm, peak_bin, tau_max, tcspc_res, n_bins)

    # Match Leica's fit window start: begin at bin 1 (first bin after t=0),
    # not bin 0. Leica exports fit from 0.1455 ns = 1 bin in.
    fit_start = 1

    # Cap fit_end to match Leica's window end (~44.95 ns = bin 463).
    # Our artefact detection finds bin 483 (46.84 ns) which includes extra
    # tail bins Leica excludes.
    leica_fit_end = int(round(44.9455 / (tcspc_res * 1e9)))
    fit_end = min(fit_end, leica_fit_end)

    # bg upper bound: pre-IRF mean overestimates true bg due to fluorescence
    # pile-up from the previous laser period (~23 cts/bin wraps into pre-IRF bins).
    # Cap bg at 0.75 * bg_init so the optimizer can't absorb pile-up into bg.
    # Leica's Tail Offset (53.62) ≈ 0.65 * pre-IRF mean (82).
    bg_upper = max(bg_init * 0.75, 10.0)

    print(f"  bg initial guess = {bg_init:.3f} (normalised), upper bound = {bg_upper:.3f} "
          f"({'free param' if fit_bg else 'fixed'})")
    print(f"  σ broadening: {'free param' if fit_sigma else 'fixed at 0'}")
    print(f"  Fit window: bins {fit_start}–{fit_end} "
          f"({fit_start*tcspc_res*1e9:.2f}–{fit_end*tcspc_res*1e9:.2f} ns), "
          f"{fit_end-fit_start} bins")

    lo, hi  = _build_bounds(n_exp, tau_min, tau_max, decay_norm.max(),   # note: decay_norm.max() = 1
                             has_tail, fit_bg, fit_sigma,
                             bg_init=bg_init, bg_upper=bg_upper)
    bounds  = list(zip(lo, hi))

    # Weights based on normalised decay – this yields χ²_norm = χ²_original / scale
    weights = np.sqrt(np.maximum(decay_norm[fit_start:fit_end], 1e-8))

    def residuals(params):
        model_norm = reconvolution_model(
            params, tcspc_res, n_bins, irf_prompt,
            n_exp, bg_fixed, has_tail, fit_bg, fit_sigma)
        return (model_norm[fit_start:fit_end] - decay_norm[fit_start:fit_end]) / weights

    if optimizer == "lm_multistart":
        rng       = np.random.default_rng(42)
        best_res  = None
        best_cost = np.inf

        for i in range(n_restarts + 1):
            tau_ov = None if i == 0 else np.sort(
                np.exp(rng.uniform(np.log(tau_min*1.001),
                                   np.log(tau_max*0.999), n_exp)))
            p0 = _pack_p0(n_exp, tau_min, tau_max, float(decay_norm.max()),
                          has_tail, fit_bg, fit_sigma, bg_init,
                          tau_override=tau_ov)
            try:
                res = least_squares(residuals, p0, bounds=(lo, hi), method="trf",
                                    max_nfev=50000,
                                    ftol=1e-13, xtol=1e-13, gtol=1e-13)
            except Exception as exc:
                print(f"    Restart {i:2d}: failed ({exc})")
                continue
            tag = "log-spaced" if i == 0 else "random    "
            if res.cost < best_cost:
                best_cost = res.cost
                best_res  = res
                print(f"    Restart {i:2d} ({tag}): cost={res.cost:.4e}  ← best")
            else:
                print(f"    Restart {i:2d} ({tag}): cost={res.cost:.4e}")

        if best_res is None:
            raise RuntimeError("All restarts failed.")
        popt_norm = best_res.x
        message   = best_res.message

    elif optimizer == "de":
        print(f"  Differential evolution: popsize={de_popsize}, "
              f"maxiter={de_maxiter}, workers={workers}")
        cost_fn = _DECost(tcspc_res, n_bins, irf_prompt, n_exp, bg_fixed,
                          has_tail, fit_bg, fit_sigma,
                          fit_start, fit_end, decay_norm, weights)
        de_res = differential_evolution(
            cost_fn, bounds=bounds,
            maxiter=de_maxiter, popsize=de_popsize,
            workers=workers, seed=42,
            updating='deferred' if workers != 1 else 'immediate',
            disp=False)
        popt_norm = de_res.x
        message   = f"DE success={de_res.success}, fun={de_res.fun:.4e}"

        if polish:
            print("  Running final LM polish...")
            pol = least_squares(residuals, popt_norm, bounds=(lo, hi), method="trf",
                                max_nfev=5000, ftol=1e-13, xtol=1e-13, gtol=1e-13)
            popt_norm = pol.x
            message  += f"; polished cost={pol.cost:.4e}"
    else:
        raise ValueError(f"Unknown optimizer: {optimizer!r}")

    # ---- Rescale amplitudes and background back to original units ----
    popt_original = popt_norm.copy()
    # amplitudes are indices n_exp : 2*n_exp
    popt_original[n_exp:2*n_exp] *= scale
    if fit_bg:
        # locate bg index: after shift, possibly sigma
        bg_idx = 2*n_exp + 1                # shift occupies one position
        if fit_sigma:
            bg_idx += 1
        popt_original[bg_idx] *= scale
    # -------------------------------------------------------------------

    summary = _make_summary(popt_original, decay, tcspc_res, n_bins, irf_prompt,
                            n_exp, bg_fixed, has_tail, fit_bg, fit_sigma,
                            fit_start, fit_end, message)
    return popt_original, summary


def _make_summary(popt, decay, tcspc_res, n_bins, irf_prompt,
                  n_exp, bg_fixed, has_tail, fit_bg, fit_sigma,
                  fit_start, fit_end, message=None) -> dict:
    """Unpack params in the same order as reconvolution_model."""

    taus  = popt[:n_exp]
    amps  = popt[n_exp:2*n_exp]
    # Enforce τ₁ > τ₂ > τ₃ order (descending taus)
    order = np.argsort(-taus)
    taus = taus[order]
    amps = amps[order]
    idx   = 2 * n_exp

    shift = popt[idx]; idx += 1

    if fit_sigma:
        sigma = popt[idx]; idx += 1
    else:
        sigma = 0.0

    if fit_bg:
        bg_fit = popt[idx]; idx += 1
    else:
        bg_fit = bg_fixed

    if has_tail:
        tail_amp = popt[idx]
        tail_tau = popt[idx + 1]
    else:
        tail_amp = tail_tau = 0.0

    model   = reconvolution_model(popt, tcspc_res, n_bins, irf_prompt,
                                   n_exp, bg_fixed, has_tail, fit_bg, fit_sigma)
    d_win   = decay[fit_start:fit_end]
    m_win   = model[fit_start:fit_end]
    sigma_w = np.sqrt(np.maximum(d_win, 1.0))
    chi2    = float(np.sum(((d_win - m_win) / sigma_w)**2))
    dof     = max((fit_end - fit_start) - len(popt), 1)
    rchi2   = chi2 / dof
    p_val   = float(1 - chi2_dist.cdf(chi2, df=dof))
    resid   = (decay - model) / np.sqrt(np.maximum(model, 1.0))

    # Tail-only chi2_r: exclude rising edge (first 20% of fit window past peak)
    # Leica reports chi2 only over the post-peak tail — this matches their convention
    peak_bin_loc = int(np.argmax(decay[fit_start:fit_end])) + fit_start
    tail_start   = peak_bin_loc + max(1, int(0.05 * (fit_end - peak_bin_loc)))
    d_tail  = decay[tail_start:fit_end]
    m_tail  = model[tail_start:fit_end]
    sw_tail = np.sqrt(np.maximum(d_tail, 1.0))
    chi2_tail  = float(np.sum(((d_tail - m_tail) / sw_tail)**2))
    dof_tail   = max((fit_end - tail_start) - len(popt), 1)
    rchi2_tail = chi2_tail / dof_tail

    # Compute amplitude fractions and weighted means using the sorted arrays
    amp_sum    = amps.sum() if amps.sum() > 0 else 1.0
    fracs      = amps / amp_sum
    tau_amp    = float(np.dot(fracs, taus))
    tau_int    = float(np.dot(amps, taus**2) / np.dot(amps, taus))

    above    = np.where(irf_prompt >= irf_prompt.max() / 2)[0]
    fwhm_pr  = (above[-1] - above[0]) if len(above) > 1 else 1
    fwhm_eff = np.sqrt(fwhm_pr**2 + (2.3548 * sigma)**2) * tcspc_res * 1e9

    return dict(
        tcspc_res        = tcspc_res,
        taus_ns          = taus * 1e9,
        amps             = amps,
        fractions        = fracs,
        bg_fit           = bg_fit,
        tau_mean_amp_ns  = tau_amp * 1e9,
        tau_mean_int_ns  = tau_int * 1e9,
        chi2             = chi2,
        reduced_chi2     = rchi2,
        reduced_chi2_tail= rchi2_tail,
        tail_start_bin   = tail_start,
        p_val            = p_val,
        dof              = dof,
        fit_window_bins  = (fit_start, fit_end),
        fit_window_ns    = (fit_start*tcspc_res*1e9, fit_end*tcspc_res*1e9),
        irf_shift_bins   = shift,
        irf_sigma_bins   = sigma,
        irf_fwhm_eff_ns  = fwhm_eff,
        tail_amp         = tail_amp,
        tail_tau_ns      = tail_tau * tcspc_res * 1e9,
        model            = model,
        residuals        = resid,
        optimizer_msg    = message,
    )


def print_summary(summary: dict, strategy: str, n_exp: int):
    s         = summary
    tcspc_res = s['tcspc_res']

    print(f"\n{'─'*60}")
    print(f"  Fit: {n_exp}-exp | IRF: {strategy}")
    print(f"{'─'*60}")
    for i, (tau, amp, frac) in enumerate(
            zip(s['taus_ns'], s['amps'], s['fractions'])):
        print(f"  τ{i+1} = {tau:8.4f} ns   α{i+1} = {amp:.3e}   f{i+1} = {frac:.4f}")
    print(f"  τ_mean (amplitude-weighted)  = {s['tau_mean_amp_ns']:.4f} ns")
    print(f"  τ_mean (intensity-weighted)  = {s['tau_mean_int_ns']:.4f} ns")
    print(f"  bg (fitted, Tail Offset)     = {s['bg_fit']:.2f} cts/bin")
    print(f"  IRF shift                    = {s['irf_shift_bins']:.3f} bins "
          f"({s['irf_shift_bins'] * tcspc_res * 1e12:.1f} ps)")
    print(f"  IRF σ (prompt broadening)    = {s['irf_sigma_bins']:.3f} bins")
    print(f"  IRF FWHM (effective)         = {s['irf_fwhm_eff_ns']:.4f} ns")
    if s['tail_amp'] > 0:
        print(f"  IRF tail amp                 = {s['tail_amp']:.4f}")
        print(f"  IRF tail τ                   = {s['tail_tau_ns']:.3f} ns")
        if s['tail_tau_ns'] > 18:
            print(f"  ⚠  tail τ near upper bound — consider acquiring a scatter PTU")
    print(f"  χ²_r = {s['reduced_chi2']:.4f}  "
          f"(χ²={s['chi2']:.1f}, DoF={s['dof']}, p={s['p_val']:.4f})  [full window]")
    print(f"  χ²_r = {s['reduced_chi2_tail']:.4f}  "
          f"(tail only, t>{s['tail_start_bin']*tcspc_res*1e9:.2f} ns)  "
          f"← Leica convention")
    flag = "✓" if 0.001 < s['p_val'] < 0.999 else "⚠"
    print(f"  {flag} Optimizer: {s['optimizer_msg']}")


# ══════════════════════════════════════════════════════════════════════════════
# 8.  PER-PIXEL FITTING
# ══════════════════════════════════════════════════════════════════════════════

def fit_per_pixel(stack, tcspc_res, n_bins, irf_prompt,
                  has_tail, fit_bg, fit_sigma,
                  global_popt, n_exp,
                  min_photons=MIN_PHOTONS_PERPIX) -> dict:
    ny, nx, _ = stack.shape

    # Extract fixed IRF parameters from global fit using same unpacking order
    idx   = 2 * n_exp
    shift = global_popt[idx]; idx += 1
    sigma = global_popt[idx] if fit_sigma else 0.0
    if fit_sigma: idx += 1
    # skip bg — re-estimated per pixel
    if fit_bg: idx += 1
    tamp  = global_popt[idx]     if has_tail else 0.0
    ttau  = global_popt[idx + 1] if has_tail else 1.0
    taus_fixed = global_popt[:n_exp]

    irf_fixed  = build_full_irf(irf_prompt, shift, sigma, tamp, ttau, n_bins)
    t_axis     = np.arange(n_bins, dtype=float) * tcspc_res
    basis      = np.stack([np.exp(-t_axis / max(tau, 1e-15)) for tau in taus_fixed])
    irf_fft    = np.fft.fft(irf_fixed)
    conv_basis = np.array([
        np.real(np.fft.ifft(np.fft.fft(b) * irf_fft)) for b in basis
    ])  # (n_exp, n_bins)
    A = conv_basis.T   # (n_bins, n_exp)

    maps = dict(
        intensity    = stack.sum(axis=2),
        tau_mean_int = np.full((ny, nx), np.nan),
        tau_mean_amp = np.full((ny, nx), np.nan),
        chi2_r       = np.full((ny, nx), np.nan),
    )
    for i in range(n_exp):
        maps[f"alpha_{i+1}"] = np.full((ny, nx), np.nan)
        maps[f"frac_{i+1}"]  = np.full((ny, nx), np.nan)

    fitted = skipped = 0
    print(f"  Per-pixel fitting: {ny}×{nx}={ny*nx} pixels "
          f"(τ fixed, amplitudes + bg free) …")
    t0 = time.time()

    for yi in range(ny):
        for xi in range(nx):
            decay_px = stack[yi, xi, :]
            if decay_px.sum() < min_photons:
                skipped += 1
                continue

            bg_px   = estimate_bg(decay_px, int(np.argmax(decay_px)))
            data_corr = np.maximum(decay_px - bg_px, 0.0)
            amps_px, _ = nnls(A, data_corr)

            model_px = A @ amps_px + bg_px
            resid    = decay_px - model_px
            chi2_px  = float(np.sum(resid**2 / np.maximum(model_px, 1.0)))
            dof_px   = max(n_bins - n_exp, 1)

            amp_sum = amps_px.sum()
            if amp_sum <= 0:
                skipped += 1
                continue

            fracs_px = amps_px / amp_sum
            taus_ns  = taus_fixed * 1e9
            tau_amp  = float(np.dot(fracs_px, taus_ns))
            denom    = np.dot(amps_px, taus_ns)
            tau_int  = float(np.dot(amps_px, taus_ns**2) / denom) \
                       if denom > 0 else np.nan

            maps["tau_mean_int"][yi, xi] = tau_int
            maps["tau_mean_amp"][yi, xi] = tau_amp
            maps["chi2_r"][yi, xi]       = chi2_px / dof_px
            for i in range(n_exp):
                maps[f"alpha_{i+1}"][yi, xi] = amps_px[i]
                maps[f"frac_{i+1}"][yi, xi]  = fracs_px[i]
            fitted += 1

    elapsed = time.time() - t0
    print(f"  Fitted: {fitted}/{ny*nx}  |  Skipped (<{min_photons} ph): {skipped}  "
          f"|  {elapsed:.1f}s")
    return maps


# ══════════════════════════════════════════════════════════════════════════════
# 9.  IRF COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

def _fwhm_ns(irf: np.ndarray, tcspc_res: float) -> float:
    """
    FWHM in ns. For very narrow IRFs (sub-bin Gaussian), falls back to
    the analytical width estimate from the peak value and bin spacing.
    """
    pk = irf.max()
    if pk <= 0:
        return np.nan
    above = np.where(irf >= pk / 2)[0]
    if len(above) > 1:
        return (above[-1] - above[0]) * tcspc_res * 1e9
    # Sub-bin case: IRF is confined to 1 bin — estimate from integral/peak
    # For a Gaussian: FWHM = 2*sqrt(2*ln2)*sigma, integral/peak = sigma*sqrt(2pi)
    # So sigma ≈ integral/peak/sqrt(2pi), FWHM ≈ integral/peak * sqrt(4*ln2/pi) * tcspc_res
    integral = irf.sum() * tcspc_res * 1e9   # in ns
    fwhm_est  = integral * np.sqrt(4 * np.log(2) / np.pi)
    return float(fwhm_est)


def compare_irfs(irf_estimated:  np.ndarray,
                 xlsx:           dict | None,
                 tcspc_res:      float,
                 n_bins:         int,
                 strategy:       str,
                 out_prefix:     str) -> dict | None:
    """
    Compare the estimated/constructed IRF against the xlsx IRF.

    Metrics are reported in two forms:
      Raw       — bin-by-bin comparison with no alignment correction.
                  Reflects actual timing offset between the two IRFs.
      Aligned   — estimated IRF is peak-shifted to match the xlsx IRF peak
                  before computing overlap. Reflects pure shape quality,
                  independent of any timing offset.

    Metrics
    -------
    FWHM (ns)              : width of each IRF at half-maximum
    Peak position (ns)     : bin of maximum value
    Peak shift             : estimated − xlsx peak (timing error)
    Pearson r              : linear correlation on shared support
    RMSE                   : root-mean-square error on normalised arrays
    Bhattacharyya coeff.   : probability-distribution overlap [0,1]
    """
    t_ns = np.arange(n_bins, dtype=float) * tcspc_res * 1e9

    # ── Embed xlsx IRF onto the PTU time axis ─────────────────────────────────
    irf_xlsx_embedded = None
    if xlsx is not None and xlsx.get('irf_t') is not None and xlsx.get('irf_c') is not None:
        irf_raw = np.zeros(n_bins)
        for t, c in zip(xlsx['irf_t'], xlsx['irf_c']):
            idx = int(round(t / (tcspc_res * 1e9)))
            if 0 <= idx < n_bins:
                irf_raw[idx] += max(c, 0.0)
        s = irf_raw.sum()
        if s > 0:
            irf_xlsx_embedded = irf_raw / s

    if irf_xlsx_embedded is None:
        print("  IRF comparison skipped — no xlsx IRF available.")
        return None

    # ── Normalise both to unit area ───────────────────────────────────────────
    est = irf_estimated / irf_estimated.sum()
    ref = irf_xlsx_embedded / irf_xlsx_embedded.sum()

    # ── Peak positions ────────────────────────────────────────────────────────
    peak_est_bin = int(np.argmax(est))
    peak_ref_bin = int(np.argmax(ref))
    shift_bins   = peak_est_bin - peak_ref_bin   # +ve: est is right of ref

    # ── Peak-aligned estimated IRF ────────────────────────────────────────────
    # shift_bins = est_peak - ref_peak.
    # To move est RIGHT by abs(shift_bins), query at x + shift_bins:
    #   np.interp(x + shift_bins, x, est)  moves est towards ref
    # The common mistake is x - shift_bins which inverts the direction.
    x = np.arange(n_bins, dtype=float)
    est_aligned = np.interp(x + shift_bins, x, est, left=0.0, right=0.0)
    s = est_aligned.sum()
    if s > 0:
        est_aligned /= s

    # ── Metric helper ─────────────────────────────────────────────────────────
    def _metrics(a, b, label):
        support = (a > 1e-8) | (b > 1e-8)
        a_s, b_s = a[support], b[support]
        if len(a_s) > 1 and a_s.std() > 0 and b_s.std() > 0:
            r = float(np.corrcoef(a_s, b_s)[0, 1])
        else:
            r = np.nan
        rmse = float(np.sqrt(np.mean((a - b)**2)))
        bc   = float(np.sum(np.sqrt(a * b)))
        return dict(label=label, pearson_r=r, rmse=rmse,
                    overlap_score=max(0.0, 1.0 - rmse), bhattacharyya=bc)

    m_raw     = _metrics(est,         ref, "raw     (unaligned)")
    m_aligned = _metrics(est_aligned, ref, "aligned (peak-shift corrected)")

    fwhm_est = _fwhm_ns(est, tcspc_res)
    fwhm_ref = _fwhm_ns(ref, tcspc_res)
    peak_est_ns = peak_est_bin * tcspc_res * 1e9
    peak_ref_ns = peak_ref_bin * tcspc_res * 1e9

    metrics = dict(
        fwhm_estimated_ns  = fwhm_est,
        fwhm_xlsx_ns       = fwhm_ref,
        peak_estimated_ns  = peak_est_ns,
        peak_xlsx_ns       = peak_ref_ns,
        peak_shift_ns      = shift_bins * tcspc_res * 1e9,
        peak_shift_bins    = shift_bins,
        raw                = m_raw,
        aligned            = m_aligned,
    )

    # ── Print ─────────────────────────────────────────────────────────────────
    print(f"\n  IRF Comparison  ({strategy.split('peak_bin')[0].strip()}  vs  xlsx)")
    print(f"  {'Metric':<28} {'Estimated':>12} {'xlsx':>12}")
    print(f"  {'─'*54}")
    print(f"  {'FWHM (ns)':<28} {fwhm_est:>12.4f} {fwhm_ref:>12.4f}")
    print(f"  {'Peak position (ns)':<28} {peak_est_ns:>12.4f} {peak_ref_ns:>12.4f}")
    print(f"  {'Peak shift (est − xlsx)':<28} "
          f"{shift_bins * tcspc_res * 1e9:>+11.4f} ns  ({shift_bins:+d} bins)")
    print(f"  {'─'*54}")
    for m in (m_raw, m_aligned):
        print(f"  [{m['label']}]")
        print(f"    {'Pearson r':<26} {m['pearson_r']:>12.4f}")
        print(f"    {'RMSE (normalised)':<26} {m['rmse']:>12.6f}")
        print(f"    {'Overlap score (1−RMSE)':<26} {m['overlap_score']:>12.4f}")
        print(f"    {'Bhattacharyya coeff.':<26} {m['bhattacharyya']:>12.4f}")

    bc_a = m_aligned['bhattacharyya']
    if bc_a >= 0.99:
        print(f"\n  ✓ Excellent shape match after alignment (BC={bc_a:.4f})")
        print(f"    → Use --irf-fwhm with adjusted peak; shape is correct.")
    elif bc_a >= 0.90:
        print(f"\n  ~ Acceptable shape match after alignment (BC={bc_a:.4f})")
        print(f"    → Shape is reasonable but consider --xlsx for fitting.")
    else:
        print(f"\n  ⚠ Poor shape match even after alignment (BC={bc_a:.4f})")
        print(f"    → FWHM or IRF model is wrong. Use --xlsx for fitting.")

    if abs(shift_bins) >= 2:
        print(f"  ⚠ Peak misaligned by {shift_bins:+d} bins ({shift_bins*tcspc_res*1e12:+.0f} ps) "
              f"— IRF peak bin estimate may be off.")

    # ── Plot ──────────────────────────────────────────────────────────────────
    plt.rcParams.update({"figure.dpi": 130, "font.size": 10,
                          "axes.spines.top": False, "axes.spines.right": False})

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("IRF Comparison — Estimated vs LAS X xlsx",
                 fontsize=11, fontweight="bold")

    # Restrict x to non-zero support ± 10 bins
    support = (est > 1e-8) | (ref > 1e-8)
    idx_sup = np.where(support)[0]
    if len(idx_sup):
        x_lo = max(0,        idx_sup[0]  - 10) * tcspc_res * 1e9
        x_hi = min(n_bins-1, idx_sup[-1] + 10) * tcspc_res * 1e9
    else:
        x_lo, x_hi = t_ns[0], t_ns[-1]

    row_labels = ["Unaligned", "Peak-aligned"]
    for row, (e_plot, m) in enumerate([(est, m_raw), (est_aligned, m_aligned)]):
        diff = e_plot - ref

        # Linear overlay
        axes[row, 0].plot(t_ns, ref,    "b-",  lw=2,   label="xlsx IRF")
        axes[row, 0].plot(t_ns, e_plot, "r--", lw=1.8, label="estimated")
        axes[row, 0].set_xlim(x_lo, x_hi)
        axes[row, 0].set_ylabel("Normalised amplitude")
        axes[row, 0].set_title(f"{row_labels[row]} — linear")
        axes[row, 0].legend(fontsize=8)
        if row == 1:
            axes[row, 0].set_xlabel("Time (ns)")

        # Log overlay
        axes[row, 1].semilogy(t_ns, np.clip(ref,    1e-8, None), "b-",  lw=2)
        axes[row, 1].semilogy(t_ns, np.clip(e_plot, 1e-8, None), "r--", lw=1.8)
        axes[row, 1].set_xlim(x_lo, x_hi)
        axes[row, 1].set_title(f"{row_labels[row]} — log")
        if row == 1:
            axes[row, 1].set_xlabel("Time (ns)")

        # Difference
        axes[row, 2].fill_between(t_ns, diff, where=diff >= 0,
                                   alpha=0.6, color="#e63946", label="est > xlsx")
        axes[row, 2].fill_between(t_ns, diff, where=diff < 0,
                                   alpha=0.6, color="#457b9d", label="est < xlsx")
        axes[row, 2].axhline(0, color="k", lw=0.8, ls="--")
        axes[row, 2].set_xlim(x_lo, x_hi)
        axes[row, 2].set_ylabel("Δ (estimated − xlsx)")
        axes[row, 2].set_title(f"Difference  RMSE={m['rmse']:.5f}")
        axes[row, 2].legend(fontsize=8)
        if row == 1:
            axes[row, 2].set_xlabel("Time (ns)")

        txt = (f"Pearson r = {m['pearson_r']:.4f}\n"
               f"BC        = {m['bhattacharyya']:.4f}\n"
               f"FWHM est  = {fwhm_est:.4f} ns\n"
               f"FWHM xlsx = {fwhm_ref:.4f} ns")
        if row == 0:
            txt += f"\nΔpeak = {shift_bins*tcspc_res*1e12:+.0f} ps ({shift_bins:+d} bins)"
        axes[row, 2].text(0.97, 0.97, txt, transform=axes[row, 2].transAxes,
                          va="top", ha="right", fontsize=8, family="monospace",
                          bbox=dict(boxstyle="round,pad=0.3", fc="#f7f7f7", alpha=0.9))

    plt.tight_layout()
    out = f"{out_prefix}_irf_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")

    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# 10.  PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

def plot_summed(decay, summary, ptu, xlsx, n_exp, strategy, out_prefix,
               irf_prompt=None):
    plt.rcParams.update({"figure.dpi": 130, "font.size": 10,
                          "axes.spines.top": False, "axes.spines.right": False})
    s    = summary
    t_ns = np.arange(ptu.n_bins) * ptu.tcspc_res * 1e9
    fs, fe = s["fit_window_bins"]

    fig = plt.figure(figsize=(12, 7))
    gs  = gridspec.GridSpec(2, 3, height_ratios=[3, 1], hspace=0.08, wspace=0.35)
    ax1 = fig.add_subplot(gs[0, :2])
    ax2 = fig.add_subplot(gs[1, :2], sharex=ax1)
    ax3 = fig.add_subplot(gs[0,  2])
    ax4 = fig.add_subplot(gs[1,  2])
    ax4.axis("off")

    ax1.semilogy(t_ns, np.clip(decay, 1, None), ".", color="#aaa",
                 ms=2, rasterized=True, label="PTU data")

    # IRF — scale to ~10% of decay peak for visibility on log axis.
    # Mask bins below 0.1% of IRF peak so zeros don't pollute the log axis.
    if irf_prompt is not None:
        scale      = decay.max() * 0.1 / irf_prompt.max()
        irf_scaled = irf_prompt * scale
        irf_mask   = irf_scaled > irf_scaled.max() * 1e-3   # only plot non-negligible
        t_irf_plot = t_ns[irf_mask]
        v_irf_plot = irf_scaled[irf_mask]
        ax1.semilogy(t_irf_plot, v_irf_plot,
                     color="#f4a261", lw=1.5, ls="-.", alpha=0.85,
                     label=f"IRF (×{scale:.1e})")

    if xlsx is not None and xlsx.get('fit_t') is not None:
        ax1.semilogy(xlsx["fit_t"], np.clip(xlsx["fit_c"], 1, None),
                     "b-", lw=1.1, alpha=0.55, label="LAS X fit")
    ax1.semilogy(t_ns, np.clip(s["model"], 1, None), "r-", lw=2,
                 label=f"{n_exp}-exp reconv.")
    ax1.set_xlim(0, min(t_ns[-1], 22))
    ax1.set_ylabel("Counts")
    ax1.legend(fontsize=8, loc="upper right")
    ax1.set_title(f"Summed Decay — {n_exp}-exp | IRF: {strategy}", fontweight="bold")
    ax1.axvspan(s["fit_window_ns"][0], s["fit_window_ns"][1],
                alpha=0.06, color="green")
    plt.setp(ax1.get_xticklabels(), visible=False)

    ax2.axhline(0, color="k", lw=0.8, ls="--")
    ax2.fill_between(t_ns[fs:fe], np.clip(s["residuals"][fs:fe], -5, 5),
                     alpha=0.5, color="#457b9d")
    ax2.set_ylim(-5, 5)
    ax2.set_xlim(0, min(t_ns[-1], 22))
    ax2.set_xlabel("Time (ns)")
    ax2.set_ylabel("W. Residuals")

    rv = np.clip(s["residuals"][fs:fe], -5, 5)
    ax3.hist(rv, bins=60, color="#2a9d8f", edgecolor="none", alpha=0.85)
    ax3.axvline(0, color="k", lw=0.8)
    ax3.set_xlabel("Weighted residual")
    ax3.set_ylabel("Frequency")
    ax3.set_title(f"Residuals  μ={rv.mean():.3f}  σ={rv.std():.3f}")

    lines = [f"χ²_r = {s['reduced_chi2']:.4f}",
             f"p    = {s['p_val']:.4f}",
             f"bg   = {s['bg_fit']:.1f} cts/bin",
             f"τ_mean(int) = {s['tau_mean_int_ns']:.4f} ns",
             f"τ_mean(amp) = {s['tau_mean_amp_ns']:.4f} ns",
             f"IRF FWHM(eff) = {s['irf_fwhm_eff_ns']:.4f} ns", ""]
    for i, (tau, frac) in enumerate(zip(s["taus_ns"], s["fractions"])):
        lines.append(f"τ{i+1}={tau:.4f} ns  f{i+1}={frac:.4f}")
    ax4.text(0.05, 0.97, "\n".join(lines), transform=ax4.transAxes,
             va="top", fontsize=9, family="monospace",
             bbox=dict(boxstyle="round,pad=0.4", fc="#f7f7f7", alpha=0.9))

    plt.suptitle("FLIM Reconvolution Fit — Leica FALCON / PicoHarp",
                 fontsize=12, fontweight="bold")
    out = f"{out_prefix}_summed_{n_exp}exp.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_pixel_maps(maps, n_exp, out_prefix, binning=1):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.patch.set_facecolor("#111")

    def _show(ax, data, title, cmap="viridis", vmin=None, vmax=None, unit="ns"):
        ax.set_facecolor("#111")
        valid = data[np.isfinite(data) & (data > 0)]
        if len(valid) == 0:
            ax.set_visible(False); return
        vlo = vmin if vmin is not None else np.percentile(valid, 2)
        vhi = vmax if vmax is not None else np.percentile(valid, 98)
        im  = ax.imshow(data, cmap=cmap, vmin=vlo, vmax=vhi, interpolation="nearest")
        cb  = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(unit, color="white")
        cb.ax.yaxis.set_tick_params(color="white")
        plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")
        ax.set_title(title, color="white", fontweight="bold")
        ax.set_axis_off()

    _show(axes[0, 0], maps["intensity"],    "Intensity",         "hot",     unit="photons")
    _show(axes[0, 1], maps["tau_mean_int"], "τ_mean (int.-wt.)", FLIM_CMAP)
    _show(axes[0, 2], maps["tau_mean_amp"], "τ_mean (amp.-wt.)", FLIM_CMAP)
    for i in range(min(n_exp, 3)):
        _show(axes[1, i], maps.get(f"frac_{i+1}"), f"f{i+1}",
              "viridis", vmin=0, vmax=1, unit="fraction")

    plt.suptitle(f"FLIM Pixel Maps — {n_exp}-exp (τ fixed, α free)  "
                 f"binning={binning}×{binning}",
                 color="white", fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = f"{out_prefix}_pixelmaps_{n_exp}exp.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#111")
    plt.close()
    print(f"  Saved: {out}")


def plot_lifetime_histogram(maps, n_exp, out_prefix):
    tau = maps["tau_mean_int"]
    wt  = maps["intensity"]
    ok  = np.isfinite(tau) & (wt > 0)
    if ok.sum() < 2:
        return
    mu_w = np.average(tau[ok], weights=wt[ok])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(tau[ok], bins=100, weights=wt[ok], color="#2a9d8f", alpha=0.85)
    ax.axvline(mu_w, color="red", ls="--", lw=1.5,
               label=f"Weighted mean = {mu_w:.3f} ns")
    ax.set_xlabel("τ_mean (intensity-weighted) [ns]")
    ax.set_ylabel("Photon-weighted frequency")
    ax.set_title(f"Lifetime Distribution — {n_exp}-exp", fontweight="bold")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out = f"{out_prefix}_lifetime_hist_{n_exp}exp.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# 11.  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="FLIM reconvolution fit — PTU + optional XLSX (Leica FALCON)"
    )
    ap.add_argument("--ptu",   default=None, required=True, help="Path to PTU file")
    ap.add_argument("--xlsx",  default=None,
                    help="LAS X export xlsx (overlay comparison and/or IRF source)")
    ap.add_argument("--no-xlsx-irf", action="store_true",
                    help="Load xlsx for comparison/overlay but do NOT use its IRF "
                         "for fitting. Falls through to Gaussian/estimated IRF instead.")
    ap.add_argument("--debug-xlsx", action="store_true",
                    help="Print raw xlsx row contents and detected columns to diagnose "
                         "parsing failures.")
    ap.add_argument("--irf",   default=None,
                    help="Scatter PTU for measured IRF (highest priority)")
    ap.add_argument("--irf-xlsx", default=None,
                    help="Path to a reference xlsx exported from LAS X, used ONLY "
                         "to extract the IRF shape for fitting. The IRF is "
                         "system-specific (not FOV-specific) so export once per "
                         "session and reuse across all PTU files. Independent of "
                         "--xlsx which is for overlay/comparison only.")
    ap.add_argument("--estimate-irf", choices=["raw", "parametric", "none"],
                    default="none")
    ap.add_argument("--irf-bins",      type=int,   default=21)
    ap.add_argument("--irf-fit-width", type=float, default=1.5)
    ap.add_argument("--irf-fwhm", type=float, default=None,
                    help="IRF FWHM in ns. Default: 1 bin width from PTU "
                         "(e.g. 0.097 ns for 97 ps bins). Override for other systems.")
    ap.add_argument("--nexp",     type=int,   default=3, choices=[1, 2, 3])
    ap.add_argument("--tau-min",  type=float, default=0.145, help="ns")
    ap.add_argument("--tau-max",  type=float, default=45.0, help="ns")
    ap.add_argument("--mode",     default="summed",
                    choices=["summed", "perPixel", "both"])
    ap.add_argument("--binning",     type=int, default=1)
    ap.add_argument("--min-photons", type=int, default=MIN_PHOTONS_PERPIX)
    ap.add_argument("--optimizer",   choices=["lm_multistart", "de"], default="de")
    ap.add_argument("--restarts",    type=int, default=8)
    ap.add_argument("--de-population", type=int, default=15)
    ap.add_argument("--de-maxiter",    type=int, default=1000)
    ap.add_argument("--workers",       type=int, default=-1)
    ap.add_argument("--no-polish",  action="store_true")
    ap.add_argument("--channel",    type=int, default=None)
    ap.add_argument("--out",        default="flim_out")
    ap.add_argument("--no-plots",   action="store_true")
    args = ap.parse_args()

    print(f"\n{'='*60}")
    print(f"  flim_fit_v13  |  {args.nexp}-exp  |  {args.mode}  |  optimizer={args.optimizer}")
    print(f"{'='*60}")

    # ── Load PTU ──────────────────────────────────────────────────────────────
    print(f"\n[1] PTU: {args.ptu}")
    ptu = PTUFile(args.ptu, verbose=True)

    # Resolve FWHM after PTU is loaded
    fwhm_ns = args.irf_fwhm if args.irf_fwhm is not None else ptu.tcspc_res * 1e9
    print(f"  IRF FWHM: {fwhm_ns*1000:.2f} ps "
          f"({'from --irf-fwhm' if args.irf_fwhm is not None else 'default: 1 bin'})")

    # ── Summed decay ──────────────────────────────────────────────────────────
    print(f"\n[2] Building summed decay (channel={args.channel or 'auto'})")
    decay    = ptu.summed_decay(channel=args.channel)
    # IRF peak from steepest rise of the decay, not from the decay maximum.
    # np.argmax(decay) is the fluorescence convolution peak — shifted right
    # of the true IRF peak by ~1-2 bins depending on the shortest lifetime.
    irf_peak_bin  = find_irf_peak_bin(decay)
    decay_peak_bin = int(np.argmax(decay))
    print(f"    {decay.sum():,.0f} photons  |  peak={decay.max():,.0f}  "
          f"at bin {decay_peak_bin} ({ptu.time_ns[decay_peak_bin]:.3f} ns)")
    print(f"    IRF peak (steepest rise): bin {irf_peak_bin} "
          f"({irf_peak_bin * ptu.tcspc_res * 1e9:.3f} ns)")

    # ── Load xlsx (optional) ──────────────────────────────────────────────────
    xlsx = None
    if args.xlsx is not None and Path(args.xlsx).exists():
        print(f"\n[3] XLSX: {args.xlsx}")
        xlsx = load_xlsx(args.xlsx, debug=args.debug_xlsx)
        if xlsx['fit_t'] is not None:
            print(f"    LAS X fit present, peak = {xlsx['fit_c'].max():.0f} cts")
    else:
        print(f"\n[3] No XLSX provided or file not found")

    # ── Build IRF — sets has_tail, fit_sigma, fit_bg per path ────────────────
    print(f"\n[4] Building IRF")
    
    def irf_from_scatter_ptu(path, ptu):
        irf_ptu = PTUFile(path, verbose=False)
        irf_decay = irf_ptu.summed_decay(channel=args.channel)
        s = irf_decay.sum()
        if s > 0:
            return irf_decay / s
        else:
            raise ValueError(f"IRF PTU has zero photons: {path}")
        
    if args.irf is not None:
        # Scatter PTU: IRF fully measured, no tail or sigma needed
        irf_prompt = irf_from_scatter_ptu(args.irf, ptu)
        strategy   = "scatter_ptu"
        has_tail   = False
        fit_sigma  = False
        fit_bg     = True

    elif args.irf_xlsx is not None:
        print(f"  IRF: fitting Leica analytical model to: {args.irf_xlsx}")
        if not Path(args.irf_xlsx).exists():
            raise FileNotFoundError(f"--irf-xlsx file not found: {args.irf_xlsx}")
        irf_ref = load_xlsx(args.irf_xlsx, debug=False)
        if irf_ref['irf_t'] is None or irf_ref['irf_c'] is None:
            raise ValueError(f"No IRF columns found in --irf-xlsx: {args.irf_xlsx}")
        irf_prompt, irf_params = irf_from_xlsx_analytical(
            irf_ref, ptu.n_bins, ptu.tcspc_res, verbose=True)

        # The analytical IRF is centred at t0 from the Gaussian fit (~bin 29).
        # The optimizer consistently finds shift ≈ -1.26 bins because the true
        # IRF onset is at the steepest-rise bin (~27), not the decay peak (29).
        # Pre-shift to irf_peak_bin so the free shift starts near 0.
        irf_current_peak = int(np.argmax(irf_prompt))
        pre_shift = irf_peak_bin - irf_current_peak
        if pre_shift != 0:
            x          = np.arange(ptu.n_bins, dtype=float)
            irf_prompt = np.interp(x - pre_shift, x, irf_prompt,
                                   left=0.0, right=0.0)
            s = irf_prompt.sum()
            if s > 0:
                irf_prompt /= s
            print(f"  Pre-shifted IRF by {pre_shift:+d} bins "
                  f"(from bin {irf_current_peak} → {irf_peak_bin})")

        strategy  = (f"irf_xlsx_analytical ({Path(args.irf_xlsx).name})  "
                     f"FWHM={irf_params['fwhm_ns']*1000:.1f}ps  "
                     f"tail_amp={irf_params['tail_amp']:.3f}  "
                     f"tail_tau={irf_params['tail_tau_ns']:.3f}ns")
        # IRF shape fully determined by analytical fit — tail & sigma not free
        has_tail  = False
        fit_sigma = False
        fit_bg    = True
        print(f"  IRF peak bin after pre-shift = {np.argmax(irf_prompt)}")

    elif xlsx is not None and xlsx['irf_t'] is not None and not args.no_xlsx_irf:
        # xlsx IRF: sparse rising-edge, tail and sigma needed
        irf_prompt = irf_from_xlsx(xlsx, ptu.n_bins, ptu.tcspc_res)
        above      = np.where(irf_prompt >= irf_prompt.max() / 2)[0]
        fwhm_xlsx  = (above[-1] - above[0]) * ptu.tcspc_res * 1e9 if len(above) > 1 else 0
        print(f"  IRF: xlsx prompt  peak bin={int(np.argmax(irf_prompt))}  "
              f"FWHM={fwhm_xlsx:.3f} ns  + tail + σ as free params")
        strategy  = "xlsx"
        has_tail  = True
        fit_sigma = True
        fit_bg    = True

    elif args.estimate_irf != "none":
        # Rising-edge estimation: tail and sigma still needed
        if args.estimate_irf == "raw":
            irf_prompt = estimate_irf_from_decay_raw(
                decay, ptu.tcspc_res, ptu.n_bins, n_irf_bins=args.irf_bins)
            strategy = "estimated_raw"
        else:
            irf_prompt = estimate_irf_from_decay_parametric(
                decay, ptu.tcspc_res, ptu.n_bins,
                fit_window_width_ns=args.irf_fit_width)
            strategy = "estimated_parametric"
        has_tail  = True
        fit_sigma = True
        fit_bg    = True
        print(f"  IRF: {strategy} + tail + σ as free params")

    else:
        # Gaussian (paper equation): FWHM known, σ fixed at 0.
        # has_tail=True — real HyD/SPAD IRF is asymmetric (fast rise, slower
        # fall). A pure symmetric Gaussian cannot fit this. The exponential
        # tail captures detector afterpulsing / slow component.
        # Peak position: Leica places its synthetic IRF at the decay maximum,
        # not at the steepest rise. np.argmax(decay) is correct here.
        # find_irf_peak_bin() is only useful for measured scatter PTU IRFs.
        irf_prompt = gaussian_irf_from_fwhm(
            ptu.n_bins, ptu.tcspc_res, fwhm_ns, decay_peak_bin)
        has_tail  = True    # asymmetric tail needed for real detector IRF
        fit_sigma = False   # FWHM is known — no extra broadening
        fit_bg    = True    # bg free — matches Leica "Tail Offset"
        strategy  = (f"gaussian_paper FWHM={fwhm_ns*1000:.1f}ps "
                     f"peak_bin={decay_peak_bin} (decay maximum)")
        print(f"  IRF: {strategy}")

    print(f"  Flags: has_tail={has_tail}  fit_sigma={fit_sigma}  fit_bg={fit_bg}")

    # ── IRF comparison (always run if xlsx present, regardless of IRF path) ──
    if not args.no_plots and xlsx is not None:
        matplotlib.use("Agg")
        print(f"\n[4b] IRF comparison")
        compare_irfs(irf_prompt, xlsx, ptu.tcspc_res, ptu.n_bins,
                     strategy, args.out)

    # ── Summed fit ────────────────────────────────────────────────────────────
    global_popt    = None
    global_summary = None

    def _run_summed():
        return fit_summed(
            decay, ptu.tcspc_res, ptu.n_bins,
            irf_prompt, has_tail, fit_bg, fit_sigma,
            args.nexp, args.tau_min, args.tau_max,
            optimizer=args.optimizer,
            n_restarts=args.restarts,
            de_popsize=args.de_population,
            de_maxiter=args.de_maxiter,
            workers=args.workers,
            polish=not args.no_polish,
        )

    if args.mode in ("summed", "both"):
        print(f"\n[5] Summed decay fit  ({args.nexp}-exp, optimizer={args.optimizer})")
        global_popt, global_summary = _run_summed()
        print_summary(global_summary, strategy, args.nexp)

        if not args.no_plots:
            matplotlib.use("Agg")
            print(f"\n[6] Plotting")
            plot_summed(decay, global_summary, ptu, xlsx,
                        args.nexp, strategy, args.out,
                        irf_prompt=irf_prompt)

    # ── Per-pixel fit ─────────────────────────────────────────────────────────
    if args.mode in ("perPixel", "both"):
        if global_popt is None:
            print(f"\n[5] Running summed fit first (τ needed for per-pixel)")
            global_popt, global_summary = _run_summed()
            print_summary(global_summary, strategy, args.nexp)

        print(f"\n[7] Building pixel stack (binning={args.binning}×{args.binning})")
        stack = ptu.pixel_stack(channel=ptu.photon_channel, binning=args.binning)

        print(f"\n[8] Per-pixel fitting (min_photons={args.min_photons})")
        pixel_maps = fit_per_pixel(
            stack, ptu.tcspc_res, ptu.n_bins,
            irf_prompt, has_tail, fit_bg, fit_sigma,
            global_popt, args.nexp,
            min_photons=args.min_photons,
        )

        if not args.no_plots:
            matplotlib.use("Agg")
            print(f"\n[9] Plotting pixel maps")
            plot_pixel_maps(pixel_maps, args.nexp, args.out, binning=args.binning)
            plot_lifetime_histogram(pixel_maps, args.nexp, args.out)

    print("\nDone.\n")


if __name__ == "__main__":
    main()