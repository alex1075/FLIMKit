import numpy as np
import pandas as pd
from phasorpy.plot import plot_phasor
from ..PTU.tools import signal_from_PTUFile
from phasorpy.phasor import phasor_from_signal
from phasorpy.phasor import phasor_to_polar, phasor_transform

def return_phasor_from_PTUFile(ptu_file):
    """Read a .ptu file, extract the signal, and compute the phasor coordinates.
    Args:
        ptu_file: Path to the .ptu file containing FLIM data.
    Returns:
        mean: 2D array of mean photon arrival times (in ns)
        real: 2D array of phasor real coordinates
        imag: 2D array of phasor imaginary coordinates
    """
    signal = signal_from_PTUFile(ptu_file, dtype=np.uint32, binning=4)
    mean, real, imag = phasor_from_signal(signal, axis='H')
    return mean, real, imag

def get_phasor_irf(irf_xlsx):
    """Read IRF data from an Excel file and return time and counts arrays.
    Args:
        irf_xlsx: Path to the Excel file containing IRF data.
    Returns:
            irf_time_ns: Array of IRF time points (in ns).
            irf_counts: Array of IRF counts.
    """
    df = pd.read_excel(irf_xlsx, sheet_name='Fit', header=None)
    irf_time_ns = pd.to_numeric(df.iloc[2:, 2], errors='coerce').dropna().values
    irf_counts  = pd.to_numeric(df.iloc[2:, 3], errors='coerce').dropna().values
    return irf_time_ns, irf_counts

def calibrate_signal_with_irf(signal, real, imag, irf_time_ns, irf_counts, frequency):
    """Calibrate phasor coordinates using the IRF data.
    Args:
        real: 2D array of phasor real coordinates to be calibrated.
        imag: 2D array of phasor imaginary coordinates to be calibrated.
        irf_time_ns: Array of IRF time points (in ns).
        irf_counts: Array of IRF counts.
        frequency: Modulation frequency (in MHz) used for the FLIM acquisition.
    Returns:
        real_cal: 2D array of calibrated phasor real coordinates.
        imag_cal: 2D array of calibrated phasor imaginary coordinates.
    """
    signal_time_ns = signal.coords['H'].values
    irf_on_signal = np.interp(signal_time_ns, irf_time_ns, irf_counts, left=0, right=0)

    # ── Compute the IRF phasor (should be near (1,0) for a delta function) ──
    mean_irf, real_irf, imag_irf = phasor_from_signal(
        irf_on_signal[np.newaxis, :], axis=-1
    )
    phase_irf, mod_irf = phasor_to_polar(real_irf.ravel(), imag_irf.ravel())
    real_cal, imag_cal = phasor_transform(real, imag, -phase_irf[0], 1.0 / mod_irf[0],)
    return real_cal, imag_cal

def plot_calibrated_phasor(real, imag, signal):
    """Plot the calibrated phasor coordinates.
    Args:
        real: 2D array of calibrated phasor real coordinates.
        imag: 2D array of calibrated phasor imaginary coordinates.
        signal: The original signal used to extract the phasor coordinates (for frequency info).
    """
    frequency = signal.attrs['frequency']
    plot_phasor(
        real,
        imag,
        frequency=frequency,
        title='Calibrated, filtered phasor coordinates',
    )

def calibrate_signal_with_machine_irf(signal, real, imag, machine_irf_npy: str,
                                       frequency: float):
    """Calibrate phasor coordinates using a pre-built machine IRF (.npy).

    The machine IRF is stored as a 1-D array sampled at the PTU time resolution.
    It is interpolated onto the signal time axis then treated identically to the
    XLSX IRF in calibrate_signal_with_irf.

    Parameters
    ----------
    signal : xarray.DataArray
        TCSPC signal with 'H' coordinate in ns.
    real, imag : ndarray
        Uncalibrated per-pixel phasor coordinates.
    machine_irf_npy : str
        Path to the .npy file produced by build_machine_irf_from_folder.
    frequency : float
        Laser repetition frequency in MHz.

    Returns
    -------
    real_cal, imag_cal : ndarray
        Calibrated phasor coordinates.
    """
    import json
    from pathlib import Path

    npy_path  = Path(machine_irf_npy)
    meta_path = npy_path.with_name(npy_path.stem + '_meta.json')

    irf_arr = np.load(str(npy_path))

    # Load tcspc_res from companion _meta.json if available, else infer from
    # the signal time axis spacing.
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        tcspc_res_ns = meta.get('tcspc_res_ns', None)
    else:
        tcspc_res_ns = None

    signal_time_ns = signal.coords['H'].values        # (n_bins,) in ns
    n_irf = len(irf_arr)

    if tcspc_res_ns is not None:
        irf_time_ns = np.arange(n_irf) * tcspc_res_ns
    else:
        # Assume same time axis length as signal — direct mapping
        irf_time_ns = np.linspace(signal_time_ns[0], signal_time_ns[-1], n_irf)

    return calibrate_signal_with_irf(signal, real, imag,
                                     irf_time_ns, irf_arr, frequency)