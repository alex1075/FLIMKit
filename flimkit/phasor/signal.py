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
    