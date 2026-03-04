"""Automatic peak-finding on 2-D phasor histograms.

Smooths the (G, S) histogram with a Gaussian kernel, detects local maxima,
and converts them to apparent phase / modulation lifetimes.

Usage
-----
::

    from pyflim.phasor.peaks import find_phasor_peaks, plot_phasor_peaks

    peaks = find_phasor_peaks(real_cal, imag_cal, mean, frequency)
    fig   = plot_phasor_peaks(peaks, real_cal, imag_cal, mean, frequency)
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, label, maximum_filter, center_of_mass

from phasorpy.phasor import phasor_to_polar
from phasorpy.lifetime import phasor_to_apparent_lifetime
from phasorpy.plot import PhasorPlot


# ─────────────────────────────────────────────────────────────
# Peak detection
# ─────────────────────────────────────────────────────────────
def find_phasor_peaks(
    real_cal: np.ndarray,
    imag_cal: np.ndarray,
    mean: np.ndarray,
    frequency: float,
    *,
    min_photons: float = 0.01,
    n_bins: int = 256,
    sigma: float = 3.0,
    neighbourhood: int = 15,
    threshold_frac: float = 0.10,
) -> dict:
    """Find peaks in the 2-D phasor histogram.

    Parameters
    ----------
    real_cal, imag_cal : ndarray
        Calibrated phasor G and S arrays.
    mean : ndarray
        Mean-intensity image (same spatial shape).
    frequency : float
        Modulation frequency in MHz.
    min_photons : float
        Minimum mean-intensity to include a pixel.
    n_bins : int
        Number of histogram bins per axis.
    sigma : float
        Gaussian smoothing σ in histogram-bin units.
    neighbourhood : int
        Local-maximum footprint size in bins.
    threshold_frac : float
        Peaks below this fraction of the global maximum are discarded.

    Returns
    -------
    dict with keys:

    * ``n_peaks`` – number of peaks found
    * ``peak_g``, ``peak_s`` – 1-D arrays of peak G / S coordinates
    * ``tau_phase``, ``tau_mod`` – apparent lifetimes (ns)
    * ``phase``, ``modulation`` – polar phasor coordinates
    * ``on_semicircle`` – boolean array (True if peak lies on the
      universal semicircle within tolerance)
    * ``hist``, ``hist_smooth`` – raw and smoothed 2-D histograms
    * ``g_centers``, ``s_centers`` – bin-centre arrays
    """
    rc = np.asarray(real_cal).squeeze().astype(float)
    ic = np.asarray(imag_cal).squeeze().astype(float)
    mn = np.asarray(mean).squeeze().astype(float)
    valid = (mn >= min_photons) & ~np.isnan(rc)

    g_vals = rc[valid]
    s_vals = ic[valid]

    # 2-D histogram
    hist, g_edges, s_edges = np.histogram2d(
        g_vals, s_vals, bins=n_bins,
        range=[[0, 1], [0, 0.55]])
    g_centers = 0.5 * (g_edges[:-1] + g_edges[1:])
    s_centers = 0.5 * (s_edges[:-1] + s_edges[1:])

    # Smooth + local-maximum detection
    hist_smooth = gaussian_filter(hist, sigma=sigma)
    local_max = maximum_filter(hist_smooth, size=neighbourhood)
    threshold = threshold_frac * hist_smooth.max()
    peak_mask = (hist_smooth == local_max) & (hist_smooth > threshold)

    labelled, n_peaks = label(peak_mask)

    # Weighted centroids → phasor coordinates
    peak_coords = center_of_mass(hist_smooth, labelled,
                                 range(1, n_peaks + 1))
    peak_g = np.empty(n_peaks)
    peak_s = np.empty(n_peaks)
    for i, (ig, is_) in enumerate(peak_coords):
        peak_g[i] = np.interp(ig, np.arange(n_bins), g_centers)
        peak_s[i] = np.interp(is_, np.arange(n_bins), s_centers)

    # Lifetimes & polar coordinates
    tau_phase, tau_mod = phasor_to_apparent_lifetime(
        peak_g, peak_s, frequency)
    phase_vals, mod_vals = phasor_to_polar(peak_g, peak_s)

    # On-semicircle test (distance from centre (0.5, 0) with r = 0.5)
    r = np.sqrt((peak_g - 0.5) ** 2 + peak_s ** 2)
    on_semicircle = np.abs(r - 0.5) < 0.02

    return dict(
        n_peaks=n_peaks,
        peak_g=peak_g,
        peak_s=peak_s,
        tau_phase=tau_phase,
        tau_mod=tau_mod,
        phase=phase_vals,
        modulation=mod_vals,
        on_semicircle=on_semicircle,
        hist=hist,
        hist_smooth=hist_smooth,
        g_centers=g_centers,
        s_centers=s_centers,
    )


# ─────────────────────────────────────────────────────────────
# Pretty-print
# ─────────────────────────────────────────────────────────────
def print_peaks(peaks: dict) -> None:
    """Print a summary table of detected peaks."""
    n = peaks['n_peaks']
    print(f"Found {n} peak(s) in the phasor histogram\n")
    print(f"{'Peak':>4}  {'G':>7}  {'S':>7}  {'τ_φ (ns)':>9}  "
          f"{'τ_m (ns)':>9}  {'phase(°)':>9}  {'mod':>6}")
    print("-" * 68)
    for i in range(n):
        print(f"{i+1:4d}  {peaks['peak_g'][i]:7.4f}  "
              f"{peaks['peak_s'][i]:7.4f}  "
              f"{peaks['tau_phase'][i]:9.3f}  "
              f"{peaks['tau_mod'][i]:9.3f}  "
              f"{np.degrees(peaks['phase'][i]):9.2f}  "
              f"{peaks['modulation'][i]:6.4f}")
    print()
    for i in range(n):
        if peaks['on_semicircle'][i]:
            print(f"Peak {i+1}: ON semicircle → single-exponential, "
                  f"τ ≈ {peaks['tau_phase'][i]:.2f} ns")
        else:
            r = abs(np.sqrt((peaks['peak_g'][i] - 0.5)**2 +
                            peaks['peak_s'][i]**2) - 0.5)
            print(f"Peak {i+1}: INSIDE semicircle (dist={r:.3f}) → "
                  f"multi-exponential mixture")
            print(f"          τ_φ = {peaks['tau_phase'][i]:.2f} ns,  "
                  f"τ_m = {peaks['tau_mod'][i]:.2f} ns")


# ─────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────
def plot_phasor_peaks(
    peaks: dict,
    real_cal: np.ndarray,
    imag_cal: np.ndarray,
    mean: np.ndarray,
    frequency: float,
    *,
    min_photons: float = 0.01,
    figsize: tuple[float, float] = (13, 5.5),
    ax_phasor=None,
) -> plt.Figure | None:
    """Plot the phasor histogram with detected peaks annotated.

    Parameters
    ----------
    peaks : dict
        Output of :func:`find_phasor_peaks`.
    real_cal, imag_cal, mean : ndarray
        Calibrated phasor + intensity data (used for hist2d on phasor plot).
    frequency : float
        Modulation frequency in MHz.
    ax_phasor : matplotlib Axes, optional
        If supplied, peak markers are added to this existing axes **only**
        (no new figure is created) and *None* is returned.
    figsize : tuple
        Size of the newly-created figure (ignored when *ax_phasor* is given).

    Returns
    -------
    fig or None
    """
    n = peaks['n_peaks']
    pg, ps = peaks['peak_g'], peaks['peak_s']
    tp = peaks['tau_phase']

    # ── Annotate an existing axes (e.g. the interactive phasor) ──
    if ax_phasor is not None:
        _annotate_peaks(ax_phasor, n, pg, ps, tp)
        ax_phasor.figure.canvas.draw_idle()
        return None

    # ── Full standalone figure ───────────────────────────────
    rc = np.asarray(real_cal).squeeze().astype(float)
    ic = np.asarray(imag_cal).squeeze().astype(float)
    mn = np.asarray(mean).squeeze().astype(float)
    valid = (mn >= min_photons) & ~np.isnan(rc)
    g_vals, s_vals = rc[valid], ic[valid]

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left – PhasorPlot hist2d with markers
    ax = axes[0]
    pp = PhasorPlot(frequency=frequency, ax=ax,
                    title='Phasor histogram – peak detection')
    pp.hist2d(g_vals, s_vals)
    _annotate_peaks(ax, n, pg, ps, tp)

    # Right – smoothed contour + peaks
    ax2 = axes[1]
    GG, SS = np.meshgrid(peaks['g_centers'], peaks['s_centers'])
    ax2.contourf(GG, SS, peaks['hist_smooth'].T, levels=30, cmap='Blues')
    ax2.contour(GG, SS, peaks['hist_smooth'].T, levels=10,
                colors='grey', linewidths=0.5)
    for i in range(n):
        ax2.plot(pg[i], ps[i], 'r*', ms=18,
                 markeredgecolor='k', markeredgewidth=0.8, zorder=20)
        ax2.annotate(
            f'Peak {i+1}\nτ_φ={tp[i]:.2f} ns\nτ_m={peaks["tau_mod"][i]:.2f} ns',
            xy=(pg[i], ps[i]), xytext=(12, 12),
            textcoords='offset points', fontsize=8, color='red',
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8),
            arrowprops=dict(arrowstyle='->', color='red', lw=1.2))
    theta = np.linspace(0, np.pi, 200)
    ax2.plot(0.5 + 0.5 * np.cos(theta), 0.5 * np.sin(theta),
             'k-', lw=1, alpha=0.5)
    ax2.set_xlabel('G (real)')
    ax2.set_ylabel('S (imag)')
    ax2.set_title('Smoothed contour + peaks')
    ax2.set_xlim(0, 1.05)
    ax2.set_ylim(-0.02, 0.55)
    ax2.set_aspect('equal')

    fig.tight_layout()
    return fig


def _annotate_peaks(ax, n, pg, ps, tp):
    """Add red-star markers + τ annotations to *ax*."""
    artists = []
    for i in range(n):
        (star,) = ax.plot(pg[i], ps[i], 'r*', ms=18,
                          markeredgecolor='k', markeredgewidth=0.8,
                          zorder=20)
        ann = ax.annotate(
            f'τ_φ={tp[i]:.2f} ns',
            xy=(pg[i], ps[i]), xytext=(10, 10),
            textcoords='offset points', fontsize=9, color='red',
            fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='red', lw=1.2))
        artists.extend([star, ann])
    return artists
