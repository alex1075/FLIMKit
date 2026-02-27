import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from code.PTU.reader import PTUFile

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

def irf_from_scatter_ptu(path: str, ptu_ref: PTUFile) -> np.ndarray:
    """Load a scatter/reflection PTU as measured IRF. Returns normalised array."""
    scatter = PTUFile(path, verbose=False)
    decay   = scatter.summed_decay()
    decay   = decay[:ptu_ref.n_bins]
    s       = decay.sum()
    if s == 0:
        raise ValueError(f"Scatter PTU {path!r} has no photons.")
    print(f"  IRF from scatter PTU: {s:,.0f} photons")
    return decay / s

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