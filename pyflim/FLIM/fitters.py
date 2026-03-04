import time
import numpy as np
from tqdm import tqdm
from scipy.optimize import least_squares, differential_evolution, nnls
from scipy.stats.distributions import chi2 as chi2_dist
from ..FLIM.irf_tools import build_full_irf
from ..FLIM.fit_tools import estimate_bg, find_fit_end, _build_bounds, _pack_p0
from ..FLIM.models import (reconvolution_model, _DECost, _DECostLogTau,
                           _DECostPoisson, _DECostPoissonLogTau)
from ..configs import MIN_PHOTONS_PERPIX

def fit_summed(decay, tcspc_res, n_bins, irf_prompt,
               has_tail, fit_bg, fit_sigma,
               n_exp, tau_min_ns, tau_max_ns,
               optimizer="de", n_restarts=8,
               de_popsize=15, de_maxiter=1000,
               workers=-1, polish=True,
               cost_function="chi2") -> tuple[np.ndarray, dict]:
    """Fit summed FLIM decay via reconvolution.

    Parameters
    ----------
    cost_function : str, optional
        ``'poisson'`` — Poisson deviance / C-statistic (recommended, default).
        ``'chi2'``    — Neyman chi-squared (legacy: weighted least-squares on
                        normalised decay).
    """

    tau_min  = tau_min_ns * 1e-9
    tau_max  = tau_max_ns * 1e-9

    # ---- Optionally normalise (legacy chi2 path only) ----
    if cost_function == "chi2":
        scale = decay.max()
        if scale <= 0:
            raise ValueError("Decay has zero maximum – cannot normalise.")
        decay_work = decay / scale          # peak = 1
    elif cost_function == "poisson":
        scale = 1.0
        decay_work = decay.astype(float)    # raw counts
    else:
        raise ValueError(f"Unknown cost_function: {cost_function!r}")

    peak_bin = int(np.argmax(decay_work))
    bg_init  = estimate_bg(decay_work, peak_bin)
    bg_fixed = bg_init if not fit_bg else 0.0

    fit_end   = find_fit_end(decay_work, peak_bin, tau_max, tcspc_res, n_bins)
    fit_start = 1    # match Leica: skip bin 0

    leica_fit_end = int(round(44.9455 / (tcspc_res * 1e9)))
    fit_end = min(fit_end, leica_fit_end)

    bg_upper = max(bg_init * 0.75, 10.0)

    print(f"  Cost function: {cost_function}")
    print(f"  bg initial guess = {bg_init:.3f}"
          f"{' (normalised)' if cost_function == 'chi2' else ' cts/bin'}"
          f", upper bound = {bg_upper:.3f} "
          f"({'free param' if fit_bg else 'fixed'})")
    print(f"  σ broadening: {'free param' if fit_sigma else 'fixed at 0'}")
    print(f"  Fit window: bins {fit_start}–{fit_end} "
          f"({fit_start*tcspc_res*1e9:.2f}–{fit_end*tcspc_res*1e9:.2f} ns), "
          f"{fit_end-fit_start} bins")

    lo, hi  = _build_bounds(n_exp, tau_min, tau_max, decay_work.max(),
                             has_tail, fit_bg, fit_sigma,
                             bg_init=bg_init, bg_upper=bg_upper)
    bounds  = list(zip(lo, hi))

    # ---- Define residual / cost functions ----
    if cost_function == "chi2":
        weights = np.sqrt(np.maximum(decay_work[fit_start:fit_end], 1e-8))

        def residuals(params):
            model_vals = reconvolution_model(
                params, tcspc_res, n_bins, irf_prompt,
                n_exp, bg_fixed, has_tail, fit_bg, fit_sigma)
            return (model_vals[fit_start:fit_end]
                    - decay_work[fit_start:fit_end]) / weights

    else:  # poisson
        def residuals(params):
            """Signed Poisson deviance residuals for LM."""
            model_vals = reconvolution_model(
                params, tcspc_res, n_bins, irf_prompt,
                n_exp, bg_fixed, has_tail, fit_bg, fit_sigma)
            n = decay_work[fit_start:fit_end]
            m = np.maximum(model_vals[fit_start:fit_end], 1e-10)
            dev = m - n
            pos = n > 0
            dev[pos] += n[pos] * np.log(n[pos] / m[pos])
            dev = np.maximum(dev, 0.0)       # numerical guard
            r = np.sqrt(2.0 * dev)
            r[m < n] *= -1                   # sign = data > model
            return r

    if optimizer == "lm_multistart":
        rng       = np.random.default_rng(42)
        best_res  = None
        best_cost = np.inf

        for i in range(n_restarts + 1):
            tau_ov = None if i == 0 else np.sort(
                np.exp(rng.uniform(np.log(tau_min*1.001),
                                   np.log(tau_max*0.999), n_exp)))
            p0 = _pack_p0(n_exp, tau_min, tau_max, float(decay_work.max()),
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
        popt_work = best_res.x
        message   = best_res.message

    elif optimizer == "de":
        print(f"  Differential evolution: popsize={de_popsize}, "
              f"maxiter={de_maxiter}, workers={workers}")

        # --- Log-tau reparameterisation for DE ---
        bounds_log = list(bounds)
        for i in range(n_exp):
            lo_tau, hi_tau = bounds[i]
            bounds_log[i] = (np.log10(lo_tau), np.log10(hi_tau))

        if cost_function == "poisson":
            cost_fn = _DECostPoissonLogTau(
                tcspc_res, n_bins, irf_prompt, n_exp, bg_fixed,
                has_tail, fit_bg, fit_sigma,
                fit_start, fit_end, decay_work)
        else:
            cost_fn = _DECostLogTau(
                tcspc_res, n_bins, irf_prompt, n_exp, bg_fixed,
                has_tail, fit_bg, fit_sigma,
                fit_start, fit_end, decay_work, weights)

        de_res = differential_evolution(
            cost_fn, bounds=bounds_log,
            maxiter=de_maxiter, popsize=de_popsize,
            workers=workers, seed=42,
            updating='deferred' if workers != 1 else 'immediate',
            init='sobol',
            disp=False)

        # Convert log₁₀(τ) → τ in the result
        popt_work = de_res.x.copy()
        popt_work[:n_exp] = 10.0 ** popt_work[:n_exp]
        message = f"DE success={de_res.success}, fun={de_res.fun:.4e}"

        if polish:
            print("  Running final LM polish...")
            pol = least_squares(residuals, popt_work, bounds=(lo, hi), method="trf",
                                max_nfev=5000, ftol=1e-13, xtol=1e-13, gtol=1e-13)
            popt_work = pol.x
            message  += f"; polished cost={pol.cost:.4e}"
    else:
        raise ValueError(f"Unknown optimizer: {optimizer!r}")

    # ---- Rescale amplitudes and background back to original units ----
    # (only needed for chi2 path which normalises; poisson works on raw counts)
    popt_original = popt_work.copy()
    if cost_function == "chi2":
        popt_original[n_exp:2*n_exp] *= scale
        if fit_bg:
            bg_idx = 2*n_exp + 1
            if fit_sigma:
                bg_idx += 1
            popt_original[bg_idx] *= scale

    summary = _make_summary(popt_original, decay, tcspc_res, n_bins, irf_prompt,
                            n_exp, bg_fixed * scale if cost_function == "chi2" else bg_fixed,
                            has_tail, fit_bg, fit_sigma,
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
    d_win   = decay[fit_start:fit_end].astype(float)
    m_win   = model[fit_start:fit_end]

    # Neyman chi-squared: sum((d - m)² / max(d, 1))
    sigma_w = np.sqrt(np.maximum(d_win, 1.0))
    chi2    = float(np.sum(((d_win - m_win) / sigma_w)**2))
    dof     = max((fit_end - fit_start) - len(popt), 1)
    rchi2   = chi2 / dof
    p_val   = float(1 - chi2_dist.cdf(chi2, df=dof))
    resid   = (decay - model) / np.sqrt(np.maximum(model, 1.0))

    # Tail-only chi2_r: exclude rising edge
    peak_bin_loc = int(np.argmax(decay[fit_start:fit_end])) + fit_start
    tail_start   = peak_bin_loc + max(1, int(0.05 * (fit_end - peak_bin_loc)))
    d_tail  = decay[tail_start:fit_end].astype(float)
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
        # Fixed tau (constant across all pixels) for component analysis
        maps[f"tau_{i+1}"]   = np.full((ny, nx), taus_fixed[i] * 1e9)  # in ns
        # Alias for save_weighted_tau_images compatibility
        maps[f"a{i+1}"]      = maps[f"alpha_{i+1}"]

    fitted = skipped = 0
    print(f"  Per-pixel fitting: {ny}×{nx}={ny*nx} pixels "
          f"(τ fixed, amplitudes + bg free) …")
    t0 = time.time()

    for yi in tqdm(range(ny), desc='  Per-pixel rows'):
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