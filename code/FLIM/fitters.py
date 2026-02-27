import time
import numpy as np
from scipy.optimize import least_squares, differential_evolution, nnls
from scipy.stats.distributions import chi2 as chi2_dist
from code.FLIM.irf_tools import build_full_irf
from code.FLIM.fit_tools import estimate_bg, find_fit_end, _build_bounds, _pack_p0
from code.FLIM.models import reconvolution_model, _DECost
from code.configs import MIN_PHOTONS_PERPIX

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