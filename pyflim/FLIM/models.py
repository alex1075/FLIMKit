import numpy as np
from ..FLIM.irf_tools import build_full_irf

def _exponential_kernel(tcspc_res, n_bins, taus, amps, bg):
    t = np.arange(n_bins, dtype=float) * tcspc_res
    return sum(a * np.exp(-t / max(tau, 1e-15))
               for a, tau in zip(amps, taus)) + bg

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


class _DECostLogTau(_DECost):
    """DE cost with tau parameterised in log10 space.

    The first ``n_exp`` elements of *params* are ``log10(tau/s)``.
    They are converted back to linear tau before calling the
    reconvolution model.  This gives the DE sampler equal
    exploration weight across all decades of lifetime.
    """

    def __call__(self, params):
        params_lin = np.array(params, dtype=float)
        params_lin[:self.n_exp] = 10.0 ** params_lin[:self.n_exp]
        return super().__call__(params_lin)
    
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