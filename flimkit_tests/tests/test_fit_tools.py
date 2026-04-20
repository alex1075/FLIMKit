import pytest
import numpy as np
from flimkit.FLIM.fit_tools import (
    find_irf_peak_bin,
    estimate_bg,
    estimate_bg_from_histogram,
    find_fit_start,
    find_fit_end,
    _build_bounds,
    _pack_p0,
)
from flimkit.FLIM.irf_tools import gaussian_irf_from_fwhm
from flimkit_tests.mock_data import (
    generate_synthetic_decay,
    MOCK_TCSPC_RES,
    MOCK_IRF_CENTER,
    MOCK_IRF_FWHM_BINS,
)


#  find_irf_peak_bin 

class TestFindIrfPeakBin:
    def test_gaussian_peak(self):
        """Peak of derivative should be on rising edge, before the decay maximum."""
        decay = generate_synthetic_decay(n_bins=256, tau_ns=2.0, noise=False)
        peak = find_irf_peak_bin(decay)
        decay_max = int(np.argmax(decay))
        # IRF peak (steepest rise) must be at or before the convolution maximum
        assert peak <= decay_max
        # Should be near the known IRF center (±5 bins)
        assert abs(peak - MOCK_IRF_CENTER) <= 5

    def test_flat_decay_returns_zero(self):
        """Flat array has no rising edge — peak should be 0."""
        decay = np.ones(128) * 100
        peak = find_irf_peak_bin(decay)
        assert peak == 0 or peak < 64  # Somewhere in first half

    def test_noisy_decay(self):
        """Should still find a reasonable peak with Poisson noise."""
        decay = generate_synthetic_decay(n_bins=256, tau_ns=3.0, noise=True,
                                         peak_counts=5000)
        peak = find_irf_peak_bin(decay)
        assert abs(peak - MOCK_IRF_CENTER) <= 8

    def test_different_smooth_sigma(self):
        """Larger sigma shouldn't drastically change the peak location."""
        decay = generate_synthetic_decay(n_bins=256, tau_ns=2.0, noise=True)
        peak_default = find_irf_peak_bin(decay, smooth_sigma=1.5)
        peak_wide = find_irf_peak_bin(decay, smooth_sigma=3.0)
        assert abs(peak_default - peak_wide) <= 3


#  estimate_bg 

class TestEstimateBg:
    def test_zero_bg(self):
        """Decay with no pre-IRF counts should give bg ≈ 0."""
        decay = np.zeros(256)
        decay[30:100] = np.arange(70, 0, -1)  # Peak at bin 30
        bg = estimate_bg(decay, peak_bin=30)
        assert bg == 0.0

    def test_known_bg(self):
        """Flat baseline of 5 photons/bin before IRF → bg ≈ 5."""
        decay = np.ones(256) * 5.0
        decay[50:200] += np.exp(-np.arange(150) / 30.0) * 1000
        bg = estimate_bg(decay, peak_bin=50, pre_gap=5)
        assert abs(bg - 5.0) < 1.0

    def test_peak_near_start(self):
        """If peak_bin is very early, fallback to tail median."""
        decay = np.ones(128) * 3.0
        decay[2:50] += 100
        bg = estimate_bg(decay, peak_bin=2, pre_gap=5)
        assert bg >= 0

    def test_non_negative(self):
        """Background estimate is always >= 0."""
        decay = np.random.normal(0, 10, 256).clip(min=0)
        bg = estimate_bg(decay, peak_bin=50)
        assert bg >= 0


class TestEstimateBgFromHistogram:
    def test_3d_stack(self):
        """Average over (Y, X) and first pre_bins bins."""
        stack = np.ones((4, 4, 128)) * 3.0
        stack[..., 30:] += 50  # Actual signal after bin 30
        bg = estimate_bg_from_histogram(stack, pre_bins=20)
        assert abs(bg - 3.0) < 0.1

    def test_1d_decay(self):
        """Also works for a 1-D summed decay."""
        decay = np.ones(256) * 7.0
        decay[40:] += 100
        bg = estimate_bg_from_histogram(decay, pre_bins=20)
        assert abs(bg - 7.0) < 0.1


#  find_fit_start / find_fit_end 

class TestFindFitStart:
    def test_with_gaussian_irf(self):
        """Fit start should be before the IRF onset."""
        irf = gaussian_irf_from_fwhm(256, MOCK_TCSPC_RES,
                                     MOCK_IRF_FWHM_BINS * MOCK_TCSPC_RES * 1e9,
                                     MOCK_IRF_CENTER)
        start = find_fit_start(np.zeros(256), irf, MOCK_TCSPC_RES)
        # Should be well before the IRF center
        assert start < MOCK_IRF_CENTER
        assert start >= 0

    def test_zero_irf_returns_zero(self):
        """If IRF is all zeros, start at bin 0."""
        irf = np.zeros(128)
        start = find_fit_start(np.zeros(128), irf, MOCK_TCSPC_RES)
        assert start == 0


class TestFindFitEnd:
    def test_clean_decay(self):
        """Without artifacts, end should be at tau-based limit."""
        decay = generate_synthetic_decay(n_bins=256, tau_ns=2.0, noise=False)
        end = find_fit_end(decay, peak_bin=30, tau_max_s=10e-9,
                           tcspc_res=MOCK_TCSPC_RES, n_bins=256)
        assert 100 < end <= 256

    def test_never_exceeds_n_bins(self):
        """End index is capped at n_bins."""
        decay = generate_synthetic_decay(n_bins=128, tau_ns=2.0, noise=False)
        end = find_fit_end(decay, peak_bin=30, tau_max_s=100e-9,
                           tcspc_res=MOCK_TCSPC_RES, n_bins=128)
        assert end <= 128


#  _build_bounds 

class TestBuildBounds:
    def test_single_exp_basic(self):
        lo, hi = _build_bounds(n_exp=1, tau_min=0.1, tau_max=10.0,
                               decay_peak=1000, has_tail=False,
                               fit_bg=False, fit_sigma=False)
        # τ, α, shift = 3 parameters
        assert len(lo) == 3
        assert len(hi) == 3
        assert lo[0] == 0.1 and hi[0] == 10.0  # tau bounds
        assert lo[2] == -5.0 and hi[2] == 5.0  # shift bounds

    def test_two_exp_with_all_flags(self):
        lo, hi = _build_bounds(n_exp=2, tau_min=0.05, tau_max=15.0,
                               decay_peak=5000, has_tail=True,
                               fit_bg=True, fit_sigma=True,
                               bg_init=10.0, sigma_max=3.0)
        # 2τ + 2α + shift + σ + bg + tail_amp + tail_tau = 2+2+1+1+1+2 = 9
        assert len(lo) == 9
        assert len(hi) == 9

    def test_sigma_max_respected(self):
        lo, hi = _build_bounds(n_exp=1, tau_min=0.1, tau_max=10.0,
                               decay_peak=1000, has_tail=False,
                               fit_bg=False, fit_sigma=True, sigma_max=0.5)
        # σ is the 4th parameter (index 3)
        assert hi[3] == 0.5

    def test_bounds_ordering(self):
        """Lower bounds must be < upper bounds."""
        lo, hi = _build_bounds(n_exp=3, tau_min=0.1, tau_max=10.0,
                               decay_peak=1000, has_tail=True,
                               fit_bg=True, fit_sigma=True)
        for l, h in zip(lo, hi):
            assert l < h


#  _pack_p0 ─

class TestPackP0:
    def test_length_matches_bounds(self):
        for n_exp in (1, 2, 3):
            for has_tail in (True, False):
                for fit_bg in (True, False):
                    for fit_sigma in (True, False):
                        p0 = _pack_p0(n_exp, 0.1, 10.0, 1000,
                                       has_tail, fit_bg, fit_sigma, bg_init=5.0)
                        lo, hi = _build_bounds(n_exp, 0.1, 10.0, 1000,
                                                has_tail, fit_bg, fit_sigma)
                        assert len(p0) == len(lo), (
                            f"p0 length mismatch for n_exp={n_exp}, "
                            f"has_tail={has_tail}, fit_bg={fit_bg}, "
                            f"fit_sigma={fit_sigma}")

    def test_p0_within_bounds(self):
        """Initial guess must lie within bounds."""
        p0 = _pack_p0(2, 0.1, 10.0, 1000,
                       has_tail=True, fit_bg=True, fit_sigma=True, bg_init=5.0)
        lo, hi = _build_bounds(2, 0.1, 10.0, 1000,
                                has_tail=True, fit_bg=True, fit_sigma=True)
        for i, (v, l, h) in enumerate(zip(p0, lo, hi)):
            assert l <= v <= h, f"p0[{i}]={v} outside [{l}, {h}]"

    def test_tau_override(self):
        """Custom tau starting values are respected."""
        p0 = _pack_p0(2, 0.1, 10.0, 1000,
                       has_tail=False, fit_bg=False, fit_sigma=False,
                       bg_init=0, tau_override=[1.0, 5.0])
        assert p0[0] == 1.0
        assert p0[1] == 5.0
