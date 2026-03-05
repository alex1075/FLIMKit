"""Ground-Truth Recovery Tests

Generate synthetic data with KNOWN parameters and verify the analysis
pipeline recovers those parameters within tight tolerances.

Covers:
  1. Single-exponential fitting  (known τ)
  2. Bi-exponential fitting      (known τ₁, τ₂, amplitudes)
  3. Stitch → fit round-trip     (known τ after stitching)
  4. Phasor peak recovery        (known G, S from analytical formula)
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import patch
import sys

# Ensure both flimkit_tests (for mock_data) and project root (for flimkit) are on path
_tests_dir = str(Path(__file__).parent.parent)
_project_root = str(Path(__file__).parent.parent.parent)
if _tests_dir not in sys.path:
    sys.path.insert(0, _tests_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
from mock_data import (
    MockPTUFile,
    generate_synthetic_decay,
    generate_synthetic_biexp_decay,
    generate_test_project,
    MOCK_TAU1_NS,
    MOCK_TAU2_NS,
    MOCK_AMP1,
    MOCK_AMP2,
    MOCK_TCSPC_RES,
    MOCK_IRF_CENTER,
    MOCK_IRF_FWHM_BINS,
)


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def _irf(n_bins, tcspc_res, fwhm_ns, center_bin):
    """Build the same Gaussian IRF the fitter will use."""
    from flimkit.FLIM.irf_tools import gaussian_irf_from_fwhm
    return gaussian_irf_from_fwhm(n_bins, tcspc_res, fwhm_ns, center_bin)


# ─────────────────────────────────────────────────────────────
# 1.  Single-exponential fitting
# ─────────────────────────────────────────────────────────────
class TestSingleExpRecovery:
    """Fit a synthetic single-exp decay and check τ matches the input."""

    @pytest.fixture(params=[1.0, 2.5, 5.0], ids=["tau1ns", "tau2p5ns", "tau5ns"])
    def known_tau(self, request):
        return request.param

    def test_single_exp_tau_recovery(self, known_tau):
        """Recovered τ must be within 15 % of the true value."""
        try:
            from flimkit.FLIM.fitters import fit_summed
        except ImportError:
            pytest.skip("fitters module not available")

        n_bins = 256
        tcspc_res = MOCK_TCSPC_RES
        irf_fwhm_ns = MOCK_IRF_FWHM_BINS * tcspc_res * 1e9  # ≈ 0.29 ns

        decay = generate_synthetic_decay(
            n_bins=n_bins,
            tcspc_res=tcspc_res,
            tau_ns=known_tau,
            bg=5.0,
            peak_counts=50_000.0,
            irf_fwhm_bins=MOCK_IRF_FWHM_BINS,
            irf_center_bin=MOCK_IRF_CENTER,
            noise=True,
        )

        # Use the TRUE IRF centre (not argmax(decay), which is shifted by convolution)
        irf = _irf(n_bins, tcspc_res, irf_fwhm_ns, MOCK_IRF_CENTER)

        _popt, summary = fit_summed(
            decay, tcspc_res, n_bins, irf,
            has_tail=False, fit_bg=True, fit_sigma=False,
            n_exp=1, tau_min_ns=0.05, tau_max_ns=15.0,
            optimizer="lm_multistart", n_restarts=5, workers=1,
        )

        recovered = summary['taus_ns'][0]
        rel_err = abs(recovered - known_tau) / known_tau
        assert rel_err < 0.15, (
            f"Single-exp recovery failed: true τ = {known_tau:.2f} ns, "
            f"recovered = {recovered:.3f} ns  (rel err {rel_err:.1%})"
        )


# ─────────────────────────────────────────────────────────────
# 2.  Bi-exponential fitting
# ─────────────────────────────────────────────────────────────
class TestBiExpRecovery:
    """Fit the mock bi-exponential and verify both τ values."""

    def test_biexp_tau_recovery(self):
        """Both lifetimes recovered within 25 % (bi-exp is harder)."""
        try:
            from flimkit.FLIM.fitters import fit_summed
        except ImportError:
            pytest.skip("fitters module not available")

        n_bins = 256
        tcspc_res = MOCK_TCSPC_RES
        irf_fwhm_ns = MOCK_IRF_FWHM_BINS * tcspc_res * 1e9

        decay = generate_synthetic_biexp_decay(
            n_bins=n_bins,
            tcspc_res=tcspc_res,
            tau1_ns=MOCK_TAU1_NS,
            tau2_ns=MOCK_TAU2_NS,
            a1=MOCK_AMP1,
            a2=MOCK_AMP2,
            bg=5.0,
            peak_counts=100_000.0,
            noise=True,
        )

        # Use the TRUE IRF centre (not argmax(decay), which is shifted by convolution)
        irf = _irf(n_bins, tcspc_res, irf_fwhm_ns, MOCK_IRF_CENTER)

        _popt, summary = fit_summed(
            decay, tcspc_res, n_bins, irf,
            has_tail=False, fit_bg=True, fit_sigma=False,
            n_exp=2, tau_min_ns=0.05, tau_max_ns=15.0,
            optimizer="lm_multistart", n_restarts=10, workers=1,
        )

        # summary['taus_ns'] is sorted descending
        taus = summary['taus_ns']
        longer  = taus[0]
        shorter = taus[1]

        rel_long  = abs(longer  - MOCK_TAU2_NS) / MOCK_TAU2_NS
        rel_short = abs(shorter - MOCK_TAU1_NS) / MOCK_TAU1_NS

        assert rel_long < 0.25, (
            f"Long τ recovery: true = {MOCK_TAU2_NS}, got {longer:.3f} "
            f"(err {rel_long:.0%})"
        )
        assert rel_short < 0.25, (
            f"Short τ recovery: true = {MOCK_TAU1_NS}, got {shorter:.3f} "
            f"(err {rel_short:.0%})"
        )

    def test_biexp_amplitude_fractions(self):
        """Amplitude fractions within 0.15 of truth."""
        try:
            from flimkit.FLIM.fitters import fit_summed
        except ImportError:
            pytest.skip("fitters module not available")

        n_bins = 256
        tcspc_res = MOCK_TCSPC_RES
        irf_fwhm_ns = MOCK_IRF_FWHM_BINS * tcspc_res * 1e9

        decay = generate_synthetic_biexp_decay(
            n_bins=n_bins,
            peak_counts=100_000.0,
            noise=True,
        )

        # Use the TRUE IRF centre (not argmax(decay), which is shifted by convolution)
        irf = _irf(n_bins, tcspc_res, irf_fwhm_ns, MOCK_IRF_CENTER)

        _popt, summary = fit_summed(
            decay, tcspc_res, n_bins, irf,
            has_tail=False, fit_bg=True, fit_sigma=False,
            n_exp=2, tau_min_ns=0.05, tau_max_ns=15.0,
            optimizer="lm_multistart", n_restarts=10, workers=1,
        )

        # fractions[0] corresponds to taus_ns[0] (the longer τ)
        frac_long  = summary['fractions'][0]
        frac_short = summary['fractions'][1]

        # Longer τ (3.0 ns) has true amplitude fraction MOCK_AMP2 = 0.4
        assert abs(frac_long - MOCK_AMP2) < 0.15, (
            f"Long-τ fraction: expected ≈{MOCK_AMP2}, got {frac_long:.3f}"
        )
        assert abs(frac_short - MOCK_AMP1) < 0.15, (
            f"Short-τ fraction: expected ≈{MOCK_AMP1}, got {frac_short:.3f}"
        )

    def test_biexp_chi2_reasonable(self):
        """Pearson reduced χ² (Leica convention: Σ(d-m)²/m) should be near 1."""
        try:
            from flimkit.FLIM.fitters import fit_summed
        except ImportError:
            pytest.skip("fitters module not available")

        n_bins = 256
        tcspc_res = MOCK_TCSPC_RES
        irf_fwhm_ns = MOCK_IRF_FWHM_BINS * tcspc_res * 1e9

        decay = generate_synthetic_biexp_decay(peak_counts=100_000.0, noise=True)
        # Use the TRUE IRF centre (not argmax(decay), which is shifted by convolution)
        irf = _irf(n_bins, tcspc_res, irf_fwhm_ns, MOCK_IRF_CENTER)

        _popt, summary = fit_summed(
            decay, tcspc_res, n_bins, irf,
            has_tail=False, fit_bg=True, fit_sigma=False,
            n_exp=2, tau_min_ns=0.05, tau_max_ns=15.0,
            optimizer="lm_multistart", n_restarts=10, workers=1,
        )

        # Leica convention: Pearson χ² (weights = √model)
        rchi2_p = summary['reduced_chi2_pearson']
        assert 0.5 < rchi2_p < 3.0, (
            f"Pearson reduced χ² (Leica) = {rchi2_p:.3f} — "
            f"expected 0.5–3.0 for correct model"
        )

        # Tail-only Pearson should be close to 1
        rchi2_tail = summary['reduced_chi2_tail_pearson']
        assert 0.3 < rchi2_tail < 3.0, (
            f"Pearson tail χ² (Leica) = {rchi2_tail:.3f} — "
            f"expected 0.3–3.0 for correct model"
        )


# ─────────────────────────────────────────────────────────────
# 3.  Stitch → fit  (end-to-end with ground truth)
# ─────────────────────────────────────────────────────────────
class TestStitchAndFitRecovery:
    """Stitch mock tiles, sum the decay, fit, and check τ."""

    def test_stitched_biexp_recovery(self):
        """τ values recovered from stitched mosaic match MockPTUFile truth."""
        try:
            from flimkit.PTU.stitch import stitch_flim_tiles, load_flim_for_fitting
            from flimkit.FLIM.fitters import fit_summed
            from flimkit.FLIM.irf_tools import gaussian_irf_from_fwhm
        except ImportError as e:
            pytest.skip(f"Module not available: {e}")

        with tempfile.TemporaryDirectory() as tmp:
            project = generate_test_project(
                Path(tmp), roi_name="R 2", n_tiles=4, layout="2x2"
            )
            for npy in project['ptu_dir'].glob('*.npy'):
                npy.rename(npy.with_suffix('.ptu'))

            out = project['base_dir'] / "stitched"

            with patch('flimkit.PTU.reader.PTUFile', MockPTUFile):
                stitch_flim_tiles(
                    xlif_path=project['xlif_path'],
                    ptu_dir=project['ptu_dir'],
                    output_dir=out,
                    ptu_basename=project['roi_name'],
                    verbose=False,
                )

            stack, tcspc_res, n_bins = load_flim_for_fitting(out, load_to_memory=True)
            decay = stack.sum(axis=(0, 1))

            irf_fwhm_ns = MOCK_IRF_FWHM_BINS * tcspc_res * 1e9
            irf = gaussian_irf_from_fwhm(n_bins, tcspc_res, irf_fwhm_ns, MOCK_IRF_CENTER)

            _popt, summary = fit_summed(
                decay, tcspc_res, n_bins, irf,
                has_tail=False, fit_bg=True, fit_sigma=False,
                n_exp=2, tau_min_ns=0.05, tau_max_ns=15.0,
                optimizer="lm_multistart", n_restarts=10, workers=1,
            )

            taus = summary['taus_ns']
            longer  = taus[0]
            shorter = taus[1]

            # After stitching 4 tiles the SNR is high → 30 % tolerance
            assert abs(longer  - MOCK_TAU2_NS) / MOCK_TAU2_NS < 0.30, (
                f"Long τ after stitch: expected ~{MOCK_TAU2_NS}, got {longer:.3f}"
            )
            assert abs(shorter - MOCK_TAU1_NS) / MOCK_TAU1_NS < 0.30, (
                f"Short τ after stitch: expected ~{MOCK_TAU1_NS}, got {shorter:.3f}"
            )


# ─────────────────────────────────────────────────────────────
# 4.  Phasor ground-truth recovery (tighter than before)
# ─────────────────────────────────────────────────────────────
class TestPhasorGroundTruth:
    """Synthetic single-exp phasor → the peak must land on the semicircle
    at the analytically known (G, S) and the phase lifetime must match."""

    @pytest.fixture
    def phasor_truth(self):
        """Generates phasor data for τ = 2.5 ns, freq = 40 MHz."""
        rng = np.random.default_rng(42)
        shape = (128, 128)          # bigger → tighter histogram peak
        tau_ns = 2.5
        frequency = 40.0            # MHz
        omega = 2 * np.pi * frequency * 1e-3  # rad/ns
        g = 1 / (1 + (omega * tau_ns) ** 2)
        s = omega * tau_ns / (1 + (omega * tau_ns) ** 2)
        return dict(
            real_cal=rng.normal(g, 0.01, shape),
            imag_cal=rng.normal(s, 0.01, shape),
            mean=rng.uniform(10, 80, shape),
            frequency=frequency,
            g_true=g,
            s_true=s,
            tau_ns=tau_ns,
        )

    def test_peak_location_tight(self, phasor_truth):
        """Detected peak within 0.03 of the analytical (G, S)."""
        try:
            from flimkit.phasor.peaks import find_phasor_peaks
        except ImportError:
            pytest.skip("phasor.peaks not available")

        peaks = find_phasor_peaks(
            phasor_truth['real_cal'],
            phasor_truth['imag_cal'],
            phasor_truth['mean'],
            phasor_truth['frequency'],
        )
        assert peaks['n_peaks'] >= 1

        dists = [
            np.sqrt((peaks['peak_g'][i] - phasor_truth['g_true']) ** 2 +
                    (peaks['peak_s'][i] - phasor_truth['s_true']) ** 2)
            for i in range(peaks['n_peaks'])
        ]
        best = min(dists)
        assert best < 0.03, (
            f"Peak distance = {best:.4f} (threshold 0.03). "
            f"Expected G={phasor_truth['g_true']:.4f}, S={phasor_truth['s_true']:.4f}"
        )

    def test_phase_lifetime_tight(self, phasor_truth):
        """Phase lifetime from nearest peak within 10 % of true τ."""
        try:
            from flimkit.phasor.peaks import find_phasor_peaks
        except ImportError:
            pytest.skip("phasor.peaks not available")

        peaks = find_phasor_peaks(
            phasor_truth['real_cal'],
            phasor_truth['imag_cal'],
            phasor_truth['mean'],
            phasor_truth['frequency'],
        )

        # Find index of closest peak
        dists = [
            np.sqrt((peaks['peak_g'][i] - phasor_truth['g_true']) ** 2 +
                    (peaks['peak_s'][i] - phasor_truth['s_true']) ** 2)
            for i in range(peaks['n_peaks'])
        ]
        idx = int(np.argmin(dists))
        tau_phase = peaks['tau_phase'][idx]

        rel_err = abs(tau_phase - phasor_truth['tau_ns']) / phasor_truth['tau_ns']
        assert rel_err < 0.10, (
            f"Phase τ = {tau_phase:.3f} ns vs true {phasor_truth['tau_ns']} ns "
            f"(err {rel_err:.1%})"
        )

    def test_peak_on_semicircle(self, phasor_truth):
        """Nearest peak is flagged as on the universal semicircle."""
        try:
            from flimkit.phasor.peaks import find_phasor_peaks
        except ImportError:
            pytest.skip("phasor.peaks not available")

        peaks = find_phasor_peaks(
            phasor_truth['real_cal'],
            phasor_truth['imag_cal'],
            phasor_truth['mean'],
            phasor_truth['frequency'],
        )
        dists = [
            np.sqrt((peaks['peak_g'][i] - phasor_truth['g_true']) ** 2 +
                    (peaks['peak_s'][i] - phasor_truth['s_true']) ** 2)
            for i in range(peaks['n_peaks'])
        ]
        idx = int(np.argmin(dists))
        assert peaks['on_semicircle'][idx], (
            f"Peak {idx} at G={peaks['peak_g'][idx]:.4f}, "
            f"S={peaks['peak_s'][idx]:.4f} not flagged on semicircle"
        )

    def test_modulation_lifetime_tight(self, phasor_truth):
        """Modulation lifetime from nearest peak within 10 % of true τ."""
        try:
            from flimkit.phasor.peaks import find_phasor_peaks
        except ImportError:
            pytest.skip("phasor.peaks not available")

        peaks = find_phasor_peaks(
            phasor_truth['real_cal'],
            phasor_truth['imag_cal'],
            phasor_truth['mean'],
            phasor_truth['frequency'],
        )
        dists = [
            np.sqrt((peaks['peak_g'][i] - phasor_truth['g_true']) ** 2 +
                    (peaks['peak_s'][i] - phasor_truth['s_true']) ** 2)
            for i in range(peaks['n_peaks'])
        ]
        idx = int(np.argmin(dists))
        tau_mod = peaks['tau_mod'][idx]

        rel_err = abs(tau_mod - phasor_truth['tau_ns']) / phasor_truth['tau_ns']
        assert rel_err < 0.10, (
            f"Mod τ = {tau_mod:.3f} ns vs true {phasor_truth['tau_ns']} ns "
            f"(err {rel_err:.1%})"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
