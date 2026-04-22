"""Tests for flimkit/phasor/fret.py — all six stages."""

import numpy as np
import pytest

from flimkit.phasor.fret import (
    FRETChannelData,
    FRETModelParameters,
    FRETBounds,
    FRETResult,
    _require_phasorpy_fret_api,
    _single_lifetime_phasor,
    _fret_donor_phasor,
    _fret_acceptor_phasor,
    predict_fret_trajectory,
    fit_donor_fret,
    fit_joint_fret,
    map_fret_efficiency,
    plot_fret_trajectory,
    plot_fret_fit,
)
from flimkit_tests.mock_data import (
    MOCK_FRET_FREQ,
    MOCK_FRET_TAU_D,
    MOCK_FRET_TAU_A,
    fret_donor_phasor_truth,
    fret_acceptor_phasor_truth,
    generate_fret_donor_image,
    generate_fret_acceptor_image,
)


# Aliases so the rest of the file reads cleanly.
FREQ  = MOCK_FRET_FREQ
TAU_D = MOCK_FRET_TAU_D
TAU_A = MOCK_FRET_TAU_A


# ---------------------------------------------------------------------------
# Mock data ground-truth verification
# (These tests prove the generators produce genuine FRET phasors.)
# ---------------------------------------------------------------------------

class TestFRETMockData:
    """Verify that the mock-data generators produce phasors that actually lie
    at the positions predicted by the phasorpy FRET forward model."""

    @pytest.mark.parametrize("E", [0.0, 0.3, 0.5, 0.8, 1.0])
    def test_donor_centroid_matches_fret_model(self, E):
        """Pixel centroid must equal phasor_from_fret_donor to within noise."""
        donor = generate_fret_donor_image(E, shape=(20, 20), noise=0.002, seed=42)
        g_expected, s_expected = fret_donor_phasor_truth(E)
        # Photon-weighted centroid (uniform weights here since all pixels identical)
        g_obs = float(np.mean(donor.real_cal))
        s_obs = float(np.mean(donor.imag_cal))
        assert g_obs == pytest.approx(g_expected, abs=0.01), (
            f"Donor G centroid {g_obs:.4f} differs from FRET model {g_expected:.4f} "
            f"for E={E}. Data was NOT generated from the FRET model."
        )
        assert s_obs == pytest.approx(s_expected, abs=0.01), (
            f"Donor S centroid {s_obs:.4f} differs from FRET model {s_expected:.4f} "
            f"for E={E}. Data was NOT generated from the FRET model."
        )

    @pytest.mark.parametrize("E", [0.0, 0.3, 0.5, 0.8, 1.0])
    def test_acceptor_centroid_matches_fret_model(self, E):
        """Pixel centroid must equal phasor_from_fret_acceptor to within noise."""
        acceptor = generate_fret_acceptor_image(E, shape=(20, 20), noise=0.002, seed=42)
        g_expected, s_expected = fret_acceptor_phasor_truth(E)
        g_obs = float(np.mean(acceptor.real_cal))
        s_obs = float(np.mean(acceptor.imag_cal))
        assert g_obs == pytest.approx(g_expected, abs=0.01), (
            f"Acceptor G centroid {g_obs:.4f} differs from FRET model {g_expected:.4f} "
            f"for E={E}. Data was NOT generated from the FRET model."
        )
        assert s_obs == pytest.approx(s_expected, abs=0.01)

    def test_donor_phasors_move_along_fret_trajectory(self):
        """As efficiency increases, donor G should increase (quenching shortens lifetime)."""
        g_low, _  = fret_donor_phasor_truth(0.1)
        g_high, _ = fret_donor_phasor_truth(0.9)
        assert g_high > g_low, (
            "Donor FRET trajectory is not monotone in G — "
            "ground-truth function may be broken."
        )

    def test_donor_and_acceptor_phasors_differ(self):
        """Donor and acceptor channels should occupy different phasor positions."""
        dg, ds = fret_donor_phasor_truth(0.5)
        ag, as_ = fret_acceptor_phasor_truth(0.5)
        dist = np.hypot(dg - ag, ds - as_)
        assert dist > 0.05, (
            "Donor and acceptor phasors are suspiciously close — "
            "acceptor generator may be calling the wrong phasorpy function."
        )

    def test_generated_images_have_correct_frequency(self):
        donor = generate_fret_donor_image(0.5)
        acceptor = generate_fret_acceptor_image(0.5)
        assert donor.frequency == FREQ
        assert acceptor.frequency == FREQ

    def test_all_pixels_above_photon_threshold(self):
        donor = generate_fret_donor_image(0.5, mean_photons=50.0)
        assert donor.valid_mask.all(), "Some pixels fell below the photon threshold."


# ---------------------------------------------------------------------------
# Stage 1: Data Contracts
# ---------------------------------------------------------------------------

class TestFRETChannelData:
    def test_valid_construction(self):
        g = np.ones((4, 4))
        d = FRETChannelData(real_cal=g, imag_cal=g.copy(), mean=g.copy(), frequency=80.0)
        assert d.real_cal.shape == (4, 4)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="same shape"):
            FRETChannelData(
                real_cal=np.ones((4, 4)),
                imag_cal=np.ones((3, 4)),
                mean=np.ones((4, 4)),
                frequency=80.0,
            )

    def test_mean_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="same shape"):
            FRETChannelData(
                real_cal=np.ones((4, 4)),
                imag_cal=np.ones((4, 4)),
                mean=np.ones((4, 5)),
                frequency=80.0,
            )

    def test_nonpositive_frequency_raises(self):
        with pytest.raises(ValueError, match="frequency"):
            FRETChannelData(
                real_cal=np.ones((4, 4)),
                imag_cal=np.ones((4, 4)),
                mean=np.ones((4, 4)),
                frequency=0.0,
            )

    def test_valid_mask_excludes_low_photons(self):
        g = np.ones((4, 4))
        mean = np.ones((4, 4))
        mean[0, 0] = 0.0  # below default threshold
        d = FRETChannelData(real_cal=g, imag_cal=g.copy(), mean=mean, frequency=80.0)
        assert not d.valid_mask[0, 0]
        assert d.valid_mask[1, 1]

    def test_valid_mask_excludes_nan(self):
        g = np.ones((4, 4))
        g[2, 2] = np.nan
        d = FRETChannelData(real_cal=g, imag_cal=np.ones((4, 4)), mean=np.ones((4, 4)) * 10, frequency=80.0)
        assert not d.valid_mask[2, 2]

    def test_valid_g_s_are_flattened_valid(self):
        d = generate_fret_donor_image(0.5)
        assert d.valid_g.ndim == 1
        assert d.valid_g.size == d.valid_mask.sum()
        assert d.valid_s.size == d.valid_mask.sum()


class TestFRETModelParameters:
    def test_valid_construction(self):
        p = FRETModelParameters(donor_lifetime=4.0, fret_efficiency=0.5)
        assert p.donor_lifetime == 4.0

    def test_nonpositive_lifetime_raises(self):
        with pytest.raises(ValueError, match="donor_lifetime"):
            FRETModelParameters(donor_lifetime=0.0)

    def test_efficiency_out_of_range_raises(self):
        with pytest.raises(ValueError, match="fret_efficiency"):
            FRETModelParameters(donor_lifetime=4.0, fret_efficiency=1.5)

    def test_fretting_out_of_range_raises(self):
        with pytest.raises(ValueError, match="donor_fretting"):
            FRETModelParameters(donor_lifetime=4.0, donor_fretting=-0.1)

    def test_acceptor_lifetime_optional(self):
        p = FRETModelParameters(donor_lifetime=4.0)
        assert p.acceptor_lifetime is None


class TestFRETBounds:
    def test_donor_only_scipy_length(self):
        b = FRETBounds()
        d = b.donor_only_scipy()
        assert len(d['lb']) == 3
        assert len(d['ub']) == 3

    def test_joint_scipy_length(self):
        b = FRETBounds()
        d = b.joint_scipy()
        assert len(d['lb']) == 6
        assert len(d['ub']) == 6

    def test_custom_bounds(self):
        b = FRETBounds(fret_efficiency=(0.1, 0.9))
        d = b.donor_only_scipy()
        assert d['lb'][0] == pytest.approx(0.1)
        assert d['ub'][0] == pytest.approx(0.9)


class TestFRETResult:
    def test_print_summary_donor_only(self, capsys):
        r = FRETResult(
            fret_efficiency=0.5, donor_fretting=1.0, donor_background=0.0,
            donor_real_model=0.5, donor_imag_model=0.4, residual=1e-5,
        )
        r.print_summary()
        out = capsys.readouterr().out
        assert "0.5000" in out
        assert "converged" in out

    def test_print_summary_joint(self, capsys):
        r = FRETResult(
            fret_efficiency=0.6, donor_fretting=1.0, donor_background=0.0,
            donor_real_model=0.6, donor_imag_model=0.48, residual=1e-6,
            acceptor_real_model=0.1, acceptor_imag_model=0.3,
        )
        r.print_summary()
        out = capsys.readouterr().out
        assert "Acceptor model" in out


# ---------------------------------------------------------------------------
# Stage 2: Compatibility Layer
# ---------------------------------------------------------------------------

class TestRequirePhasorpyFretApi:
    def test_passes_with_installed_phasorpy(self):
        _require_phasorpy_fret_api()  # should not raise


class TestSingleLifetimePhasor:
    def test_unquenched_on_semicircle(self):
        g, s = _single_lifetime_phasor(FREQ, TAU_D)
        assert 0.0 < g < 1.0
        assert 0.0 < s < 0.5
        # Single-exp phasors lie on the semicircle: g^2 + s^2 = g
        assert (g**2 + s**2) == pytest.approx(g, abs=1e-6)

    def test_longer_lifetime_smaller_g(self):
        g_short, _ = _single_lifetime_phasor(FREQ, 1.0)
        g_long, _ = _single_lifetime_phasor(FREQ, 8.0)
        assert g_long < g_short


class TestFretDonorPhasor:
    def test_zero_efficiency_equals_free_donor(self):
        g_fret, s_fret = _fret_donor_phasor(FREQ, TAU_D, fret_efficiency=0.0)
        g_free, s_free = _single_lifetime_phasor(FREQ, TAU_D)
        assert g_fret == pytest.approx(g_free, abs=1e-6)
        assert s_fret == pytest.approx(s_free, abs=1e-6)

    def test_higher_efficiency_increases_g(self):
        g_low, _ = _fret_donor_phasor(FREQ, TAU_D, fret_efficiency=0.1)
        g_high, _ = _fret_donor_phasor(FREQ, TAU_D, fret_efficiency=0.9)
        assert g_high > g_low

    def test_returns_scalars(self):
        g, s = _fret_donor_phasor(FREQ, TAU_D, fret_efficiency=0.5)
        assert np.ndim(g) == 0
        assert np.ndim(s) == 0


class TestFretAcceptorPhasor:
    def test_returns_scalars(self):
        g, s = _fret_acceptor_phasor(FREQ, TAU_D, TAU_A, fret_efficiency=0.5)
        assert np.ndim(g) == 0
        assert np.ndim(s) == 0

    def test_finite_values(self):
        g, s = _fret_acceptor_phasor(FREQ, TAU_D, TAU_A, fret_efficiency=0.5)
        assert np.isfinite(g)
        assert np.isfinite(s)


class TestPredictFretTrajectory:
    def test_donor_trajectory_shape(self):
        t = predict_fret_trajectory(FREQ, TAU_D, n_points=50)
        assert t['donor_g'].shape == (50,)
        assert t['donor_s'].shape == (50,)

    def test_no_acceptor_returns_none(self):
        t = predict_fret_trajectory(FREQ, TAU_D)
        assert t['acceptor_g'] is None
        assert t['acceptor_s'] is None

    def test_acceptor_trajectory_when_supplied(self):
        t = predict_fret_trajectory(FREQ, TAU_D, acceptor_lifetime=TAU_A, n_points=40)
        assert t['acceptor_g'].shape == (40,)
        assert t['acceptor_s'].shape == (40,)

    def test_efficiency_range(self):
        t = predict_fret_trajectory(FREQ, TAU_D, n_points=100)
        assert t['efficiency'][0] == pytest.approx(0.0)
        assert t['efficiency'][-1] == pytest.approx(1.0)

    def test_donor_g_increases_with_efficiency(self):
        t = predict_fret_trajectory(FREQ, TAU_D, n_points=100)
        assert t['donor_g'][-1] > t['donor_g'][0]


# ---------------------------------------------------------------------------
# Stage 3: Donor-Only Solver
# ---------------------------------------------------------------------------

class TestFitDonorFret:
    @pytest.mark.parametrize("E_true", [0.2, 0.5, 0.8])
    def test_recovers_efficiency(self, E_true):
        donor = generate_fret_donor_image(E_true, shape=(10, 10), noise=0.003)
        params = FRETModelParameters(donor_lifetime=TAU_D, fret_efficiency=0.1)
        result = fit_donor_fret(donor, params)
        assert result.fret_efficiency == pytest.approx(E_true, abs=0.02)

    def test_converged(self):
        donor = generate_fret_donor_image(0.5)
        params = FRETModelParameters(donor_lifetime=TAU_D)
        result = fit_donor_fret(donor, params)
        assert result.converged

    def test_model_point_is_finite(self):
        donor = generate_fret_donor_image(0.4)
        params = FRETModelParameters(donor_lifetime=TAU_D)
        result = fit_donor_fret(donor, params)
        assert np.isfinite(result.donor_real_model)
        assert np.isfinite(result.donor_imag_model)

    def test_acceptor_fields_zero_for_donor_only(self):
        donor = generate_fret_donor_image(0.5)
        params = FRETModelParameters(donor_lifetime=TAU_D)
        result = fit_donor_fret(donor, params)
        assert result.donor_bleedthrough == 0.0
        assert result.acceptor_real_model is None

    def test_default_bounds_applied(self):
        donor = generate_fret_donor_image(0.5)
        params = FRETModelParameters(donor_lifetime=TAU_D)
        result = fit_donor_fret(donor, params)
        assert 0.0 <= result.fret_efficiency <= 1.0
        assert 0.0 <= result.donor_fretting <= 1.0

    def test_custom_bounds_respected(self):
        donor = generate_fret_donor_image(0.5)
        params = FRETModelParameters(donor_lifetime=TAU_D)
        bounds = FRETBounds(fret_efficiency=(0.0, 0.3))
        result = fit_donor_fret(donor, params, bounds)
        assert result.fret_efficiency <= 0.3 + 1e-6

    def test_unweighted_mode(self):
        donor = generate_fret_donor_image(0.5)
        params = FRETModelParameters(donor_lifetime=TAU_D)
        result = fit_donor_fret(donor, params, weight_by_photons=False)
        assert result.converged

    def test_low_photon_pixels_excluded(self):
        donor = generate_fret_donor_image(0.5, shape=(6, 6))
        donor.mean[:] = 0.0  # all pixels below threshold
        params = FRETModelParameters(donor_lifetime=TAU_D)
        em = map_fret_efficiency(donor, params)
        # No valid pixels -> all NaN, no crash
        assert np.all(np.isnan(em['efficiency']))
        assert not em['converged'].any()


# ---------------------------------------------------------------------------
# Stage 4: Joint Donor+Acceptor Solver
# ---------------------------------------------------------------------------

class TestFitJointFret:
    @pytest.mark.parametrize("E_true", [0.3, 0.6])
    def test_recovers_efficiency(self, E_true):
        donor = generate_fret_donor_image(E_true, shape=(10, 10), noise=0.003)
        acceptor = generate_fret_acceptor_image(E_true, shape=(10, 10), noise=0.003)
        params = FRETModelParameters(donor_lifetime=TAU_D, acceptor_lifetime=TAU_A, fret_efficiency=0.1)
        result = fit_joint_fret(donor, acceptor, params)
        assert result.fret_efficiency == pytest.approx(E_true, abs=0.02)

    def test_converged(self):
        donor = generate_fret_donor_image(0.5)
        acceptor = generate_fret_acceptor_image(0.5)
        params = FRETModelParameters(donor_lifetime=TAU_D, acceptor_lifetime=TAU_A)
        result = fit_joint_fret(donor, acceptor, params)
        assert result.converged

    def test_acceptor_model_populated(self):
        donor = generate_fret_donor_image(0.5)
        acceptor = generate_fret_acceptor_image(0.5)
        params = FRETModelParameters(donor_lifetime=TAU_D, acceptor_lifetime=TAU_A)
        result = fit_joint_fret(donor, acceptor, params)
        assert result.acceptor_real_model is not None
        assert result.acceptor_imag_model is not None

    def test_missing_acceptor_lifetime_raises(self):
        donor = generate_fret_donor_image(0.5)
        acceptor = generate_fret_acceptor_image(0.5)
        params = FRETModelParameters(donor_lifetime=TAU_D)  # no acceptor_lifetime
        with pytest.raises(ValueError, match="acceptor_lifetime"):
            fit_joint_fret(donor, acceptor, params)

    def test_shape_mismatch_raises(self):
        donor = generate_fret_donor_image(0.5, shape=(6, 6))
        acceptor = generate_fret_acceptor_image(0.5, shape=(8, 8))
        params = FRETModelParameters(donor_lifetime=TAU_D, acceptor_lifetime=TAU_A)
        with pytest.raises(ValueError, match="shape"):
            fit_joint_fret(donor, acceptor, params)

    def test_frequency_mismatch_raises(self):
        donor = generate_fret_donor_image(0.5)
        acceptor = FRETChannelData(
            real_cal=np.ones((8, 8)),
            imag_cal=np.ones((8, 8)),
            mean=np.ones((8, 8)) * 100,
            frequency=40.0,  # different
        )
        params = FRETModelParameters(donor_lifetime=TAU_D, acceptor_lifetime=TAU_A)
        with pytest.raises(ValueError, match="frequency"):
            fit_joint_fret(donor, acceptor, params)


# ---------------------------------------------------------------------------
# Stage 5: Pixelwise Maps
# ---------------------------------------------------------------------------

class TestMapFretEfficiency:
    def test_output_shape(self):
        donor = generate_fret_donor_image(0.5, shape=(5, 5))
        params = FRETModelParameters(donor_lifetime=TAU_D)
        em = map_fret_efficiency(donor, params)
        assert em['efficiency'].shape == (5, 5)
        assert em['fretting'].shape == (5, 5)
        assert em['residual'].shape == (5, 5)
        assert em['converged'].shape == (5, 5)

    def test_mean_efficiency_close_to_true(self):
        donor = generate_fret_donor_image(0.5, shape=(6, 6), noise=0.003)
        params = FRETModelParameters(donor_lifetime=TAU_D)
        em = map_fret_efficiency(donor, params)
        assert np.nanmean(em['efficiency']) == pytest.approx(0.5, abs=0.03)

    def test_masked_pixels_are_nan(self):
        donor = generate_fret_donor_image(0.5, shape=(4, 4))
        donor.mean[0, 0] = 0.0
        params = FRETModelParameters(donor_lifetime=TAU_D)
        em = map_fret_efficiency(donor, params)
        assert np.isnan(em['efficiency'][0, 0])

    def test_all_valid_pixels_converged(self):
        donor = generate_fret_donor_image(0.5, shape=(4, 4), noise=0.003)
        params = FRETModelParameters(donor_lifetime=TAU_D)
        em = map_fret_efficiency(donor, params)
        assert em['converged'][donor.valid_mask].all()

    def test_joint_mode_with_acceptor(self):
        # Per-pixel joint fitting: fix nuisance terms near zero so the solver
        # focuses on efficiency recovery.  Use a good starting point.
        donor = generate_fret_donor_image(0.6, shape=(6, 6), noise=0.002)
        acceptor = generate_fret_acceptor_image(0.6, shape=(6, 6), noise=0.002)
        params = FRETModelParameters(
            donor_lifetime=TAU_D, acceptor_lifetime=TAU_A,
            fret_efficiency=0.6,
        )
        bounds = FRETBounds(
            donor_background=(0.0, 1e-4),
            donor_bleedthrough=(0.0, 1e-4),
            acceptor_bleedthrough=(0.0, 1e-4),
            acceptor_background=(0.0, 1e-4),
        )
        em = map_fret_efficiency(donor, params, bounds=bounds, acceptor=acceptor)
        assert np.nanmean(em['efficiency']) == pytest.approx(0.6, abs=0.05)


# ---------------------------------------------------------------------------
# Stage 6: Visualization
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _headless_mpl():
    import matplotlib
    matplotlib.use('Agg')


class TestPlotFretTrajectory:
    def test_returns_ax_and_lines(self):
        ax, lines = plot_fret_trajectory(FREQ, TAU_D)
        import matplotlib.axes
        assert isinstance(ax, matplotlib.axes.Axes)
        assert len(lines) == 1

    def test_two_lines_with_acceptor(self):
        _, lines = plot_fret_trajectory(FREQ, TAU_D, acceptor_lifetime=TAU_A)
        assert len(lines) == 2

    def test_accepts_existing_axes(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax_out, _ = plot_fret_trajectory(FREQ, TAU_D, ax=ax)
        assert ax_out is ax
        plt.close(fig)

    def test_donor_kw_applied(self):
        _, lines = plot_fret_trajectory(FREQ, TAU_D, donor_kw={'color': 'green'})
        assert lines[0].get_color() == 'green'

    def test_n_points_controls_data_length(self):
        _, lines = plot_fret_trajectory(FREQ, TAU_D, n_points=37)
        assert len(lines[0].get_xdata()) == 37


class TestPlotFretFit:
    def test_returns_ax_and_artists(self):
        donor = generate_fret_donor_image(0.5)
        params = FRETModelParameters(donor_lifetime=TAU_D)
        result = fit_donor_fret(donor, params)
        ax, artists = plot_fret_fit(donor, result, FREQ, TAU_D)
        assert 'donor_scatter' in artists
        assert 'donor_model' in artists

    def test_joint_includes_acceptor_artists(self):
        donor = generate_fret_donor_image(0.5)
        acceptor = generate_fret_acceptor_image(0.5)
        params = FRETModelParameters(donor_lifetime=TAU_D, acceptor_lifetime=TAU_A)
        result = fit_joint_fret(donor, acceptor, params)
        _, artists = plot_fret_fit(
            donor, result, FREQ, TAU_D,
            acceptor=acceptor, acceptor_lifetime=TAU_A,
        )
        assert 'acceptor_scatter' in artists
        assert 'acceptor_model' in artists

    def test_accepts_existing_axes(self):
        import matplotlib.pyplot as plt
        donor = generate_fret_donor_image(0.5)
        params = FRETModelParameters(donor_lifetime=TAU_D)
        result = fit_donor_fret(donor, params)
        fig, ax = plt.subplots()
        ax_out, _ = plot_fret_fit(donor, result, FREQ, TAU_D, ax=ax)
        assert ax_out is ax
        plt.close(fig)
