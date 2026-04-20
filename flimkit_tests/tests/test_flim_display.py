import pytest
import numpy as np
from flimkit.UI.flim_display import (
    compute_intensity_weighted_lifetime,
    apply_color_scale,
)


class TestComputeIntensityWeightedLifetime:
    def test_precomputed_tau_mean_int(self):
        """If pixel_maps already has tau_mean_int, return it directly."""
        tau_map = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        pixel_maps = {'tau_mean_int': tau_map}
        intensity = np.ones((2, 2))
        result = compute_intensity_weighted_lifetime(pixel_maps, intensity)
        np.testing.assert_array_equal(result, tau_map)

    def test_precomputed_tau_mean_amp(self):
        """If pixel_maps has tau_mean_amp, return it with zeros→NaN."""
        tau_map = np.array([[1.5, 0.0], [2.5, 3.0]], dtype=np.float32)
        pixel_maps = {'tau_mean_amp': tau_map}
        intensity = np.ones((2, 2))
        result = compute_intensity_weighted_lifetime(pixel_maps, intensity)
        assert np.isnan(result[0, 1])  # 0.0 → NaN
        assert result[0, 0] == pytest.approx(1.5)

    def test_component_weighted_average(self):
        """Compute from individual tau/amp components."""
        pixel_maps = {
            'tau1': np.array([[1.0, 2.0]]),
            'a1':   np.array([[100.0, 50.0]]),
            'tau2': np.array([[3.0, 4.0]]),
            'a2':   np.array([[100.0, 50.0]]),
        }
        intensity = np.ones((1, 2))
        result = compute_intensity_weighted_lifetime(pixel_maps, intensity, n_exp=2)
        # Pixel (0,0): (1*100 + 3*100) / 200 = 2.0
        assert result[0, 0] == pytest.approx(2.0)
        # Pixel (0,1): (2*50 + 4*50) / 100 = 3.0
        assert result[0, 1] == pytest.approx(3.0)

    def test_unfitted_pixels_are_nan(self):
        """Pixels with zero amplitude sum → NaN."""
        pixel_maps = {
            'tau1': np.array([[np.nan]]),
            'a1':   np.array([[0.0]]),
        }
        intensity = np.zeros((1, 1))
        result = compute_intensity_weighted_lifetime(pixel_maps, intensity, n_exp=1)
        assert np.isnan(result[0, 0])

    def test_empty_pixel_maps(self):
        """No matching keys → all NaN."""
        result = compute_intensity_weighted_lifetime({}, np.ones((2, 2)), n_exp=2)
        assert np.all(np.isnan(result))


class TestApplyColorScale:
    def test_auto_range(self):
        """Without vmin/vmax, auto-detect from percentiles."""
        img = np.array([[0.0, 5.0], [10.0, 15.0]])
        result = apply_color_scale(img)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_explicit_range(self):
        """Explicit vmin/vmax clips and normalizes."""
        img = np.array([[0.0, 5.0], [10.0, 20.0]])
        result = apply_color_scale(img, vmin=0, vmax=20, gamma=1.0)
        assert result[0, 0] == pytest.approx(0.0)
        assert result[1, 1] == pytest.approx(1.0)
        assert result[0, 1] == pytest.approx(0.25)

    def test_gamma_correction(self):
        """Gamma > 1 boosts mid-range values (applies x^(1/gamma))."""
        img = np.array([[0.0, 0.5, 1.0]])
        linear = apply_color_scale(img, vmin=0, vmax=1, gamma=1.0)
        boosted = apply_color_scale(img, vmin=0, vmax=1, gamma=2.0)
        # gamma=2.0 → x^(1/2) = sqrt → 0.5^0.5 ≈ 0.71 > 0.5
        assert boosted[0, 1] > linear[0, 1]

    def test_nan_preserved(self):
        """NaN pixels remain NaN after scaling."""
        img = np.array([[1.0, np.nan], [3.0, 4.0]])
        result = apply_color_scale(img, vmin=0, vmax=5, gamma=1.0)
        assert np.isnan(result[0, 1])
        assert not np.isnan(result[0, 0])

    def test_flat_image(self):
        """All same values → vmax == vmin → output all zeros."""
        img = np.full((3, 3), 5.0)
        result = apply_color_scale(img, vmin=5, vmax=5)
        np.testing.assert_array_equal(result, 0.0)
