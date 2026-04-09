import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from mock_data import generate_mock_ptu_tiles, MockPTUFile
from flimkit.PTU.reader import PTUFile


class TestDecode:
    """Test suite for decode module."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        shutil.rmtree(temp_path)
    
    @pytest.fixture
    def mock_ptu(self):
        """Create a mock PTU file."""
        return MockPTUFile(n_y=512, n_x=512, n_bins=256)
    
    def test_create_time_axis(self):
        """Test time axis creation."""
        from flimkit.PTU.decode import create_time_axis
        
        n_bins = 256
        tcspc_res = 97e-12  # 97 ps
        
        time_axis = create_time_axis(n_bins, tcspc_res)
        
        # Check shape
        assert len(time_axis) == n_bins
        
        # Check values
        assert time_axis[0] == 0.0
        assert abs(time_axis[1] - 0.097) < 0.001  # 97 ps in ns
        assert abs(time_axis[-1] - 24.7) < 0.1    # ~25 ns total
        
        # Check monotonic increasing
        assert np.all(np.diff(time_axis) > 0)
    
    def test_mock_ptu_structure(self, mock_ptu):
        """Test that mock PTU has correct structure."""
        # Check attributes
        assert hasattr(mock_ptu, 'n_y')
        assert hasattr(mock_ptu, 'n_x')
        assert hasattr(mock_ptu, 'n_bins')
        assert hasattr(mock_ptu, 'tcspc_res')
        assert hasattr(mock_ptu, 'frequency')
        
        # Check methods
        assert hasattr(mock_ptu, 'summed_decay')
        assert hasattr(mock_ptu, 'pixel_stack')
        
        # Check dimensions
        assert mock_ptu.n_y == 512
        assert mock_ptu.n_x == 512
        assert mock_ptu.n_bins == 256
    
    def test_summed_decay(self, mock_ptu):
        """Test summed decay extraction."""
        decay = mock_ptu.summed_decay()
        
        # Check shape
        assert decay.shape == (mock_ptu.n_bins,)
        
        # Check values are positive
        assert np.all(decay >= 0)
        
        # Check it's not all zeros
        assert np.sum(decay) > 0
        
        # Check decay shape (should have a peak and decay)
        assert np.argmax(decay) > 0  # Peak not at first bin
        assert np.argmax(decay) < len(decay) - 1  # Peak not at last bin
    
    def test_pixel_stack(self, mock_ptu):
        """Test pixel stack extraction."""
        stack = mock_ptu.pixel_stack()
        
        # Check shape
        assert stack.shape == (mock_ptu.n_y, mock_ptu.n_x, mock_ptu.n_bins)
        
        # Check values
        assert np.all(stack >= 0)
        assert np.sum(stack) > 0
        
        # Check summed equals summed_decay
        decay_from_stack = stack.sum(axis=(0, 1))
        decay_direct = mock_ptu.summed_decay()
        np.testing.assert_array_almost_equal(decay_from_stack, decay_direct)
    
    def test_pixel_stack_binning(self, mock_ptu):
        """Test pixel stack with binning."""
        binning = 2
        stack = mock_ptu.pixel_stack(binning=binning)
        
        # Check shape is reduced
        expected_y = mock_ptu.n_y // binning
        expected_x = mock_ptu.n_x // binning
        assert stack.shape == (expected_y, expected_x, mock_ptu.n_bins)
        
        # Check total photons are conserved
        stack_full = mock_ptu.pixel_stack(binning=1)
        assert abs(stack.sum() - stack_full.sum()) < 1  # Within rounding
    
    def test_histogram_properties(self, mock_ptu):
        """Test that histogram has realistic FLIM properties."""
        decay = mock_ptu.summed_decay()
        
        # Find peak
        peak_idx = np.argmax(decay)
        peak_time_ns = peak_idx * mock_ptu.tcspc_res * 1e9
        
        # Peak should be in reasonable range (IRF delay)
        assert 1.0 < peak_time_ns < 5.0
        
        # Decay should decrease after peak
        tail = decay[peak_idx:]
        # Check that values generally decrease (allowing some noise)
        decreasing_ratio = np.sum(np.diff(tail) < 0) / len(np.diff(tail))
        assert decreasing_ratio > 0.7  # Most bins should decrease


class TestDecodeIntegration:
    """Integration tests with actual decode functions."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        shutil.rmtree(temp_path)
    
    def test_mock_ptu_save_load(self, temp_dir):
        """Test saving and loading mock PTU files."""
                
        # Generate mock PTU
        ptu_files = generate_mock_ptu_tiles(
            temp_dir,
            ptu_basename="R 2",
            n_tiles=1,
            tile_shape=(256, 256),
            n_bins=128
        )
        
        assert len(ptu_files) == 1
        assert ptu_files[0].exists()
        
        # Load it back
        mock_ptu = PTUFile(str(ptu_files[0]), verbose=False)
        
        # Check properties
        assert mock_ptu.n_y == 256
        assert mock_ptu.n_x == 256
        assert mock_ptu.n_bins == 128
        
        # Check data
        decay = mock_ptu.summed_decay()
        assert len(decay) == 128
        assert decay.sum() > 0


def test_normalise_flim():
    """Test FLIM normalization function."""
    from flimkit.PTU.decode import normalise_flim
    
    # Test 4D -> 3D
    data_4d = np.random.rand(1, 512, 512, 256)
    result = normalise_flim(data_4d)
    assert result.shape == (512, 512, 256)
    
    # Test 3D -> 3D (no change)
    data_3d = np.random.rand(512, 512, 256)
    result = normalise_flim(data_3d)
    assert result.shape == (512, 512, 256)
    
    # Test 2D -> None
    data_2d = np.random.rand(512, 512)
    result = normalise_flim(data_2d)
    assert result is None
    
    # Test None -> None
    result = normalise_flim(None)
    assert result is None


def test_estimate_bg_from_histogram():
    """Test background estimation."""
    from flimkit.FLIM.fit_tools import estimate_bg_from_histogram
    
    # Create synthetic histogram with known background
    hist = np.random.poisson(5, size=(512, 512, 256))  # bg=5
    
    # Add decay on top
    hist[:, :, 50:150] += np.random.poisson(100, size=(512, 512, 100))
    
    bg = estimate_bg_from_histogram(hist, pre_bins=20)
    
    # Should be close to 5
    assert 3 < bg < 7  # Allow some variation due to Poisson


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
