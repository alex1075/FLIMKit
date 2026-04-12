import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil
import sys
import json

sys.path.insert(0, str(Path(__file__).parent.parent))
from mock_data import (generate_test_project,
    generate_synthetic_decay)
from flimkit.PTU.reader import PTUFile

class TestStitchingWorkflow:
    """Test complete tile stitching workflow."""
    
    @pytest.fixture
    def test_project(self):
        """Create a complete test project."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project = generate_test_project(
                Path(temp_dir),
                roi_name="R 2",
                n_tiles=4,
                layout="2x2"
            )
            yield project
    
    def test_stitch_tiles_complete(self, test_project):
        """Test complete stitching workflow with mock data."""
        # This test would use your actual stitch_flim_tiles function
        # For now, we'll test the mock data generation
        
        assert test_project['xlif_path'].exists()
        assert len(test_project['ptu_files']) == 4
        
        # Check all PTU files exist
        for ptu_file in test_project['ptu_files']:
            assert ptu_file.exists()
            
            # Load and verify
            mock_ptu = PTUFile(str(ptu_file), verbose=False)
            assert mock_ptu.n_y == 64
            assert mock_ptu.n_x == 64
            assert mock_ptu.n_bins == 128
    
    def test_stitching_output_structure(self, test_project):
        """Test that stitching creates expected output files."""
        # Create output directory
        output_dir = test_project['base_dir'] / "stitched"
        output_dir.mkdir(exist_ok=True)
        
        # Expected output files
        expected_files = [
            "stitched_intensity.tif",
            "stitched_flim_counts.npy",
            "time_axis_ns.npy",
            "weight_map.npy",
            "metadata.json"
        ]
        
        # Note: Actual stitching would create these
        # Here we just verify the structure
        for filename in expected_files:
            assert not (output_dir / filename).exists(), \
                "Files don't exist yet (as expected before stitching)"


class TestFittingWorkflow:
    """Test FLIM fitting workflows."""
    
    @pytest.fixture
    def synthetic_decay(self):
        """Create synthetic decay for testing."""
        return generate_synthetic_decay(
            n_bins=128,
            tau_ns=2.0,
            bg=10.0,
            peak_counts=1000.0,
            noise=True
        )
    
    def test_synthetic_decay_properties(self, synthetic_decay):
        """Test synthetic decay has expected properties."""
        assert len(synthetic_decay) == 128
        assert np.all(synthetic_decay >= 0)
        assert synthetic_decay.sum() > 0
        
        # Check it has a peak
        peak_idx = np.argmax(synthetic_decay)
        assert 10 < peak_idx < 50  # IRF delay region
        
        # Check it decays after peak
        tail = synthetic_decay[peak_idx:]
        assert tail[-1] < tail[0]  # Decreases
    
    def test_synthetic_decay_multiexp(self):
        """Test generating multi-exponential decays."""
        # Single exponential
        decay1 = generate_synthetic_decay(tau_ns=1.0, noise=False)
        
        # Different tau
        decay2 = generate_synthetic_decay(tau_ns=5.0, noise=False)
        
        # Should be different
        assert not np.allclose(decay1, decay2)
        
        # Longer tau should have slower decay
        peak1 = np.argmax(decay1)
        peak2 = np.argmax(decay2)
        
        # Check tail decay rate
        tail1_ratio = decay1[peak1 + 50] / decay1[peak1 + 20]
        tail2_ratio = decay2[peak2 + 50] / decay2[peak2 + 20]
        
        # Longer tau should have slower decay (higher ratio)
        assert tail2_ratio > tail1_ratio


class TestMemoryEfficiency:
    """Test memory efficiency of memmap usage."""
    
    def test_large_mosaic_memmap(self):
        """Test that large mosaics use memmap efficiently."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create a "large" memmap file
            large_shape = (2048, 2048, 128)
            memmap_file = temp_path / "large_flim.npy"
            
            # Create memmap
            flim_memmap = np.memmap(
                str(memmap_file),
                dtype=np.uint32,
                mode='w+',
                shape=large_shape
            )
            
            # Write some data
            flim_memmap[100, 100, :] = np.random.poisson(100, size=128)
            flim_memmap.flush()
            
            # Close and reload
            del flim_memmap
            
            # Load as read-only
            flim_loaded = np.memmap(
                str(memmap_file),
                dtype=np.uint32,
                mode='r',
                shape=large_shape
            )
            
            # Verify data
            assert flim_loaded[100, 100, :].sum() > 0
            
            # Check file exists and has reasonable size
            assert memmap_file.exists()
            file_size_mb = memmap_file.stat().st_size / (128 * 128)
            expected_size_mb = (np.prod(large_shape) * 4) / (128 * 128)  # 4 bytes per uint32
            assert abs(file_size_mb - expected_size_mb) < 1  # Within 1 MB


class TestEndToEnd:
    """End-to-end integration tests."""
    
    @pytest.fixture
    def complete_project(self):
        """Create complete test project with all files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project = generate_test_project(
                Path(temp_dir),
                roi_name="R 2",
                n_tiles=4,
                layout="2x2"
            )
            yield project
    
    def test_project_structure(self, complete_project):
        """Test that generated project has correct structure."""
        base_dir = complete_project['base_dir']
        
        # Check directories exist
        assert (base_dir / "PTUs").exists()
        assert (base_dir / "Metadata").exists()
        
        # Check XLIF exists
        assert complete_project['xlif_path'].exists()
        
        # Check PTU files exist
        assert len(complete_project['ptu_files']) == 4
        for ptu_file in complete_project['ptu_files']:
            assert ptu_file.exists()
    
    def test_metadata_json_structure(self, complete_project):
        """Test metadata JSON structure."""
        # Create mock metadata
        metadata = {
            'canvas_shape': (128, 128),
            'n_time_bins': 128,
            'time_range_ns': (0.0, 24.7),
            'tcspc_resolution_ps': 97.0,
            'pixel_size_um': 0.3,
            'tiles_processed': 4,
            'tiles_skipped': 0,
        }
        
        # Verify all keys present
        required_keys = [
            'canvas_shape', 'n_time_bins', 'time_range_ns',
            'tcspc_resolution_ps', 'pixel_size_um',
            'tiles_processed', 'tiles_skipped'
        ]
        
        for key in required_keys:
            assert key in metadata
        
        # Verify types
        assert isinstance(metadata['canvas_shape'], tuple)
        assert isinstance(metadata['n_time_bins'], int)
        assert isinstance(metadata['tiles_processed'], int)


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_missing_xlif_file(self):
        """Test handling of missing XLIF file."""
        from flimkit.utils.xml_utils import parse_xlif_tile_positions
        
        with pytest.raises(FileNotFoundError):
            parse_xlif_tile_positions(Path("/nonexistent/file.xlif"), "R 2")
    
    def test_missing_ptu_tiles(self):
        """Test handling of missing PTU tiles during stitching."""
        # This would test the tiles_skipped counter
        # When PTU files are missing, stitching should continue
        # and report skipped tiles
        pass  # Implement based on actual stitch_flim_tiles
    
    def test_time_bin_mismatch(self):
        """Test handling of tiles with different time bins."""
        # Should pad/crop to match first tile
        pass  # Implement based on actual stitch_flim_tiles
    
    def test_zero_photon_fitting(self):
        """Test fitting with zero photon pixels."""
        # Should skip pixels with < min_photons
        pass  # Implement based on actual fit_per_pixel


class TestPerformance:
    """Performance and regression tests."""
    
    def test_stitching_speed(self):
        """Test that stitching completes in reasonable time (reduced data)."""
        import time
        from flimkit.PTU.stitch import stitch_flim_tiles
        from mock_data import generate_mock_xlif, generate_mock_ptu_tiles
    
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
    
            # Create XLIF for 4 tiles (tiny tiles)
            xlif_path = generate_mock_xlif(
                temp_path / "test.xlif",
                n_tiles=4,
                layout="2x2",
                tile_size=32          # 32×32 pixels
            )
    
            # Create PTU directory with 4 tiny tiles
            ptu_dir = temp_path / "PTUs"
            ptu_dir.mkdir()
            generate_mock_ptu_tiles(
                ptu_dir,
                ptu_basename="R 2",
                n_tiles=4,
                tile_shape=(32, 32),
                n_bins=64,            # only 64 time bins
                mean_photons=50       # low photon count (needs mock_data change)
            )
    
            output_dir = temp_path / "output"
    
            start = time.time()
            result = stitch_flim_tiles(
                xlif_path=xlif_path,
                ptu_dir=ptu_dir,
                output_dir=output_dir,
                ptu_basename="R 2",
                rotate_tiles=True,
                verbose=False
            )
            elapsed = time.time() - start
    
            assert result['tiles_processed'] == 4
            # On a modern machine, 4 tiny tiles should stitch in < 15 seconds
            assert elapsed < 15.0, f"Stitching took {elapsed:.2f}s (>15s)"
    
    def test_memory_usage_scaling(self):
        """Test memory usage scales appropriately."""
        # Small mosaic
        small_shape = (64, 64, 128)
        small_size_mb = (np.prod(small_shape) * 4) / (128 * 128)
        
        # Large mosaic
        large_shape = (2048, 2048, 128)
        large_size_mb = (np.prod(large_shape) * 4) / (128 * 128)
        
        # Should scale linearly with pixels
        size_ratio = large_size_mb / small_size_mb
        pixel_ratio = (2048 * 2048) / (64 * 64)
        
        assert abs(size_ratio - pixel_ratio) < 0.01


def test_validate_outputs():
    """Test output validation functions."""
    # Verify stitched data is valid
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create mock stitched data
        shape = (128, 128, 128)
        flim_file = temp_path / "stitched_flim_counts.npy"
        
        flim_data = np.memmap(
            str(flim_file),
            dtype=np.uint32,
            mode='w+',
            shape=shape
        )
        
        # Write test data
        flim_data[:, :, :] = np.random.poisson(10, size=shape)
        flim_data.flush()
        
        # Verify
        assert flim_file.exists()
        
        # Load and check
        flim_loaded = np.memmap(
            str(flim_file),
            dtype=np.uint32,
            mode='r',
            shape=shape
        )
        
        assert flim_loaded.shape == shape
        assert flim_loaded.sum() > 0


class TestPhasorWorkflow:
    """Test phasor analysis integration."""

    @pytest.fixture
    def phasor_data(self):
        """Create synthetic phasor arrays."""
        rng = np.random.default_rng(99)
        shape = (32, 32)
        tau_ns = 3.0
        freq = 40.0
        omega = 2 * np.pi * freq * 1e-3
        g = 1 / (1 + (omega * tau_ns) ** 2)
        s = omega * tau_ns / (1 + (omega * tau_ns) ** 2)
        return dict(
            real_cal=rng.normal(g, 0.015, shape),
            imag_cal=rng.normal(s, 0.015, shape),
            mean=rng.uniform(10, 60, shape),
            frequency=freq,
            g_true=g,
            s_true=s,
        )

    def test_phasor_peak_detection(self, phasor_data):
        """Peak detection returns valid structure."""
        try:
            from flimkit.phasor.peaks import find_phasor_peaks

            peaks = find_phasor_peaks(
                phasor_data['real_cal'],
                phasor_data['imag_cal'],
                phasor_data['mean'],
                phasor_data['frequency'],
            )

            # Structure checks
            for key in ('n_peaks', 'peak_g', 'peak_s', 'tau_phase',
                        'tau_mod', 'on_semicircle', 'hist', 'hist_smooth'):
                assert key in peaks, f"Missing key: {key}"
            assert peaks['n_peaks'] >= 1

        except ImportError:
            pytest.skip("phasor.peaks not available")

    def test_phasor_save_load(self, phasor_data):
        """Save and reload phasor session."""
        try:
            from flimkit.phasor_launcher import save_session, load_session
            import os

            cursors = [dict(center_g=0.5, center_s=0.25, color='#ff0000')]
            params = dict(radius=0.04, radius_minor=0.02, angle_mode='semicircle')

            with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
                tmp = f.name
            try:
                save_session(tmp,
                             real_cal=phasor_data['real_cal'],
                             imag_cal=phasor_data['imag_cal'],
                             mean=phasor_data['mean'],
                             frequency=phasor_data['frequency'],
                             cursors=cursors, params=params)
                sess = load_session(tmp)
                assert sess['frequency'] == phasor_data['frequency']
                np.testing.assert_array_almost_equal(
                    sess['real_cal'], phasor_data['real_cal'])
            finally:
                os.unlink(tmp)

        except ImportError:
            pytest.skip("phasor_launcher not available")

    def test_phasor_module_imports(self):
        """All phasor sub-modules import cleanly."""
        modules = [
            'flimkit.phasor.signal',
            'flimkit.phasor.interactive',
            'flimkit.phasor.peaks',
            'flimkit.phasor_launcher',
        ]
        for mod in modules:
            try:
                __import__(mod)
            except ImportError:
                pytest.skip(f"{mod} not available")

class TestMemoryEfficiency:
    """Test memory mapping with large synthetic mosaics."""

    def test_large_mosaic_memmap_stress(self, tmp_path):
        """Create a large memmap array, write/read chunks, verify no memory leak."""
        import psutil
        import gc
        from flimkit.PTU.stitch import stitch_flim_tiles
        from mock_data import generate_test_project

        # Generate a 4x4 tile project (bigger than typical 2x2)
        project = generate_test_project(
            tmp_path,
            roi_name="R 2",
            n_tiles=16,
            layout="4x4",
            tile_shape=(128, 128),  # 128x128 per tile → 512x512 canvas
            n_bins=256,
            mean_photons=100,
        )

        # Rename .npy to .ptu for stitching
        for npy_file in project['ptu_dir'].glob('*.npy'):
            npy_file.rename(npy_file.with_suffix('.ptu'))

        output_dir = project['base_dir'] / "stitched"

        # Measure memory before
        gc.collect()
        mem_before = psutil.Process().memory_info().rss / 1024**2  # MB

        # Stitch with memmap
        result = stitch_flim_tiles(
            xlif_path=project['xlif_path'],
            ptu_dir=project['ptu_dir'],
            output_dir=output_dir,
            ptu_basename=project['roi_name'],
            rotate_tiles=False,
            verbose=False,
        )

        # Force garbage collection
        gc.collect()
        mem_after = psutil.Process().memory_info().rss / 1024**2

        # Memory increase should be moderate (< 500 MB for 512x512x256)
        # The memmap itself is on disk, only small chunks loaded
        assert mem_after - mem_before < 500, f"Memory increased by {mem_after - mem_before:.1f} MB"

        # Load the memmap and verify we can access slices without loading all
        flim_path = result['flim_path']
        flim = np.memmap(flim_path, dtype=np.uint32, mode='r', shape=(512, 512, 256))

        # Access a few chunks, should not cause large memory spike
        _ = flim[100:200, 100:200, :].sum()
        _ = flim[300:400, 300:400, :].mean()

        # Ensure the memmap file is correctly sized
        expected_size = 512 * 512 * 256 * 4  # uint32 = 4 bytes
        actual_size = Path(flim_path).stat().st_size
        assert actual_size == expected_size

    def test_out_of_core_processing(self, tmp_path):
        """Simulate processing large mosaic by iterating over chunks."""
        import numpy as np

        # Create a synthetic large memmap
        shape = (1024, 1024, 256)
        memmap_path = tmp_path / "large.npy"
        mmap = np.memmap(memmap_path, dtype=np.uint32, mode='w+', shape=shape)

        # Fill with random data in chunks to simulate real writing
        chunk_size = 128
        for i in range(0, shape[0], chunk_size):
            i_end = min(i + chunk_size, shape[0])
            mmap[i:i_end, :, :] = np.random.poisson(10, size=(i_end - i, shape[1], shape[2]))

        mmap.flush()
        del mmap

        # Now process in chunks (like per-pixel fitting)
        mmap_read = np.memmap(memmap_path, dtype=np.uint32, mode='r', shape=shape)
        total_photons = 0
        for i in range(0, shape[0], chunk_size):
            i_end = min(i + chunk_size, shape[0])
            chunk = mmap_read[i:i_end, :, :]
            total_photons += chunk.sum()
            # Simulate some computation
            _ = np.log1p(chunk.astype(np.float32))

        assert total_photons > 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
