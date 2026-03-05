"""Comprehensive End-to-End Tests for Complete FLIM Pipeline

Tests the full workflow including interactive.py functions,
tile stitching, and FLIM fitting integration.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil
import sys
import json

sys.path.insert(0, str(Path(__file__).parent.parent))
from mock_data import (
    generate_test_project,
    generate_synthetic_decay,
    MockPTUFile,
    load_mock_ptu_file
)


class TestCompleteStitchingPipeline:
    """Test complete tile stitching pipeline."""
    
    @pytest.fixture
    def test_project_2x2(self):
        """Create 2x2 test project."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project = generate_test_project(
                Path(temp_dir),
                roi_name="R 2",
                n_tiles=4,
                layout="2x2"
            )
            yield project
    
    @pytest.fixture
    def test_project_3x3(self):
        """Create 3x3 test project."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project = generate_test_project(
                Path(temp_dir),
                roi_name="R 3",
                n_tiles=9,
                layout="3x3"
            )
            yield project
    
    def test_stitch_2x2_tiles(self, test_project_2x2):
        """Test stitching 2x2 tile layout."""
        try:
            from flimkit.PTU.stitch import stitch_flim_tiles
            
            output_dir = test_project_2x2['base_dir'] / "stitched"
            
            result = stitch_flim_tiles(
                xlif_path=test_project_2x2['xlif_path'],
                ptu_dir=test_project_2x2['ptu_dir'],
                output_dir=output_dir,
                ptu_basename=test_project_2x2['roi_name'],
                rotate_tiles=True,
                verbose=False
            )
            
            # Verify result structure
            assert 'intensity_path' in result
            assert 'flim_path' in result
            assert 'time_axis_path' in result
            assert 'metadata_path' in result
            
            # Verify files exist
            assert result['intensity_path'].exists()
            assert result['flim_path'].exists()
            assert result['time_axis_path'].exists()
            assert result['metadata_path'].exists()
            
            # Verify dimensions
            assert result['canvas_shape'] == (1024, 1024)  # 2x2 * 512
            assert result['n_time_bins'] == 256
            assert result['tiles_processed'] == 4
            
        except ImportError as e:
            pytest.skip(f"Required module not available: {e}")
    
    def test_stitch_3x3_tiles(self, test_project_3x3):
        """Test stitching 3x3 tile layout."""
        try:
            from flimkit.PTU.stitch import stitch_flim_tiles
            
            output_dir = test_project_3x3['base_dir'] / "stitched"
            
            result = stitch_flim_tiles(
                xlif_path=test_project_3x3['xlif_path'],
                ptu_dir=test_project_3x3['ptu_dir'],
                output_dir=output_dir,
                ptu_basename=test_project_3x3['roi_name'],
                rotate_tiles=True,
                verbose=False
            )
            
            # Verify 3x3 dimensions
            assert result['canvas_shape'] == (1536, 1536)  # 3x3 * 512
            assert result['tiles_processed'] == 9
            
        except ImportError:
            pytest.skip("Required module not available")
    
    def test_load_stitched_data(self, test_project_2x2):
        """Test loading stitched data."""
        try:
            from flimkit.PTU.stitch import stitch_flim_tiles, load_stitched_flim
            
            output_dir = test_project_2x2['base_dir'] / "stitched"
            
            # Stitch first
            stitch_flim_tiles(
                xlif_path=test_project_2x2['xlif_path'],
                ptu_dir=test_project_2x2['ptu_dir'],
                output_dir=output_dir,
                ptu_basename=test_project_2x2['roi_name'],
                verbose=False
            )
            
            # Load back
            flim, time_axis, intensity, metadata = load_stitched_flim(output_dir)
            
            # Verify shapes
            assert flim.shape == (1024, 1024, 256)
            assert len(time_axis) == 256
            assert intensity.shape == (1024, 1024)
            
            # Verify metadata
            assert metadata['canvas_shape'] == [1024, 1024]
            assert metadata['n_time_bins'] == 256
            
            # Verify data integrity
            assert flim.sum() > 0
            assert intensity.sum() > 0
            assert np.all(time_axis >= 0)
            
        except ImportError:
            pytest.skip("Required module not available")
    
    def test_load_for_fitting(self, test_project_2x2):
        """Test loading data ready for fitting."""
        try:
            from flimkit.PTU.stitch import stitch_flim_tiles, load_flim_for_fitting
            
            output_dir = test_project_2x2['base_dir'] / "stitched"
            
            # Stitch
            stitch_flim_tiles(
                xlif_path=test_project_2x2['xlif_path'],
                ptu_dir=test_project_2x2['ptu_dir'],
                output_dir=output_dir,
                ptu_basename=test_project_2x2['roi_name'],
                verbose=False
            )
            
            # Load for fitting
            stack, tcspc_res, n_bins = load_flim_for_fitting(
                output_dir,
                load_to_memory=True
            )
            
            # Verify format ready for fitting
            assert stack.shape == (1024, 1024, 256)
            assert isinstance(tcspc_res, float)
            assert isinstance(n_bins, int)
            assert tcspc_res > 0
            assert n_bins == 256
            
            # Verify can extract decay
            decay = stack.sum(axis=(0, 1))
            assert len(decay) == n_bins
            assert decay.sum() > 0
            
        except ImportError:
            pytest.skip("Required module not available")


class TestCompleteFittingPipeline:
    """Test complete FLIM fitting pipeline with stitched data."""
    
    @pytest.fixture
    def stitched_project(self):
        """Create and stitch a test project."""
        try:
            from flimkit.PTU.stitch import stitch_flim_tiles
            
            with tempfile.TemporaryDirectory() as temp_dir:
                project = generate_test_project(
                    Path(temp_dir),
                    roi_name="R 2",
                    n_tiles=4,
                    layout="2x2"
                )
                
                output_dir = project['base_dir'] / "stitched"
                
                stitch_flim_tiles(
                    xlif_path=project['xlif_path'],
                    ptu_dir=project['ptu_dir'],
                    output_dir=output_dir,
                    ptu_basename=project['roi_name'],
                    verbose=False
                )
                
                project['output_dir'] = output_dir
                yield project
                
        except ImportError:
            pytest.skip("Required module not available")
    
    def test_fit_stitched_summed_decay(self, stitched_project):
        """Test fitting summed decay from stitched data."""
        try:
            from flimkit.PTU.stitch import load_flim_for_fitting
            from flimkit.FLIM.irf_tools import gaussian_irf_from_fwhm
            from flimkit.FLIM.fitters import fit_summed
            
            # Load data
            stack, tcspc_res, n_bins = load_flim_for_fitting(
                stitched_project['output_dir'],
                load_to_memory=True
            )
            
            # Get summed decay
            decay = stack.sum(axis=(0, 1))
            
            # Create IRF
            irf = gaussian_irf_from_fwhm(
                n_bins=n_bins,
                tcspc_res=tcspc_res,
                fwhm_ns=0.3,
                center_bin=20
            )
            
            # Fit (using simple settings for speed)
            popt, summary = fit_summed(
                decay=decay,
                tcspc_res=tcspc_res,
                n_bins=n_bins,
                irf_prompt=irf,
                has_tail=False,
                fit_bg=True,
                fit_sigma=False,
                n_exp=1,  # Single exp for speed
                tau_min_ns=0.1,
                tau_max_ns=10.0,
                optimizer="lm_multistart",
                n_restarts=2,  # Fewer restarts for speed
                workers=1
            )
            
            # Verify fit results
            assert popt is not None
            assert summary is not None
            assert 'tau_1' in summary
            assert 'chi2r' in summary
            
            # Check tau is reasonable
            tau_ns = summary['tau_1']
            assert 0.1 < tau_ns < 10.0
            
            # Check chi2r is reasonable
            chi2r = summary['chi2r']
            assert 0.5 < chi2r < 5.0  # Reasonable fit quality
            
        except ImportError as e:
            pytest.skip(f"Required module not available: {e}")
    
    def test_full_workflow_summed_only(self, stitched_project):
        """Test complete workflow: stitch → load → fit (summed only)."""
        try:
            from flimkit.PTU.stitch import load_flim_for_fitting
            from flimkit.FLIM.irf_tools import gaussian_irf_from_fwhm
            from flimkit.FLIM.fitters import fit_summed
            
            # This simulates what the user would do
            
            # Step 1: Load stitched data
            stack, tcspc_res, n_bins = load_flim_for_fitting(
                stitched_project['output_dir']
            )
            
            # Step 2: Prepare for fitting
            decay = stack.sum(axis=(0, 1))
            irf = gaussian_irf_from_fwhm(n_bins, tcspc_res, 0.3, 20)
            
            # Step 3: Fit
            popt, summary = fit_summed(
                decay, tcspc_res, n_bins, irf,
                has_tail=False, fit_bg=True, fit_sigma=False,
                n_exp=1, tau_min_ns=0.1, tau_max_ns=10.0,
                optimizer="lm_multistart", n_restarts=2, workers=1
            )
            
            # Step 4: Verify results
            assert summary is not None
            assert 'tau_1' in summary
            assert 0.1 < summary['tau_1'] < 10.0
            
            print(f"✓ Complete workflow successful!")
            print(f"  Fitted lifetime: {summary['tau_1']:.3f} ns")
            print(f"  Chi2r: {summary['chi2r']:.3f}")
            
        except ImportError:
            pytest.skip("Required module not available")


class TestInteractiveFunctions:
    """Test interactive.py workflow functions."""
    
    @pytest.fixture
    def test_project(self):
        """Create test project."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project = generate_test_project(
                Path(temp_dir),
                roi_name="R 2",
                n_tiles=4,
                layout="2x2"
            )
            yield project
    
    def test_stitch_tiles_workflow(self, test_project):
        """Test stitch_tiles workflow from interactive.py."""
        try:
            from flimkit.interactive import _run_tile_stitch
            import argparse
            
            # Build args
            args = argparse.Namespace()
            args.xlif = str(test_project['xlif_path'])
            args.ptu_dir = str(test_project['ptu_dir'])
            args.output_dir = str(test_project['base_dir'] / "stitched")
            args.ptu_basename = test_project['roi_name']
            args.rotate_tiles = True
            
            # Run workflow
            result = _run_tile_stitch(args)
            
            # Verify output
            assert result is not None
            assert result['tiles_processed'] == 4
            assert Path(args.output_dir).exists()
            
        except ImportError:
            pytest.skip("interactive.py not available")
    
    def test_stitch_and_fit_workflow(self, test_project):
        """Test complete stitch + fit workflow from interactive.py."""
        try:
            from flimkit.interactive import _run_stitch_and_fit
            import argparse
            
            # Build args
            args = argparse.Namespace()
            
            # Stitching params
            args.xlif = str(test_project['xlif_path'])
            args.ptu_dir = str(test_project['ptu_dir'])
            args.output_dir = str(test_project['base_dir'] / "results")
            args.ptu_basename = test_project['roi_name']
            args.rotate_tiles = True
            
            # Fitting params
            args.irf = None
            args.estimate_irf = "gaussian"
            args.nexp = 1
            args.tau_min = 0.1
            args.tau_max = 10.0
            args.mode = "summed"  # Faster
            args.binning = 2  # Faster
            args.min_photons = 50
            args.optimizer = "lm_multistart"
            args.restarts = 2
            args.de_population = 10
            args.de_maxiter = 100
            args.workers = 1
            args.no_polish = False
            args.channel = None
            args.irf_fwhm = 0.3
            args.irf_bins = 50
            args.irf_fit_width = 1.5
            args.no_plots = True  # Skip plotting for tests
            
            # Run complete workflow
            _run_stitch_and_fit(args)
            
            # Verify outputs exist
            output_dir = Path(args.output_dir)
            assert output_dir.exists()
            assert (output_dir / "stitched_intensity.tif").exists()
            assert (output_dir / "stitched_flim_counts.npy").exists()
            assert (output_dir / "metadata.json").exists()
            
            print("✓ Complete stitch + fit workflow successful!")
            
        except ImportError as e:
            pytest.skip(f"interactive.py not available: {e}")


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_missing_tiles(self):
        """Test handling of missing PTU tiles."""
        try:
            from flimkit.PTU.stitch import stitch_flim_tiles
            from mock_data import generate_mock_xlif
            
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Create XLIF for 4 tiles
                xlif_path = generate_mock_xlif(
                    temp_path / "test.xlif",
                    n_tiles=4,
                    layout="2x2"
                )
                
                # Create PTU directory with only 2 tiles
                ptu_dir = temp_path / "PTUs"
                ptu_dir.mkdir()
                
                # Create only 2 of 4 tiles
                (ptu_dir / "R 2_s1.ptu").touch()
                (ptu_dir / "R 2_s2.ptu").touch()
                # Missing: s3, s4
                
                output_dir = temp_path / "output"
                
                # Should still work but report skipped tiles
                result = stitch_flim_tiles(
                    xlif_path=xlif_path,
                    ptu_dir=ptu_dir,
                    output_dir=output_dir,
                    ptu_basename="R 2",
                    verbose=False
                )
                
                # Should process 0 tiles (they're empty files)
                # but shouldn't crash
                assert result['tiles_skipped'] >= 2
                
        except ImportError:
            pytest.skip("stitch module not available")
    
    def test_empty_decay(self):
        """Test fitting with zero photons."""
        try:
            from flimkit.FLIM.fitters import fit_summed
            from flimkit.FLIM.irf_tools import gaussian_irf_from_fwhm
            
            # Create zero decay
            decay = np.zeros(256)
            tcspc_res = 97e-12
            n_bins = 256
            
            irf = gaussian_irf_from_fwhm(n_bins, tcspc_res, 0.3, 20)
            
            # Should handle gracefully
            with pytest.raises((ValueError, RuntimeError)):
                fit_summed(
                    decay, tcspc_res, n_bins, irf,
                    has_tail=False, fit_bg=True, fit_sigma=False,
                    n_exp=1, tau_min_ns=0.1, tau_max_ns=10.0,
                    optimizer="lm_multistart", n_restarts=1, workers=1
                )
                
        except ImportError:
            pytest.skip("fitters module not available")
    
    def test_single_tile_as_mosaic(self):
        """Test that single tile (no stitching needed) works."""
        try:
            from flimkit.PTU.stitch import stitch_flim_tiles
            from mock_data import generate_mock_xlif, generate_mock_ptu_tiles
            
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Create XLIF for single tile
                xlif_path = generate_mock_xlif(
                    temp_path / "test.xlif",
                    n_tiles=1,
                    layout="1x1"
                )
                
                # Create single PTU
                ptu_dir = temp_path / "PTUs"
                generate_mock_ptu_tiles(
                    ptu_dir,
                    "R 2",
                    n_tiles=1
                )
                
                output_dir = temp_path / "output"
                
                # Should work fine
                result = stitch_flim_tiles(
                    xlif_path=xlif_path,
                    ptu_dir=ptu_dir,
                    output_dir=output_dir,
                    ptu_basename="R 2",
                    verbose=False
                )
                
                assert result['tiles_processed'] == 1
                assert result['canvas_shape'] == (512, 512)
                
        except ImportError:
            pytest.skip("stitch module not available")


class TestDataIntegrity:
    """Test data integrity throughout pipeline."""
    
    def test_photon_conservation(self):
        """Test that photons are conserved during stitching."""
        try:
            from flimkit.PTU.stitch import stitch_flim_tiles, load_stitched_flim
            from mock_data import generate_test_project, load_mock_ptu_file
            
            with tempfile.TemporaryDirectory() as temp_dir:
                project = generate_test_project(
                    Path(temp_dir),
                    roi_name="R 2",
                    n_tiles=4,
                    layout="2x2"
                )
                
                # Count photons in original tiles
                original_photons = 0
                for ptu_file in project['ptu_files']:
                    mock_ptu = load_mock_ptu_file(ptu_file)
                    original_photons += mock_ptu.summed_decay().sum()
                
                # Stitch
                output_dir = project['base_dir'] / "stitched"
                stitch_flim_tiles(
                    xlif_path=project['xlif_path'],
                    ptu_dir=project['ptu_dir'],
                    output_dir=output_dir,
                    ptu_basename=project['roi_name'],
                    verbose=False
                )
                
                # Count photons in stitched
                flim, _, _, _ = load_stitched_flim(output_dir)
                stitched_photons = flim.sum()
                
                # Should be equal (or very close)
                photon_ratio = stitched_photons / original_photons
                assert 0.95 < photon_ratio < 1.05, \
                    f"Photon conservation failed: {photon_ratio:.3f}"
                
        except ImportError:
            pytest.skip("Required modules not available")
    
    def test_time_axis_consistency(self):
        """Test time axis is consistent throughout pipeline."""
        try:
            from flimkit.PTU.stitch import stitch_flim_tiles, load_stitched_flim
            from flimkit.PTU.decode import create_time_axis
            from mock_data import generate_test_project
            
            with tempfile.TemporaryDirectory() as temp_dir:
                project = generate_test_project(
                    Path(temp_dir),
                    roi_name="R 2",
                    n_tiles=4,
                    layout="2x2"
                )
                
                # Stitch
                output_dir = project['base_dir'] / "stitched"
                result = stitch_flim_tiles(
                    xlif_path=project['xlif_path'],
                    ptu_dir=project['ptu_dir'],
                    output_dir=output_dir,
                    ptu_basename=project['roi_name'],
                    verbose=False
                )
                
                # Load time axis
                _, time_axis, _, metadata = load_stitched_flim(output_dir)
                
                # Recreate from metadata
                tcspc_res = metadata['tcspc_resolution_ps'] * 1e-12
                n_bins = metadata['n_time_bins']
                time_axis_recreated = create_time_axis(n_bins, tcspc_res)
                
                # Should match
                np.testing.assert_array_almost_equal(
                    time_axis,
                    time_axis_recreated,
                    decimal=6
                )
                
        except ImportError:
            pytest.skip("Required modules not available")


def test_installation_check():
    """Test that all required modules can be imported."""
    required_modules = [
        'flimkit.utils.xml_utils',
        'flimkit.PTU.decode',
        'flimkit.PTU.stitch',
    ]
    
    optional_modules = [
        'flimkit.interactive',
        'flimkit.FLIM.fitters',
        'flimkit.FLIM.irf_tools',
        'flimkit.phasor.signal',
        'flimkit.phasor.interactive',
        'flimkit.phasor.peaks',
        'flimkit.phasor_launcher',
    ]
    
    missing_required = []
    missing_optional = []
    
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_required.append(module)
    
    for module in optional_modules:
        try:
            __import__(module)
        except ImportError:
            missing_optional.append(module)
    
    if missing_required:
        pytest.fail(
            f"Missing required modules: {', '.join(missing_required)}\n"
            f"Make sure simplified integration is installed!"
        )
    
    if missing_optional:
        print(f"Note: Some optional modules not available: {', '.join(missing_optional)}")


class TestPhasorPipeline:
    """Test phasor analysis pipeline end-to-end."""

    @pytest.fixture
    def synthetic_phasor(self):
        """Generate synthetic single-exponential phasor data."""
        rng = np.random.default_rng(42)
        shape = (64, 64)
        tau_ns = 2.5
        frequency = 40.0
        omega = 2 * np.pi * frequency * 1e-3
        g_true = 1 / (1 + (omega * tau_ns) ** 2)
        s_true = omega * tau_ns / (1 + (omega * tau_ns) ** 2)

        return dict(
            real_cal=rng.normal(g_true, 0.02, shape),
            imag_cal=rng.normal(s_true, 0.02, shape),
            mean=rng.uniform(5, 50, shape),
            frequency=frequency,
            g_true=g_true,
            s_true=s_true,
            tau_ns=tau_ns,
        )

    def test_find_phasor_peaks(self, synthetic_phasor):
        """Peak detection finds a peak near the expected phasor location."""
        try:
            from flimkit.phasor.peaks import find_phasor_peaks

            peaks = find_phasor_peaks(
                synthetic_phasor['real_cal'],
                synthetic_phasor['imag_cal'],
                synthetic_phasor['mean'],
                synthetic_phasor['frequency'],
            )

            assert peaks['n_peaks'] >= 1
            best = min(
                np.sqrt((peaks['peak_g'][i] - synthetic_phasor['g_true']) ** 2 +
                        (peaks['peak_s'][i] - synthetic_phasor['s_true']) ** 2)
                for i in range(peaks['n_peaks']))
            assert best < 0.1, f"Peak distance {best:.3f} > 0.1"

        except ImportError:
            pytest.skip("phasor.peaks not available")

    def test_peak_lifetime_values(self, synthetic_phasor):
        """Detected peak lifetimes are consistent with the input tau."""
        try:
            from flimkit.phasor.peaks import find_phasor_peaks

            peaks = find_phasor_peaks(
                synthetic_phasor['real_cal'],
                synthetic_phasor['imag_cal'],
                synthetic_phasor['mean'],
                synthetic_phasor['frequency'],
            )

            # Phase lifetime should be in a reasonable neighbourhood
            tau_phase = peaks['tau_phase']
            assert np.any(np.abs(tau_phase - synthetic_phasor['tau_ns']) < 1.5), \
                f"No phase lifetime near expected {synthetic_phasor['tau_ns']} ns"

        except ImportError:
            pytest.skip("phasor.peaks not available")

    def test_save_load_session_roundtrip(self, synthetic_phasor):
        """save_session / load_session preserves all data."""
        try:
            from flimkit.phasor_launcher import save_session, load_session
            import os

            cursors = [dict(center_g=0.4, center_s=0.3, color='#1f77b4')]
            params = dict(radius=0.05, radius_minor=0.03, angle_mode='semicircle')

            with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
                tmp = f.name

            try:
                save_session(
                    tmp,
                    real_cal=synthetic_phasor['real_cal'],
                    imag_cal=synthetic_phasor['imag_cal'],
                    mean=synthetic_phasor['mean'],
                    frequency=synthetic_phasor['frequency'],
                    cursors=cursors,
                    params=params,
                    ptu_file='test.ptu',
                )

                sess = load_session(tmp)
                assert sess['frequency'] == synthetic_phasor['frequency']
                assert len(sess['cursors']) == 1
                np.testing.assert_array_almost_equal(
                    sess['real_cal'], synthetic_phasor['real_cal'])
            finally:
                os.unlink(tmp)

        except ImportError:
            pytest.skip("phasor_launcher not available")

    def test_print_peaks(self, synthetic_phasor, capsys):
        """print_peaks produces output without error."""
        try:
            from flimkit.phasor.peaks import find_phasor_peaks, print_peaks

            peaks = find_phasor_peaks(
                synthetic_phasor['real_cal'],
                synthetic_phasor['imag_cal'],
                synthetic_phasor['mean'],
                synthetic_phasor['frequency'],
            )
            print_peaks(peaks)
            captured = capsys.readouterr()
            assert "Peak" in captured.out or "peak" in captured.out

        except ImportError:
            pytest.skip("phasor.peaks not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
