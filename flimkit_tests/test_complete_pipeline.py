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
    MOCK_IRF_CENTER,   # IRF peak bin used by MockPTUFile (= 30)
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
                layout="2x2",
                mean_photons=500
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
                register_tiles=False,  
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
            assert result['canvas_shape'] == (128, 128)  # 2x2 * 512
            assert result['n_time_bins'] == 128
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
                register_tiles=False,
                verbose=False
            )
            
            # Verify 3x3 dimensions
            assert result['canvas_shape'] == (192, 192)  # 3x3 * 512
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
                rotate_tiles=True,
                register_tiles=False,
                verbose=False
            )
            
            # Load back
            flim, time_axis, intensity, metadata = load_stitched_flim(output_dir)
            
            # Verify shapes
            assert flim.shape == (128, 128, 128)
            assert len(time_axis) == 128
            assert intensity.shape == (128, 128)
            
            # Verify metadata
            assert metadata['canvas_shape'] == [128, 128]
            assert metadata['n_time_bins'] == 128
            
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
                register_tiles=False,
                verbose=False
            )
            
            # Load for fitting
            stack, tcspc_res, n_bins = load_flim_for_fitting(
                output_dir,
                load_to_memory=True
            )
            
            # Verify format ready for fitting
            assert stack.shape == (128, 128, 128)
            assert isinstance(tcspc_res, float)
            assert isinstance(n_bins, int)
            assert tcspc_res > 0
            assert n_bins == 128
            
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
                    layout="2x2",
                    mean_photons=500,
                    n_bins=256)  # 24.8 ns window; wrap ~0.025%
                
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
                peak_bin=MOCK_IRF_CENTER   # must match MockPTUFile (bin 30)
            )
            
            # Fit (using simple settings for speed).
            popt, summary = fit_summed(
                decay=decay,
                tcspc_res=tcspc_res,
                n_bins=n_bins,
                irf_prompt=irf,
                has_tail=False,
                fit_bg=True,
                fit_sigma=False,
                n_exp=2,  # Bi-exp to match mock data
                tau_min_ns=0.1,
                tau_max_ns=10.0,
                optimizer="de",
                workers=-1
            )

            # Verify fit results
            assert popt is not None
            assert summary is not None
            assert 'taus_ns' in summary
            assert 'reduced_chi2_tail' in summary
            
            # Check tau is reasonable
            tau_ns = summary['taus_ns'][0]
            assert 0.1 < tau_ns < 10.0
            
            # Check chi2r is reasonable
            chi2r = summary['reduced_chi2_tail']
            assert 0.5 < chi2r < 10.0  # Reasonable fit quality (bi-exp)
            
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
            assert 'taus_ns' in summary
            assert 0.1 < summary['taus_ns'][0] < 10.0
            
            print(f"✓ Complete workflow successful!")
            print(f"  Fitted lifetime: {summary['taus_ns'][0]:.3f} ns")
            print(f"  Chi2r: {summary['reduced_chi2_pearson']:.3f}")
            
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
                layout="2x2",
                mean_photons=500
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
            
            # Verify outputs exist.
            # The stitcher prefixes all output files with the ROI name
            # (spaces replaced by underscores), e.g. "R_2_stitched_intensity.tif".
            output_dir = Path(args.output_dir)
            roi_prefix = test_project['roi_name'].replace(' ', '_')
            assert output_dir.exists()
            assert (output_dir / f"{roi_prefix}_stitched_intensity.tif").exists()
            assert (output_dir / f"{roi_prefix}_stitched_flim_counts.npy").exists()
            assert (output_dir / f"{roi_prefix}_metadata.json").exists()
            
            print("✓ Complete stitch + fit workflow successful!")
            
        except ImportError as e:
            pytest.skip(f"interactive.py not available: {e}")


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_missing_tiles(self):
        """Test handling of missing PTU tiles."""
        try:
            from flimkit.PTU.stitch import stitch_flim_tiles
            from mock_data import generate_mock_xlif, generate_mock_ptu_tiles
            
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
    
                # Generate valid PTU files for the first two tiles only
                generate_mock_ptu_tiles(
                    ptu_dir,
                    ptu_basename="R 2",
                    n_tiles=2,
                    tile_shape=(64, 64),
                    n_bins=128,
                    mean_photons=500
                )
                # Tiles s3 and s4 are intentionally missing
    
                output_dir = temp_path / "output"
    
                result = stitch_flim_tiles(
                    xlif_path=xlif_path,
                    ptu_dir=ptu_dir,
                    output_dir=output_dir,
                    ptu_basename="R 2",
                    verbose=False
                )
    
                # Should process 2 tiles and skip 2
                assert result['tiles_processed'] == 2
                assert result['tiles_skipped'] == 2
                
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
                assert result['canvas_shape'] == (64, 64)
                
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
                    layout="2x2", mean_photons=500)
                
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
                assert 0.95 < photon_ratio < 1.05,                     f"Photon conservation failed: {photon_ratio:.3f}"
                
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
                , mean_photons=500)
                
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


class TestPerTileFitPipeline:
    """Test the per-tile fit → assemble → save pipeline.

    The tests are structured in three layers:
      1. Component tests: call assemble_tile_maps / derive_global_tau /
         save_assembled_maps directly with synthetic data — no PTU I/O needed.
      2. Integration test: patch fit_flim_tiles to return synthetic tile_results
         and verify _run_tile_fit completes end-to-end and saves the right files.
      3. Machine-IRF variant: same integration test with estimate_irf='machine_irf'.
    """

    N_BINS  = 256
    TCSPC   = 97e-12   # seconds per bin
    TILE_H  = 64       # small tiles for speed
    TILE_W  = 64

    #  helpers 

    def _pixel_maps(self, h=None, w=None, n_exp=1):
        """Build a minimal pixel_maps dict that assemble_tile_maps accepts."""
        h = h or self.TILE_H
        w = w or self.TILE_W
        pm = {
            'intensity': np.random.poisson(500, (h, w)).astype(np.float32),
            'chi2':      np.random.uniform(0.8, 1.5, (h, w)).astype(np.float32),
        }
        for k in range(1, n_exp + 1):
            pm[f'tau{k}'] = np.random.uniform(1.0, 4.0, (h, w)).astype(np.float32)
            pm[f'a{k}']   = np.random.uniform(0.3, 0.7, (h, w)).astype(np.float32)

        # Compute amplitude-weighted mean tau (required for assemble_tile_maps)
        if n_exp >= 1:
            taus = np.stack([pm[f'tau{k}'] for k in range(1, n_exp+1)], axis=0)
            amps = np.stack([pm[f'a{k}']   for k in range(1, n_exp+1)], axis=0)
            sum_amps = np.sum(amps, axis=0)
            with np.errstate(divide='ignore', invalid='ignore'):
                tau_mean_amp = np.sum(taus * amps, axis=0) / sum_amps
                tau_mean_amp[sum_amps == 0] = np.nan
            pm['tau_mean_amp'] = tau_mean_amp.astype(np.float32)
        else:
            pm['tau_mean_amp'] = pm['tau1'].copy()
        return pm

    def _tile_results(self, layout="2x2", n_exp=1):
        """Build synthetic tile_results for a given layout."""
        rows, cols = map(int, layout.split('x'))
        results = []
        for r in range(rows):
            for c in range(cols):
                pm = self._pixel_maps(n_exp=n_exp)
                results.append({
                    'ptu_name':       f'R_2_s{r*cols+c+1}.ptu',
                    'pixel_y':        r * self.TILE_H,
                    'pixel_x':        c * self.TILE_W,
                    'tile_h':         self.TILE_H,
                    'tile_w':         self.TILE_W,
                    'pixel_maps':     pm,
                    'global_summary': {
                        'n_pixels_fitted':       self.TILE_H * self.TILE_W,
                        'tau_mean_amp_global_ns': 2.1,
                        'tau_std_amp_global_ns':  0.3,
                        **{f'tau{k}_mean_ns': 2.0 for k in range(1, n_exp + 1)},
                        **{f'a{k}_mean_frac': 1.0 / n_exp for k in range(1, n_exp + 1)},
                    },
                    'strategy': 'parametric',
                })
        canvas_h = rows * self.TILE_H
        canvas_w = cols * self.TILE_W
        return results, canvas_h, canvas_w

    def _fit_flim_tiles_return_value(self, tile_results, canvas_h, canvas_w, n_exp=1):
        """Build full 9-element return tuple for fit_flim_tiles mock (new signature)."""
        import numpy as np
        
        # Create synthetic pooled decay (1024 bins)
        pooled_decay = np.random.exponential(scale=100, size=1024).astype(np.float32)
        pooled_decay = np.maximum(pooled_decay, 1.0)  # Ensure positivity
        
        # Create synthetic pooled IRF
        pooled_irf = np.random.exponential(scale=10, size=1024).astype(np.float32)
        pooled_irf = pooled_irf / pooled_irf.sum()  # Normalize to probability
        
        # Use standard TCSPC resolution (~50 ps/bin)
        tcspc_ref = 50e-12  # 50 ps in seconds
        
        # Create synthetic global_popt (number of params varies with n_exp)
        # Standard exponential fit: tau + amplitude + bg + sigma + irf_shift
        n_params = 2 * n_exp + 3  # taus + amplitudes + bg + sigma + irf_shift
        global_popt = np.random.uniform(0.5, 5.0, n_params).astype(np.float32)
        
        # Use global_summary from first tile result
        global_summary = {
            'taus_ns': [2.0 + 0.1*k for k in range(n_exp)],
            'amplitudes': [1.0 / n_exp for k in range(n_exp)],
            'reduced_chi2': 1.2,
            'reduced_chi2_tail': 1.1,
            'n_pixels_fitted': len(tile_results) * self.TILE_H * self.TILE_W,
            'tau_mean_amp_global_ns': 2.1,
            'tau_std_amp_global_ns': 0.3,
        }
        
        # Create corrected_positions (dummy, matching tile_results)
        corrected_positions = [{'file': f'R_2_s{i+1}.ptu'} for i in range(len(tile_results))]
        
        return tile_results, canvas_h, canvas_w, corrected_positions, pooled_decay, pooled_irf, tcspc_ref, global_popt, global_summary


    def _base_args(self, output_dir, n_exp=1):
        """Minimal args namespace for _run_tile_fit."""
        import argparse
        try:
            from flimkit.configs import (
                Tau_min, Tau_max, IRF_BINS, IRF_FIT_WIDTH, IRF_FWHM,
                channels, lm_restarts, de_population, de_maxiter,
            )
        except ImportError:
            pytest.skip("flimkit.configs not available")

        return argparse.Namespace(
            xlif='dummy.xlif', ptu_dir='dummy_ptu', ptu_basename='R 2',
            output_dir=str(output_dir), rotate_tiles=True,
            estimate_irf='parametric', irf=None, irf_xlsx=None,
            irf_xlsx_dir=None, irf_xlsx_map=None, machine_irf=None,
            irf_bins=IRF_BINS, irf_fit_width=IRF_FIT_WIDTH, irf_fwhm=IRF_FWHM,
            no_xlsx_irf=True, nexp=n_exp, tau_min=Tau_min, tau_max=Tau_max,
            mode='both', binning=4, min_photons=10,
            optimizer='lm_multistart', restarts=1,
            de_population=de_population, de_maxiter=de_maxiter,
            workers=1, no_polish=True, channel=channels, no_plots=True,
            cell_mask=False, debug_xlsx=False, print_config=False,
            xlsx=None, out=None, intensity_threshold=None,
            tau_display_min=None, tau_display_max=None,
        )

    #  1. component tests

    def test_assemble_tile_maps_basic(self):
        """assemble_tile_maps places tiles correctly on the canvas."""
        try:
            from flimkit.FLIM.assemble import assemble_tile_maps
        except ImportError as e:
            pytest.skip(f"assemble not available: {e}")

        tile_results, ch, cw = self._tile_results("2x2", n_exp=1)
        canvas = assemble_tile_maps(tile_results, ch, cw, n_exp=1)

        assert 'intensity'    in canvas
        assert 'tau_mean_amp' in canvas
        assert 'tau1'         in canvas
        assert canvas['intensity'].shape    == (ch, cw)
        assert canvas['tau_mean_amp'].shape == (ch, cw)
        # All pixels should have been filled (no NaN gaps for a gapless 2×2)
        assert np.all(canvas['intensity'] >= 0)
        assert np.sum(np.isfinite(canvas['tau1'])) > 0

    def test_assemble_tile_maps_biexp(self):
        """assemble_tile_maps works for 2-component fits."""
        try:
            from flimkit.FLIM.assemble import assemble_tile_maps
        except ImportError as e:
            pytest.skip(f"assemble not available: {e}")

        tile_results, ch, cw = self._tile_results("2x2", n_exp=2)
        canvas = assemble_tile_maps(tile_results, ch, cw, n_exp=2)

        assert 'tau1' in canvas and 'tau2' in canvas
        assert 'a1'   in canvas and 'a2'   in canvas
        assert canvas['tau1'].shape == (ch, cw)

    def test_derive_global_tau_single_exp(self):
        """derive_global_tau extracts meaningful stats from a canvas."""
        try:
            from flimkit.FLIM.assemble import assemble_tile_maps, derive_global_tau
        except ImportError as e:
            pytest.skip(f"assemble not available: {e}")

        tile_results, ch, cw = self._tile_results("2x2", n_exp=1)
        canvas = assemble_tile_maps(tile_results, ch, cw, n_exp=1)
        gs = derive_global_tau(canvas, n_exp=1)

        assert 'n_pixels_fitted'       in gs
        assert 'tau_mean_amp_global_ns' in gs
        assert 'tau1_mean_ns'           in gs
        assert gs['n_pixels_fitted']   > 0
        tau = gs['tau_mean_amp_global_ns']
        assert 0.1 < tau < 15.0, f"Unreasonable global tau: {tau}"

    def test_save_assembled_maps_creates_files(self):
        """save_assembled_maps writes intensity TIFF, tau TIFF, numpy maps, and summary."""
        try:
            from flimkit.FLIM.assemble import assemble_tile_maps, derive_global_tau, save_assembled_maps
        except ImportError as e:
            pytest.skip(f"assemble not available: {e}")

        tile_results, ch, cw = self._tile_results("2x2", n_exp=1)
        canvas = assemble_tile_maps(tile_results, ch, cw, n_exp=1)
        gs     = derive_global_tau(canvas, n_exp=1)

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            save_assembled_maps(canvas, gs, out, roi_name='R_2', n_exp=1)

            assert (out / 'R_2_intensity.tif').exists(),     "intensity TIFF missing"
            assert (out / 'R_2_tau_mean_amp.tif').exists(),  "tau TIFF missing"
            assert (out / 'R_2_intensity.npy').exists(),     "intensity .npy missing"
            assert (out / 'R_2_tau_mean_amp.npy').exists(),  "tau .npy missing"
            assert (out / 'R_2_global_summary.txt').exists(),"global summary missing"

    #  2. integration test with patched fit_flim_tiles 

    def test_run_tile_fit_end_to_end(self):
        """_run_tile_fit completes and writes output files when fit_flim_tiles is patched."""
        try:
            from flimkit.interactive import _run_tile_fit
            from flimkit.FLIM.assemble import assemble_tile_maps, derive_global_tau
        except ImportError as e:
            pytest.skip(f"Required module not available: {e}")

        from unittest.mock import patch

        tile_results, ch, cw = self._tile_results("2x2", n_exp=1)
        fit_flim_tiles_return = self._fit_flim_tiles_return_value(tile_results, ch, cw, n_exp=1)

        with tempfile.TemporaryDirectory() as tmp:
            args = self._base_args(Path(tmp))

            with patch('flimkit.PTU.stitch.fit_flim_tiles',
                       return_value=fit_flim_tiles_return):
                fit_result = _run_tile_fit(args)

            # _run_tile_fit now returns a dict
            assert isinstance(fit_result, dict)
            canvas = fit_result['canvas']
            global_summary = fit_result['global_summary']
            
            assert 'intensity'    in canvas
            assert 'tau_mean_amp' in canvas
            assert global_summary['n_pixels_fitted'] > 0
            tau = global_summary['tau_mean_amp_global_ns']
            assert 0.1 < tau < 15.0, f"Unreasonable tau: {tau}"

            # Output files must exist
            out = Path(args.output_dir)
            roi = args.ptu_basename.replace(' ', '_')
            assert (out / f'{roi}_intensity.tif').exists(),    "intensity TIFF missing"
            assert (out / f'{roi}_tau_mean_amp.tif').exists(), "tau TIFF missing"
            assert (out / f'{roi}_global_summary.txt').exists()

    def test_run_tile_fit_canvas_dimensions(self):
        """Canvas is sized to the tile layout, not a fixed constant."""
        try:
            from flimkit.interactive import _run_tile_fit
        except ImportError as e:
            pytest.skip(f"Required module not available: {e}")

        from unittest.mock import patch

        # Use a 3×2 layout to verify non-square canvas handling
        tile_results, ch, cw = self._tile_results("3x2", n_exp=1)
        fit_flim_tiles_return = self._fit_flim_tiles_return_value(tile_results, ch, cw, n_exp=1)
        
        UPSAMPLE = 4
        expected_h = 3 * self.TILE_H * UPSAMPLE
        expected_w = 2 * self.TILE_W * UPSAMPLE

        with tempfile.TemporaryDirectory() as tmp:
            args = self._base_args(Path(tmp))
            args.ptu_basename = 'R 3'

            with patch('flimkit.PTU.stitch.fit_flim_tiles',
                    return_value=fit_flim_tiles_return):
                fit_result = _run_tile_fit(args)
                canvas = fit_result['canvas']

            h, w = canvas['intensity'].shape
            assert h == expected_h and w == expected_w, (
                f"Expected {expected_h}×{expected_w}, got {h}×{w}"
            )

    def test_run_tile_fit_biexp_summary(self):
        """With nexp=2 the global summary exposes tau1, tau2, a1, a2 stats."""
        try:
            from flimkit.interactive import _run_tile_fit
        except ImportError as e:
            pytest.skip(f"Required module not available: {e}")

        from unittest.mock import patch

        tile_results, ch, cw = self._tile_results("2x2", n_exp=2)
        fit_flim_tiles_return = self._fit_flim_tiles_return_value(tile_results, ch, cw, n_exp=2)

        with tempfile.TemporaryDirectory() as tmp:
            args = self._base_args(Path(tmp), n_exp=2)

            with patch('flimkit.PTU.stitch.fit_flim_tiles',
                       return_value=fit_flim_tiles_return):
                fit_result = _run_tile_fit(args)
                global_summary = fit_result['global_summary']

            assert 'tau1_mean_ns' in global_summary, "tau1_mean_ns missing"
            assert 'tau2_mean_ns' in global_summary, "tau2_mean_ns missing"

    #  3. machine-IRF variant

    def test_run_tile_fit_with_machine_irf(self):
        """estimate_irf='machine_irf' path does not crash and produces valid output."""
        try:
            from flimkit.interactive import _run_tile_fit
        except ImportError as e:
            pytest.skip(f"Required module not available: {e}")

        from unittest.mock import patch

        tile_results, ch, cw = self._tile_results("2x2", n_exp=1)

        with tempfile.TemporaryDirectory() as tmp:
            # Write a Gaussian machine IRF .npy
            sigma   = 3.0 / 2.3548
            bins    = np.arange(self.N_BINS, dtype=float)
            irf_arr = np.exp(-0.5 * ((bins - 30) / sigma) ** 2)
            irf_arr /= irf_arr.sum()
            mirf_path = Path(tmp) / 'machine_irf_test.npy'
            np.save(str(mirf_path), irf_arr.astype(np.float64))

            args = self._base_args(Path(tmp))
            args.estimate_irf = 'machine_irf'
            args.machine_irf  = str(mirf_path)

            with patch('flimkit.PTU.stitch.fit_flim_tiles',
                       return_value=self._fit_flim_tiles_return_value(tile_results, ch, cw, n_exp=1)):
                fit_result = _run_tile_fit(args)
            canvas = fit_result['canvas']
            gs = fit_result['global_summary']

            assert 'intensity' in canvas
            assert gs['n_pixels_fitted'] > 0


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
            f"Missing required modules: {', '.join(missing_required)}"
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
            assert np.any(np.abs(tau_phase - synthetic_phasor['tau_ns']) < 1.5),                 f"No phase lifetime near expected {synthetic_phasor['tau_ns']} ns"

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
