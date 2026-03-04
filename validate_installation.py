#!/usr/bin/env python3
"""
Installation Validation Script

Checks that all components are properly installed and working.
Runs quick sanity tests without requiring pytest.
"""

import sys
import traceback
from pathlib import Path
from unittest.mock import patch

# Ensure the project root is on the path
project_root = str(Path(__file__).parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now we can import from pyflim and mock_data
from pyflim_tests.mock_data import MockPTUFile, generate_test_project


class Colors:
    """ANSI color codes."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'
    BOLD = '\033[1m'


def print_header(text):
    """Print section header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text:^60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}\n")


def print_success(text):
    """Print success message."""
    print(f"{Colors.GREEN}✓{Colors.END} {text}")


def print_error(text):
    """Print error message."""
    print(f"{Colors.RED}✗{Colors.END} {text}")


def print_warning(text):
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠{Colors.END} {text}")


def check_dependencies():
    """Check that required dependencies are installed."""
    print_header("Checking Dependencies")
    
    required = {
        'numpy': 'NumPy',
        'scipy': 'SciPy',
        'tifffile': 'TiffFile',
        'phasorpy': 'PhasorPy',
        'xarray': 'xarray',
        'inquirer': 'Inquirer',
    }
    
    optional = {
        'pytest': 'Pytest (for testing)',
        'matplotlib': 'Matplotlib (for plotting)',
        'ipywidgets': 'ipywidgets (notebook interactive)',
        'ipympl': 'ipympl (notebook matplotlib backend)',
        'pandas': 'Pandas (IRF Excel reading)',
    }
    
    all_ok = True
    
    for module, name in required.items():
        try:
            __import__(module)
            print_success(f"{name} installed")
        except ImportError:
            print_error(f"{name} NOT installed (required)")
            all_ok = False
    
    for module, name in optional.items():
        try:
            __import__(module)
            print_success(f"{name} installed")
        except ImportError:
            print_warning(f"{name} not installed (optional)")
    
    return all_ok


def check_simplified_integration():
    """Check that simplified integration files are present."""
    print_header("Checking Simplified Integration")
    
    required_files = [
        'pyflim/utils/xml_utils.py',
        'pyflim/PTU/decode.py',
        'pyflim/PTU/stitch.py',
    ]
    
    all_ok = True
    
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            print_success(f"{file_path} found")
        else:
            print_error(f"{file_path} NOT found")
            all_ok = False
    
    return all_ok


def check_modules_import():
    """Check that modules can be imported."""
    print_header("Checking Module Imports")
    
    modules = [
        ('pyflim.utils.xml_utils', 'XML/XLIF parsing'),
        ('pyflim.PTU.decode', 'PTU decoding'),
        ('pyflim.PTU.stitch', 'Tile stitching'),
        ('pyflim.PTU.tools', 'PTU signal tools'),
    ]
    
    optional_modules = [
        ('pyflim.interactive', 'Interactive FLIM workflows'),
        ('pyflim.FLIM.fitters', 'FLIM fitting'),
        ('pyflim.FLIM.irf_tools', 'IRF tools'),
        ('pyflim.PTU.reader', 'PTUFile reader'),
        ('pyflim.phasor.signal', 'Phasor signal processing'),
        ('pyflim.phasor.interactive', 'Phasor interactive tool'),
        ('pyflim.phasor.peaks', 'Phasor peak detection'),
        ('pyflim.phasor_launcher', 'Phasor launcher'),
    ]
    
    all_ok = True
    
    for module, description in modules:
        try:
            __import__(module)
            print_success(f"{description} ({module})")
        except ImportError as e:
            print_error(f"{description} ({module}): {e}")
            all_ok = False
    
    for module, description in optional_modules:
        try:
            __import__(module)
            print_success(f"{description} ({module})")
        except ImportError:
            print_warning(f"{description} ({module}) not available")
    
    return all_ok


def test_xml_parsing():
    """Test XML/XLIF parsing functionality."""
    print_header("Testing XML/XLIF Parsing")
    
    try:
        from pyflim.utils.xml_utils import parse_xlif_tile_positions, compute_tile_pixel_positions
        import tempfile
        from pyflim_tests.mock_data import generate_mock_xlif
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Generate mock XLIF
            xlif_path = generate_mock_xlif(
                Path(temp_dir) / "test.xlif",
                n_tiles=4,
                layout="2x2"
            )
            
            # Parse it
            tiles = parse_xlif_tile_positions(xlif_path, "R 2")
            
            assert len(tiles) == 4, "Expected 4 tiles"
            assert all('file' in t for t in tiles), "Missing file field"
            
            # Compute pixel positions
            tiles, width, height = compute_tile_pixel_positions(
                tiles, pixel_size_m=3e-7, tile_size=512
            )
            
            assert width == 1024, f"Expected width 1024, got {width}"
            assert height == 1024, f"Expected height 1024, got {height}"
            
            print_success("XLIF parsing works correctly")
            return True
            
    except Exception as e:
        print_error(f"XLIF parsing test failed: {e}")
        traceback.print_exc()
        return False


def test_mock_data():
    """Test mock data generation."""
    print_header("Testing Mock Data Generation")
    
    try:
        from pyflim_tests.mock_data import MockPTUFile, generate_synthetic_decay
        import numpy as np
        
        # Test MockPTUFile
        ptu = MockPTUFile(n_y=256, n_x=256, n_bins=128)
        decay = ptu.summed_decay()
        stack = ptu.pixel_stack()
        
        assert decay.shape == (128,), f"Wrong decay shape: {decay.shape}"
        assert stack.shape == (256, 256, 128), f"Wrong stack shape: {stack.shape}"
        assert decay.sum() > 0, "Decay has no photons"
        
        print_success("MockPTUFile works correctly")
        
        # Test synthetic decay
        syn_decay = generate_synthetic_decay(
            n_bins=256,
            tau_ns=2.0,
            noise=True
        )
        
        assert len(syn_decay) == 256, "Wrong synthetic decay length"
        assert syn_decay.sum() > 0, "Synthetic decay has no photons"
        
        print_success("Synthetic decay generation works correctly")
        
        return True
        
    except Exception as e:
        print_error(f"Mock data test failed: {e}")
        traceback.print_exc()
        return False


def test_stitching():
    """Test tile stitching functionality."""
    print_header("Testing Tile Stitching")
    
    try:
        from pyflim.PTU.stitch import stitch_flim_tiles, load_stitched_flim
        import tempfile
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Generate test project
            project = generate_test_project(
                Path(temp_dir),
                roi_name="R 2",
                n_tiles=4,
                layout="2x2"
            )
            
            # Rename .npy mock files to .ptu (the stitching code expects .ptu)
            for npy_file in project['ptu_dir'].glob('*.npy'):
                ptu_file = npy_file.with_suffix('.ptu')
                npy_file.rename(ptu_file)
            
            output_dir = project['base_dir'] / "stitched"
            
            # Patch PTUFile with MockPTUFile so the .ptu files (actually .npy) are read correctly
            with patch('pyflim.PTU.reader.PTUFile', MockPTUFile):
                result = stitch_flim_tiles(
                    xlif_path=project['xlif_path'],
                    ptu_dir=project['ptu_dir'],
                    output_dir=output_dir,
                    ptu_basename=project['roi_name'],
                    rotate_tiles=True,
                    verbose=False
                )
            
            assert result['tiles_processed'] == 4, "Not all tiles processed"
            assert result['canvas_shape'] == (1024, 1024), "Wrong canvas size"
            
            print_success(f"Stitched 4 tiles → {result['canvas_shape']}")
            
            # Load back (no patch needed for reading, because the files are now .npy memmaps)
            flim, time_axis, intensity, metadata = load_stitched_flim(output_dir)
            
            assert flim.shape == (1024, 1024, 256), f"Wrong FLIM shape: {flim.shape}"
            assert len(time_axis) == 256, "Wrong time axis length"
            assert intensity.shape == (1024, 1024), "Wrong intensity shape"
            
            print_success("Stitched data loads correctly")
            return True
            
    except Exception as e:
        print_error(f"Stitching test failed: {e}")
        traceback.print_exc()
        return False


def test_complete_workflow():
    """Test complete stitch + fit workflow."""
    print_header("Testing Complete Workflow")
    
    try:
        from pyflim.PTU.stitch import stitch_flim_tiles, load_flim_for_fitting
        import tempfile
        import numpy as np
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Generate and stitch
            project = generate_test_project(
                Path(temp_dir),
                roi_name="R 2",
                n_tiles=4,
                layout="2x2"
            )
            
            # Rename .npy to .ptu
            for npy_file in project['ptu_dir'].glob('*.npy'):
                ptu_file = npy_file.with_suffix('.ptu')
                npy_file.rename(ptu_file)
            
            output_dir = project['base_dir'] / "stitched"
            
            with patch('pyflim.PTU.reader.PTUFile', MockPTUFile):
                result = stitch_flim_tiles(
                    xlif_path=project['xlif_path'],
                    ptu_dir=project['ptu_dir'],
                    output_dir=output_dir,
                    ptu_basename=project['roi_name'],
                    verbose=False
                )
            
            print_success(f"Step 1: Stitched {result['tiles_processed']} tiles")
            
            # Load for fitting (no patch needed here because load_flim_for_fitting reads .npy directly)
            stack, tcspc_res, n_bins = load_flim_for_fitting(
                output_dir,
                load_to_memory=True
            )
            
            assert stack.shape == (1024, 1024, 256), "Wrong stack shape"
            assert tcspc_res > 0, "Invalid TCSPC resolution"
            assert n_bins == 256, "Wrong number of bins"
            
            print_success("Step 2: Loaded data for fitting")
            
            # Check can extract decay
            decay = stack.sum(axis=(0, 1))
            assert decay.sum() > 0, "No photons in decay"
            
            print_success(f"Step 3: Extracted decay ({decay.sum():.0f} photons)")
            
            # Try fitting if available — use a standalone synthetic decay with
            # realistic photon counts so the fitter works in its intended regime.
            # (The stitched mosaic has billions of photons which inflates χ².)
            try:
                from pyflim.FLIM.fitters import fit_summed
                from pyflim.FLIM.irf_tools import gaussian_irf_from_fwhm
                from pyflim_tests.mock_data import (
                    MOCK_TAU1_NS, MOCK_TAU2_NS, MOCK_IRF_FWHM_BINS,
                    MOCK_IRF_CENTER, MOCK_TCSPC_RES,
                    generate_synthetic_biexp_decay,
                )
                
                fit_n_bins   = 256
                fit_tcspc    = MOCK_TCSPC_RES
                irf_fwhm_ns  = MOCK_IRF_FWHM_BINS * fit_tcspc * 1e9
                irf = gaussian_irf_from_fwhm(fit_n_bins, fit_tcspc, irf_fwhm_ns,
                                             MOCK_IRF_CENTER)

                fit_decay = generate_synthetic_biexp_decay(
                    n_bins=fit_n_bins, tcspc_res=fit_tcspc,
                    peak_counts=100_000.0, noise=True,
                )

                popt, summary = fit_summed(
                    fit_decay, fit_tcspc, fit_n_bins, irf,
                    has_tail=False, fit_bg=True, fit_sigma=False,
                    n_exp=2, tau_min_ns=0.05, tau_max_ns=15.0,
                    optimizer="lm_multistart", n_restarts=10, workers=1,
                    cost_function="poisson",
                )

                assert summary is not None, "Fit failed"
                taus   = summary['taus_ns']        # sorted descending
                chi2_r = summary['reduced_chi2_tail_pearson']  # Leica X2

                # Ground-truth comparison (15 % for realistic photon count)
                rel_long  = abs(taus[0] - MOCK_TAU2_NS) / MOCK_TAU2_NS
                rel_short = abs(taus[1] - MOCK_TAU1_NS) / MOCK_TAU1_NS
                assert rel_long  < 0.15, f"Long τ err {rel_long:.0%}"
                assert rel_short < 0.15, f"Short τ err {rel_short:.0%}"
                assert chi2_r < 3.0, f"Pearson tail χ²_r = {chi2_r:.2f}"

                print_success(
                    f"Step 4: Bi-exp fit "
                    f"(τ₁={taus[1]:.2f} vs {MOCK_TAU1_NS} ns [{rel_short:.0%}], "
                    f"τ₂={taus[0]:.2f} vs {MOCK_TAU2_NS} ns [{rel_long:.0%}], "
                    f"χ²_r={chi2_r:.2f})"
                )
                                
            except ImportError:
                print_warning("Step 4: Fitting not available (fitters module missing)")
            
            return True
            
    except Exception as e:
        print_error(f"Complete workflow test failed: {e}")
        traceback.print_exc()
        return False


def test_phasor_pipeline():
    """Test phasor analysis pipeline with synthetic data."""
    print_header("Testing Phasor Pipeline")

    try:
        import numpy as np

        # ── Synthetic phasor data (no real PTU needed) ───────
        rng = np.random.default_rng(42)
        shape = (64, 64)
        # Single-exponential → point ON the semicircle
        tau_ns = 2.5
        frequency = 40.0  # MHz
        omega = 2 * np.pi * frequency * 1e-3  # rad/ns
        g_true = 1 / (1 + (omega * tau_ns) ** 2)
        s_true = omega * tau_ns / (1 + (omega * tau_ns) ** 2)

        real_cal = rng.normal(g_true, 0.02, shape)
        imag_cal = rng.normal(s_true, 0.02, shape)
        mean = rng.uniform(5, 50, shape)

        print_success("Step 1: Generated synthetic phasor data")

        # ── Peak detection ───────────────────────────────────
        from pyflim.phasor.peaks import find_phasor_peaks

        peaks = find_phasor_peaks(real_cal, imag_cal, mean, frequency)
        assert peaks['n_peaks'] >= 1, "No peaks found"
        print_success(f"Step 2: Found {peaks['n_peaks']} peak(s)")

        # Peak should be near the expected (g_true, s_true)
        best_idx = 0
        best_dist = float('inf')
        for i in range(peaks['n_peaks']):
            d = np.sqrt((peaks['peak_g'][i] - g_true) ** 2 +
                        (peaks['peak_s'][i] - s_true) ** 2)
            if d < best_dist:
                best_dist = d
                best_idx = i
        assert best_dist < 0.05, f"Closest peak too far from expected: {best_dist:.3f}"
        print_success(f"Step 3: Peak distance = {best_dist:.4f} (< 0.05)")

        # Phase lifetime should match input τ within 15 %
        tau_phase = peaks['tau_phase'][best_idx]
        rel_err = abs(tau_phase - tau_ns) / tau_ns
        assert rel_err < 0.15, f"Phase τ = {tau_phase:.3f} vs true {tau_ns} (err {rel_err:.0%})"
        print_success(f"Step 3b: Phase τ = {tau_phase:.3f} ns vs true {tau_ns} (err {rel_err:.1%})")

        # ── Save / load session ──────────────────────────────
        from pyflim.phasor_launcher import save_session, load_session
        import tempfile, os

        cursors = [dict(center_g=float(peaks['peak_g'][0]),
                        center_s=float(peaks['peak_s'][0]),
                        color='#d62728')]
        params = dict(radius=0.05, radius_minor=0.03, angle_mode='semicircle')

        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            tmp_path = f.name

        try:
            save_session(tmp_path,
                         real_cal=real_cal, imag_cal=imag_cal, mean=mean,
                         frequency=frequency, cursors=cursors, params=params,
                         ptu_file='synthetic.ptu')

            sess = load_session(tmp_path)
            assert sess['frequency'] == frequency
            assert len(sess['cursors']) == 1
            np.testing.assert_array_almost_equal(sess['real_cal'], real_cal)
            print_success("Step 4: Save / load session round-trip OK")
        finally:
            os.unlink(tmp_path)

        return True

    except Exception as e:
        print_error(f"Phasor pipeline test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all validation checks."""
    print(f"\n{Colors.BOLD}FLIM Pipeline Installation Validation{Colors.END}")
    print(f"{Colors.BOLD}{'='*60}{Colors.END}\n")
    
    results = []
    
    results.append(("Dependencies", check_dependencies()))
    results.append(("Simplified Integration Files", check_simplified_integration()))
    results.append(("Module Imports", check_modules_import()))
    results.append(("XML/XLIF Parsing", test_xml_parsing()))
    results.append(("Mock Data Generation", test_mock_data()))
    results.append(("Tile Stitching", test_stitching()))
    results.append(("Complete Workflow", test_complete_workflow()))
    results.append(("Phasor Pipeline", test_phasor_pipeline()))
    
    print_header("Validation Summary")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        if result:
            print_success(f"{name}: PASSED")
        else:
            print_error(f"{name}: FAILED")
    
    print(f"\n{Colors.BOLD}Results: {passed}/{total} checks passed{Colors.END}\n")
    
    if passed == total:
        print(f"{Colors.GREEN}{Colors.BOLD}✓ All checks passed! Your installation is working correctly.{Colors.END}\n")
        return 0
    else:
        print(f"{Colors.RED}{Colors.BOLD}✗ Some checks failed. See errors above.{Colors.END}\n")
        print("Common fixes:")
        print("- Ensure you have installed all the required dependencies (see the first check).")
        print("- Check that you are using the correct Python environment where the package is installed.")
        print("- Pray that the error messages above give a clue about what is missing or misconfigured.")
        print("- Consider all your life choices that led to this moment. Just kidding, but seriously, check the error messages and fix the issues.")
        return 1


if __name__ == "__main__":
    sys.exit(main())