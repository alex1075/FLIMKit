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
    }
    
    optional = {
        'pytest': 'Pytest (for testing)',
        'matplotlib': 'Matplotlib (for plotting)',
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
    ]
    
    optional_modules = [
        ('pyflim.interactive', 'Interactive workflows'),
        ('pyflim.FLIM.fitters', 'FLIM fitting'),
        ('pyflim.FLIM.irf_tools', 'IRF tools'),
        ('pyflim.PTU.reader', 'PTUFile reader'),
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
            
            # Try fitting if available
            try:
                from pyflim.FLIM.fitters import fit_summed
                from pyflim.FLIM.irf_tools import gaussian_irf_from_fwhm
                
                irf = gaussian_irf_from_fwhm(n_bins, tcspc_res, 0.3, 20)
                
                popt, summary = fit_summed(
                    decay, tcspc_res, n_bins, irf,
                    has_tail=False, fit_bg=True, fit_sigma=False,
                    n_exp=1, tau_min_ns=0.1, tau_max_ns=10.0,
                    optimizer="lm_multistart", n_restarts=2, workers=1
                )

                assert summary is not None, "Fit failed"
                assert 'taus_ns' in summary, "Missing taus in results"
                assert len(summary['taus_ns']) > 0, "No tau values"
                tau = summary['taus_ns'][0]          # first component
                chi2 = summary['reduced_chi2']       # reduced chi‑squared
                assert 0.1 < tau < 10.0, f"Tau {tau} out of expected range"

                print_success(f"Step 4: Fitted (τ={tau:.3f} ns, χ²={chi2:.3f})")
                                
            except ImportError:
                print_warning("Step 4: Fitting not available (fitters module missing)")
            
            return True
            
    except Exception as e:
        print_error(f"Complete workflow test failed: {e}")
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