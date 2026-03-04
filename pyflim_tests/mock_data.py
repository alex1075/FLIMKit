"""Mock Data Generator for FLIM Testing

Creates synthetic PTU files, XLIF metadata, and test data
for testing the FLIM pipeline without requiring real microscopy data.
"""

import numpy as np
import json
from pathlib import Path
from typing import Tuple, List, Dict, Any
import xml.etree.ElementTree as ET


# ─── Ground-truth constants (also used by MockPTUFile._generate_synthetic_data) ─
MOCK_TAU1_NS       = 0.5     # Short component lifetime (ns)
MOCK_TAU2_NS       = 3.0     # Long component lifetime (ns)
MOCK_AMP1          = 0.6     # Amplitude fraction of τ₁
MOCK_AMP2          = 0.4     # Amplitude fraction of τ₂
MOCK_IRF_CENTER    = 30      # IRF peak (bin index)
MOCK_IRF_FWHM_BINS = 3.0    # IRF FWHM in bins
MOCK_TCSPC_RES     = 97e-12  # Default TCSPC resolution (s)
MOCK_FREQUENCY     = 19.5e6  # Default laser repetition rate (Hz)
MOCK_MEAN_PHOTONS  = 500     # Mean photons per pixel


class MockPTUFile:
    """Mock PTUFile class for testing."""
    
    def __init__(self, path=None, verbose=False, **kwargs):
        """
        Create a mock PTU file.
        
        If path is provided, load mock data from a .npy file.
        Otherwise, generate synthetic data using kwargs.
        """
        if path is not None:
            # Load from .npy file
            data = np.load(path, allow_pickle=True).item()
            self.n_y = data['stack'].shape[0]
            self.n_x = data['stack'].shape[1]
            self.n_bins = data['stack'].shape[2]
            self.tcspc_res = data['tcspc_res']
            self.frequency = data['frequency']
            self.sync_rate = data['frequency']       # alias used by decode
            self.time_ns = np.arange(self.n_bins) * self.tcspc_res * 1e9
            self.photon_channel = 1
            self._stack = data['stack'].astype(np.float32)
        else:
            # Generate synthetic data
            self.n_y = kwargs.get('n_y', 512)
            self.n_x = kwargs.get('n_x', 512)
            self.n_bins = kwargs.get('n_bins', 256)
            self.tcspc_res = kwargs.get('tcspc_res', 97e-12)
            self.frequency = kwargs.get('frequency', 19.5e6)
            self.sync_rate = self.frequency              # alias used by decode
            self.time_ns = np.arange(self.n_bins) * self.tcspc_res * 1e9
            self.photon_channel = 1
            self._generate_synthetic_data()
    
    def _generate_synthetic_data(self):
        """Generate synthetic FLIM histogram with realistic decay."""
        t = np.arange(self.n_bins) * self.tcspc_res

        # Bi‑exponential decay (uses module-level ground-truth constants)
        tau1 = MOCK_TAU1_NS * 1e-9
        tau2 = MOCK_TAU2_NS * 1e-9
        a1 = MOCK_AMP1
        a2 = MOCK_AMP2
        decay_kernel = a1 * np.exp(-t / tau1) + a2 * np.exp(-t / tau2)

        # IRF: Gaussian centered at MOCK_IRF_CENTER, FWHM = MOCK_IRF_FWHM_BINS
        irf_center = MOCK_IRF_CENTER
        irf_sigma = MOCK_IRF_FWHM_BINS / 2.355
        x = np.arange(self.n_bins)
        irf = np.exp(-0.5 * ((x - irf_center) / irf_sigma) ** 2)
        irf = irf / irf.sum()

        # Convolve (mode='same' aligns the peak with the IRF center)
        decay_profile = np.convolve(decay_kernel, irf, mode='same')
        decay_profile = decay_profile / decay_profile.max()

        # Force peak to bin 30
        desired_peak = 30
        current_peak = np.argmax(decay_profile)
        shift = desired_peak - current_peak
        if shift != 0:
            decay_profile = np.roll(decay_profile, shift)
        # Spatial pattern (circular gradient)
        y, x = np.ogrid[:self.n_y, :self.n_x]
        cy, cx = self.n_y // 2, self.n_x // 2
        r = np.sqrt((y - cy)**2 + (x - cx)**2)
        spatial = 1.0 - 0.5 * (r / (self.n_y / 2))
        spatial = np.clip(spatial, 0.2, 1.0)

        mean_photons = 500
        self._stack = np.zeros((self.n_y, self.n_x, self.n_bins), dtype=np.float32)
        for i in range(self.n_y):
            for j in range(self.n_x):
                intensity = mean_photons * spatial[i, j]
                photon_dist = intensity * decay_profile
                self._stack[i, j, :] = np.random.poisson(photon_dist)
    
    def summed_decay(self, channel=None):
        return self._stack.sum(axis=(0, 1))
    
    def pixel_stack(self, channel=None, binning=1):
        if binning == 1:
            return self._stack
        ny, nx, nt = self._stack.shape
        new_ny = ny // binning
        new_nx = nx // binning
        binned = np.zeros((new_ny, new_nx, nt), dtype=self._stack.dtype)
        for i in range(new_ny):
            for j in range(new_nx):
                binned[i, j, :] = self._stack[
                    i*binning:(i+1)*binning,
                    j*binning:(j+1)*binning,
                    :
                ].sum(axis=(0, 1))
        return binned


def generate_mock_xlif(
    output_path: Path,
    n_tiles: int = 4,
    tile_size: int = 512,
    pixel_size_m: float = 3e-7,
    layout: str = "2x2"
) -> Path:
    """
    Generate a mock XLIF metadata file.
    
    Args:
        output_path: Where to save the XLIF file
        n_tiles: Number of tiles
        tile_size: Size of each tile in pixels
        pixel_size_m: Pixel size in meters
        layout: Tile layout ("2x2", "3x3", "1x4", etc.)
    
    Returns:
        Path to created XLIF file
    """
    # Parse layout
    if 'x' in layout:
        rows, cols = map(int, layout.split('x'))
    else:
        rows, cols = 1, n_tiles
    
    if rows * cols != n_tiles:
        raise ValueError(f"Layout {layout} doesn't match n_tiles={n_tiles}")
    
    # Create XML structure
    root = ET.Element("LMSDataContainerHeader")
    
    # Add basic metadata
    version = ET.SubElement(root, "Version")
    version.text = "1.0"
    
    # Add dimension description
    dimensions = ET.SubElement(root, "Dimensions")
    dim_desc = ET.SubElement(dimensions, "DimensionDescription", 
                             DimID="1", NumberOfElements=str(tile_size))
    dim_desc.set("Length", str(tile_size * pixel_size_m))
    
    # Add TileScanInfo
    tile_scan_info = ET.SubElement(root, "Attachment", Name="TileScanInfo")
    
    # Add tile positions
    tile_idx = 0
    for row in range(rows):
        for col in range(cols):
            pos_x = col * tile_size * pixel_size_m
            pos_y = row * tile_size * pixel_size_m
            
            tile = ET.SubElement(tile_scan_info, "Tile",
                                FieldX=str(tile_idx),
                                PosX=str(pos_x),
                                PosY=str(pos_y))
            tile_idx += 1
    
    # Write to file
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    
    return output_path


def generate_mock_ptu_tiles(
    output_dir: Path,
    ptu_basename: str,
    n_tiles: int = 4,
    tile_shape: Tuple[int, int] = (512, 512),
    n_bins: int = 256
) -> List[Path]:
    """
    Generate mock PTU tile files.
    
    Args:
        output_dir: Directory to save PTU files
        ptu_basename: Base name (e.g., "R 2")
        n_tiles: Number of tiles
        tile_shape: (height, width) of each tile
        n_bins: Number of time bins
    
    Returns:
        List of paths to created PTU files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    ptu_files = []
    
    for tile_idx in range(n_tiles):
        # Create mock PTU data
        mock_ptu = MockPTUFile(
            n_y=tile_shape[0],
            n_x=tile_shape[1],
            n_bins=n_bins
        )
        
        # Save as numpy array (in real code, this would be PTU format)
        filename = f"{ptu_basename}_s{tile_idx + 1}.npy"
        filepath = output_dir / filename
        
        # Save stack and metadata
        data = {
            'stack': mock_ptu._stack,
            'tcspc_res': mock_ptu.tcspc_res,
            'n_bins': mock_ptu.n_bins,
            'frequency': mock_ptu.frequency,
        }
        
        np.save(filepath, data, allow_pickle=True)
        ptu_files.append(filepath)
    
    return ptu_files


def generate_test_project(
    base_dir: Path,
    roi_name: str = "R 2",
    n_tiles: int = 4,
    layout: str = "2x2"
) -> Dict[str, Any]:
    """
    Generate a complete test project with XLIF and PTU files.
    
    Args:
        base_dir: Base directory for test project
        roi_name: ROI identifier
        n_tiles: Number of tiles
        layout: Tile layout
    
    Returns:
        Dict with paths to all generated files
    """
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Create directory structure
    ptu_dir = base_dir / "PTUs"
    metadata_dir = base_dir / "Metadata"
    ptu_dir.mkdir(exist_ok=True)
    metadata_dir.mkdir(exist_ok=True)
    
    # Generate XLIF
    xlif_path = generate_mock_xlif(
        metadata_dir / f"{roi_name}.xlif",
        n_tiles=n_tiles,
        layout=layout
    )
    
    # Generate PTU tiles
    ptu_files = generate_mock_ptu_tiles(
        ptu_dir,
        roi_name,
        n_tiles=n_tiles
    )
    
    return {
        'base_dir': base_dir,
        'xlif_path': xlif_path,
        'ptu_dir': ptu_dir,
        'ptu_files': ptu_files,
        'roi_name': roi_name,
        'n_tiles': n_tiles,
    }


def generate_synthetic_decay(
    n_bins: int = 256,
    tcspc_res: float = MOCK_TCSPC_RES,
    tau_ns: float = 2.0,
    bg: float = 10.0,
    peak_counts: float = 1000.0,
    irf_fwhm_bins: float = MOCK_IRF_FWHM_BINS,
    irf_center_bin: int = MOCK_IRF_CENTER,
    noise: bool = True,
) -> np.ndarray:
    """Generate a synthetic single-exp decay using circular FFT reconvolution.

    Uses the **same** forward model as the fitter
    (``ifft(fft(kernel) * fft(irf))``), so the fitter can recover the
    known lifetime.

    Parameters
    ----------
    n_bins : int
        Number of TCSPC time bins.
    tcspc_res : float
        Bin width in seconds.
    tau_ns : float
        Lifetime in nanoseconds.
    bg : float
        Background counts per bin added **after** convolution.
    peak_counts : float
        Peak value of the decay (before noise).
    irf_fwhm_bins : float
        IRF FWHM in bins.
    irf_center_bin : int
        IRF peak position (bin index).
    noise : bool
        Add Poisson noise.

    Returns
    -------
    np.ndarray, shape (n_bins,)
    """
    t = np.arange(n_bins, dtype=float) * tcspc_res
    tau_s = tau_ns * 1e-9

    # Kernel: same as fitter's _exponential_kernel (pure exp from t = 0)
    kernel = np.exp(-t / tau_s)

    # Gaussian IRF centred at irf_center_bin (same formula as
    # gaussian_irf_from_fwhm / gaussian_irf in irf_tools.py)
    sigma = irf_fwhm_bins / 2.3548
    bins = np.arange(n_bins, dtype=float)
    irf = np.exp(-0.5 * ((bins - irf_center_bin) / sigma) ** 2)
    irf /= irf.sum()

    # Circular FFT convolution — matches reconvolution_model exactly
    model = np.real(np.fft.ifft(np.fft.fft(kernel) * np.fft.fft(irf)))

    # Scale to desired peak counts and add flat background
    model = model / model.max() * peak_counts + bg

    if noise:
        rng = np.random.default_rng(0)
        model = rng.poisson(np.maximum(model, 0)).astype(float)

    return model


def generate_synthetic_biexp_decay(
    n_bins: int = 256,
    tcspc_res: float = MOCK_TCSPC_RES,
    tau1_ns: float = MOCK_TAU1_NS,
    tau2_ns: float = MOCK_TAU2_NS,
    a1: float = MOCK_AMP1,
    a2: float = MOCK_AMP2,
    bg: float = 5.0,
    peak_counts: float = 50_000.0,
    irf_fwhm_bins: float = MOCK_IRF_FWHM_BINS,
    irf_center_bin: int = MOCK_IRF_CENTER,
    noise: bool = True,
) -> np.ndarray:
    """Generate a bi-exponential decay using circular FFT reconvolution.

    Uses the **same** forward model as the fitter
    (``ifft(fft(kernel) * fft(irf))``), so the fitter can recover the
    known lifetimes and amplitude fractions.

    Returns
    -------
    np.ndarray, shape (n_bins,)
        Photon-count histogram (float).
    """
    t = np.arange(n_bins, dtype=float) * tcspc_res
    tau1_s = tau1_ns * 1e-9
    tau2_s = tau2_ns * 1e-9

    # Kernel: same as fitter's _exponential_kernel
    kernel = a1 * np.exp(-t / tau1_s) + a2 * np.exp(-t / tau2_s)

    # Gaussian IRF centred at irf_center_bin
    sigma = irf_fwhm_bins / 2.3548
    bins = np.arange(n_bins, dtype=float)
    irf = np.exp(-0.5 * ((bins - irf_center_bin) / sigma) ** 2)
    irf /= irf.sum()

    # Circular FFT convolution — matches reconvolution_model exactly
    model = np.real(np.fft.ifft(np.fft.fft(kernel) * np.fft.fft(irf)))

    # Scale to desired peak counts and add flat background
    model = model / model.max() * peak_counts + bg

    if noise:
        rng = np.random.default_rng(0)
        model = rng.poisson(np.maximum(model, 0)).astype(float)

    return model


def load_mock_ptu_file(ptu_path: Path) -> MockPTUFile:
    """
    Load a mock PTU file created by generate_mock_ptu_tiles.
    
    Args:
        ptu_path: Path to .npy file
    
    Returns:
        MockPTUFile instance
    """
    data = np.load(ptu_path, allow_pickle=True).item()
    
    mock_ptu = MockPTUFile(
        n_y=data['stack'].shape[0],
        n_x=data['stack'].shape[1],
        n_bins=data['stack'].shape[2],
        tcspc_res=data['tcspc_res'],
        frequency=data['frequency']
    )
    
    mock_ptu._stack = data['stack']
    
    return mock_ptu
