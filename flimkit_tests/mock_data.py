import numpy as np
import json
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
import xml.etree.ElementTree as ET


# Ground-truth constants (also used by MockPTUFile._generate_synthetic_data)
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
            self.mean_photons = kwargs.get('mean_photons', 500)
            self.sync_rate = self.frequency              # alias used by decode
            self.time_ns = np.arange(self.n_bins) * self.tcspc_res * 1e9
            self.photon_channel = 1
            self._generate_synthetic_data()
    
    def _generate_synthetic_data(self):
        """Generate synthetic FLIM histogram with realistic decay."""
        t = np.arange(self.n_bins, dtype=float) * self.tcspc_res

        # Bi‑exponential decay (uses module-level ground-truth constants)
        tau1 = MOCK_TAU1_NS * 1e-9
        tau2 = MOCK_TAU2_NS * 1e-9
        a1 = MOCK_AMP1
        a2 = MOCK_AMP2
        decay_kernel = a1 * np.exp(-t / tau1) + a2 * np.exp(-t / tau2)

        # IRF: Gaussian centered at MOCK_IRF_CENTER, FWHM = MOCK_IRF_FWHM_BINS
        irf_sigma = MOCK_IRF_FWHM_BINS / 2.3548
        bins = np.arange(self.n_bins, dtype=float)
        irf = np.exp(-0.5 * ((bins - MOCK_IRF_CENTER) / irf_sigma) ** 2)
        irf = irf / irf.sum()

        # Circular FFT convolution — matches reconvolution_model exactly
        decay_profile = np.real(np.fft.ifft(
            np.fft.fft(decay_kernel) * np.fft.fft(irf)))
        decay_profile = decay_profile / decay_profile.max()

        # Spatial pattern (circular gradient)
        y, x = np.ogrid[:self.n_y, :self.n_x]
        cy, cx = self.n_y // 2, self.n_x // 2
        r = np.sqrt((y - cy)**2 + (x - cx)**2)
        spatial = 1.0 - 0.5 * (r / (self.n_y / 2))
        spatial = np.clip(spatial, 0.2, 1.0)

        self._stack = np.zeros((self.n_y, self.n_x, self.n_bins), dtype=np.float32)
        for i in range(self.n_y):
            for j in range(self.n_x):
                intensity = self.mean_photons * spatial[i, j]
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
    tile_shape: Tuple[int, int] = (64, 64),
    n_bins: int = 128,
    mean_photons: int = 50
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
    
    from flimkit.PTU.reader import PTUFile as _PTUFile

    ptu_files = []

    for tile_idx in range(n_tiles):
        mock_ptu = MockPTUFile(
            n_y=tile_shape[0],
            n_x=tile_shape[1],
            n_bins=n_bins,
            mean_photons=mean_photons,
        )

        filepath = output_dir / f"{ptu_basename}_s{tile_idx + 1}.ptu"

        _PTUFile.write(
            filepath,
            mock_ptu._stack.astype(np.uint16),
            tcspc_res=mock_ptu.tcspc_res,
            frequency=mock_ptu.frequency,
        )

        ptu_files.append(filepath)

    return ptu_files


def generate_test_project(
    base_dir: Path,
    roi_name: str = "R 2",
    n_tiles: int = 4,
    layout: str = "2x2",
    tile_shape: Tuple[int, int] = (64, 64),
    n_bins: int = 128,
    mean_photons: int = 50,
    tile_size: Optional[int] = None
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
    _tile_size = tile_size if tile_size is not None else tile_shape[0]
    xlif_path = generate_mock_xlif(
        metadata_dir / f"{roi_name}.xlif",
        n_tiles=n_tiles,
        tile_size=_tile_size,
        layout=layout
    )
    
    # Generate PTU tiles
    ptu_files = generate_mock_ptu_tiles(
        ptu_dir,
        roi_name,
        n_tiles=n_tiles,
        tile_shape=tile_shape,
        n_bins=n_bins,
        mean_photons=mean_photons
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
    n_bins: int = 128,
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
    n_bins: int = 128,
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


def load_mock_ptu_file(ptu_path: Path):
    """Load a PTU file written by generate_mock_ptu_tiles() and return an object with .summed_decay()."""
    from flimkit.PTU.reader import PTUFile
    ptu = PTUFile(str(ptu_path), verbose=False)
    class _Wrapper:
        def __init__(self, ptu):
            self.ptu = ptu
        def summed_decay(self, channel=None):
            return self.ptu.summed_decay(channel)
    return _Wrapper(ptu)


# ---------------------------------------------------------------------------
# FRET phasor mock data
# ---------------------------------------------------------------------------

# Ground-truth FRET parameters used throughout the FRET test suite.
MOCK_FRET_FREQ   = 80.0   # MHz  — laser repetition rate
MOCK_FRET_TAU_D  = 4.0    # ns   — unquenched donor lifetime
MOCK_FRET_TAU_A  = 3.0    # ns   — acceptor lifetime


def fret_donor_phasor_truth(efficiency: float) -> Tuple[float, float]:
    """Return the exact phasorpy FRET donor ``(G, S)`` for a given efficiency.

    This is the ground truth that ``generate_fret_donor_image`` centres its
    pixels on.  Tests can compare fit results directly against this value.
    """
    from phasorpy.lifetime import phasor_from_fret_donor
    g, s = phasor_from_fret_donor(
        MOCK_FRET_FREQ, MOCK_FRET_TAU_D,
        fret_efficiency=efficiency,
        unit_conversion=1e-3,
    )
    return float(g), float(s)


def fret_acceptor_phasor_truth(efficiency: float) -> Tuple[float, float]:
    """Return the exact phasorpy FRET acceptor ``(G, S)`` for a given efficiency."""
    from phasorpy.lifetime import phasor_from_fret_acceptor
    g, s = phasor_from_fret_acceptor(
        MOCK_FRET_FREQ, MOCK_FRET_TAU_D, MOCK_FRET_TAU_A,
        fret_efficiency=efficiency,
        unit_conversion=1e-3,
    )
    return float(g), float(s)


def generate_fret_donor_image(
    efficiency: float,
    shape: Tuple[int, int] = (8, 8),
    noise: float = 0.005,
    mean_photons: float = 100.0,
    seed: int = 0,
):
    """Generate a synthetic calibrated donor-channel phasor image.

    Each pixel is drawn from a Gaussian centred on the phasorpy FRET donor
    model prediction for *efficiency*.  The ground-truth phasor coordinate
    can be recovered with :func:`fret_donor_phasor_truth`.

    Parameters
    ----------
    efficiency : float
        FRET efficiency in [0, 1].
    shape : tuple of int
        Spatial shape ``(Y, X)`` of the output image.
    noise : float
        Std-dev of the per-pixel Gaussian noise (phasor units).
    mean_photons : float
        Uniform mean-photon value assigned to every pixel.
    seed : int
        NumPy random seed for reproducibility.

    Returns
    -------
    FRETChannelData
    """
    from flimkit.phasor.fret import FRETChannelData
    rng = np.random.default_rng(seed)
    g0, s0 = fret_donor_phasor_truth(efficiency)
    g = np.full(shape, g0) + rng.normal(0, noise, shape)
    s = np.full(shape, s0) + rng.normal(0, noise, shape)
    mean = np.full(shape, mean_photons)
    return FRETChannelData(real_cal=g, imag_cal=s, mean=mean, frequency=MOCK_FRET_FREQ)


def generate_fret_acceptor_image(
    efficiency: float,
    shape: Tuple[int, int] = (8, 8),
    noise: float = 0.005,
    mean_photons: float = 100.0,
    seed: int = 1,
):
    """Generate a synthetic calibrated acceptor-channel phasor image.

    Each pixel is drawn from a Gaussian centred on the phasorpy FRET acceptor
    model prediction for *efficiency*.  The ground-truth phasor coordinate
    can be recovered with :func:`fret_acceptor_phasor_truth`.

    Parameters
    ----------
    efficiency : float
        FRET efficiency in [0, 1].
    shape, noise, mean_photons, seed
        See :func:`generate_fret_donor_image`.

    Returns
    -------
    FRETChannelData
    """
    from flimkit.phasor.fret import FRETChannelData
    rng = np.random.default_rng(seed)
    g0, s0 = fret_acceptor_phasor_truth(efficiency)
    g = np.full(shape, g0) + rng.normal(0, noise, shape)
    s = np.full(shape, s0) + rng.normal(0, noise, shape)
    mean = np.full(shape, mean_photons)
    return FRETChannelData(real_cal=g, imag_cal=s, mean=mean, frequency=MOCK_FRET_FREQ)
