"""Enhanced Output Functions for FLIM Fitting Results

Adds functionality to save:
- Fit parameters to text file
- Intensity-weighted tau images
- Amplitude-weighted tau images
"""

import numpy as np
import tifffile
from pathlib import Path
from typing import Dict, Any, Optional


def save_fit_summary_txt(
    summary: Dict[str, Any],
    output_path: Path,
    n_exp: int = 2,
    strategy: str = "gaussian",
    metadata: Optional[Dict] = None
):
    """
    Save fit results to a human-readable text file.
    
    Args:
        summary: Fit summary dictionary from fit_summed
        output_path: Path for output file (e.g., "results/R_002_fit_summary.txt")
        n_exp: Number of exponential components
        strategy: IRF strategy used
        metadata: Optional metadata (canvas size, photon counts, etc.)
    
    Example:
        >>> save_fit_summary_txt(summary, Path("R_002_fit_summary.txt"), n_exp=2)
    """
    output_path = Path(output_path)
    
    with open(output_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("FLIM FIT RESULTS SUMMARY\n")
        f.write("="*60 + "\n\n")
        
        # Metadata
        if metadata:
            f.write("ROI Information:\n")
            f.write("-"*60 + "\n")
            if 'canvas_shape' in metadata:
                f.write(f"Canvas size: {metadata['canvas_shape'][0]} × {metadata['canvas_shape'][1]} pixels\n")
            if 'tiles_processed' in metadata:
                f.write(f"Tiles processed: {metadata['tiles_processed']}\n")
            if 'total_photons' in metadata:
                f.write(f"Total photons: {metadata['total_photons']:,.0f}\n")
            f.write("\n")
        
        # Fit parameters
        f.write("Fit Parameters:\n")
        f.write("-"*60 + "\n")
        f.write(f"Number of exponentials: {n_exp}\n")
        f.write(f"IRF strategy: {strategy}\n")
        f.write(f"Chi-squared (reduced): {summary.get('chi2r', 0):.6f}\n")
        f.write(f"Background: {summary.get('bg', 0):.3f}\n")
        if 'sigma' in summary:
            f.write(f"IRF broadening (σ): {summary['sigma']:.3f} ns\n")
        f.write("\n")
        
        # Lifetime components
        f.write("Lifetime Components:\n")
        f.write("-"*60 + "\n")
        for i in range(1, n_exp + 1):
            tau_key = f'tau_{i}'
            amp_key = f'a{i}'
            
            if tau_key in summary:
                tau = summary[tau_key]
                f.write(f"Component {i}:\n")
                f.write(f"  τ{i} = {tau:.4f} ns\n")
                
                if amp_key in summary:
                    amp = summary[amp_key]
                    f.write(f"  A{i} = {amp:.4f}\n")
                    
                    # Calculate fractional intensity
                    total_amp = sum(summary.get(f'a{j}', 0) for j in range(1, n_exp + 1))
                    if total_amp > 0:
                        frac = amp / total_amp * 100
                        f.write(f"  Fractional intensity: {frac:.1f}%\n")
                
                f.write("\n")
        
        # Average lifetime
        if n_exp > 1:
            tau_avg = 0
            total_amp = 0
            for i in range(1, n_exp + 1):
                tau = summary.get(f'tau_{i}', 0)
                amp = summary.get(f'a{i}', 0)
                tau_avg += tau * amp
                total_amp += amp
            
            if total_amp > 0:
                tau_avg /= total_amp
                f.write(f"Average lifetime (amplitude-weighted): {tau_avg:.4f} ns\n\n")
        
        # All parameters (raw)
        f.write("Raw Parameters:\n")
        f.write("-"*60 + "\n")
        for key, value in sorted(summary.items()):
            if isinstance(value, (int, float)):
                f.write(f"{key}: {value:.6e}\n")
            else:
                f.write(f"{key}: {value}\n")
        
        f.write("\n" + "="*60 + "\n")
    
    print(f"✓ Fit summary saved: {output_path}")


def save_weighted_tau_images(
    pixel_maps: Dict[str, np.ndarray],
    output_dir: Path,
    roi_name: str = "ROI",
    n_exp: int = 2,
    save_intensity: bool = True,
    save_amplitude: bool = True
):
    """
    Save intensity-weighted and/or amplitude-weighted tau images.
    
    Args:
        pixel_maps: Dictionary from fit_per_pixel containing:
            - 'tau_1', 'tau_2', ... : Lifetime maps (Y, X)
            - 'a1', 'a2', ... : Amplitude maps (Y, X)
            - 'intensity' : Total intensity map (Y, X)
        output_dir: Directory for output files
        roi_name: ROI identifier for filename
        n_exp: Number of exponential components
        save_intensity: Save intensity-weighted tau image
        save_amplitude: Save amplitude-weighted tau image
    
    Creates:
        - {roi_name}_tau_intensity_weighted.tif
        - {roi_name}_tau_amplitude_weighted.tif
        - {roi_name}_intensity.tif (optional)
    
    Example:
        >>> save_weighted_tau_images(
        ...     pixel_maps,
        ...     Path("results/R_002/"),
        ...     roi_name="R_002",
        ...     n_exp=2
        ... )
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get intensity map (total photon counts per pixel)
    if 'intensity' in pixel_maps:
        intensity = pixel_maps['intensity']
    else:
        # Calculate from amplitudes if not provided
        intensity = np.zeros_like(pixel_maps['tau_1'])
        for i in range(1, n_exp + 1):
            amp_key = f'a{i}'
            if amp_key in pixel_maps:
                intensity += pixel_maps[amp_key]
    
    # Save intensity image
    if save_intensity:
        intensity_path = output_dir / f"{roi_name}_intensity.tif"
        tifffile.imwrite(str(intensity_path), intensity.astype(np.float32))
        print(f"✓ Intensity image saved: {intensity_path}")
    
    # Compute intensity-weighted average lifetime
    if save_intensity and n_exp > 1:
        tau_intensity_weighted = np.zeros_like(intensity, dtype=np.float32)
        
        for i in range(1, n_exp + 1):
            tau_key = f'tau_{i}'
            amp_key = f'a{i}'
            
            if tau_key in pixel_maps and amp_key in pixel_maps:
                tau = pixel_maps[tau_key]
                amp = pixel_maps[amp_key]
                
                # Weight by amplitude (which represents intensity contribution)
                tau_intensity_weighted += tau * amp
        
        # Normalize by total intensity
        mask = intensity > 0
        tau_intensity_weighted[mask] /= intensity[mask]
        tau_intensity_weighted[~mask] = 0
        
        # Save
        tau_int_path = output_dir / f"{roi_name}_tau_intensity_weighted.tif"
        tifffile.imwrite(str(tau_int_path), tau_intensity_weighted)
        print(f"✓ Intensity-weighted tau image saved: {tau_int_path}")
        print(f"  Range: {tau_intensity_weighted[mask].min():.3f} - {tau_intensity_weighted[mask].max():.3f} ns")
    
    # Compute amplitude-weighted average lifetime
    if save_amplitude and n_exp > 1:
        tau_amplitude_weighted = np.zeros_like(intensity, dtype=np.float32)
        total_amplitude = np.zeros_like(intensity, dtype=np.float32)
        
        for i in range(1, n_exp + 1):
            tau_key = f'tau_{i}'
            amp_key = f'a{i}'
            
            if tau_key in pixel_maps and amp_key in pixel_maps:
                tau = pixel_maps[tau_key]
                amp = pixel_maps[amp_key]
                
                # Weight by amplitude (fractional contribution)
                tau_amplitude_weighted += tau * amp
                total_amplitude += amp
        
        # Normalize by total amplitude
        mask = total_amplitude > 0
        tau_amplitude_weighted[mask] /= total_amplitude[mask]
        tau_amplitude_weighted[~mask] = 0
        
        # Save
        tau_amp_path = output_dir / f"{roi_name}_tau_amplitude_weighted.tif"
        tifffile.imwrite(str(tau_amp_path), tau_amplitude_weighted)
        print(f"✓ Amplitude-weighted tau image saved: {tau_amp_path}")
        print(f"  Range: {tau_amplitude_weighted[mask].min():.3f} - {tau_amplitude_weighted[mask].max():.3f} ns")
    
    # For single exponential, just save the single tau map
    if n_exp == 1:
        tau_single = pixel_maps['tau_1']
        tau_path = output_dir / f"{roi_name}_tau.tif"
        tifffile.imwrite(str(tau_path), tau_single.astype(np.float32))
        print(f"✓ Lifetime image saved: {tau_path}")
        mask = tau_single > 0
        print(f"  Range: {tau_single[mask].min():.3f} - {tau_single[mask].max():.3f} ns")


def save_individual_tau_maps(
    pixel_maps: Dict[str, np.ndarray],
    output_dir: Path,
    roi_name: str = "ROI",
    n_exp: int = 2
):
    """
    Save individual lifetime component maps.
    
    Args:
        pixel_maps: Dictionary from fit_per_pixel
        output_dir: Directory for output files
        roi_name: ROI identifier
        n_exp: Number of exponential components
    
    Creates:
        - {roi_name}_tau1.tif
        - {roi_name}_tau2.tif
        - {roi_name}_a1.tif
        - {roi_name}_a2.tif
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i in range(1, n_exp + 1):
        # Save tau map
        tau_key = f'tau_{i}'
        if tau_key in pixel_maps:
            tau = pixel_maps[tau_key]
            tau_path = output_dir / f"{roi_name}_tau{i}.tif"
            tifffile.imwrite(str(tau_path), tau.astype(np.float32))
            print(f"✓ τ{i} map saved: {tau_path}")
        
        # Save amplitude map
        amp_key = f'a{i}'
        if amp_key in pixel_maps:
            amp = pixel_maps[amp_key]
            amp_path = output_dir / f"{roi_name}_a{i}.tif"
            tifffile.imwrite(str(amp_path), amp.astype(np.float32))
            print(f"✓ A{i} map saved: {amp_path}")


def create_complete_output_package(
    summary: Dict[str, Any],
    pixel_maps: Optional[Dict[str, np.ndarray]],
    output_dir: Path,
    roi_name: str,
    n_exp: int,
    strategy: str,
    metadata: Optional[Dict] = None,
    save_individual_components: bool = True
):
    """
    Create complete output package with all results.
    
    Args:
        summary: Fit summary from fit_summed
        pixel_maps: Optional pixel maps from fit_per_pixel
        output_dir: Output directory
        roi_name: ROI identifier
        n_exp: Number of exponentials
        strategy: IRF strategy
        metadata: Optional metadata
        save_individual_components: Save individual tau/amplitude maps
    
    Creates comprehensive output package with all results organized.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Creating complete output package: {roi_name}")
    print(f"{'='*60}\n")
    
    # Save fit summary
    summary_path = output_dir / f"{roi_name}_fit_summary.txt"
    save_fit_summary_txt(summary, summary_path, n_exp, strategy, metadata)
    
    # Save pixel maps if available
    if pixel_maps is not None:
        # Weighted averages
        save_weighted_tau_images(
            pixel_maps,
            output_dir,
            roi_name,
            n_exp,
            save_intensity=True,
            save_amplitude=True
        )
        
        # Individual components
        if save_individual_components:
            save_individual_tau_maps(pixel_maps, output_dir, roi_name, n_exp)
    
    print(f"\n{'='*60}")
    print(f"Output package complete: {output_dir}")
    print(f"{'='*60}\n")
