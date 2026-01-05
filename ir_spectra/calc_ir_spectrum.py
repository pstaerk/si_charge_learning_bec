import spectrakit
import numpy as np
from ase.io import read
from pathlib import Path
import argparse

from maicos.lib.math import FT, iFT
from maicos.lib.util import (
    bin,
    charge_neutral,
    citation_reminder,
    get_compound,
    render_docs,
)


def calculate_spectrum_from_dipole_bm(
    dipole_moment: np.ndarray,
    dt: float,
    volume: float,
    temperature: float,
    segs: int | None = None,
    df: float | None = None,
    bins: int = 200,
    binafter: float = 20,
    nobin: bool = False,
) -> dict[str, np.ndarray]:
    """
    Dielectric susceptibility from dipole-moment trajectory using FDT.

    Modifications from the spectrakit implementation:
    - per-segment mean removal
    - per-segment linear detrend
    - Hann window applied
    - normalization preserved
    - safer Welford variance handling
    - FT/iFT used consistently
    """
    import numpy as np
    import logging
    import scipy.constants

    P = dipole_moment.copy()
    n_frames = len(P)

    # ------------------------------------------------------------
    # Determine number of segments
    # ------------------------------------------------------------
    if df is not None:
        segs = np.max([int(n_frames * dt * df), 2])
    elif segs is None:
        segs = 20

    seglen = int(n_frames / segs)

    # ------------------------------------------------------------
    # Prefactor for susceptibility
    # ------------------------------------------------------------
    pref = (scipy.constants.e)**2 * scipy.constants.angstrom**2
    pref /= 3 * volume * scipy.constants.angstrom**3
    pref /= scipy.constants.k * temperature
    pref /= scipy.constants.epsilon_0

    # ------------------------------------------------------------
    # Time array and frequency grid from FT
    # ------------------------------------------------------------
    t = dt * np.arange(seglen)
    nu = FT(t, P[:seglen, 0])[0]  # rad/ps

    # Output arrays
    susc = np.zeros(seglen, dtype=complex)
    dsusc = np.zeros(seglen, dtype=complex)

    # Precompute Hann window
    win = np.hanning(seglen)
    wnorm = np.sum(win**2) / seglen

    # ------------------------------------------------------------
    # Loop over segments
    # ------------------------------------------------------------
    for s in range(segs):
        logging.info(f"\rSegment {s + 1} of {segs}")

        seg = P[s * seglen:(s + 1) * seglen]
        if seg.shape[0] != seglen:
            break

        # ----------------------------------------
        # Detrend and window each Cartesian component
        # ----------------------------------------
        ss = np.zeros(seglen, dtype=complex)

        for i in range(3):
            x = seg[:, i]

            # Mean removal
            x = x - np.mean(x)

            # Linear detrend
            tnorm = np.linspace(-0.5, 0.5, seglen)
            coeff = np.sum(tnorm * x) / np.sum(tnorm**2)
            x = x - coeff * tnorm

            # Apply window
            xw = x * win

            # FFT
            FP = FT(t, xw, indvar=False)

            # |FFT|²
            ss += (FP.real**2 + FP.imag**2) / wnorm

        # Multiply by i ω  (fluctuation–dissipation relation)
        ss *= 1j * nu

        # ----------------------------------------
        # Kramers–Kronig for the real part
        # ----------------------------------------
        hilb = iFT(
            t,
            1j * np.sign(nu) * FT(nu, ss, indvar=False),
            indvar=False,
        )
        ss.real = hilb.imag

        # ----------------------------------------
        # Accumulate mean and variance (Welford)
        # ----------------------------------------
        if s == 0:
            susc = ss.copy()
            continue

        mean_prev = susc / s

        delta = ss - mean_prev
        susc += ss
        mean_new = susc / (s + 1)

        # Proper complex-valued variance update
        dsusc += delta * np.conj(ss - mean_new)


    # ----------------------------------------
    # Finalize variance
    # ----------------------------------------
    dsusc.real = np.sqrt(dsusc.real)
    dsusc.imag = np.sqrt(dsusc.imag)

    # ----------------------------------------
    # Normalize susceptibility
    # ----------------------------------------
    norm = pref / (seglen * segs * dt)
    susc *= norm
    dsusc *= norm

    # Convert rad/ps → THz
    nu = nu / (2 * np.pi)

    # Keep positive frequencies
    pos = nu >= 0
    nu = nu[pos]
    susc = susc[pos]
    dsusc = dsusc[pos]

    results = {"t": t, "nu": nu, "susc": susc, "dsusc": dsusc}

    logging.info(f"Length of segments: {seglen} frames, {seglen * dt:.0f} ps")
    logging.info(f"Frequency spacing: ~ {segs / (n_frames * dt):.5f} THz")

    # ------------------------------------------------------------
    # Optional binning (unchanged from your version)
    # ------------------------------------------------------------
    if not nobin and seglen > bins:
        bin_indices = np.logspace(
            np.log10(binafter),
            np.log10(len(susc)),
            bins - binafter + 1,
        ).astype(int)
        bin_indices = np.unique(np.append(np.arange(binafter), bin_indices))[:-1]

        results["nu_binned"] = bin(nu, bin_indices)
        results["susc_binned"] = bin(susc, bin_indices)
        results["dsusc_binned"] = bin(dsusc, bin_indices)

        logging.info(f"Binning data above datapoint {binafter} in log-spaced bins")
        logging.info(f"Binned data consists of {len(susc)} datapoints")
    else:
        logging.info(f"Not binning data: there are {len(susc)} datapoints")

    return results


def calculate_ir_spectrum_from_mdot(M_dot, dt=0.5, temperature=300, 
                                    segs=2, bins=200, volume=1,
                                    output_path=None):
    """
    Calculate IR spectrum from pre-computed dipole moment derivative.

    Parameters:
    -----------
    M_dot : np.ndarray
        Time derivative of dipole moment, shape (n_frames, 3)
    dt : float
        Time step in fs (default: 0.5)
    temperature : float
        Temperature in K (default: 300)
    segs : int
        Number of segments for spectrum calculation (default: 2)
    bins : int
        Number of bins for spectrum (default: 200)
    volume : float
        Volume for spectrum calculation (default: 1)
    output_path : str or Path, optional
        Path to save the spectrum results

    Returns:
    --------
    spectrum : dict
        Calculated IR spectrum from spectrakit
    """
    # Calculate dipole moment time series from M_dot
    print("Calculating dipole moment time series from M_dot...")
    M_timeseries = np.cumsum(M_dot * dt, axis=0)

    # write out how many frames we have
    print(f"Number of frames in M_timeseries: {M_timeseries.shape[0]}")

    # Calculate spectrum
    print("Calculating IR spectrum...")
    spectrum = calculate_spectrum_from_dipole_bm(
        M_timeseries,
        dt=dt,  # Convert fs to ps
        volume=volume,
        temperature=temperature,
        segs=segs,
        bins=bins
    )

    spectrum_processed = process_spectrum(spectrum, fmin=1e-3,
                                          fmax=2e2, num=300)

    # Save results if output path is provided
    if output_path:
        output_path = Path(output_path)
        print(f"Saving spectrum to {output_path}...")
        np.savez(
            output_path,
            nu=spectrum_processed['nu'],
            susc_imag=spectrum_processed['susc_imag'],
            dsusc_imag=spectrum_processed['dsusc_imag'],
            susc_real=spectrum_processed['susc_real'],
            dsusc_real=spectrum_processed['dsusc_real']
        )

    return spectrum_processed


def process_spectrum(spectra_dict, fmin=1e-3, fmax=2e2, num=300):
    """Process the raw spectrum data into binned format.

    Parameters:
    -----------
    spectra_dict : dict
        Dictionary containing raw spectrum data with keys:
        'nu', 'susc', 'dsusc'
    fmin : float
        Minimum frequency for binning (default: 1e-3)
    fmax : float
        Maximum frequency for binning (default: 2e2)
    num : int
        Number of bins (default: 300)

    Returns:
    --------
    spectra_dict : dict
        Dictionary containing binned spectrum data with keys:
        'nu', 'susc_imag', 'dsusc_imag', 'susc_real', 'dsusc_real'
    """
    spectra_dict = dict(spectra_dict)

    nu_raw = spectra_dict["nu"]
    susc_imag_raw = spectra_dict["susc"].imag
    dsusc_imag_raw = spectra_dict["dsusc"].imag
    susc_real_raw = spectra_dict["susc"].real
    dsusc_real_raw = spectra_dict["dsusc"].real

    # bin data on geometric grid
    bins = np.geomspace(fmin, fmax, num=num)
    nu_binned = 0.5 * (bins[1:] + bins[:-1])

    susc_imag_binned, _ = np.histogram(nu_raw, bins=bins, weights=susc_imag_raw)
    dsusc_imag_binned, _ = np.histogram(nu_raw, bins=bins, weights=dsusc_imag_raw)
    susc_real_binned, _ = np.histogram(nu_raw, bins=bins, weights=susc_real_raw)
    dsusc_real_binned, _ = np.histogram(nu_raw, bins=bins, weights=dsusc_real_raw)

    norm, _ = np.histogram(nu_raw, bins=bins)
    with np.errstate(divide='ignore', invalid='ignore'):
        susc_imag_binned /= norm
        dsusc_imag_binned /= norm
        susc_real_binned /= norm
        dsusc_real_binned /= norm

    # remove nan values from empty bins
    valid = ~np.isnan(susc_imag_binned)
    nu_binned = nu_binned[valid]
    susc_imag_binned = susc_imag_binned[valid]
    dsusc_imag_binned = dsusc_imag_binned[valid]
    susc_real_binned = susc_real_binned[valid]
    dsusc_real_binned = dsusc_real_binned[valid]

    spectra_dict["nu"] = nu_binned
    spectra_dict["susc_imag"] = susc_imag_binned
    spectra_dict["dsusc_imag"] = dsusc_imag_binned
    spectra_dict["susc_real"] = susc_real_binned
    spectra_dict["dsusc_real"] = dsusc_real_binned

    return spectra_dict


def collect_and_average_spectra(base_paths, seed_pattern='seed_*', dt=0.0005, 
                                temperature=10, volume=1, output_file=None,
                                max_time=50_000):
    """
    Calculate spectra from dipole derivatives for multiple seeds and compute mean/std across seeds.
    
    """
    from pathlib import Path
    
    results = {}
    
    for model_name, base_path in base_paths.items():
        print(f"\nProcessing {model_name}...")
        base_path = Path(base_path)
        
        # Find all seed directories
        seed_dirs = sorted(base_path.glob(seed_pattern))
        
        if not seed_dirs:
            print(f"  Warning: No {seed_pattern} directories found in {base_path}")
            continue
        
        print(f"  Found {len(seed_dirs)} seeds: {[d.name for d in seed_dirs]}")
        
        # Collect spectra from each seed
        all_spectra = []
        nu_reference = None
        
        for seed_dir in seed_dirs:
            # Look for dipole_deriv.bin file
            dipole_file = seed_dir / 'dipole_deriv.bin'
            
            if not dipole_file.exists():
                print(f"  Warning: No dipole_deriv.bin found in {seed_dir.name}")
                continue
            
            try:
                # Load dipole derivative
                print(f"  Loading {seed_dir.name}...")
                mdot = np.fromfile(dipole_file, dtype=np.float32).reshape(-1, 3)
                
                # Calculate dipole moment time series
                M = np.cumsum(mdot * dt, axis=0)
                M = M[:max_time]  # Limit to max_time frames
                
                # Calculate spectrum
                res = calculate_spectrum_from_dipole_bm(M, dt=dt, volume=volume, temperature=temperature)
                
                nu = res['nu']
                
                # Check frequency alignment
                if nu_reference is None:
                    nu_reference = nu
                else:
                    if not np.allclose(nu, nu_reference, rtol=1e-10):
                        raise ValueError(
                            f"Frequency mismatch in {seed_dir.name}!\n"
                            f"Expected shape: {nu_reference.shape}, got: {nu.shape}\n"
                            f"Max difference: {np.max(np.abs(nu - nu_reference))}"
                        )
                
                all_spectra.append(res)
                print(f"    Calculated spectrum with {len(M)} frames")
                
            except Exception as e:
                print(f"  Error processing {seed_dir.name}: {e}")
                continue
        
        if not all_spectra:
            print(f"  Warning: No valid spectra loaded for {model_name}")
            continue
        
        # Compute statistics across seeds
        print(f"  Computing statistics across {len(all_spectra)} seeds...")
        
        model_results = {'nu': nu_reference, 'num_seeds': len(all_spectra)}
        
        # Find all keys that contain spectral data
        spectral_keys = [key for key in all_spectra[0].keys() 
                        if key not in ['nu', 'num_seeds']]
        
        for key in spectral_keys:
            # Stack arrays from all seeds
            try:
                # Handle complex arrays (like 'susc')
                if hasattr(all_spectra[0][key], 'imag'):
                    # Complex array - separate real and imaginary parts
                    stacked_real = np.array([s[key].real for s in all_spectra])
                    stacked_imag = np.array([s[key].imag for s in all_spectra])
                    
                    model_results[f'mean_{key}_real'] = np.mean(stacked_real, axis=0)
                    model_results[f'std_{key}_real'] = np.std(stacked_real, axis=0)
                    model_results[f'stderr_{key}_real'] = np.std(stacked_real, axis=0) / np.sqrt(len(all_spectra))
                    
                    model_results[f'mean_{key}_imag'] = np.mean(stacked_imag, axis=0)
                    model_results[f'std_{key}_imag'] = np.std(stacked_imag, axis=0)
                    model_results[f'stderr_{key}_imag'] = np.std(stacked_imag, axis=0) / np.sqrt(len(all_spectra))
                    
                    print(f"    Averaged {key} (complex): shape {stacked_real.shape}")
                else:
                    # Real array
                    stacked = np.array([s[key] for s in all_spectra])
                    
                    model_results[f'mean_{key}'] = np.mean(stacked, axis=0)
                    model_results[f'std_{key}'] = np.std(stacked, axis=0)
                    model_results[f'stderr_{key}'] = np.std(stacked, axis=0) / np.sqrt(len(all_spectra))
                    
                    print(f"    Averaged {key}: shape {stacked.shape}")
            except Exception as e:
                print(f"    Warning: Could not average {key}: {e}")
        
        results[model_name] = model_results
    
    # Save combined results if requested
    if output_file:
        print(f"\nSaving combined results to {output_file}...")
        save_dict = {}
        for model_name, model_data in results.items():
            for key, value in model_data.items():
                if isinstance(value, (int, float, np.integer, np.floating)):
                    save_dict[f'{model_name}_{key}'] = value
                elif isinstance(value, np.ndarray):
                    save_dict[f'{model_name}_{key}'] = value
        np.savez(output_file, **save_dict)
        print(f"Saved {len(save_dict)} arrays to {output_file}")
    
    return results
