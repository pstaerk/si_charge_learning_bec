from calc_ir_spectrum import (calculate_ir_spectrum_from_mdot,
                              collect_and_average_spectra)
import numpy as np
from pathlib import Path
import glob
from ase.io import read


def load_dipole_deriv_with_checkpoints(base_path):
    """
    Load dipole derivative data, including all checkpoint files if they exist.
    
    Parameters:
    -----------
    base_path : str or Path
        Path to the base dipole_deriv.bin file (e.g., '/path/to/dipole_deriv.bin')
    
    Returns:
    --------
    mdot : np.ndarray
        Concatenated dipole derivative data from all files
    """
    base_path = Path(base_path)
    parent_dir = base_path.parent
    base_name = base_path.stem  # e.g., 'dipole_deriv'
    
    # Find all matching files: dipole_deriv.bin, dipole_deriv_1.bin, dipole_deriv_2.bin, etc.
    all_files = sorted(parent_dir.glob(f"{base_name}*.bin"))
    
    # Extract checkpoint numbers and sort
    file_dict = {}
    for file_path in all_files:
        name = file_path.stem
        if name == base_name:
            file_dict[0] = file_path
        else:
            # Extract number from pattern like 'dipole_deriv_1', 'dipole_deriv_2', etc.
            try:
                num = int(name.split('_')[-1])
                file_dict[num] = file_path
            except (ValueError, IndexError):
                continue
    
    if not file_dict:
        raise FileNotFoundError(f"No dipole derivative files found at {base_path}")
    
    # Load and concatenate in order
    mdot_segments = []
    for checkpoint_num in sorted(file_dict.keys()):
        file_path = file_dict[checkpoint_num]
        print(f"Loading {file_path.name}...")
        data = np.fromfile(file_path, dtype=np.float32).reshape(-1, 3)
        mdot_segments.append(data)
    
    mdot = np.concatenate(mdot_segments, axis=0)
    print(f"Total loaded: {len(mdot)} frames from {len(mdot_segments)} file(s)")
    
    return mdot


if __name__ == "__main__":
    cutoff_time = 200

    mdot = load_dipole_deriv_with_checkpoints('/auto.ant/work/pstaerk/md_runs_ml/lorem_i_bulk_300K/dipole_deriv.bin')
    mdot = mdot[cutoff_time:]
    spectrum = calculate_ir_spectrum_from_mdot(
        mdot, dt=0.0005, temperature=300, segs=2, bins=200, volume=1,
        output_path='spectrum_bulk_from_mdot'
    )

    mdot = load_dipole_deriv_with_checkpoints('/auto.ant/work/pstaerk/md_runs_ml/lorem_ii_bulk_300K/dipole_deriv.bin')
    mdot = mdot[cutoff_time:]
    spectrum = calculate_ir_spectrum_from_mdot(
        mdot, dt=0.0005, temperature=300, segs=2, bins=200, volume=1,
        output_path='spectrum_lii_bulk_from_mdot'
    )

    # this needs to be re-scaled from the uncorrected Z
    gamma_bulk_global = -0.18940874127933097
    mdot = load_dipole_deriv_with_checkpoints('/auto.ant/work/pstaerk/md_runs_ml/lorem_ii_global_bulk_300K/dipole_deriv.bin')
    mdot *= gamma_bulk_global
    mdot = mdot[cutoff_time:]
    spectrum = calculate_ir_spectrum_from_mdot(
        mdot, dt=0.0005, temperature=300, segs=2, bins=200, volume=1,
        output_path='spectrum_lii_global_bulk_from_mdot'
    )

    #####################################################################
    #                   Physical Models                                 #
    #####################################################################

    # for model ii global
    gamma_phys_global = 1.974

    mdot = load_dipole_deriv_with_checkpoints('/auto.ant/work/pstaerk/md_runs_ml/physical_i_bulk_300K_singlerun/dipole_deriv.bin')
    mdot = mdot[cutoff_time:]
    np.save('mdot_ph_i_bulk.npy', mdot)
    spectrum = calculate_ir_spectrum_from_mdot(
        mdot, dt=0.0005, temperature=300, segs=2, bins=200, volume=1,
        output_path='spectrum_ph_i_bulk_from_mdot'
    )

    mdot = load_dipole_deriv_with_checkpoints('/auto.ant/work/pstaerk/md_runs_ml/physical_ii_bulk_300K_singlerun/dipole_deriv.bin')
    mdot = mdot[cutoff_time:]
    np.save('mdot_ph_ii_bulk.npy', mdot)
    spectrum = calculate_ir_spectrum_from_mdot(
        mdot, dt=0.0005, temperature=300, segs=2, bins=200, volume=1,
        output_path='spectrum_ph_ii_bulk_from_mdot'
    )

    mdot = load_dipole_deriv_with_checkpoints('/auto.ant/work/pstaerk/md_runs_ml/physical_ii_global_bulk_300K_singlerun/dipole_deriv.bin')
    mdot = mdot[cutoff_time:]
    mdot *= gamma_phys_global
    np.save('mdot_ph_ii_global_bulk.npy', mdot)
    spectrum = calculate_ir_spectrum_from_mdot(
        mdot, dt=0.0005, temperature=300, segs=2, bins=200, volume=1,
        output_path='spectrum_ph_ii_global_bulk_from_mdot'
    )


    # We combine multiple runs for better statistics
    base_paths = {
        'global': '/auto.ant/work/pstaerk/md_runs_ml/phys_global_cage',
        'i': '/auto.ant/work/pstaerk/md_runs_ml/phys_uncoupled_cage',
        'ii': '/auto.ant/work/pstaerk/md_runs_ml/phys_coupl_local_cage'
    }

    results = collect_and_average_spectra(
        base_paths, dt=0.0005, temperature=10,
        output_file=None)

    for model_name in results.keys():
        output_file = f'combined_spectrum_ph_{model_name}_hex_cage_from_mdot.npz'
        model_data = results[model_name]

        if 'global' in model_name:
            # apply a-posteriori scaling for global model
            gamma_phys_global = 1.974
            model_data['mean_susc_imag'] *= gamma_phys_global**2
            model_data['std_susc_imag'] *= gamma_phys_global**2
            model_data['mean_susc_real'] *= gamma_phys_global**2
            model_data['std_susc_real'] *= gamma_phys_global**2

        save_dict = {}
        for key, value in model_data.items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                save_dict[key] = value
            elif isinstance(value, np.ndarray):
                save_dict[key] = value
        np.savez(output_file, **save_dict)
        print(f"Saved {len(save_dict)} arrays to {output_file}")

    base_paths = {
        'global': '/auto.ant/work/pstaerk/md_runs_ml/phys_global_book',
        'i': '/auto.ant/work/pstaerk/md_runs_ml/phys_uncoupled_book',
        'ii': '/auto.ant/work/pstaerk/md_runs_ml/phys_coupl_local_book'
    }

    results = collect_and_average_spectra(base_paths, dt=0.0005, temperature=10, 
                                          output_file=None)

    for model_name in results.keys():
        output_file = f'combined_spectrum_ph_{model_name}_hex_book_from_mdot.npz'
        model_data = results[model_name]

        if 'global' in model_name:
            # apply a-posteriori scaling for global model
            gamma_phys_global = 1.974
            model_data['mean_susc_imag'] *= gamma_phys_global**2
            model_data['std_susc_imag'] *= gamma_phys_global**2
            model_data['mean_susc_real'] *= gamma_phys_global**2
            model_data['std_susc_real'] *= gamma_phys_global**2

        save_dict = {}
        for key, value in model_data.items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                save_dict[key] = value
            elif isinstance(value, np.ndarray):
                save_dict[key] = value
        np.savez(output_file, **save_dict)
        print(f"Saved {len(save_dict)} arrays to {output_file}")

    base_paths = {
        'global': '/auto.ant/work/pstaerk/md_runs_ml/lorem_global_cage_com',
        'i': '/auto.ant/work/pstaerk/md_runs_ml/lorem_uncoupled_cage_com',
        'ii': '/auto.ant/work/pstaerk/md_runs_ml/lorem_coupl_local_cage_com'
    }

    results = collect_and_average_spectra(
        base_paths, dt=0.0005, temperature=10,
        output_file=None)

    for model_name in results.keys():
        output_file = f'combined_spectrum_lorem_{model_name}_hex_cage_from_mdot.npz'
        model_data = results[model_name]

        if 'global' in model_name:
            # apply a-posteriori scaling for global model
            gamma_lorem_global = -0.189
            model_data['mean_susc_imag'] *= gamma_lorem_global**2
            model_data['std_susc_imag'] *= gamma_lorem_global**2
            model_data['mean_susc_real'] *= gamma_lorem_global**2
            model_data['std_susc_real'] *= gamma_lorem_global**2

        save_dict = {}
        for key, value in model_data.items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                save_dict[key] = value
            elif isinstance(value, np.ndarray):
                save_dict[key] = value
        np.savez(output_file, **save_dict)
        print(f"Saved {len(save_dict)} arrays to {output_file}")

    base_paths = {
        'global': '/auto.ant/work/pstaerk/md_runs_ml/lorem_global_book_com',
        'i': '/auto.ant/work/pstaerk/md_runs_ml/lorem_uncoupled_book_com',
        'ii': '/auto.ant/work/pstaerk/md_runs_ml/lorem_coupl_local_book_com'
    }

    results = collect_and_average_spectra(
        base_paths, dt=0.0005, temperature=10,
        output_file=None)

    for model_name in results.keys():
        output_file = f'combined_spectrum_lorem_{model_name}_hex_book_from_mdot.npz'
        model_data = results[model_name]

        if 'global' in model_name:
            # apply a-posteriori scaling for global model
            gamma_lorem_global = -0.189
            model_data['mean_susc_imag'] *= gamma_lorem_global**2
            model_data['std_susc_imag'] *= gamma_lorem_global**2
            model_data['mean_susc_real'] *= gamma_lorem_global**2
            model_data['std_susc_real'] *= gamma_lorem_global**2

        save_dict = {}
        for key, value in model_data.items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                save_dict[key] = value
            elif isinstance(value, np.ndarray):
                save_dict[key] = value
        np.savez(output_file, **save_dict)
        print(f"Saved {len(save_dict)} arrays to {output_file}")
