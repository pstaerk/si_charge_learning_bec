import yaml
from marathon.io import read_yaml

md_settings = read_yaml("md_settings.yaml")
checkpoint = md_settings['io_definitions']['model_checkpoint']
model_definitions = md_settings['io_definitions']['model_definition']
import sys
import pathlib

# go from relative to absolute path
model_definitions = pathlib.Path(model_definitions).resolve()
if str(model_definitions) not in sys.path:
    sys.path.append(str(model_definitions))
from ase.io import read, write
from ase.md.bussi import Bussi
from ase import units
from ase.io.trajectory import Trajectory
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from rich.progress import track
from calculator import Calculator
import numpy as np
from pathlib import Path
import glob

class BinaryPropertyWriter:
    def __init__(self, filename, get_property_func, atoms):
        self.fh = open(filename, "ab")  # append in binary mode
        self.get_property = get_property_func
        self.atoms = atoms

    def __call__(self):
        props = self.get_property(self.atoms).astype(np.float32)  # shape (n_atoms, 3, 3)
        props.tofile(self.fh)  # write raw binary, no header

    def close(self):
        self.fh.close()


def get_apt(atoms):
    apts = atoms.calc.results['apt']
    return apts


def get_dipole_moment_derivative(atoms):
    """
    Calculate the time derivative of the dipole moment.
    
    dM/dt = sum_i (Z_i * v_i)
    where Z_i is the atomic polar tensor and v_i is the velocity
    
    Returns:
    --------
    dM_dt : np.ndarray
        Time derivative of dipole moment, shape (3,)
    """
    apts = atoms.calc.results['apt']  # shape (n_atoms, 3, 3)
    velocities = atoms.get_velocities()  # shape (n_atoms, 3)
    
    # Matrix-vector multiplication: Z_i @ v_i for each atom, then sum
    dM_dt = np.einsum('ijk,ik->j', apts, velocities)
    
    return dM_dt


def find_restart_configuration(out_dir, out_path):
    """
    Find the latest checkpoint to restart from.
    
    Returns:
    --------
    restart_atoms : ase.Atoms or None
        Atoms object to restart from, or None if no restart
    checkpoint_number : int
        The checkpoint number to continue from (0 if fresh start)
    """
    out_dir = Path(out_dir)
    
    # Look for existing trajectories with pattern: traj.traj, traj_1.traj, traj_2.traj, etc.
    base_name = out_path.stem  # e.g., 'traj'
    extension = out_path.suffix  # e.g., '.traj'
    
    # Find all matching trajectory files
    existing_trajs = sorted(out_dir.glob(f"{base_name}*.traj"))
    existing_trajs = [t for t in existing_trajs if t != out_path or t.exists()]
    
    if not existing_trajs:
        return None, 0
    
    # Extract checkpoint numbers
    checkpoint_numbers = []
    for traj_path in existing_trajs:
        name = traj_path.stem
        if name == base_name:
            checkpoint_numbers.append(0)
        else:
            # Extract number from pattern like 'traj_1', 'traj_2', etc.
            try:
                num = int(name.split('_')[-1])
                checkpoint_numbers.append(num)
            except (ValueError, IndexError):
                continue
    
    if not checkpoint_numbers:
        return None, 0
    
    # Get the highest checkpoint number
    max_checkpoint = max(checkpoint_numbers)
    
    # Determine the corresponding trajectory file
    if max_checkpoint == 0:
        latest_traj = out_dir / f"{base_name}{extension}"
    else:
        latest_traj = out_dir / f"{base_name}_{max_checkpoint}{extension}"
    
    print(f"[green]Found existing trajectory: {latest_traj}[/green]")
    print(f"[green]Restarting from checkpoint {max_checkpoint}[/green]")
    
    # Read the last frame
    atoms = read(latest_traj, index=-1)
    
    return atoms, max_checkpoint


io_definitions = md_settings['io_definitions']
md_definitions = md_settings['md_definitions']

out_path = Path(io_definitions.get('out_path', './traj.traj'))
out_dir = out_path.parent

# Check for restart
restart_atoms, checkpoint_number = find_restart_configuration(out_dir, out_path)

if restart_atoms is not None:
    # Restarting from checkpoint
    atoms_start = restart_atoms
    new_checkpoint_number = checkpoint_number + 1
    
    # Create new output paths
    base_name = out_path.stem
    extension = out_path.suffix
    out_path = out_dir / f"{base_name}_{new_checkpoint_number}{extension}"
    apt_output = out_dir / f"apts_{new_checkpoint_number}.bin"
    dipole_output = out_dir / f"dipole_deriv_{new_checkpoint_number}.bin"
    
    print(f"[yellow]Restarting simulation from checkpoint {checkpoint_number}[/yellow]")
    print(f"[yellow]New outputs will be saved with suffix _{new_checkpoint_number}[/yellow]")
else:
    # Fresh start
    start_index = io_definitions.get('start_index', 0)
    print(f"Using start index {start_index} for reading the initial structure.")
    start_struct = io_definitions.get('start_struct', 'initial_structure.traj')
    
    atoms_start = read(start_struct, index=start_index)
    new_checkpoint_number = 0
    apt_output = out_dir / "apts.bin"
    dipole_output = out_dir / "dipole_deriv.bin"
    
    print("[green]Starting fresh simulation[/green]")

model_dir = io_definitions.get('model_checkpoint', 'model_dir')
field_vector = md_definitions.get('field_vector', [0.0, 0.0, 0.0])

atoms_start.info['electric_field'] = np.array(field_vector)
calc = Calculator.from_checkpoint(model_dir, field=field_vector)

timestep = md_definitions.get('timestep', 0.5) * units.fs
temperature = md_definitions.get('temperature', 300)
thermostat_tau = md_definitions.get('thermostat_tau', 100) * units.fs
total_steps = md_definitions.get('total_steps', 10000)
steps_per_frame = md_definitions.get('steps_per_frame', 2000)

from rich.panel import Panel
from rich.table import Table
from rich.console import Console

# Create a table for the MD parameters
table = Table.grid(expand=True, padding=(0, 1))
table.add_column(justify="right", style="cyan", no_wrap=True)
table.add_column(justify="left", style="white")

# Add rows to the table
table.add_row("Checkpoint Number:", f"{new_checkpoint_number}")
table.add_row("Timestep:", f"{timestep:.2f} internal units")
table.add_row("Temperature:", f"{temperature} K")
table.add_row("Thermostat Tau:", f"{thermostat_tau:.2f} internal units")
table.add_row("Total Steps:", f"{total_steps}")
table.add_row("Steps per Frame:", f"{steps_per_frame}")
table.add_row("Output Path:", f"'{out_path}'")
table.add_row("APT Output:", f"'{apt_output}'")
table.add_row("Dipole Deriv Output:", f"'{dipole_output}'")
table.add_row("Electric Field:", f"{field_vector} V/Å")

# Create a panel with the table
panel = Panel(
    table,
    title="[bold magenta]MD Simulation Parameters[/bold magenta]",
    border_style="green",
    expand=False
)

# Print the panel
console = Console()
console.print(panel)

# Initialize APT arrays if not present
if 'apt' not in atoms_start.arrays:
    atoms_start.arrays['apt'] = np.zeros((len(atoms_start), 3, 3))
if 'apt_charge' not in atoms_start.arrays:
    atoms_start.arrays['apt_charge'] = np.zeros((len(atoms_start)))

atoms = atoms_start.copy()
atoms.calc = calc

# Only set velocities if starting fresh (not restarting)
if restart_atoms is None:
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)

dyn = Bussi(atoms, timestep, temperature, thermostat_tau)

workdir = out_dir
traj = Trajectory(out_path, 'w', atoms,
                  properties=['energy', 'forces', 'charges', 'apt'])
N_per_step = steps_per_frame
dyn.attach(traj.write, interval=N_per_step)

N_steps = total_steps

apt_writer = BinaryPropertyWriter(str(apt_output), get_apt, atoms)
dyn.attach(apt_writer, interval=N_per_step)

# Add dipole moment derivative writer
dipole_deriv_writer = BinaryPropertyWriter(str(dipole_output), get_dipole_moment_derivative, atoms)
dyn.attach(dipole_deriv_writer, interval=N_per_step)

dyn.run(N_per_step*N_steps)

traj.close()
apt_writer.close()
dipole_deriv_writer.close()

print(f"[green]Simulation completed successfully![/green]")
print(f"[green]Outputs saved to checkpoint {new_checkpoint_number}[/green]")
