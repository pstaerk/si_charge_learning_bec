import sys
import pathlib
import numpy as np
from pathlib import Path
import os
from rich.panel import Panel
from rich.table import Table
from rich.console import Console
from marathon.io import read_yaml

# Parse command line arguments for seed and output directory override
import argparse
parser = argparse.ArgumentParser(description='Run MD simulation with APT tracking')
parser.add_argument('--seed', type=int, default=None, 
                    help='Random seed for velocity initialization (default: None, uses random)')
parser.add_argument('--out-dir', type=str, default=None,
                    help='Override output directory from config (useful for parallel runs)')
args = parser.parse_args()

md_settings = read_yaml('md_settings.yaml')

checkpoint = md_settings['io_definitions']['model_checkpoint']
model_definitions = md_settings['io_definitions']['model_definition']

# go from relative to absolute path
model_definitions = pathlib.Path(model_definitions).resolve()
if str(model_definitions) not in sys.path:
    sys.path.append(str(model_definitions))

from ase.io import read, write
from ase.md.bussi import Bussi
from ase import units
from ase.io.trajectory import Trajectory
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, ZeroRotation
from rich.progress import track
from calculator import Calculator
import glob

console = Console()

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


def remove_com(atoms):
    """Set the positions such that the COM is at the origin."""
    current_com = atoms.get_center_of_mass()
    positions = atoms.get_positions()
    new_positions = positions - current_com
    atoms.set_positions(new_positions)


def remove_com_velocity(atoms):
    """Set the velocities such that the COM velocity is zero."""
    current_com_velocity = atoms.get_momenta().sum(axis=0) / atoms.get_masses().sum()
    velocities = atoms.get_velocities()
    new_velocities = velocities - current_com_velocity
    atoms.set_velocities(new_velocities)

io_definitions = md_settings['io_definitions']
md_definitions = md_settings['md_definitions']

# Determine output directory
out_path = Path(io_definitions.get('out_path', './traj.traj'))
if args.out_dir is not None:
    # Override output directory if provided via command line
    out_dir = Path(args.out_dir)
    out_path = out_dir / out_path.name
else:
    out_dir = out_path.parent

# Check if output directory exists and has contents
if out_dir.exists():
    # Check if directory has any files
    contents = list(out_dir.iterdir())
    if contents:
        console.print(f"[bold red]ERROR:[/bold red] Output directory '{out_dir}' already exists and contains files!")
        console.print(f"[yellow]Found {len(contents)} items in directory.[/yellow]")
        console.print("[yellow]Please either:[/yellow]")
        console.print("  1. Remove or rename the existing directory")
        console.print("  2. Specify a different output directory using --out-dir")
        sys.exit(1)
    else:
        console.print(f"[yellow]Output directory '{out_dir}' exists but is empty. Proceeding...[/yellow]")
else:
    # Create output directory
    out_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"[green]Created output directory: {out_dir}[/green]")

start_index = io_definitions.get('start_index', 0)
console.print(f"Using start index {start_index} for reading the initial structure.")
start_struct = io_definitions.get('start_struct', 'initial_structure.traj')

atoms_start = read(start_struct, index=start_index)
new_checkpoint_number = 0
apt_output = out_dir / "apts.bin"
dipole_output = out_dir / "dipole_deriv.bin"

console.print("[green]Starting fresh simulation[/green]")

model_dir = io_definitions.get('model_checkpoint', 'model_dir')
field_vector = md_definitions.get('field_vector', [0.0, 0.0, 0.0])

atoms_start.info['electric_field'] = np.array(field_vector)
calc = Calculator.from_checkpoint(model_dir, field=field_vector)

timestep = md_definitions.get('timestep', 0.5) * units.fs
temperature = md_definitions.get('temperature', 300)
thermostat_tau = md_definitions.get('thermostat_tau', 100) * units.fs
total_steps = md_definitions.get('total_steps', 10000)
steps_per_frame = md_definitions.get('steps_per_frame', 2000)
com_removal_interval = md_definitions.get('com_removal_interval', 1)  # Remove COM drift every N steps
com_velocity_removal_interval = md_definitions.get('com_velocity_removal_interval', 100)  # Remove COM velocity every N steps

from rich.panel import Panel
from rich.table import Table
from rich.console import Console

# Create a table for the MD parameters
table = Table.grid(expand=True, padding=(0, 1))
table.add_column(justify="right", style="cyan", no_wrap=True)
table.add_column(justify="left", style="white")

# Add rows to the table
table.add_row("Random Seed:", f"{args.seed if args.seed is not None else 'None (random)'}")
table.add_row("Checkpoint Number:", f"{new_checkpoint_number}")
table.add_row("Timestep:", f"{timestep:.2f} internal units")
table.add_row("Temperature:", f"{temperature} K")
table.add_row("Thermostat Tau:", f"{thermostat_tau:.2f} internal units")
table.add_row("Total Steps:", f"{total_steps}")
table.add_row("Steps per Frame:", f"{steps_per_frame}")
table.add_row("Output Directory:", f"'{out_dir}'")
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

# Initialize velocities with optional seed
if args.seed is not None:
    console.print(f"[cyan]Initializing velocities with seed {args.seed}[/cyan]")
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature, rng=np.random.RandomState(args.seed))
    ZeroRotation(atoms)
else:
    console.print("[cyan]Initializing velocities with random seed[/cyan]")
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)
    ZeroRotation(atoms)

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


# Attach COM motion remover
dyn.attach(remove_com, interval=com_removal_interval, atoms=atoms)
dyn.attach(remove_com_velocity, interval=com_velocity_removal_interval,
           atoms=atoms)
dyn.attach(ZeroRotation, interval=1, atoms=atoms)

dyn.run(N_per_step*N_steps)

traj.close()
apt_writer.close()
dipole_deriv_writer.close()

console.print(f"[green]Simulation completed successfully![/green]")
console.print(f"[green]Outputs saved to {out_dir}[/green]")
