import yaml
from marathon.io import read_yaml

md_settings = read_yaml("md_settings.yaml")
checkpoint = md_settings['io_definitions']['model_checkpoint']
model_definitions = md_settings['io_definitions']['model_definition']
import sys
import pathlib

from batching import get_batch
from transforms import ToSample as ToSampleLoremII, SetUpEwald

cutoff = 5.
to_sample_lorem_ii = ToSampleLoremII(cutoff=cutoff,
                                     forces=False, energy=False,
                                     stress=False, apt=False)
batch_style = "batch_shape"
from rich.console import Console

valid_samples = []

from marathon.extra.hermes.pain import Record, RecordMetadata

# Remove this line - batcher will be created inside Calculator.setup()
# batcher = get_batcher()
prepare_ewald_lorem_ii = SetUpEwald(lr_wavelength=cutoff / 8, smearing=cutoff / 4)

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
import numpy as np
from pathlib import Path
import glob

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_default_matmul_precision", "highest")

from ase.calculators.abc import GetPropertiesMixin
from ase.calculators.calculator import PropertyNotImplementedError, compare_atoms

from marathon import comms
from marathon.data.batching import next_multiple

console = Console()

def z_i_pbc(batch, params, q_function):
    mask = batch.unit_cell_mask
    to_replicate = batch.to_replicate_idx
    nr_nodes = batch.unfolded_nodes.shape[0]

    def calc_q_sc_sum(batch, params, mask):
        rijs = (
            batch.unfolded_positions[batch.unfolded_others]
            - batch.unfolded_positions[batch.unfolded_centers]
        )
        qs = q_function(batch, params, rijs)
        qs *= mask
        return jnp.sum(qs), qs

    # gradient of sum_j q_j wrt. r_i,beta
    d_sumq_drib, qs = jax.grad(calc_q_sc_sum, allow_int=True, has_aux=True, argnums=0)(
        batch, params, mask
    )

    # outer product of all positions with the corresponding gradient
    outer_product = (
        batch.unfolded_positions[:, :, None]
        * d_sumq_drib.unfolded_positions[:, None, :]
    )

    # sum of replica positions * gradient
    grad_q_sc_sum = jax.ops.segment_sum(
        outer_product,
        to_replicate,
        num_segments=nr_nodes,
    )

    # For the Barycenter calculation, we use positions which are not in the AD
    # graph in order to only be able to calc the derivative of the sum.
    stopgrad_positions = jax.lax.stop_gradient(batch.unfolded_positions)

    def z_alpha(alpha):
        def barycenter(batch, params, mask):
            rijs = (
                batch.unfolded_positions[batch.unfolded_others]
                - batch.unfolded_positions[batch.unfolded_centers]
            )
            qs = q_function(batch, params, rijs)
            qs *= mask  # Only restrict to simulation cell atoms, no ghosts

            return jnp.sum(qs[..., None] * stopgrad_positions, axis=0)[alpha]

        z_alpha = jax.grad(
            lambda b: barycenter(b, params, mask), allow_int=True, argnums=0
        )(batch).unfolded_positions

        z_alpha = jax.ops.segment_sum(
            z_alpha,
            to_replicate,
            num_segments=nr_nodes,
        )
        return z_alpha

    s1 = jax.vmap(lambda a: z_alpha(a))(jnp.arange(3)).transpose((1, 0, 2))

    Z = s1 - grad_q_sc_sum

    # add q_i * delta_alpha_beta
    Z = Z + jnp.einsum("i,ab->iab", qs, jnp.eye(3))
    return Z


def calc_q(params, batch, rijs, apply_fn):
    # rijs = (batch.positions[batch.others] -
    #         batch.positions[batch.centers])
    _, q = apply_fn(
        params,
        rijs,
        # batch.edges,
        batch.unfolded_centers,
        batch.unfolded_others,
        batch.unfolded_nodes,
        batch.node_to_graph,
        batch.unfolded_edge_mask,
        batch.unfolded_node_mask,
    )
    # print(f'{q.shape=}')
    # jax.debug.print("q[:9] = {}", q[:9])
    return q


def calc_apt(params, batch, calc_q):
    return z_i_pbc(batch, params, calc_q)


class Calculator(GetPropertiesMixin):
    # ase/vibes compatibility. not used!
    name = "marathon"
    parameters = {}

    def todict(self):
        return self.parameters

    implemented_properties = [
        "energy",
        "forces",
        "stress",
        "charges",
        "apt",
    ]

    def __init__(
        self,
        pred_fn,
        species_weights,
        params,
        cutoff,
        calc_apt_fn,
        to_sample,
        prepare_ewald,
        atoms=None,
        stress=False,
        add_offset=True,
        next_multiple=16,
        field=None,
    ):
        self.params = params
        self.cutoff = cutoff
        self.add_offset = add_offset
        self.next_multiple = next_multiple
        self.field = field

        if not stress:
            self.implemented_properties = ["energy", "forces", "charges",
                                           "apt"]

        predict_fn = lambda params, batch: pred_fn(
            params, batch,
            )

        self.predict_fn = jax.jit(predict_fn)
        self.calc_apt = jax.jit(calc_apt_fn)  # <-- Use calc_apt_fn parameter, not global calc_apt
        self.species_weights = species_weights

        self.atoms = None
        self.batch = None
        self.results = {}
        if atoms is not None:
            self.setup(atoms)

        self.batcher = None
        self.num_edges = 0
        self.num_nodes = 0

        # Use the passed to_sample and prepare_ewald
        self.to_sample = to_sample
        self.prepare_ewald = prepare_ewald

    @classmethod
    def from_checkpoint(
        cls,
        folder,
        to_sample,  # Add to_sample as a parameter
        prepare_ewald,  # Add prepare_ewald as a parameter
        **kwargs,
    ):
        from pathlib import Path

        from marathon.io import read_yaml, from_dict

        folder = Path(folder)

        model = from_dict(read_yaml(folder / "model/model.yaml"))

        _ = model.init(jax.random.key(1), *model.dummy_inputs())

        baseline = read_yaml(folder / "model/baseline.yaml")
        species_to_weight = baseline["elemental"]

        from marathon.emit.checkpoint import read_msgpack

        params = read_msgpack(folder / "model/model.msgpack")

        electric_field = (jnp.array(kwargs['field'], dtype=jnp.float32)
                          if kwargs['field'] is not None else None)

        from predict import get_predict_fn

        electric_field = (jnp.array(kwargs['field'], dtype=jnp.float32)
                          if kwargs['field'] is not None else None)

        predict_fn = get_predict_fn(model.apply, electrostatics='ewald',
                                    electric_field=electric_field,
                                    )
   
        # Create a fully reduced calc_apt_fn that only takes params and batch
        def calc_apt_fn_reduced(p, b):
            # Create the calc_q function with model.apply bound
            calc_q_fn = lambda params, batch, rijs: calc_q(batch, params,
                                                           rijs, model.apply)
            # Now call calc_apt with all three arguments
            return calc_apt(p, b, calc_q_fn)

        return cls(predict_fn, species_to_weight, params, model.cutoff,
                   calc_apt_fn=calc_apt_fn_reduced,
                   to_sample=to_sample,
                   prepare_ewald=prepare_ewald,
                   **kwargs)

    def update(self, atoms):
        changes = compare_atoms(self.atoms, atoms)

        if len(changes) > 0:
            self.results = {}
            self.atoms = atoms.copy()
            self.setup(atoms)

    def setup(self, atoms):
        sample = self.to_sample.map(atoms)

        n_edges = len(sample.graph.unfolded_centers)
        n_nodes = len(sample.graph.unfolded_nodes)

        if n_edges + 1 > self.num_edges or n_nodes + 1 > self.num_nodes:
            num_edges = next_multiple(n_edges, self.next_multiple)
            num_nodes = next_multiple(n_nodes, self.next_multiple)

            self.batcher = lambda x: get_batch(
                [x], num_nodes, num_edges, [], num_graphs=2
            )
            

        self.batch = self.prepare_ewald.map(self.batcher(sample))

    def calculate(
        self,
        atoms=None,
        properties=None,
        system_changes=None,
        **kwargs,
    ):
        self.update(atoms)

        results = self.predict_fn(self.params, self.batch)

        z = self.calc_apt(self.params, self.batch)

        actual_results = {}
        for key in self.implemented_properties:
            if key == "energy":
                actual_results[key] = float(results[key][self.batch.graph_mask].squeeze())
            elif key == "forces":
                actual_results[key] = np.array(results[key][self.batch.node_mask].reshape(-1, 3))
            elif key == "charges":
                actual_results[key] = np.array(results[key][self.batch.node_mask])
            elif key == "stress":
                raise KeyError

        actual_results['apt'] = np.array(z.reshape(-1, 3, 3)[self.batch.node_mask])

        if self.add_offset:
            energy_offset = np.sum(
                [self.species_weights[Z] for Z in atoms.get_atomic_numbers()]
            )
            actual_results["energy"] += energy_offset

        self.results = actual_results
        return actual_results

    def get_property(self, name, atoms=None, allow_calculation=True):
        if name not in self.implemented_properties:
            raise PropertyNotImplementedError(f"{name} property not implemented")

        self.update(atoms)

        if name not in self.results:
            if not allow_calculation:
                return None
            self.calculate(atoms=atoms)

        if name not in self.results:
            # For some reason the calculator was not able to do what we want,
            # and that is OK.
            raise PropertyNotImplementedError(
                f"{name} property not present in results!"
            )

        result = self.results[name]
        if isinstance(result, np.ndarray):
            result = result.copy()
        return result

    def get_potential_energy(self, atoms=None):
        return self.get_property(name="energy", atoms=atoms)

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


io_definitions = md_settings['io_definitions']
md_definitions = md_settings['md_definitions']

out_path = Path(io_definitions.get('out_path', './traj.traj'))
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

# Fresh start
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
calc = Calculator.from_checkpoint(
    model_dir, 
    to_sample=to_sample_lorem_ii,
    prepare_ewald=prepare_ewald_lorem_ii,
    field=field_vector
)

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

console.print(f"[green]Simulation completed successfully![/green]")
console.print(f"[green]Outputs saved to checkpoint {new_checkpoint_number}[/green]")
