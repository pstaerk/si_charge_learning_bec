import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_default_matmul_precision", "highest")

from ase.calculators.abc import GetPropertiesMixin
from ase.calculators.calculator import PropertyNotImplementedError, compare_atoms

from marathon import comms
from marathon.data.batching import next_multiple

from batching import get_batch
from transforms import ToSample, SetUpEwald


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
        self.species_weights = species_weights

        self.atoms = None
        self.batch = None
        self.results = {}
        if atoms is not None:
            self.setup(atoms)

        self.batcher = None
        self.num_edges = 0
        self.num_nodes = 0

        self.to_sample = ToSample(
            cutoff=cutoff, forces=False, energy=False, stress=False
        )
        self.prepare_ewald = SetUpEwald(lr_wavelength=cutoff / 8, smearing=cutoff / 4)

    @classmethod
    def from_checkpoint(
        cls,
        folder,
        **kwargs,
    ):
        from pathlib import Path

        from myrto.engine import from_dict, read_yaml

        folder = Path(folder)

        model = from_dict(read_yaml(folder / "model/model.yaml"))

        _ = model.init(jax.random.key(1), *model.dummy_inputs())

        baseline = read_yaml(folder / "model/baseline.yaml")
        species_to_weight = baseline["elemental"]

        from marathon.emit.checkpoint import read_msgpack

        params = read_msgpack(folder / "model/model.msgpack")

        # from predict import get_predict_fn

        electric_field = (jnp.array(kwargs['field'], dtype=jnp.float32)
                          if kwargs['field'] is not None else None)

        predict_fn = lambda params, batch: model.predict(
            params,
            batch,
            electric_field=electric_field,
            excess_charge_neutralization=True,
        )

        return cls(predict_fn, species_to_weight, params, model.cutoff,
                   **kwargs)

    def update(self, atoms):
        changes = compare_atoms(self.atoms, atoms)

        if len(changes) > 0:
            self.results = {}
            self.atoms = atoms.copy()
            self.setup(atoms)

    def setup(self, atoms):
        sample = self.to_sample.map(atoms)

        n_edges = len(sample.graph.centers)
        n_nodes = len(sample.graph.nodes)

        # we need to check if full_edges exist and use that for the edges
        n_edges = (len(sample.graph.full_edges)
                   if sample.graph.full_edges is not None else n_edges)

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

        actual_results = {}
        for key in self.implemented_properties:
            if key == "energy":
                actual_results[key] = float(results[key][self.batch.graph_mask].squeeze())
            elif key == "forces":
                actual_results[key] = np.array(results[key][self.batch.node_mask].reshape(-1, 3))
            elif key == "charges":
                actual_results[key] = np.array(results[key][self.batch.node_mask])
            elif key == 'apt':
                actual_results[key] = np.array(results[key][self.batch.node_mask].reshape(-1, 3, 3))
            elif key == "stress":
                raise KeyError

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