import numpy as np

from dataclasses import dataclass

from marathon.extra.hermes.pain import (
    FilterTransform,
    MapTransform,
    RandomMapTransform,
    Record,
)


@dataclass(frozen=True)
class ToSample(MapTransform):
    cutoff: float
    energy: bool = True
    forces: bool = True
    stress: bool = False
    apt: bool = False

    def map(self, atoms):
        from sample import to_sample

        return to_sample(
            atoms,
            self.cutoff,
            energy=self.energy,
            forces=self.forces,
            stress=self.stress,
            apt=self.apt,
        )


@dataclass(frozen=True)
class SetUpEwald(MapTransform):
    lr_wavelength: float | None
    smearing: float | None

    def map(self, batch):
        # no-op for non-periodic case
        if batch.cell is not None:
            return batch._replace(
                k_grid=get_kgrid_ewald(batch.cell, self.lr_wavelength),
                smearing=self.smearing,
            )
        else:
            return batch

@dataclass(frozen=True)
class SetUpEwaldFixedKgrid(MapTransform):
    smearing: float | None
    k_grid: np.ndarray | None = None

    def map(self, batch):
        # no-op for non-periodic case
        if batch.cell is None:
            return batch

        k_grid = self.k_grid
        if k_grid is None:
            k_grid = get_kgrid_ewald(batch.cell, self.lr_wavelength)

        return batch._replace(
            k_grid=k_grid,
            smearing=self.smearing,
        )

def get_kgrid_ewald(cell, lr_wavelength):
    ns = np.ceil(np.linalg.norm(cell, axis=-1) / lr_wavelength)
    return np.ones((int(ns[0]), int(ns[1]), int(ns[2])))


@dataclass(frozen=True)
class ToFixedShapeBatch:
    num_nodes: int
    num_edges: int
    keys: tuple = ("energy", "forces", "apt")
    num_graphs: int = 2  # must be 2 for periodic (one real, one padding)

    def __call__(self, input_iterator):
        records_to_batch = []
        num_nodes = 0
        num_edges = 0
        last_record_metadata = None
        for input_record in input_iterator:
            this_record_metadata = input_record.metadata

            this_data = input_record.data
            this_num_nodes = this_data.graph.nodes.shape[0]
            this_num_edges = this_data.graph.edges.shape[0]

            if (
                num_nodes + this_num_nodes + 1 > self.num_nodes
                or num_edges + this_num_edges + 1 > self.num_edges
                or len(records_to_batch) + 1 == self.num_graphs
            ):
                batch = self._batch(records_to_batch)
                records_to_batch = []
                num_nodes = 0
                num_edges = 0
                yield Record(last_record_metadata.remove_record_key(), batch)

            records_to_batch.append(this_data)
            num_nodes += this_num_nodes
            num_edges += this_num_edges
            last_record_metadata = this_record_metadata

        # we exhausted the iterator, let's return the rest
        if records_to_batch:
            yield Record(
                last_record_metadata.remove_record_key(),
                self._batch(records_to_batch),
            )

    def _batch(self, records_to_batch):
        from batching import get_batch

        return get_batch(
            records_to_batch,
            self.num_nodes,
            self.num_edges,
            self.keys,
            num_graphs=self.num_graphs,
        )