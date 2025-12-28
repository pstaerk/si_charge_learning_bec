import numpy as np

from collections import namedtuple

from marathon.data.sample import to_labels #  , Sample

Sample = namedtuple("Sample", ("graph", "labels"))
Graph = namedtuple(
    "Graph",
    (
        "edges",
        "nodes",
        "centers",
        "others",
        "info",
        "full_edges",
        "full_centers",
        "full_others",
    ),
)


def to_sample(atoms, cutoff, energy=True, forces=True, stress=False,
              k_grid=np.ones((8, 8, 8)),
              smearing=1.8,
              apt=True,
              ):
    graph = to_graph(atoms, cutoff)

    labels = {}

    if energy:
        labels["energy"] = np.array(atoms.get_potential_energy())

    if forces:
        labels["forces"] = atoms.get_forces()

    if apt:
        # read the apt array from the atoms object
        labels['apt'] = atoms.arrays["apt"].reshape(-1, 3, 3)

    if stress:
        raw_stress = np.array(
            [atoms.get_stress(voigt=False, include_ideal_gas=False) * atoms.get_volume()]
        )

        # special case: FHI-aims + vibes return precisely zero if stress was not computed;
        # in this case we set it to nan so we can mask it out later
        if (raw_stress == 0.0).all():
            raw_stress *= float("nan")

        labels["stress"] = raw_stress

    return Sample(graph, labels)

def to_graph(atoms, cutoff):
    from vesin import ase_neighbor_list as neighbor_list

    if 'electric_field' in atoms.info:
        electric_field = atoms.info["electric_field"]

    if atoms.pbc.all():
        i, j, D, S = neighbor_list(
            "ijDS", atoms, cutoff
        )  # they follow the R_ij = R_j - R_i convention
        Z = atoms.get_atomic_numbers().astype(int)

        sort_idx = np.argsort(i)

        info = {"cell_shifts": S[sort_idx], "cell": atoms.get_cell().array, "pbc": True}
        if 'electric_field' in atoms.info:
            electric_field = atoms.info["electric_field"]
            info['electric_field'] = electric_field

        full_i = None
        full_j = None
        full_D = None

    else:
        assert not atoms.pbc.any()  # can't treat mixed pbc yet

        i, j, D = neighbor_list(
            "ijD", atoms, cutoff
        )  # they follow the R_ij = R_j - R_i convention
        Z = atoms.get_atomic_numbers().astype(int)

        sort_idx = np.argsort(i)
        info = {"pbc": False}

        N = len(atoms)
        full_i = np.arange(N).repeat(N)
        full_j = np.tile(np.arange(N), N)
        full_D = atoms.get_all_distances(vector=True).reshape(N * N, 3)

    # special case for sn2 dataset: empty neighborlists get forcibly extended,
    # the cutoff function should take care of it
    if len(i) == 0:
        if len(atoms) > 1:
            i = np.array([0, 1])
            j = np.array([1, 0])
            sort_idx = np.array([0, 1])
            d = atoms.get_distance(0, 1, mic=True, vector=True)
            D = np.array([d, -d])

    if len(i) > 0:
        info["max_neighbors"] = np.unique(i, return_counts=True)[1].max()
    else:
        info["max_neighbors"] = 0

    info["positions"] = atoms.get_positions()

    return Graph(
        D[sort_idx],
        Z,
        i[sort_idx],
        j[sort_idx],
        info,
        full_D,
        full_i,
        full_j,
    )