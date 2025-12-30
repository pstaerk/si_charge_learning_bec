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
        "unfolded_nodes",
        "unfolded_positions",
        "unfolded_edges",
        "unit_cell_mask",
        "unfolded_centers",
        "unfolded_others",
        "to_replicate_idx",
    ),
)


def to_sample(atoms, cutoff, energy=True, forces=True, stress=False, apt=True):
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

        (all_nodes, all_positions, sD,
         si, sj, unit_cell_mask, to_replicate_idx) = (
             unfolded_ghosts(atoms, cutoff)
         )

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

        # In non-periodic Systems these are not needed
        (all_nodes, all_positions, sD, si, sj, unit_cell_mask,
         to_replicate_idx) = (
            None, None, None, None, None, None, None
        )

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
        all_nodes,
        all_positions,
        sD,
        unit_cell_mask,
        si,
        sj,
        to_replicate_idx,
    )


def unfolded_ghosts(atoms, cutoff,):
    """Builds all ghost atoms explicitly.

    :atoms: ase.Atoms object with periodic boundary conditions
    :cutoff: float, cutoff for neighbor list, needs to be effective cutoff

    returns:
    :all_nodes: (N, ) int array of atomic numbers including ghosts
    :all_positions: (N, 3) float array of positions including ghosts
    :sD: (E, 3) float array of all edge vectors including ghosts
    :si: (E, ) int array of source node indices for edges including ghosts
    :sj: (E, ) int array of target node indices for edges including ghosts
    :unit_cell_mask: (N, ) bool array, True for atoms in original unit cell
    :to_replicate: (N, ) int array, mapping from all_positions to original
                         atoms

    """
    import vesin
    nl_calc = vesin.NeighborList(cutoff=cutoff, full_list=True)

    # 1. Find all neighbors, including those in periodic images
    positions = atoms.positions
    i_p, j_p, S_p = nl_calc.compute(
        points=positions,
        box=atoms.cell,
        periodic=True,
        quantities="ijS"
    )
    # Find all unique (shift, atom_index) pairs
    replicas = np.unique(np.concatenate([S_p, j_p[:, None]], axis=-1), axis=0)
    is_original_cell = np.all(replicas[:, :3] == 0, axis=1)

    replicas = replicas[~is_original_cell]  # only replicas, not originals

    # Separate shifts from atom indices
    cell_shifts, to_replicate = np.split(replicas, [3], axis=-1)
    to_replicate = to_replicate.flatten().astype(int)

    # 3. Construct the supercell positions and node types
    # print(f'{np.zeros((len(positions), 3), dtype=int).shape=} {cell_shifts.shape=}')
    cell_shifts = np.concatenate([np.zeros((len(positions), 3), dtype=int),
                                  cell_shifts], axis=0)
    to_replicate = np.concatenate([np.arange(len(positions), dtype=int),
                                   np.array(to_replicate, dtype=int)],
                                   dtype=int)

    unit_cell_mask = np.zeros(len(to_replicate), dtype=bool)
    unit_cell_mask[: len(positions)] = True

    offsets = np.einsum("pA,Aa->pa", cell_shifts, atoms.cell)
    all_positions = atoms.positions[to_replicate] + offsets
    all_nodes = atoms.get_atomic_numbers()[to_replicate]

    # 3. Construct the supercell positions and node types
    offsets = np.einsum("pA,Aa->pa", cell_shifts, atoms.cell)
    all_positions = atoms.positions[to_replicate] + offsets
    all_nodes = atoms.get_atomic_numbers()[to_replicate]

    # 5. Compute the new, non-periodic neighbor list for the supercell
    super_box = atoms.cell * np.max(np.abs(cell_shifts), axis=0) * 2
    si, sj, sD = nl_calc.compute(
        points=all_positions,
        periodic=False,
        quantities="ijD",
        box=super_box,
    )
    return all_nodes, all_positions, sD, si, sj, unit_cell_mask, to_replicate
