import numpy as np

from collections import namedtuple

from marathon.data.batching import next_multiple

Batch = namedtuple(
    "Batch",
    (
        "nodes",  # Z_i
        "edges",  # R_ij
        "centers",  # i
        "others",  # j
        "node_to_graph",  # map nodes to original graphs
        "edge_to_graph",  # map edges to original graphs
        "graph_mask",  # False for padding
        "node_mask",  # False for padding
        "edge_mask",  # False for padding
        "cell_shifts",
        "labels",
        "electric_field",
        # for periodic, we set all this != None:
        "positions",  # -- assuming uni batch --
        "cell",  # -- assuming uni batch --
        "k_grid",  # set downstream
        "smearing",  # set downstream
        # else, we set this != None:
        "full_edges",  # R_ij all-to-all
        "full_centers",  # i all-to-all
        "full_others",  # j all-to-all
        "full_edge_to_graph",  # map edges to original graphs all-to-all
        "full_edge_mask",  # False for padding all-to-all
        "unfolded_nodes",  # Z_i for unfolded supercell
        "unfolded_positions",  # positions for unfolded supercell
        "unit_cell_mask",  # mask for the "real atoms" in the simulations cell
        "unfolded_centers",  # i for unfolded supercell
        "unfolded_others",  # j for unfolded supercell
        "to_replicate_idx",  # Index for which "original" each ghost belongs to
        "unfolded_node_mask",  # False for padding
        "unfolded_edge_mask",  # False for padding
    ),
)


def get_batch(samples, num_nodes, num_edges, keys, num_graphs=None):
    periodic = samples[0].graph.info["pbc"]
    if periodic:
        assert num_graphs == 2
        assert len(samples) == 1

    else:
        if num_graphs is None:
            num_graphs = len(samples) + 1
        else:
            num_input_graphs = len(samples)
            assert num_input_graphs + 1 <= num_graphs

        # note: this will cause recompiles, but this strategy seems to work ok for now
        # full_num_edges = get_size(sum([s.graph.full_edges.shape[0] for s in samples]))
        full_num_edges = num_edges

    nodes = np.zeros(num_nodes, dtype=int)
    edges = np.zeros((num_edges, 3), dtype=float)
    centers = np.zeros(num_edges, dtype=int)
    others = np.zeros(num_edges, dtype=int)
    electric_field = np.zeros((num_graphs, 3), dtype=float)
    node_to_graph = np.zeros(num_nodes, dtype=int)
    edge_to_graph = np.zeros(num_edges, dtype=int)
    graph_mask = np.zeros(num_graphs, dtype=bool)
    node_mask = np.zeros(num_nodes, dtype=bool)
    edge_mask = np.zeros(num_edges, dtype=bool)
    positions = np.zeros((num_nodes, 3), dtype=float)

    if periodic:
        cell = samples[0].graph.info["cell"]
        cell_shifts = np.zeros((num_edges, 3), dtype=int)
        unfolded_nodes = np.zeros(num_nodes, dtype=int)
        unfolded_positions = np.zeros((num_nodes, 3), dtype=float)
        unfolded_edges = np.zeros((num_edges, 3), dtype=int)
        unit_cell_masks = np.zeros(num_nodes, dtype=bool)
        unfolded_centers = np.zeros(num_edges, dtype=int)
        unfolded_others = np.zeros(num_edges, dtype=int)
        to_replicate_idx = np.zeros(num_nodes, dtype=int)

        unfolded_node_mask = np.zeros(num_nodes, dtype=bool)
        unfolded_edge_mask = np.zeros(num_edges, dtype=bool)

        unfolded_node_offset = 0
        unfolded_edges_offset = 0
    else:
        full_edges = np.zeros((full_num_edges, 3), dtype=float)
        full_centers = np.zeros(full_num_edges, dtype=int)
        full_others = np.zeros(full_num_edges, dtype=int)
        full_edge_to_graph = np.zeros(full_num_edges, dtype=int)
        full_edge_mask = np.zeros(full_num_edges, dtype=bool)

    labels = {}
    if "energy" in keys:
        labels["energy"] = np.zeros(num_graphs, dtype=float)
        labels["energy_mask"] = labels["energy"].astype(bool)
    if "forces" in keys:
        labels["forces"] = np.zeros((num_nodes, 3), dtype=float)
        labels["forces_mask"] = labels["forces"].astype(bool)
    if "apt" in keys:
        labels["apt"] = np.zeros((num_nodes, 3, 3), dtype=float)
        labels["apt_mask"] = labels["apt"].astype(bool)
    if "stress" in keys:
        labels["stress"] = np.zeros((num_graphs, 3, 3), dtype=float)
        labels["stress_mask"] = labels["stress"].astype(bool)

    node_offset = 0
    edge_offset = 0
    full_edge_offset = 0
    for i, sample in enumerate(samples):
        g = sample.graph
        l = sample.labels

        num_n = sample.graph.nodes.shape[0]
        num_e = sample.graph.edges.shape[0]

        if periodic:
            num_unfolded_n = sample.graph.unfolded_nodes.shape[0]
            unfolded_node_slice = slice(
                unfolded_node_offset, unfolded_node_offset + num_unfolded_n
            )

            num_unfolded_e = sample.graph.unfolded_edges.shape[0]
            unfolded_edge_slice = slice(
                unfolded_edges_offset, unfolded_edges_offset + num_unfolded_e
            )

        node_slice = slice(node_offset, node_offset + num_n)
        edge_slice = slice(edge_offset, edge_offset + num_e)

        nodes[node_slice] = g.nodes
        edges[edge_slice] = g.edges
        centers[edge_slice] = g.centers + node_offset
        others[edge_slice] = g.others + node_offset
        positions[node_slice] = g.info["positions"]

        if periodic:
            cell_shifts[edge_slice] = g.info["cell_shifts"]

            unfolded_nodes[unfolded_node_slice] = g.unfolded_nodes
            unfolded_positions[unfolded_node_slice] = g.unfolded_positions
            unfolded_edges[unfolded_edge_slice] = g.unfolded_edges
            unfolded_centers[unfolded_edge_slice] = (g.unfolded_centers +
                                                     unfolded_node_offset)
            unfolded_others[unfolded_edge_slice] = (g.unfolded_others +
                                                    unfolded_node_offset)
            unit_cell_masks[unfolded_node_slice] = g.unit_cell_mask
            to_replicate_idx[unfolded_node_slice] = (g.to_replicate_idx +
                                                     unfolded_node_offset)
        else:
            num_full_e = sample.graph.full_edges.shape[0]
            full_edge_slice = slice(full_edge_offset, full_edge_offset + num_full_e)
            full_edges[full_edge_slice] = g.full_edges
            full_centers[full_edge_slice] = g.full_centers + node_offset
            full_others[full_edge_slice] = g.full_others + node_offset
            full_edge_to_graph[full_edge_slice] = i
            full_edge_mask[full_edge_slice] = True
            full_edge_offset += num_full_e

        node_to_graph[node_slice] = i
        edge_to_graph[edge_slice] = i

        graph_mask[i] = True
        node_mask[node_slice] = True
        edge_mask[edge_slice] = True

        if periodic:
            unfolded_node_mask[unfolded_node_slice] = True
            unfolded_edge_mask[unfolded_edge_slice] = True

        # NaNs get replaced with zero, and then
        # later masked out in the loss
        if "energy" in keys:
            if not np.isnan(l["energy"]).any():
                labels["energy"][i] = l["energy"]
                labels["energy_mask"][i] = True
            else:
                labels["energy"][i] = 0.0

        if "forces" in keys:
            if not np.isnan(l["forces"]).any():
                labels["forces"][node_slice] = l["forces"]
                labels["forces_mask"][node_slice] = True
            else:
                labels["forces"][node_slice] = 0.0

        if "apt" in keys:
            if not np.isnan(l["apt"]).any():
                labels["apt"][node_slice, :] = l["apt"]
                labels["apt_mask"][node_slice, :] = True
            else:
                labels["apt"][i] = 0.0

        if "stress" in keys:
            if not np.isnan(l["stress"]).any():
                labels["stress"][i] = l["stress"]
                labels["stress_mask"][i] = True
            else:
                labels["stress"][i] = 0.0

        node_offset += num_n
        edge_offset += num_e

        if periodic:
            unfolded_edges_offset += num_unfolded_e
            unfolded_node_offset += num_unfolded_n

    # now we add the padding

    nodes[node_offset:] = samples[0].graph.nodes[0]  # todo: is this ok?
    # skip edges -- already zero
    centers[edge_offset:] = node_offset
    others[edge_offset:] = node_offset

    node_to_graph[node_offset:] = num_graphs - 1
    edge_to_graph[edge_offset:] = num_graphs - 1

    if not periodic:
        full_centers[full_edge_offset:] = node_offset
        full_others[full_edge_offset:] = node_offset
        full_edge_to_graph[full_edge_offset:] = num_graphs - 1

    # skip masks -- already False

    # skip labels -- already zero

    if periodic:
        return Batch(
            nodes,
            edges,
            centers,
            others,
            node_to_graph,
            edge_to_graph,
            graph_mask,
            node_mask,
            edge_mask,
            cell_shifts,
            labels,
            electric_field,
            positions,
            cell,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            unfolded_nodes,
            unfolded_positions,
            unit_cell_masks,
            unfolded_centers,
            unfolded_others,
            to_replicate_idx,
            unfolded_node_mask,
            unfolded_edge_mask,
        )
    else:
        return Batch(
            nodes,
            edges,
            centers,
            others,
            node_to_graph,
            edge_to_graph,
            graph_mask,
            node_mask,
            edge_mask,
            None,
            labels,
            electric_field,
            positions,
            None,
            None,
            None,
            full_edges,
            full_centers,
            full_others,
            full_edge_to_graph,
            full_edge_mask,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def get_size(n):
    if n <= 16:
        return next_multiple(n, 4)

    if n <= 64:
        return next_multiple(n, 16)

    if n <= 256:
        return next_multiple(n, 64)

    if n <= 1024:
        return next_multiple(n, 256)

    if n <= 4096:
        return next_multiple(n, 1024)

    if n <= 32768:
        return next_multiple(n, 4096)

    return next_multiple(n, 16384)
