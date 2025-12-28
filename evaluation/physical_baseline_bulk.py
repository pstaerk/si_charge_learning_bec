import matplotlib.pyplot as plt
import numpy as np
import yaml
import pathlib
import sys
import jax
import jax.numpy as jnp
from ase.io import read, write
from ase import Atoms
import os
from importlib import reload

from marathon.extra.hermes import (
    DataLoader,
    DataSource,
    FilterEmpty,
    IndexSampler,
    ToStack,
    prefetch_to_device,
)
from pathlib import Path
import time


def load_model(folder, model_definitions, predict_i=True, **kwargs):
    model_definitions = pathlib.Path(model_definitions).resolve()
    if str(model_definitions) not in sys.path:
        sys.path.append(str(model_definitions))

    from pathlib import Path
    from myrto.engine import from_dict, read_yaml

    folder = Path(folder)

    model = from_dict(read_yaml(folder / "model/model.yaml"))

    _ = model.init(jax.random.key(1), *model.dummy_inputs())

    baseline = read_yaml(folder / "model/baseline.yaml")
    species_to_weight = baseline["elemental"]

    from marathon.emit.checkpoint import read_msgpack
    from predict import get_predict_fn

    params = read_msgpack(folder / "model/model.msgpack")

    pred_fn = get_predict_fn(model.apply)

    return pred_fn, model.apply, species_to_weight, params, model.cutoff

pred_fn, apply_fn, species_to_weight, params, cutoff = load_model(
    '../training_runs/model_lr_alldata/run/checkpoints/R2_E+F/',
    '../model_lr_only/', field=None, predict_i=False)

def sum_q_sc_screen(batch, params, screen):
    mask = batch.unit_cell_mask
    to_replicate = batch.to_replicate_idx
    nr_nodes = batch.unfolded_nodes.shape[0]

    def calc_q_sc_sum(batch, params, mask):
        rijs = (batch.unfolded_positions[batch.unfolded_others] -
                batch.unfolded_positions[batch.unfolded_centers])
        qs = calc_q_superbatch(batch, params, rijs)
        qs *= screen
        qs *= mask
        return jnp.sum(qs), qs

    # gradient of sum_j q_j wrt. r_i,beta
    d_sumq_drib, qs = jax.grad(calc_q_sc_sum, allow_int=True,
                               has_aux=True, argnums=0)(batch, params, mask)

    # outer product of all positions with the corresponding gradient
    outer_product = (batch.unfolded_positions[:, :, None] *
                     d_sumq_drib.unfolded_positions[:, None, :])

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
            rijs = (batch.unfolded_positions[batch.unfolded_others] -
                    batch.unfolded_positions[batch.unfolded_centers])
            qs = calc_q_superbatch(batch, params, rijs)
            qs *= screen
            qs *= mask  # Only restrict to simulation cell atoms, no ghosts

            return jnp.sum(qs[:, None] * stopgrad_positions, axis=0)[alpha]

        z_alpha = jax.grad(lambda b: barycenter(b, params, mask),
                           allow_int=True, argnums=0)(batch).unfolded_positions

        z_alpha = jax.ops.segment_sum(
            z_alpha,
            to_replicate,
            num_segments=nr_nodes,
            )
        return z_alpha

    s1 = jax.vmap(lambda a: z_alpha(a))(jnp.arange(3)).transpose((1,0,2))  # (N, 3, 3)

    Z = s1 - grad_q_sc_sum

    # add q_i * delta_alpha_beta
    Z = Z + jnp.einsum('i,ab->iab', qs, jnp.eye(3))
    return Z

@jax.jit
def calc_q_superbatch(batch, params, rijs):
    # output of the model is energy, apt, q
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
    return q


cutoff = 5.
num_edges = 150_000
num_nodes = 5000
num_graphs = 2
# load the path from the model_ii directory, because we want the unfolded
# batching_path = os.path.abspath('../model_ii_local_screen/')
batching_path = os.path.abspath('../lorem_ii/')
sys.path.insert(0, batching_path)
from transforms import ToSample, SetUpEwald, ToFixedShapeBatch

to_sample = ToSample(cutoff=cutoff,
                     stress=False, apt=True)
batch_style = "batch_shape"

valid_samples = []

data_valid = Path("/work/pstaerk/datasets_ml/bingqing_finite_apt/valid")
source_valid = DataSource(data_valid)
n_valid = len(source_valid)

from marathon.extra.hermes.pain import Record, RecordMetadata


def valid_iterator():
    filterer = FilterEmpty()
    for i in range(n_valid):
        sample = to_sample.map(source_valid[i])
        if filterer.filter(sample):
            valid_samples.append(sample)
            yield Record(
                data=sample, metadata=RecordMetadata(index=i, record_key=i)
            )


def get_batcher():
    assert batch_style == "batch_shape"
    return ToFixedShapeBatch(
        num_graphs=num_graphs, num_edges=num_edges, num_nodes=num_nodes,
        # keys=('energy', 'forces', 'apt'),
        keys=('energy', 'forces', 'apt'),
    )


batcher = get_batcher()
prepare_ewald = SetUpEwald(lr_wavelength=cutoff / 8, smearing=cutoff / 4)
data_valid = [prepare_ewald.map(b.data) for b in batcher(valid_iterator())]

sys.path.remove(batching_path)  # avoid conflicts with other batching.py

# model_path = os.path.abspath('../model_lr_only/')
# sys.path.insert(0, model_path)

# remove from path again to avoid conflicts

from loss import get_loss_fn

loss_weights = {"energy": 1., "forces": 1.}

import itertools
iter_valid_with_prefetch = prefetch_to_device(itertools.cycle(data_valid), 2)

def make_serializable(data):
    """Convert arrays to serializable format."""
    return {
        k: v.tolist() if isinstance(v, (np.ndarray, jnp.ndarray)) else v 
        for k, v in data.items()
    }


def extract_results(results, batch, labels, params):
    """Extract and format results from model predictions."""
    actual_results = {}

    # Extract standard predictions
    for key in ['energy', 'forces', 'apt', 'charges']:
        if key == "energy":
            if key in results:
                actual_results[key] = float(results[key][batch.graph_mask].squeeze())
        elif key == "forces":
            if key in results:
                actual_results[key] = np.array(results[key][batch.node_mask].reshape(-1, 3))
        elif key == "charges":
            if key in results:
                actual_results[key] = np.array(results[key][batch.node_mask])
        elif key == 'apt':
            if key in results:
                actual_results[key] = np.array(results[key][batch.node_mask].reshape(-1, 3, 3))
    
    # Calculate APT from charges (periodic BC)
    apt_pbc = sum_q_sc_screen(batch, params, 1.)
    apt_pbc *= batch.node_mask[..., None, None]
    actual_results['apt_from_charges'] = np.array(apt_pbc[batch.node_mask].reshape(-1, 3, 3))
    
    # Extract labels
    labels_extracted = {}
    for key in ['energy', 'forces', 'apt']:
        if key == "energy":
            labels_extracted[key] = float(labels[key][batch.graph_mask].squeeze())
        elif key == "forces":
            labels_extracted[key] = np.array(labels[key][batch.node_mask].reshape(-1, 3))
        elif key == 'apt':
            labels_extracted[key] = np.array(labels[key][batch.node_mask].reshape(-1, 3, 3))
            
    return actual_results, labels_extracted


# Create output directory
output_path = Path('/work/pstaerk/outs/physical_baseline_bulk/')
os.makedirs(output_path, exist_ok=True)

# Warmup
print("Warming up JIT...")
batch = data_valid[0]
out = pred_fn(params, batch)
jax.tree_util.tree_map(lambda x: x.block_until_ready(), out)
apt = sum_q_sc_screen(batch, params, 1.)
apt.block_until_ready()

# Evaluate
print(f"Evaluating {len(data_valid)} batches...")
timings = []

for i in range(len(data_valid)):
    batch = data_valid[i]
    
    start = time.perf_counter()
    results = pred_fn(params, batch)
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), results)
    end = time.perf_counter()
    
    execution_time = end - start
    timings.append(execution_time)
    
    print(f"Batch {i+1}/{len(data_valid)} - Execution time: {execution_time:.6f} s")
    
    actual_results, labels_extracted = extract_results(
        results, batch, batch.labels, params
    )
    
    # Save results
    output_data = {
        'predictions': make_serializable(actual_results),
        'labels': make_serializable(labels_extracted),
        'execution_time': execution_time,
    }
    
    output_file = output_path / f'results_dict{i}.yaml'
    with open(output_file, 'w') as file:
        yaml.dump(output_data, file)

    translate_symbol = {1: 'H', 8: 'O'}
    # Convert batch.nodes to numpy array first, then filter
    nodes_array = np.array(batch.nodes)
    node_mask_array = np.array(batch.node_mask)
    symbols = [translate_symbol[int(Z)] for Z in nodes_array[node_mask_array]]

    atoms = source_valid[i]
    # atoms = Atoms(
    #     symbols=symbols,
    #     positions=np.array(batch.positions[batch.node_mask]),
    #     cell=np.array([batch.cell[0, 0],
    #                    batch.cell[1, 1],
    #                    batch.cell[2, 2]]),
    # )
    atoms.arrays['charges'] = actual_results.get('charges', np.zeros(len(atoms)))
    atoms.arrays['screen'] = actual_results.get('screen', np.ones(len(atoms)))

    # save this via extxyz writer from ase
    struct_file = Path(output_path) / f'structure_with_charges_{i}.xyz'
    write(struct_file, atoms,
          columns=['symbols', 'positions', 'charges', 'screen'],
          format='extxyz') 

# Save timing statistics
timing_stats = {
    'total_time': sum(timings),
    'mean_time': np.mean(timings),
    'std_time': np.std(timings),
    'min_time': min(timings),
    'max_time': max(timings),
    'all_timings': timings,
}

timing_file = output_path / 'timing_statistics.yaml'
with open(timing_file, 'w') as file:
    yaml.dump(timing_stats, file)

print(f"\nEvaluation complete. Results saved to {output_path}")
print(f"Total time: {timing_stats['total_time']:.3f} s")
print(f"Mean time per batch: {timing_stats['mean_time']:.6f} s ± {timing_stats['std_time']:.6f} s")
