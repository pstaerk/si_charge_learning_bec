import matplotlib.pyplot as plt
import numpy as np
import yaml
import pathlib
import sys
import jax
import jax.numpy as jnp
from ase.io import read, write
import os
import time
import itertools
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

    params = read_msgpack(folder / "model/model.msgpack")

    return model.predict, model.apply, species_to_weight, params, model.cutoff

pred_fn, apply_fn, species_to_weight, params, cutoff = load_model(
    '../training_runs/lorem_baseline_clusters/run/checkpoints/R2_E+F/',
    '../lorem_baseline/', field=None, predict_i=False)


@jax.jit
def calc_rijs(batch):
    rij = batch.positions[batch.others] - batch.positions[batch.centers]
    return rij


@jax.jit
def pol_function_npbc(params, batch, alpha):
    """Calculate the polarization for non-periodic boundary conditions.
    This is a dummy function that does not use the periodicity of the system.
    """
    rijs = calc_rijs(batch)
    _, q = apply_fn(
        params,
        rijs,
        batch.centers,
        batch.others,
        batch.nodes,
        batch.edge_mask,
        batch.node_mask,
        batch.positions,
        batch.cell,
        batch.k_grid,
        batch.smearing,
        batch.full_edges,
        batch.full_centers,
        batch.full_others,
        batch.full_edge_mask,
    )

    charges = q.flatten()
    return jnp.sum(batch.positions[:, alpha] * charges * batch.node_mask)


@jax.jit
def z_i_alpha_beta_npbc(params, batch):
    """Calculate the z_i_alpha_beta for non-periodic boundary conditions.
    This is a dummy function that does not use the periodicity of the system.
    """
    deriv_p_npbc = jax.value_and_grad(pol_function_npbc, has_aux=False,
                                            argnums=1, allow_int=True)
    z_i_ab = jnp.zeros((batch.positions.shape[0], 3, 3), dtype=jnp.float32)

    for alpha in range(3):
        # we want the derivative of P_alpha with respect to r_beta
        _, deriv_alpha = deriv_p_npbc(params, batch, alpha)
        for beta in range(3):
            component = deriv_alpha.positions[:, beta]
            z_i_ab = z_i_ab.at[:, alpha, beta].set(component)
    return z_i_ab


cutoff = 5.
num_edges = 10_000
num_nodes = 200
num_graphs = 2
# load the path from the model_ii directory, because we want the unfolded
batching_path = os.path.abspath('../lorem_ii/')
sys.path.insert(0, batching_path)
from transforms import ToSample, SetUpEwald, ToFixedShapeBatch

to_sample = ToSample(cutoff=cutoff,
                     stress=False, apt=True)
batch_style = "batch_shape"

valid_samples = []

data_valid = Path("/work/pstaerk/datasets_ml/water_clusters/valid")
# data_valid = Path("/work/amam/cps2815/datasets_ml/water_clusters/valid")
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
# prepare_ewald = SetUpEwald(lr_wavelength=cutoff / 8, smearing=cutoff / 4)
data_valid = [b.data for b in batcher(valid_iterator())]

sys.path.remove(batching_path)  # avoid conflicts with other batching.py

# model_path = os.path.abspath('../model_lr_only/')
# sys.path.insert(0, model_path)

# remove from path again to avoid conflicts

from loss import get_loss_fn

loss_weights = {"energy": 1., "forces": 1.}

iter_valid_with_prefetch = prefetch_to_device(itertools.cycle(data_valid), 2)

print(next(iter_valid_with_prefetch).unfolded_nodes)


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
                actual_results[key] = np.array(results[key][batch.node_mask])[..., 0]
        elif key == 'apt':
            if key in results:
                actual_results[key] = np.array(results[key][batch.node_mask].reshape(-1, 3, 3))
    
    # Calculate APT from charges (non-periodic BC)
    apt_npbc = z_i_alpha_beta_npbc(params, batch)
    apt_npbc *= batch.node_mask[..., None, None]
    actual_results['apt_from_charges'] = np.array(apt_npbc[batch.node_mask].reshape(-1, 3, 3))
    
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
output_path = Path('/work/pstaerk/outs/lorem_baseline_clusters/')
# output_path = Path('/work/amam/cps2815/outs/lorem_baseline_clusters/')
os.makedirs(output_path, exist_ok=True)

# Warmup
print("Warming up JIT...")
pred_fn = jax.jit(pred_fn)
batch = next(iter_valid_with_prefetch)
out = pred_fn(params, batch)
jax.tree_util.tree_map(lambda x: x.block_until_ready(), out)
apt = z_i_alpha_beta_npbc(params, batch)
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

    atoms = source_valid[i]
    atoms.arrays['charges'] = actual_results.get('charges', np.zeros(len(atoms)))
    atoms.arrays['screen'] = actual_results.get('screen', np.ones(len(atoms)))

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
