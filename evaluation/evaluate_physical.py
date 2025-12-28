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
from dataclasses import dataclass
from typing import Optional, Dict, List, Any
from pathlib import Path

from ase import Atoms
from marathon.extra.hermes import (
    DataLoader,
    DataSource,
    FilterEmpty,
    IndexSampler,
    ToStack,
    prefetch_to_device,
)
from marathon.extra.hermes.pain import Record, RecordMetadata


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""
    # Model paths
    checkpoint_folder: str
    model_definitions: str
    
    # Data paths
    validation_data_path: str
    output_path: str
    
    # Model parameters
    cutoff: float = 5.0
    field: Optional[List[float]] = None
    predict_i: bool = False
    
    # Batching parameters
    num_edges: int = 150_000
    num_nodes: int = 5000
    num_graphs: int = 2
    batch_style: str = "batch_shape"
    
    # Ewald parameters
    lr_wavelength_factor: float = 8.0
    smearing_factor: float = 4.0
    
    # APT calculation parameters
    calculate_apt_from_charges: bool = False  # Calculate APT from charges
    use_pbc: bool = True  # Use periodic boundary conditions (True) or non-periodic (False)
    apt_batching_path: Optional[str] = None  # Path to batching with unfolded structure (for PBC)
    
    # Other parameters
    stress: bool = False
    apt: bool = True
    prefetch_size: int = 2
    n_max: Optional[int] = None  # Maximum number of structures to evaluate
    output_screen: bool = False


class ModelEvaluator:
    """Handles model loading and evaluation."""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.pred_fn = None
        self.apply_fn = None
        self.species_to_weight = None
        self.params = None
        self.model_cutoff = None
        self.valid_samples = []
        
    def load_model(self):
        """Load the model and its parameters."""
        model_definitions = pathlib.Path(self.config.model_definitions).resolve()
        if str(model_definitions) not in sys.path:
            sys.path.append(str(model_definitions))

        from myrto.engine import from_dict, read_yaml
        from marathon.emit.checkpoint import read_msgpack
        from predict import get_predict_fn  # Import get_predict_fn

        folder = Path(self.config.checkpoint_folder)

        model = from_dict(read_yaml(folder / "model/model.yaml"))
        _ = model.init(jax.random.key(1), *model.dummy_inputs())

        baseline = read_yaml(folder / "model/baseline.yaml")
        self.species_to_weight = baseline["elemental"]

        self.params = read_msgpack(folder / "model/model.msgpack")
        
        # Use get_predict_fn instead of model.predict
        if self.config.predict_i:
            self.pred_fn = get_predict_fn(
                apply_fn=model.apply,
                # output_screen=self.config.output_screen,
                # predict_i=self.config.predict_i,
                electric_field=self.config.field
                # electric_field=self.config.field
            )
        else:
            self.pred_fn = get_predict_fn(
                apply_fn=model.apply,
                output_screening=self.config.output_screen,
                cutoff=self.config.cutoff,
            )
        
        self.apply_fn = model.apply  # Store for APT calculation
        self.model_cutoff = model.cutoff
        
        return self
    
    def setup_apt_calculation(self):
        """Setup functions for APT calculation from charges."""
        if not self.config.calculate_apt_from_charges:
            return
        
        # Store apply_fn for calc_q_superbatch
        apply_fn = self.apply_fn
        
        # Setup for periodic boundary conditions (unfolded batching)
        if self.config.use_pbc:
            @jax.jit
            def calc_q_superbatch(batch, params, rijs):
                energies, q = apply_fn(
                    params,
                    rijs,
                    batch.unfolded_centers,
                    batch.unfolded_others,
                    batch.unfolded_nodes,
                    batch.unfolded_edge_mask,
                    batch.unfolded_node_mask,
                    batch.positions,
                    batch.cell,
                    batch.k_grid,
                    batch.smearing,
                    batch.full_edges,
                    batch.full_centers,
                    batch.full_others,
                    batch.full_edge_mask,
                )
                return q[..., 0]
            
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

            self.sum_q_sc_screen = sum_q_sc_screen

        # Setup for non-periodic boundary conditions
        else:  # not self.config.use_pbc
            def calc_rijs(batch):
                rij = batch.positions[batch.others] - batch.positions[batch.centers]
                return rij

            def pol_function_npbc(params, batch, alpha):
                """Calculate the polarization for non-periodic boundary conditions."""
                rijs = calc_rijs(batch)
                _, q = apply_fn(
                    params,
                    rijs,
                    batch.centers,
                    batch.others,
                    batch.nodes,
                    batch.node_to_graph,
                    batch.edge_mask,
                    batch.node_mask,
                )

                charges = q.flatten()
                return jnp.sum(batch.positions[:, alpha] * charges * batch.node_mask)

            def z_i_alpha_beta_npbc(params, batch):
                """Calculate the z_i_alpha_beta for non-periodic boundary conditions."""
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
            
            self.z_i_alpha_beta_npbc = z_i_alpha_beta_npbc

    def prepare_data(self):
        """Prepare validation data with batching and transforms."""
        # Determine which batching to use
        if self.config.calculate_apt_from_charges and self.config.use_pbc and self.config.apt_batching_path:
            batching_path = os.path.abspath(self.config.apt_batching_path)
            print(f"Using custom batching from: {batching_path}")
        else:
            batching_path = os.path.abspath(self.config.model_definitions)
        
        # Remove old transforms from sys.modules if it exists
        if 'transforms' in sys.modules:
            del sys.modules['transforms']
        
        # Insert batching path at the beginning
        sys.path.insert(0, batching_path)
        
        # Now import fresh
        from transforms import ToSample, SetUpEwald, ToFixedShapeBatch
        
        to_sample = ToSample(
            cutoff=self.config.cutoff,
            stress=self.config.stress,
            apt=self.config.apt
        )
        
        data_valid = Path(self.config.validation_data_path)
        source_valid = DataSource(data_valid)
        n_valid = len(source_valid)
    
        # for saving the strucuttures with charges later
        self.source_valid = source_valid
        
        # Apply n_max limit if specified
        if self.config.n_max is not None:
            n_valid = min(n_valid, self.config.n_max)
            print(f"Limiting evaluation to {n_valid} structures (n_max={self.config.n_max})")
        
        def valid_iterator():
            filterer = FilterEmpty()
            for i in range(n_valid):
                sample = to_sample.map(source_valid[i])
                if filterer.filter(sample):
                    self.valid_samples.append(sample)
                    yield Record(
                        data=sample, 
                        metadata=RecordMetadata(index=i, record_key=i)
                    )
        
        batcher = ToFixedShapeBatch(
            num_graphs=self.config.num_graphs,
            num_edges=self.config.num_edges,
            num_nodes=self.config.num_nodes,
            keys=('energy', 'forces', 'apt'),
        )
        
        prepare_ewald = SetUpEwald(
            lr_wavelength=self.config.cutoff / self.config.lr_wavelength_factor,
            smearing=self.config.cutoff / self.config.smearing_factor
        )
        
        data_valid = [prepare_ewald.map(b.data) for b in batcher(valid_iterator())]
        
        # Remove batching path to avoid conflicts
        sys.path.remove(batching_path)
        
        return data_valid
    
    def warmup(self, data_batch):
        """Warmup JIT compilation."""
        self.pred_fn = jax.jit(self.pred_fn)
        out = self.pred_fn(self.params, data_batch)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), out)
        
        if self.config.calculate_apt_from_charges:
            if self.config.use_pbc:
                apt = self.sum_q_sc_screen(data_batch, self.params, screen=1.0)
                apt.block_until_ready()
            else:
                apt = self.z_i_alpha_beta_npbc(self.params, data_batch)
                apt.block_until_ready()
        
    def extract_results(self, results: Dict, batch, labels: Dict) -> tuple:
        """Extract and format results from model predictions."""
        actual_results = {}
        for key in ['energy', 'forces', 'apt', 'charges', 'screen']:
            if key == "energy":
                actual_results[key] = float(results[key][batch.graph_mask].squeeze())
            elif key == "forces":
                actual_results[key] = np.array(results[key][batch.node_mask].reshape(-1, 3))
            elif key == "charges":
                if key in results:
                    actual_results[key] = np.array(results[key][batch.node_mask])
            elif key == "screen":
                if key in results:
                    actual_results[key] = np.array(results[key][batch.node_mask])
            elif key == 'apt':
                if key in results:
                    actual_results[key] = np.array(results[key][batch.node_mask].reshape(-1, 3, 3))
        
        # Calculate APT from charges if requested
        if self.config.calculate_apt_from_charges:
            if self.config.use_pbc:
                # Periodic boundary conditions
                apt_from_charges = self.sum_q_sc_screen(batch, self.params, screen=1.0)
                actual_results['apt_from_charges'] = np.array(apt_from_charges[batch.node_mask].reshape(-1, 3, 3))
            else:
                # Non-periodic boundary conditions
                apt_npbc = self.z_i_alpha_beta_npbc(self.params, batch)
                apt_npbc *= batch.node_mask[..., None, None]
                actual_results['apt_from_charges'] = np.array(apt_npbc[batch.node_mask].reshape(-1, 3, 3))
        
        # energy_offset = np.sum(
        #     [self.species_to_weight[int(Z)] for Z in batch.nodes]
        # )
        # actual_results["energy"] += energy_offset
        
        labels_extracted = {}
        for key in ['energy', 'forces', 'apt']:
            if key == "energy":
                labels_extracted[key] = float(labels[key][batch.graph_mask].squeeze())
            elif key == "forces":
                labels_extracted[key] = np.array(labels[key][batch.node_mask].reshape(-1, 3))
            elif key == 'apt':
                labels_extracted[key] = np.array(labels[key][batch.node_mask].reshape(-1, 3, 3))

        # labels_extracted['positions'] = np.array(batch.positions[batch.node_mask].reshape(-1, 3))
                
        return actual_results, labels_extracted
    
    def make_serializable(self, data: Dict) -> Dict:
        """Convert arrays to serializable format."""
        return {
            k: v.tolist() if isinstance(v, (np.ndarray, jnp.ndarray)) else v 
            for k, v in data.items()
        }
    
    def evaluate(self):
        """Run full evaluation pipeline."""
        print("Loading model...")
        self.load_model()
        
        if self.config.calculate_apt_from_charges:
            print("Setting up APT calculation from charges...")
            self.setup_apt_calculation()
        
        print("Preparing data...")
        data_valid = self.prepare_data()
        
        print("Setting up data iterator...")
        iter_valid = prefetch_to_device(
            itertools.cycle(data_valid), 
            self.config.prefetch_size
        )
        
        print("Warming up JIT...")
        print(f'{next(iter_valid)=}')
        self.warmup(next(iter_valid))
        
        print(f"Evaluating {len(data_valid)} batches...")
        os.makedirs(self.config.output_path, exist_ok=True)

        nr_dats = (len(data_valid) if self.config.n_max is None else
                   min(len(data_valid), self.config.n_max))
        
        timings = []
        
        for i in range(nr_dats):
            batch = data_valid[i]
            
            start = time.perf_counter()
            results = self.pred_fn(self.params, batch)
            jax.tree_util.tree_map(lambda x: x.block_until_ready(), results)
            end = time.perf_counter()
            
            execution_time = end - start
            timings.append(execution_time)
            
            print(f"Batch {i+1}/{nr_dats} - Execution time: {execution_time:.6f} s")
            
            actual_results, labels_extracted = self.extract_results(
                results, batch, batch.labels
            )
            
            # Save results
            output_data = {
                'predictions': self.make_serializable(actual_results),
                'labels': self.make_serializable(labels_extracted),
                'execution_time': execution_time,
            }

            output_file = Path(self.config.output_path) / f'results_dict{i}.yaml'
            with open(output_file, 'w') as file:
                yaml.dump(output_data, file)

            # also output the structure, create it from batch to have exactly the same
            translate_symbol = {1: 'H', 8: 'O'}
            # Convert batch.nodes to numpy array first, then filter
            nodes_array = np.array(batch.nodes)
            node_mask_array = np.array(batch.node_mask)
            symbols = [translate_symbol[int(Z)] for Z in nodes_array[node_mask_array]]

            # source valid is now the same items as batch, and already ase
            atoms = self.source_valid[i]
            # atoms = Atoms(
            #     symbols=symbols,
            #     positions=np.array(batch.positions[batch.node_mask]),
            #     # cell=np.array(batch.cell),
            # )
            atoms.arrays['charges'] = actual_results.get('charges', np.zeros(len(atoms)))
            atoms.arrays['screen'] = actual_results.get('screen', np.ones(len(atoms)))
            
            # save this via extxyz writer from ase
            struct_file = Path(self.config.output_path) / f'structure_with_charges_{i}.xyz'
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
        
        timing_file = Path(self.config.output_path) / 'timing_statistics.yaml'
        with open(timing_file, 'w') as file:
            yaml.dump(timing_stats, file)
        
        print(f"\nEvaluation complete. Results saved to {self.config.output_path}")
        print(f"Total time: {timing_stats['total_time']:.3f} s")
        print(f"Mean time per batch: {timing_stats['mean_time']:.6f} s ± {timing_stats['std_time']:.6f} s")


def main():
    """Main entry point with example configuration."""
    # Example 1: Standard evaluation
    # config = EvaluationConfig(
    #     checkpoint_folder='../training_runs/lorem_ii_mixedpbc/run/checkpoints/R2_E+F+A/',
    #     model_definitions='../lorem_ii/',
    #     validation_data_path='/work/amam/cps2815/datasets_ml/water_vapor/valid',
    #     output_path='/work/amam/cps2815/outs/lorem_ii_interface/',
    #     cutoff=5.0,
    #     n_max=100,
    # )
    
    # Example 2: Calculate APT from charges using unfolded batching (periodic BC)
    # config = EvaluationConfig(
    #     checkpoint_folder='../training_runs/lorem_baseline_bulk/run/checkpoints/R2_E+F/',
    #     model_definitions='../lorem_baseline/',
    #     validation_data_path='/work/pstaerk/datasets_ml/bingqing_finite_apt/valid',
    #     output_path='./outs/pure_lorem_bulk/',
    #     cutoff=5.0,
    #     calculate_apt_from_charges=True,
    #     use_pbc=True,
    #     apt_batching_path='../lorem_ii/',  # Use lorem_ii's batching for unfolded structure
    #     n_max=10,
    # )
    
    # Example 3: Calculate APT from charges for non-periodic systems (clusters)
    # config = EvaluationConfig(
    #     checkpoint_folder='../training_runs/model_lr_clusters/run/checkpoints/R2_E+F/',
    #     model_definitions='../model_lr_only/',
    #     validation_data_path='/work/pstaerk_m3/pyiron/ml_nacl/datasets_ml/water_clusters/valid',
    #     output_path='./outs/baseline_cluster/',
    #     cutoff=5.0,
    #     calculate_apt_from_charges=True,
    #     use_pbc=False,  # Non-periodic boundary conditions
    #     num_edges=10_000,
    #     num_nodes=200,
    #     n_max=10,
    # )
    
    evaluator = ModelEvaluator(config)
    evaluator.evaluate()


if __name__ == "__main__":
    main()
