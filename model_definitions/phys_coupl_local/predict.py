import jax
import jax.numpy as jnp
import e3x

from functools import partial

import jaxpme
from jaxpme.utils import safe_norm
import vesin
from batching import Batch


def get_predict_fn(apply_fn, electrostatics="ewald",
                   lr_cutoff=None, excess_charge_neutralization=False,
                   cutoff=None, output_screening=False, electric_field=None):
    assert electrostatics in ["ewald", None], "PME is not yet supported"
    assert cutoff is not None, "cutoff must be provided"  # Force explicitly

    if electrostatics is not None:
        if electrostatics == "ewald":
            # units?
            coulomb_calc = jaxpme.Ewald(prefactor=jaxpme.prefactors.eV_A,
                                        exclusion_radius=lr_cutoff,
                                        full_neighbor_list=True)
    if electric_field is None:
        electric_field = jnp.zeros((3,))
    else:
        # check that it is a 3-vector
        assert electric_field.shape == (3,), "Electric field must be a 3-vector"

    def energy_fn(params, batch):
        # Short range energies + charges
        pbc = True if batch.full_edges is None else False
        rijs = calc_rijs(batch, pbc=pbc)
        energies, q, screen = apply_fn(
            params,
            rijs,
            batch.centers,
            batch.others,
            batch.nodes,
            batch.node_to_graph,
            batch.edge_mask,
            batch.node_mask,
        )
        energies *= batch.node_mask
        U_sr = jnp.sum(energies)

        # Calculation of Ewald
        if batch.full_edges is not None:
            U_coul = sc_coulomb(batch.node_mask, batch.positions, q,
                                batch.full_centers, batch.full_others,)
        else:
            distances = safe_norm(batch.edges, axis=-1)
            U_coul = coulomb_calc.energy(
                q,
                batch.cell,
                batch.positions,
                batch.centers,
                batch.others,
                None,
                batch.k_grid,
                batch.smearing,
                atom_mask=batch.node_mask,
                pair_mask=batch.edge_mask,
                distances=distances,
            )

        if batch.full_edges is None:
            # If we have pbc
            Z_i = sum_q_sc(batch, params)
            Z_i *= batch.unfolded_node_mask[..., None, None]  # apply mask
        else:
            # If we have non-periodic boundary conditions
            Z_i = z_i_alpha_beta_npbc(params, batch)
            Z_i *= batch.node_mask[..., None, None]

        if excess_charge_neutralization:
            excess_charge = acoustic_sum_rule(Z_i)
            # redistribute excess charge
            num_active_atoms = jnp.sum(batch.node_mask)
            charges_to_redistribute = jnp.where(
                num_active_atoms > 0,
                excess_charge / num_active_atoms,
                jnp.zeros_like(excess_charge)  # Or another appropriate default if no atoms are active
            )  # Shape: (3, 3)
            Z_i -= charges_to_redistribute
            Z_i *= batch.node_mask[..., None, None]  # apply mask

        Z_i = Z_i[:batch.node_mask.shape[0]]

        # E_padded = jnp.zeros((batch.nodes.shape[0], 3))
        # E_padded = E_padded.at[:].set(batch.electric_field[batch.node_to_graph])

        # F_ext = jnp.einsum('...ij,...j->...i', Z_i, E_padded)
        # U_ext = jnp.sum(F_ext * batch.positions)

        # total_energy = U_sr + U_coul + U_ext
        total_energy = U_sr + U_coul
        return total_energy, (energies, q, Z_i, screen)

    def predict(params, batch):
        energy_and_derivatives_fn = jax.value_and_grad(
            energy_fn, allow_int=True, has_aux=True, argnums=1
        )

        (total_energy, (energies, q_i, apt, screen)), grads = (
            energy_and_derivatives_fn(params, batch)
        )

        energy = jax.ops.segment_sum(
            energies, batch.node_to_graph, batch.graph_mask.shape[0]
        )

        forces = -grads.positions * batch.node_mask[..., None]

        E_padded = jnp.zeros((batch.nodes.shape[0], 3))
        # set all vectors to the electric field, which we assume is vector 3
        E_padded = E_padded.at[batch.node_to_graph].set(
            electric_field
        )
        F_ext = jnp.einsum('...ij,...j->...i', apt, E_padded)
        forces += F_ext
        forces *= batch.node_mask[..., None]

        results = {"energy": energy, "forces": forces}
        results["charges"] = q_i
        results["apt"] = apt
        if output_screening:
            results['screen'] = screen
        return results

    def pol_function_npbc(params, batch, alpha):
        """Calculate the polarization for non-periodic boundary conditions.
        This is a dummy function that does not use the periodicity of the system.
        """
        rijs = calc_rijs(batch, pbc=False)
        _, q, screen = apply_fn(
            params,
            rijs,
            batch.centers,
            batch.others,
            batch.nodes,
            batch.node_to_graph,
            batch.edge_mask,
            batch.node_mask,
        )

        # TODO Check this
        charges = q.flatten() * screen
        return jnp.sum(batch.positions[:, alpha] * charges * batch.node_mask)

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

    @jax.jit
    def calc_q_superbatch(batch, params, rijs):
        _, q, screen = apply_fn(
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
        q *= screen
        return q

    def sum_q_sc(batch, params):
        mask = batch.unit_cell_mask
        to_replicate = batch.to_replicate_idx
        nr_nodes = batch.unfolded_nodes.shape[0]

        def calc_q_sc_sum(batch, params, mask):
            rijs = (batch.unfolded_positions[batch.unfolded_others] -
                    batch.unfolded_positions[batch.unfolded_centers])
            qs = calc_q_superbatch(batch, params, rijs)
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

    return predict


def acoustic_sum_rule(apts):
    # calculate the acoustic sum rule for the given apts
    # this means per component sum over all particles
    # any excess charge can then be redistributed, which is the return value
    # apts: (nr_batches, N, 3, 3), want (nr_batches, 3)
    excess_charge = jnp.einsum("...nij->...ij", apts)  # sum over i
    return excess_charge


def calc_rijs(batch, pbc=True):
    rij = batch.positions[batch.others] - batch.positions[batch.centers]
    if pbc:
        rij += jnp.einsum("pA,Aa->pa", batch.cell_shifts, batch.cell)
    return rij

@jax.jit
def sc_coulomb(node_mask, positions, q_i, centers, others):
    full_edges = positions[others] - positions[centers]

    num_total_nodes = q_i.shape[0]  # Use static shape for JIT
    full_r_ij = e3x.ops.norm(full_edges, axis=-1)

    # Avoid division by zero for self-interactions
    mask = full_r_ij < 1e-6
    masked_r_ij = jnp.where(mask, 1.0, full_r_ij)  # Prevent NaNs in gradient
    one_over_r = jnp.where(mask, 0.0, 1 / masked_r_ij)

    # Calculate potentials on all nodes (active and inactive)
    potentials = jax.ops.segment_sum(
        q_i[others] * one_over_r,
        centers,
        num_segments=num_total_nodes,
        indices_are_sorted=False,
    )

    potentials *= jaxpme.prefactors.eV_A

    # Calculate energy only for active nodes and sum
    energies = q_i * potentials * node_mask
    return jnp.sum(energies)
