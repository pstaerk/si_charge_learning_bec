import jax
import jax.numpy as jnp
import numpy as np

from flax.core import FrozenDict
import flax.linen as nn
import e3x

from jaxtyping import Array, Bool, Float, Int, Num, Shaped
import functools
from collections.abc import Sequence

from jaxpme import Ewald
import jaxpme


class Lorem(nn.Module):
    cutoff: float = 5.0
    max_degree: int = 6
    max_degree_lr: int = 2
    num_features: int = 128
    num_radial: int = 32
    num_species: int = 8
    num_spherical_features: int = 8
    cutoff_fn: str = "cosine_cutoff"
    radial_basis: str = "basic_bernstein"
    lr: bool = True
    num_message_passing: int = 0
    equivariant_message_passing: bool = False
    initialize_node_features: bool = False

    @nn.compact
    def __call__(
        self,
        R_ij,
        i,
        j,
        Z_i,
        pair_mask,
        node_mask,
        # inputs for Ewald (periodic) or None (non-periodic)
        # if Ewald is used, the batch may only contain one real sample
        positions,
        cell,
        k_grid,  # only .shape matters (see jax-pme)
        smearing,
        # all-to-all edges (non-periodic) or None (periodic)
        full_R_ij,
        full_i,
        full_j,
        full_edge_mask,
    ):
        num_nodes = Z_i.shape[0]
        num_pairs = R_ij.shape[0]

        max_degree = self.max_degree
        max_degree_lr = self.max_degree_lr
        num_l = self.max_degree + 1
        num_lm = int((self.max_degree + 1) ** 2)

        d = self.num_features
        s = self.num_spherical_features

        # empirical factors to make var of equivariant norm more uniform across l
        l_factors = (
            jnp.array([(2 * l + 1) for l in range(max_degree + 1)], dtype=float) ** 0.25
        )

        # -- initial embeddings --
        radial, spherical, species, cutoffs, r_ij = Initial(
            cutoff=self.cutoff,
            max_degree=self.max_degree,
            num_features=self.num_features,
            num_radial=self.num_radial,
            num_species=self.num_species,
            num_spherical_features=self.num_spherical_features,
            cutoff_fn=self.cutoff_fn,
            radial_basis=self.radial_basis,
        )(
            R_ij,
            Z_i,
            pair_mask,
            node_mask,
        )

        # -- learned linear transformation of radial expansion --
        edges_scalar = RadialCoefficients(d)(
            jnp.concatenate([species[i], species[j]], axis=-1),
            radial,
            cutoffs,
            pair_mask,
        )

        # -- initial scalar and equivariant (spherical) node features
        if self.initialize_node_features:
            nodes_scalar = masked(nn.Dense(d, use_bias=True), species, node_mask)
        else:
            nodes_scalar = jnp.zeros((num_nodes, d), dtype=species.dtype)

        updates = (
            jax.ops.segment_sum(
                masked(nn.Dense(d, use_bias=False), edges_scalar, pair_mask),
                i,
                num_segments=num_nodes,
            )
            * node_mask[..., None]
        )
        nodes_scalar = Update(d)(nodes_scalar, updates, node_mask)

        coefficients = masked(
            nn.Dense(num_l * s, use_bias=False), edges_scalar, pair_mask
        ).reshape(num_pairs, num_l, s)
        coefficients = degree_wise_repeat_last_axis(coefficients, max_degree)
        edges_spherical = jnp.einsum("plf,pl->plf", coefficients, spherical)

        nodes_spherical = (
            jax.ops.segment_sum(
                edges_spherical.reshape(num_pairs, 1, num_lm, s),
                i,
                num_segments=num_nodes,
            )
            * node_mask[..., None, None, None]
        )
        nodes_spherical = e3x.nn.TensorDense(use_bias=False, include_pseudotensors=False)(
            nodes_spherical
        )

        # -- mix equivariant information into scalar node features --
        norms = spherical_norm_last_axis(nodes_spherical, max_degree)
        updates = (norms * l_factors[None, None, :, None]).reshape(num_nodes, -1)

        nodes_scalar = Update(d)(nodes_scalar, updates, node_mask)

        # -- initial prediction --
        energy = masked(MLP(features=[d, d, 1]), nodes_scalar, node_mask)[..., 0]

        # -- message passing (if turned on) --
        for _ in range(self.num_message_passing):
            edges_scalar = RadialCoefficients(d)(
                jnp.concatenate([nodes_scalar[i], nodes_scalar[j]], axis=-1),
                radial,
                cutoffs,
                pair_mask,
            )
            updates = (
                jax.ops.segment_sum(
                    masked(nn.Dense(d, use_bias=False), edges_scalar, pair_mask),
                    i,
                    num_segments=num_nodes,
                )
                * node_mask[..., None]
            )
            nodes_scalar = Update(d)(nodes_scalar, updates, node_mask)

            if self.equivariant_message_passing:
                coefficients = masked(
                    nn.Dense(num_l * s, use_bias=False), edges_scalar, pair_mask
                ).reshape(num_pairs, num_l, s)
                coefficients = degree_wise_repeat_last_axis(coefficients, max_degree)
                edges_spherical = jnp.einsum(
                    "plf,pl->plf", coefficients, spherical
                ).reshape(num_pairs, 1, num_lm, s)

                messages = (
                    e3x.nn.MessagePass(include_pseudotensors=False)(
                        nodes_spherical,
                        edges_spherical,
                        dst_idx=i,
                        src_idx=j,
                    )
                    * node_mask[..., None, None, None]
                )
                nodes_spherical = e3x.nn.Tensor(include_pseudotensors=False)(
                    e3x.nn.Dense(use_bias=False, features=s)(nodes_spherical),
                    e3x.nn.Dense(use_bias=False, features=s)(messages),
                )

                norms = spherical_norm_last_axis(nodes_spherical, max_degree)
                updates = (norms * l_factors[None, None, :, None]).reshape(num_nodes, -1)
                nodes_scalar = Update(d)(nodes_scalar, updates, node_mask)

            # -- residual prediction --
            energy += masked(MLP(features=[d, d, 1]), nodes_scalar, node_mask)[..., 0]

        scalar_charges = masked(MLP(features=[2 * d, 1]), nodes_scalar,
                                node_mask)
        scalar_screen = masked(MLP(features=[2 * d, 1]), nodes_scalar,
                               node_mask)

        if self.lr:
            # -- compute LR potentials --
            spherical_charges = e3x.nn.TensorDense(
                features=1,
                use_bias=False,
                max_degree=max_degree_lr,
                include_pseudotensors=False,
            )(nodes_spherical).reshape(num_nodes, -1)
            charges = jnp.concatenate([scalar_charges, spherical_charges], axis=-1)

            if k_grid is not None:  # if periodic
                calculator = Ewald(full_neighbor_list=True,
                                   prefactor=jaxpme.prefactors.eV_A)
                potentials = jax.vmap(
                    lambda q: calculator.potentials(
                        q,
                        cell,
                        positions,
                        i,
                        j,
                        None,
                        k_grid,
                        smearing,
                        atom_mask=node_mask,
                        pair_mask=pair_mask,
                        distances=r_ij,
                    ),
                    in_axes=-1,
                    out_axes=-1,
                )(charges)
            elif full_R_ij is not None:  # if non-periodic
                full_r_ij = e3x.ops.norm(full_R_ij, axis=-1)
                mask = full_r_ij == 0
                masked_r_ij = jnp.where(mask, 1e-6, full_r_ij)
                one_over_r = jnp.where(mask, 0.0, 1 / masked_r_ij)
                potentials = jax.ops.segment_sum(
                    charges[full_j] * one_over_r[..., None], full_i,
                    num_segments=num_nodes
                )
                potentials *= jaxpme.prefactors.eV_A / 2
            else:
                print(f'{full_R_ij, k_grid=}')

            scalar_potential = potentials[..., 0][..., None]
            spherical_potential = potentials[..., 1:].reshape(num_nodes, 1, -1, 1)

            # -- combine LR potentials back into local features --
            spherical_potential = e3x.nn.Dense(s, use_bias=False)(spherical_potential)
            spherical_updates = e3x.nn.Tensor(include_pseudotensors=False)(
                spherical_potential, nodes_spherical
            )

            norms = spherical_norm_last_axis(spherical_updates, max_degree)
            norms = (norms * l_factors[None, None, :, None]).reshape(num_nodes, -1)
            updates = jnp.concatenate([scalar_potential, norms], axis=-1)
            nodes_scalar = Update(d)(nodes_scalar, updates, node_mask)

            # -- residual prediction --
            energy += masked(MLP(features=[d, d, 1]), nodes_scalar, node_mask)[..., 0]

        return energy, scalar_charges, scalar_screen

    def dummy_inputs(self, dtype=jnp.float32):
        return (
            jnp.array([[0, 0, 0], [1, 1, 1], [0.5, 1, 1], [0, 1, 0]], dtype=dtype),
            jnp.array([0, 1, 2, 2]),
            jnp.array([1, 0, 2, 2]),
            jnp.array([0, 0, 0]),
            jnp.array([True, True, False, False]),
            jnp.array([True, True, False]),
            jnp.array([[0, 0, 0], [1, 1, 1], [0.5, 1, 1]], dtype=dtype),
            jnp.eye(3),
            jnp.ones((4, 4, 4)),
            jnp.array(1.0),
            jnp.array([[0, 0, 0], [1, 1, 1], [0.5, 1, 1]], dtype=dtype),
            jnp.array([0, 1, 2]),
            jnp.array([1, 0, 2]),
            jnp.array([True, True, False]),
        )

    def energy(self, params, batch):
        energies, scalar_charges, scalar_screen = self.apply(
            params,
            batch.edges,
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
        energies *= batch.node_mask
        scalar_charges *= batch.node_mask[..., None]
        scalar_screen *= batch.node_mask[..., None]

        return jnp.sum(energies), (energies, scalar_charges, scalar_screen)

    def z_i_pbc(self, batch, params):
        mask = batch.unit_cell_mask
        to_replicate = batch.to_replicate_idx
        nr_nodes = batch.unfolded_nodes.shape[0]

        @jax.jit
        def calc_q_superbatch(batch, params, rijs):
            self.lr = False
            _, q, screen = self.apply(
                params,
                rijs,
                batch.unfolded_centers,
                batch.unfolded_others,
                batch.unfolded_nodes,
                batch.unfolded_edge_mask,
                batch.unfolded_node_mask,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )
            self.lr = True
            q = q[...,0] * screen[...,0]
            return q

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

                return jnp.sum(qs[..., None] * stopgrad_positions, axis=0)[alpha]

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

    def pol_function_npbc(self, params, batch, alpha):
        """Calculate the polarization for non-periodic boundary conditions.
        This is a dummy function that does not use the periodicity of the system.
        """
        rijs = calc_rijs(batch, pbc=False)
        self.lr = False
        _, q, screen = self.apply(
            params,
            rijs,
            batch.centers,
            batch.others,
            batch.nodes,
            batch.edge_mask,
            batch.node_mask,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
        self.lr = True

        # TODO Check this
        charges = q[...,0] * screen[...,0]
        return jnp.sum(batch.positions[:, alpha] * charges * batch.node_mask)

    def z_i_alpha_beta_npbc(self, params, batch):
        """Calculate the z_i_alpha_beta for non-periodic boundary conditions.
        This is a dummy function that does not use the periodicity of the system.
        """
        deriv_p_npbc = jax.value_and_grad(self.pol_function_npbc, has_aux=False,
                                          argnums=1, allow_int=True)
        z_i_ab = jnp.zeros((batch.positions.shape[0], 3, 3), dtype=jnp.float32)

        for alpha in range(3):
            # we want the derivative of P_alpha with respect to r_beta
            _, deriv_alpha = deriv_p_npbc(params, batch, alpha)
            for beta in range(3):
                component = deriv_alpha.positions[:, beta]
                z_i_ab = z_i_ab.at[:, alpha, beta].set(component)
        return z_i_ab

    def predict(self, params, batch, stress=False,
                excess_charge_neutralization=True,
                pme_neutralization=True,
                electric_field=None,
                output_screen=False,):
        # Setup the field
        if electric_field is None:
            electric_field = jnp.zeros((3,))
        else:
            # check that it is a 3-vector
            assert electric_field.shape == (3,), "Electric field must be a 3-vector"

        energy_and_derivatives_fn = jax.value_and_grad(
            self.energy, allow_int=True, has_aux=True, argnums=1
        )
        batch_energy_and_atom_energies, grads = energy_and_derivatives_fn(params, batch)
        _, (energies, q, screen) = batch_energy_and_atom_energies

        if batch.full_edges is None:
            # If we have pbc
            apt = self.z_i_pbc(batch, params)
            apt *= batch.unfolded_node_mask[..., None, None]  # apply mask
        else:
            # If we have non-periodic boundary conditions
            apt = self.z_i_alpha_beta_npbc(params, batch)
            apt *= batch.node_mask[..., None, None]

        # redistribute excess charge per component in apt
        if excess_charge_neutralization:
            excess_charge = acoustic_sum_rule(apt)
            num_active_atoms = jnp.sum(batch.node_mask)
            charges_to_redistribute = jnp.where(
                num_active_atoms > 0,
                excess_charge / num_active_atoms,
                jnp.zeros_like(excess_charge)  # Or another appropriate default if no atoms are active
            )  # Shape: (3, 3)
            apt -= charges_to_redistribute
            apt *= batch.node_mask[..., None, None]  # apply mask

        # redistribute excess charge in q
        if pme_neutralization:
            q *= batch.node_mask
            excess_pme = q.sum()
            num_active_atoms = jnp.sum(batch.node_mask)
            q -= excess_pme / num_active_atoms

        energy = jax.ops.segment_sum(
            energies, batch.node_to_graph, batch.graph_mask.shape[0]
        )

        dR_ij = grads.edges * batch.edge_mask[..., None]
        forces_1 = jax.ops.segment_sum(
            dR_ij, batch.centers, batch.nodes.shape[0], indices_are_sorted=False
        )
        forces_2 = jax.ops.segment_sum(
            dR_ij, batch.others, batch.nodes.shape[0], indices_are_sorted=False
        )

        forces = (forces_1 - forces_2) * batch.node_mask[..., None]

        if batch.positions is not None:
            forces_3 = -grads.positions * batch.node_mask[..., None]

            forces += forces_3 * batch.node_mask[..., None]

        elif batch.full_edges is not None:
            full_dR_ij = grads.full_edges * batch.full_edge_mask[..., None]
            forces_3 = jax.ops.segment_sum(
                full_dR_ij,
                batch.full_centers,
                batch.nodes.shape[0],
                indices_are_sorted=False,
            )
            forces_4 = jax.ops.segment_sum(
                full_dR_ij,
                batch.full_others,
                batch.nodes.shape[0],
                indices_are_sorted=False,
            )

            forces += (forces_3 - forces_4) * batch.node_mask[..., None]

        E_padded = jnp.zeros((batch.nodes.shape[0], 3))
        # set all vectors to the electric field, which we assume is vector 3
        E_padded = E_padded.at[:].set(
            electric_field
        )
        # E_padded = E_padded.at[:].set(batch.electric_field[batch.node_to_graph])

        # TODO: This needs to change
        F_ext = jnp.einsum('...ij,...j->...i', apt, E_padded)
        forces += F_ext
        forces *= batch.node_mask[..., None]

        # if batch.positions is not None and batch.full_edges is not None:
        #     raise ValueError

        results = {"energy": energy, "forces": forces, "apt": apt,
                   "charges": q}
        if output_screen:
            results['screen'] = screen

        return results


# -- initial embeddings --


class Initial(nn.Module):
    cutoff: float = 5.0
    max_degree: int = 4
    num_features: int = 128
    num_radial: int = 32
    num_species: int = 8
    num_spherical_features: int = 4
    cutoff_fn: str = "cosine_cutoff"
    radial_basis: str = "basic_bernstein"

    @nn.compact
    def __call__(
        self,
        R_ij,
        Z_i,
        pair_mask,
        node_mask,
    ):
        cutoff_fn = getattr(e3x.nn.functions, self.cutoff_fn)

        R_ij, r_ij = e3x.ops.normalize_and_return_norm(R_ij, axis=-1)
        R_ij *= pair_mask[..., None]

        cutoffs = cutoff_fn(r_ij, cutoff=self.cutoff) * pair_mask  # -> [pairs]

        radial_expansion = (
            RadialEmbedding(
                self.num_radial,
                self.cutoff,
                function=self.radial_basis,
            )(r_ij)
            * cutoffs[..., None]
        )

        spherical_expansion = e3x.so3.spherical_harmonics(
            R_ij, self.max_degree, r_is_normalized=True
        )
        spherical_expansion *= pair_mask[..., None]

        species_expansion = (
            ChemicalEmbedding(num_features=self.num_species)(Z_i) * node_mask[..., None]
        )

        return radial_expansion, spherical_expansion, species_expansion, cutoffs, r_ij


class ChemicalEmbedding(nn.Module):
    num_features: int
    total_species: int = 100

    @nn.compact
    def __call__(self, species):
        return nn.Embed(num_embeddings=self.total_species, features=self.num_features)(
            species
        )


class RadialEmbedding(nn.Module):
    num_features: int
    cutoff: int
    function: str = "basic_gaussian"
    args: FrozenDict = FrozenDict({})
    learned_transform: bool = False

    @nn.compact
    def __call__(self, r):
        function = getattr(e3x.nn.functions, self.function)

        expansion = function(
            r, **{"limit": self.cutoff, "num": self.num_features, **self.args}
        )

        if self.learned_transform:
            expansion = nn.Dense(features=self.num_features, use_bias=False)(expansion)

        return expansion


# -- basic modules --


class MLP(nn.Module):
    features: Sequence[int]
    activation: str = "silu"
    use_bias: bool = True

    @nn.compact
    def __call__(self, x):
        activation = getattr(jax.nn, self.activation)
        num_layers = len(self.features)

        for i, f in enumerate(self.features):
            x = nn.Dense(features=f, use_bias=self.use_bias)(x)
            if i != num_layers - 1:
                x = activation(x)

        return x


class Update(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x, y, node_mask):
        x += masked(
            MLP(features=[2 * self.features, self.features]),
            y,
            node_mask,
        )
        x = masked(nn.LayerNorm(), x, node_mask)
        x += masked(MLP(features=[2 * self.features, self.features]), x, node_mask)
        x = masked(nn.LayerNorm(), x, node_mask)

        return x


# -- other modules --


class RadialCoefficients(nn.Module):
    features: int

    @nn.compact
    def __call__(self, pair_features, radial_expansion, cutoffs, pair_mask):
        num_radial = radial_expansion.shape[-1]

        coefficients = masked(
            MLP(
                features=[
                    self.features,
                    num_radial * self.features,
                ]
            ),
            pair_features,
            pair_mask,
        )
        coefficients = coefficients.reshape(-1, num_radial, self.features)
        coefficients = jnp.einsum("prf,pr->pf", coefficients, radial_expansion)

        return coefficients


class PerParticleTensorPredictor(nn.Module):
    features: int = 128

    @nn.compact
    def __call__(self, spherical_features):
        # Weighting, linear combinations of spherical features.
        x = e3x.nn.Dense(features=self.features)(
            spherical_features
        )  # -> [...,1 or 2,(l+1)**2,sp_features]

        x = e3x.nn.activations.silu(x)

        x = e3x.nn.Dense(features=self.features)(
            spherical_features
        )  # -> [...,1 or 2,(l+1)**2,sp_features]

        x = e3x.nn.activations.silu(x)

        x = e3x.nn.Dense(features=self.features)(
            spherical_features
        )  # -> [...,1 or 2,(l+1)**2,sp_features]

        # coupling and weighting
        x = e3x.nn.TensorDense(
            features=1,
            max_degree=2,
        )(
            x
        )  # -> [...,N,2,9,1]

        # Take only the non-pseudovector channels, feature away
        x = x[..., 0, :, 0]  # -> [...,N,9]

        # Taken from e3x tutorial, combination of l <= 2 spherical harmonics
        # to 3x3 matrix, done per particle
        cg = e3x.so3.clebsch_gordan(
            max_degree1=1, max_degree2=1, max_degree3=2
        )  # Shape (4, 4, 9).
        y = jnp.einsum("...l,nml->...nm", x, cg[1:, 1:, :])  # Shape (..., 3, 3).
        return y

# -- helpers to deal with spherical features --


def degree_wise_trace(
    x,
    max_degree,
):
    segments = np.concatenate(
        [np.array([l] * (2 * l + 1)) for l in range(max_degree + 1)]
    ).reshape(-1)

    return jax.vmap(
        lambda _x: jax.ops.segment_sum(_x, segments, num_segments=(max_degree + 1)),
    )(x)


def degree_wise_repeat(x, max_degree, axis):
    repeats = np.array([2 * l + 1 for l in range(max_degree + 1)])

    return jnp.repeat(x, repeats, total_repeat_length=repeats.sum(), axis=axis)


def degree_wise_repeat_last_axis(x, max_degree: int):
    return jax.vmap(
        lambda y: degree_wise_repeat(y, max_degree, -1), in_axes=-1, out_axes=-1
    )(x)


@functools.partial(jax.custom_jvp, nondiff_argnums=(1,))
def spherical_norm(X, max_degree):
    squared = jax.lax.square(X)
    trace = degree_wise_trace(squared, max_degree)
    norm = jnp.sqrt(trace)
    return norm


@spherical_norm.defjvp
def spherical_norm_jvp(max_degree, primals, tangents):
    (x,) = primals
    (x_dot,) = tangents
    primal_out = spherical_norm(x, max_degree)

    x_hat = x / degree_wise_repeat(jnp.where(primal_out > 0, primal_out, 1), max_degree, -1)

    tangent_out = degree_wise_trace(x_dot * x_hat, max_degree)
    return primal_out, tangent_out


def spherical_norm_last_axis(X, max_degree):
    # X is a e3x-style array, i.e. [batch, 1|2, lm, features]:
    # we vmap over parity and feature dimensions
    return jax.vmap(
        lambda z: jax.vmap(
            lambda x: spherical_norm(x, max_degree), in_axes=-1, out_axes=-1
        )(z),
        in_axes=1,
        out_axes=1,
    )(X)


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


def masked(
    fn: callable,
    x: Shaped[Array, "*shared features"],
    mask: Bool[Array, " *shared"],
    fn_value: float = 0.0,  # value to be passed into fn
    return_value: float = 0.0,  # value to be returned
):
    fn_value = jnp.array(fn_value, dtype=x.dtype)
    return_value = jnp.array(return_value, dtype=x.dtype)

    return jnp.where(
        mask[..., None], fn(jnp.where(mask[..., None], x, fn_value)), return_value
    )
