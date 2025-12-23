import jax
import jax.numpy as jnp

import e3x
import flax.linen as nn
from flax.core import FrozenDict
from collections.abc import Sequence

from jaxtyping import Array, Bool, Float, Int, Num, Shaped
import numpy as np



class Critias(nn.Module):
    # an unremarkable student of So3krates
    # ... field edition

    cutoff: float = 5.0
    max_degree: int = 4
    num_radial: int = 64
    num_species: int = 4
    num_spherical: int = 10
    num_scalar: int = 16
    cutoff_fn: str = "cosine_cutoff"
    radial_basis: str = "basic_bernstein"
    activation: str = "silu"
    semilocal_radial: bool = True
    semilocal_spherical: bool = False
    message_degree: int = 1

    old_efield: bool = True
    n_rad_e: int = 10
    features_efield: int = 64
    max_degree_sphere: int = 4

    @nn.compact
    def __call__(
        self,
        R_ij: Float[Array, "pairs 3"],
        i: Int[Array, " pairs"],
        j: Int[Array, " pairs"],
        Z_i: Int[Array, " nodes"],
        node_to_graph: Int[Array, " nodes"],
        pair_mask: Bool[Array, " pairs"],
        node_mask: Bool[Array, " nodes"],
    ):
        num_nodes = Z_i.shape[0]
        num_pairs = R_ij.shape[0]
        num_radial = self.num_radial
        num_spherical = self.num_spherical
        num_scalar = self.num_scalar
        max_degree = self.max_degree
        num_l = self.max_degree + 1
        num_lm = int((self.max_degree + 1) ** 2)

        cutoff_fn = getattr(e3x.nn.functions, self.cutoff_fn)

        species_embedding = ChemicalEmbedding(num_features=self.num_species)(
            Z_i
        )  # -> [nodes, num_species]
        species_embedding *= node_mask[..., None]

        r_ij = e3x.ops.norm(R_ij, axis=-1)  # -> [pairs]

        # e = e3x.ops.norm(E)

        # TODO: (Maybe) try something like this
        # the length of E:
        # radial_expansion_e = RadialEmbedding(
        #     num_radial,
        #     self.cutoff,
        #     function=self.radial_basis,
        # )(e)

        cutoffs = cutoff_fn(r_ij, cutoff=self.cutoff)  # -> [pairs]
        cutoffs *= pair_mask

        neighborhood_sizes = jax.ops.segment_sum(cutoffs, i, num_segments=num_nodes)[
            i
        ]  # -> [pairs]
        neighborhood_sizes += 1.0  # this catches the case of almost-empty neighbourhoods
        pair_scale = masked(
            lambda x: 1 / x, neighborhood_sizes[:, None], pair_mask,
            fn_value=1.,
        ).reshape(-1)

        radial_expansion = RadialEmbedding(
            num_radial,
            self.cutoff,
            function=self.radial_basis,
        )(r_ij)
        radial_expansion *= pair_mask[..., None]  # -> [pairs, num_radial]

        spherical_expansion = SolidHarmonics(self.max_degree)(R_ij)
        spherical_expansion *= pair_scale[..., None]  # -> [pairs, (max_degree+1)**2]

        # scalar features: linear combinations of radial expansion
        # Philipp: Linear learned weights given by species embedding --> weight MLP
        coefficients = RadicalCoefficients(features=num_scalar)(
            jnp.concatenate([species_embedding[i], species_embedding[j]], axis=-1),
            radial_expansion,
            cutoffs,
            pair_mask,
        )
        coefficients *= pair_scale[..., None]
        scalar_features = (
            jax.ops.segment_sum(coefficients, i, num_segments=num_nodes)
            * node_mask[..., None]
        )

        # spherical features: learned per-degree coefficients based on radial basis,
        #                     summed up over neighbourhoods
        coefficients = RadicalCoefficients(features=num_l * num_spherical)(
            jnp.concatenate([scalar_features[i], species_embedding[j]], axis=-1),
            radial_expansion,
            cutoffs,
            pair_mask,
        ).reshape(
            -1, num_l, num_spherical
        )  # -> [nodes, num_l, num_spherical]
        coefficients = degree_wise_repeat_last_axis(coefficients, max_degree)

        spherical_features = jax.ops.segment_sum(
            jnp.einsum("pla,pl->pla", coefficients, spherical_expansion),
            i,
            num_segments=num_nodes,
        )
        spherical_features *= node_mask[..., None, None]

        spherical_features = spherical_features.reshape(num_nodes, num_lm, num_spherical)

        # mix: squared trace (SOAP)
        trace = degree_wise_trace_last_axis(
            jax.lax.square(spherical_features), max_degree
        ).reshape(num_nodes, -1)

        d_scalar_features = masked(MLP(features=[num_scalar]), trace, node_mask)
        scalar_features += d_scalar_features
        scalar_features = masked(nn.LayerNorm(), scalar_features, node_mask)

        # predictor of atomic polar tensors
        # ... we now switch to e3x convention for spherical features
        spherical_features = spherical_features.reshape(num_nodes, 1, num_lm, num_spherical)

        # direct internal charge
        q = masked(
            MLP(features=[num_scalar, 1]),
            scalar_features,
            node_mask,
        )

        if self.semilocal_radial:
            # invariant MP: radial update as before, but now using neighbour
            # information
            radial = RadicalCoefficients(features=num_scalar)(
                jnp.concatenate([scalar_features[i], scalar_features[j]], axis=-1),
                radial_expansion,
                cutoffs,
                pair_mask,
            )
            radial *= pair_scale[..., None]
            d_scalar_features = (
                jax.ops.segment_sum(radial, i, num_segments=num_nodes)
                * node_mask[..., None]
            )
            new_scalar_features = scalar_features + d_scalar_features
            new_scalar_features = masked(nn.LayerNorm(), new_scalar_features,
                                         node_mask)

        if self.semilocal_spherical:
            # equivariant MP: linearly transform spherical features w/
            # coefficients then do tensor product update

            if self.semilocal_radial:
                pairs = jnp.concatenate(
                    [new_scalar_features[i], scalar_features[j]], axis=-1
                )
            else:
                pairs = jnp.concatenate([scalar_features[i],
                                         scalar_features[j]], axis=-1)

            coefficients = RadicalCoefficients(features=num_l * num_spherical * 2)(
                pairs,
                radial_expansion,
                cutoffs,
                pair_mask,
            ).reshape(-1, num_l, num_spherical * 2)
            coefficients = degree_wise_repeat_last_axis(coefficients, max_degree)
            coefficients = coefficients.reshape(-1, num_lm, num_spherical, 2)

            spherical_features = spherical_features.reshape(num_nodes, 1,
                                                            num_lm,
                                                            num_spherical)
            modified_spherical_expansion = jnp.einsum(
                "pla,pl->pla", coefficients[:, :, :, 0], spherical_expansion
            ).reshape(num_pairs, 1, num_lm, num_spherical)

            modified_spherical_features = (
                coefficients[:, :, :, 1] * spherical_features[j][:, 0, :, :]
            ).reshape(num_pairs, 1, num_lm, num_spherical)
            central_features = (spherical_features[i][:,0,:,:] * pair_mask[..., None, None]).reshape(
                num_pairs, 1, num_lm, num_spherical
            )

            messages = e3x.nn.Tensor(
                max_degree=self.message_degree, include_pseudotensors=True
            )(modified_spherical_features, modified_spherical_expansion)
            messages = e3x.nn.Tensor(
                max_degree=self.message_degree, include_pseudotensors=True
            )(central_features, messages)

            messages = messages.reshape(num_pairs, -1, num_spherical)

            m = jax.ops.segment_sum(
                messages,
                i,
                num_segments=num_nodes,
            )
            m *= node_mask[..., None, None]

            m = m.reshape(num_nodes, 2, -1, num_spherical)

            square = jax.lax.square(m)
            trace1 = degree_wise_trace_last_axis(
                square[:, 0, :, :], self.message_degree
            ).reshape(num_nodes, -1)
            trace2 = degree_wise_trace_last_axis(
                square[:, 1, :, :], self.message_degree
            ).reshape(num_nodes, -1)
            # todo: deeper?
            d_scalar_features = masked(MLP(features=[num_scalar]), trace, node_mask)
            d_scalar_features += masked(MLP(features=[num_scalar]), trace2, node_mask)

        if self.semilocal_radial:
            scalar_features = new_scalar_features

        if self.semilocal_spherical:
            scalar_features += d_scalar_features
            scalar_features = masked(nn.LayerNorm(), scalar_features, node_mask)

        energies = masked(
            MLP(features=[num_scalar, 1]),
            scalar_features,
            node_mask,
        )

        return energies[..., 0], q[...,0]

    def dummy_inputs(self, dtype=jnp.float32):
        return (
            jnp.array([[1, 0, 0], [1, 1, 1], [0.5, 1, 1], [0, 1, 0]], dtype=dtype),
            jnp.array([0, 1, 2, 2]),
            jnp.array([1, 0, 2, 2]),
            jnp.array([0, 0, 0]),
            jnp.array([0, 0, 0]),
            jnp.array([True, True, False, False]),
            jnp.array([True, True, False]),
        )


class RadicalCoefficients(nn.Module):
    features: int = 64

    @nn.compact
    def __call__(self, pair_features, radial_expansion, cutoffs, pair_mask):
        num_radial = radial_expansion.shape[-1]

        coefficients = masked(
            MLP(
                features=[
                    num_radial * self.features,
                ]
            ),
            pair_features,
            pair_mask,
        )
        coefficients = coefficients.reshape(-1, num_radial, self.features)
        coefficients = jnp.einsum("prf,pr->pf", coefficients, radial_expansion)
        coefficients *= cutoffs[..., None]

        return coefficients


def degree_wise_repeat_last_axis(
    x: Num[Array, " ... ( max_degree+1) dim"], max_degree: int
) -> Num[Array, " ... ( max_degree+1)**2 dim"]:
    return jax.vmap(
        lambda y: degree_wise_repeat(y, max_degree, -1), in_axes=-1, out_axes=-1
    )(x)


def degree_wise_trace_last_axis(
    x: Num[Array, " dim ( max_degree+1)**2 dim"],
    max_degree: int,
) -> Num[Array, "dim max_degree+1 dim"]:
    return jax.vmap(lambda y: degree_wise_trace(y, max_degree), in_axes=-1, out_axes=-1)(x)


class SolidHarmonics(nn.Module):
    max_degree: int

    def __call__(
        self, R: Float[Array, "pairs 3"]
    ) -> Float[Array, " pairs ( max_degree+1)**2"]:
        return e3x.so3.solid_harmonics(R, self.max_degree)


class PerParticleTensorPredictor(nn.Module):
    features = 20

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
        # x = x[..., :, :, 0]  # -> [...,N,9]
        x = x[..., 0, :, 0]  # -> [...,N,9]

        # Taken from e3x tutorial, combination of l <= 2 spherical harmonics
        # to 3x3 matrix, done per particle
        cg = e3x.so3.clebsch_gordan(
            max_degree1=1, max_degree2=1, max_degree3=2
        )  # Shape (4, 4, 9).
        y = jnp.einsum("...l,nml->...nm", x, cg[1:, 1:, :])  # Shape (..., 3, 3).
        # y = jnp.sum(y, axis=-3)
        return y

class ChemicalEmbedding(nn.Module):
    num_features: int
    total_species: int = 100

    @nn.compact
    def __call__(self, species: Int[Array, " nodes"]) -> Float[Array, "nodes num_features"]:
        return nn.Embed(num_embeddings=self.total_species, features=self.num_features)(
            species
        )


class SphericalHarmonics(nn.Module):
    max_degree: int

    def __call__(
        self, R: Float[Array, "pairs 3"]
    ) -> Float[Array, " pairs ( max_degree+1)**2"]:
        return spherical_harmonics(R, self.max_degree)


class RadialEmbedding(nn.Module):
    num_features: int
    cutoff: int
    function: str = "basic_gaussian"
    args: FrozenDict = FrozenDict({})
    learned_transform: bool = False

    @nn.compact
    def __call__(self, r: Float[Array, " pairs"]) -> Float[Array, "pairs num_features"]:
        function = getattr(e3x.nn.functions, self.function)

        expansion = function(
            r, **{"limit": self.cutoff, "num": self.num_features, **self.args}
        )

        if self.learned_transform:
            expansion = nn.Dense(features=self.num_features, use_bias=False)(expansion)

        return expansion

def infer_max_degree(x: Num[Array, " ( max_degree+1)**2"], axis: int = 0) -> int:
    return int(np.sqrt(x.shape[axis]) - 1)


def degree_wise_trace(
    x: Num[Array, " dim ( max_degree+1)**2"],
    max_degree: int,
) -> Num[Array, "dim max_degree+1"]:
    segments = np.concatenate(
        [np.array([l] * (2 * l + 1)) for l in range(max_degree + 1)]
    ).reshape(-1)

    return jax.vmap(
        lambda _x: jax.ops.segment_sum(_x, segments, num_segments=(max_degree + 1)),
    )(x)


def degree_wise_repeat(
    x: Num[Array, " ... ( max_degree+1)"], max_degree: int, axis: int = 0
) -> Num[Array, " ... ( max_degree+1)**2"]:
    repeats = np.array([2 * l + 1 for l in range(max_degree + 1)])

    return jnp.repeat(x, repeats, total_repeat_length=repeats.sum(), axis=axis)


def split_into_heads(
    x: Num[Array, "... dim*num_heads"], num_heads: int
) -> Num[Array, "... dim num_heads"]:
    return x.reshape(*x.shape[:-1], num_heads, -1)


def concatenate_heads(
    inputs: Num[Array, "... dim*num_heads"],
) -> Num[Array, "... dim num_heads"]:
    return inputs.reshape(*inputs.shape[:-2], -1)


def constant(value):
    def _constant(*no, **thanks):
        return jnp.array(value, dtype=jnp.float32)

    return _constant

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
