import jax
import jax.numpy as jnp


def get_loss_fn(predict_fn, weights={"energy": 1.0, "forces": 1.0},
                neutrality_weight=1.0, abs_charge_weight=0.0):
    """Get a loss function.

    A loss function is something that ingests a Batch and returns
    a MSE loss (for optimisation) and summed residuals (for other metrics).

    The assumption is that this can be vmapped or scanned across a whole
    bunch of batches at once.

    Hardcoded decisions for now:
        - Energy loss is always scaled by number of atoms.
        - We take all the means over flattened data, i.e. each
            atom contributes with the same weight, as opposed to
            averaging over structures (graphs) first, which would
            weigh smaller structures higher.
        - We expect the loss weights to take care of variance scaling.

    """

    def loss_fn(params, batch):
        _, num_nodes_by_graph = jnp.unique(
            batch.node_to_graph, size=batch.graph_mask.shape[0],
            return_counts=True,
        )

        predictions = predict_fn(params, batch)

        residuals = {
            key: predictions[key] - batch.labels[key]
            for key in predictions.keys()
            if not (key == "charges"
                    or key == "total_charge"
                    or key == "apt_charge")  # we have no labels for charge
        }

        residuals["energy"] = residuals["energy"] / num_nodes_by_graph

        loss = jnp.array(0.0)
        for key, weight in weights.items():
            se = jax.lax.square(residuals[key]) * batch.labels[key + "_mask"]
            loss += weight * jnp.mean(se)

        if neutrality_weight and "charges" in predictions:
            # we want to penalize any net charge so we can sum over all
            abs_charge = jax.lax.square(
                jnp.sum(jnp.abs(predictions['charges'])))

            loss += neutrality_weight * jnp.mean(
                jax.lax.square(jnp.sum(predictions["charges"]))
            )/(abs_charge + 1e-12)

        if abs_charge_weight and "charges" in predictions:
            abs_charge = jnp.sum(jnp.abs(predictions['charges']))
            loss += (abs_charge_weight *
                     (abs_charge - batch.labels["abs_charge"])**2)

        # if neutrality_weight and "total_charge" in predictions:
        #     loss += jnp.sum(neutrality_weight *
        #                     predictions["total_charge"] ** 2)

        aux = {}
        for key, residual in residuals.items():
            mask = batch.labels[key + "_mask"]

            aux[f"{key}_abs"] = jnp.abs(residual * mask).sum(axis=0)
            aux[f"{key}_sq"] = jax.lax.square(residual * mask).sum(axis=0)

            # we need to count samples. so we reshape the mask to
            # [samples, flattened components] to give us the "real" samples
            mask = batch.labels[key + "_mask"]
            mask = mask.reshape(mask.shape[0], -1)
            aux[f"{key}_n"] = jnp.sum(mask.all(axis=1))

        return loss, aux

    return loss_fn
