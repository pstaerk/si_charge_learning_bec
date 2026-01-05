#!/usr/bin/env python3

"""Plot MAE errors shown in Figure 4 of the manuscript."""

import matplotlib.pyplot as plt
import ase.io
import mpltex
import numpy as np
from PySnips.plotting import add_subplotlabels

clusters = ase.io.read("data/cluster_eval_valid_summarized.xyz", ":")
bulk = ase.io.read("data/bulk_eval_valid_summarized.xyz", ":")
paired = plt.get_cmap("Paired")


model_names_bulk = [
    "lorem_uncoup",
    "lorem_coup_global",
    "lorem_coup_local",
    "physical_uncoup",
    "physical_coup_global",
    "physical_coup_local",
]
model_names_clusters = [
    "lorem_uncoup",
    "lorem_coup_global",
    "lorem_coup_local",
    "physical_uncoup",
    "physical_coup_global",
    "physical_coup_local",
]


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


traj_bulk = bulk
traj_clusters = clusters


def compute_metrics_for_dataset(traj, model_names, dataset_type):
    """
    Compute MAE metrics for a given trajectory dataset.

    Parameters:
    -----------
    traj : list of Atoms
        Trajectory data containing predictions and labels
    model_names : list of str
        Names of models to evaluate
    dataset_type : str
        Type of dataset ('bulk' or 'clusters')

    Returns:
    --------
    dict : Dictionary containing metrics for each model
    """
    results = {}

    for model in model_names:
        results[model] = {}
        results[model]["valid"] = {}

        # Initialize lists to collect all predictions and labels
        all_forces_pred = []
        all_forces_label = []
        all_energy_pred = []
        all_energy_label = []
        all_apt_pred = []
        all_apt_label = []

        # Loop over all 300 validation structures
        for i in range(len(traj)):
            all_forces_label.append(traj[i].arrays["force_label"])
            all_forces_pred.append(traj[i].arrays[f"{model}_force_pred"])
            all_energy_label.append(traj[i].info["energy_label"])
            all_energy_pred.append(traj[i].info[f"{model}_energy_pred"])
            all_apt_label.append(traj[i].arrays["apt_label"])
            all_apt_pred.append(traj[i].arrays[f"{model}_apt_pred"])

        # Compute MAE over all structures
        if all_forces_pred:
            all_forces_pred = np.concatenate(all_forces_pred)
            all_forces_label = np.concatenate(all_forces_label)
            force_mae = mae(all_forces_label, all_forces_pred) * 1000
            results[model]["valid"]["forces"] = {"mae": force_mae}

        if all_energy_pred:
            all_energy_pred = np.array(all_energy_pred)
            all_energy_label = np.array(all_energy_label)
            energy_mae = mae(all_energy_label, all_energy_pred)
            results[model]["valid"]["energy"] = {"mae": energy_mae}

        if all_apt_pred:
            all_apt_pred = np.concatenate(all_apt_pred)
            all_apt_label = np.concatenate(all_apt_label)
            apt_mae = mae(all_apt_label, all_apt_pred) * 1000
            results[model]["valid"]["apt"] = {"mae": apt_mae}

    return results


# Compute metrics for both datasets
metrics_bulk = compute_metrics_for_dataset(traj_bulk, model_names_bulk, "bulk")
metrics_clusters = compute_metrics_for_dataset(
    traj_clusters, model_names_clusters, "clusters"
)

# Store in the global metrics dictionary if needed
metrics = {"bulk": metrics_bulk, "clusters": metrics_clusters}


@mpltex.acs_decorator
def tmp():
    fig = plt.figure()
    height = fig.get_figheight()
    fig.set_figheight(height * 1.8)
    fig.set_figwidth(3.5)

    # Main gridspec with 2 rows: [main plots + top broken axis, bottom broken axis]
    gs_main = fig.add_gridspec(2, 1, height_ratios=[3.5, 1], hspace=0.02)

    # Nested gridspec for energy, forces, and bottom part of APT
    gs_top = gs_main[0].subgridspec(3, 1, hspace=0.0, height_ratios=[1, 1, 0.2])

    # Create the three main axes + top part of broken axis
    ax = [fig.add_subplot(gs_top[0, 0]), fig.add_subplot(gs_top[1, 0])]

    # Top part of broken APT axis (200-250)
    ax_top = fig.add_subplot(gs_top[2, 0])

    # Bottom part of broken APT axis (0-99)
    ax_bottom = fig.add_subplot(gs_main[1, 0])

    # Define datasets and models
    datasets = ["bulk", "clusters"]
    labels = ["Bulk", "Clusters"]

    model_order = [
        "physical_coup_global",
        "physical_coup_local",
        "lorem_coup_local",
        "physical_uncoup",
        "lorem_uncoup",
    ]

    label_dict = {
        "physical_uncoup": r"Uncoupl. (Phys.)",
        "lorem_uncoup": r"Uncoupl. (LOREM)",
        "physical_coup_global": r"Coupl. $\gamma$ (Phys.)",
        "physical_coup_local": r"Coupl. $\gamma_i$ (Phys.)",
        "lorem_coup_local": r"Coupl. $\gamma_i$ (LOREM)",
    }

    color_map = {
        "physical_uncoup": paired(1),
        "lorem_uncoup": paired(0),
        "physical_coup_global": paired(5),
        "physical_coup_local": paired(3),
        "lorem_coup_local": paired(2),
    }

    bar_width = 0.15
    x = np.arange(len(datasets))

    # Row 0: Energy (U) in meV
    handles = []
    for i, model in enumerate(model_order):
        values = [
            metrics[ds][model]["valid"]["energy"]["mae"] for ds in datasets
        ]  # Convert to meV
        bars = ax[0].bar(
            x + i * bar_width,
            values,
            bar_width,
            label=label_dict[model],
            color=color_map[model],
        )
        handles.append(bars)
    ax[0].set_ylabel("$U$ [meV]")

    # Row 1: Forces (F) in meV/Å
    for i, model in enumerate(model_order):
        values = [metrics[ds][model]["valid"]["forces"]["mae"] for ds in datasets]
        ax[1].bar(x + i * bar_width, values, bar_width, color=color_map[model])
    ax[1].set_ylabel("$F$ [meV/Å]")

    # APT top part (200-250) - same data, different y-limits
    for i, model in enumerate(model_order):
        values = np.array(
            [metrics[ds][model]["valid"]["apt"]["mae"] for ds in datasets]
        )
        ax_top.bar(x + i * bar_width, values, bar_width, color=color_map[model])

    # APT bottom part (0-99) - BOTTOM axis
    for i, model in enumerate(model_order):
        values = np.array(
            [metrics[ds][model]["valid"]["apt"]["mae"] for ds in datasets]
        )
        ax_bottom.bar(x + i * bar_width, values, bar_width, color=color_map[model])

    ax_bottom.set_ylabel("$Z$ [me]")

    # Hide spines between broken parts
    ax_top.spines["bottom"].set_visible(False)
    ax_bottom.spines["top"].set_visible(False)
    ax_top.tick_params(labelbottom=False, bottom=False)

    # Add break indicators
    d = 0.008
    kwargs = dict(transform=ax_top.transAxes, color="k", clip_on=False)
    ax_top.plot((-d, +d), (-10 * d, +10 * d), **kwargs)
    ax_top.plot((1 - d, 1 + d), (-10 * d, +10 * d), **kwargs)

    kwargs = dict(transform=ax_bottom.transAxes, color="k", clip_on=False)
    ax_bottom.plot((-d, +d), (1 - 3 * d, 1 + 3 * d), **kwargs)
    ax_bottom.plot((1 - d, 1 + d), (1 - 3 * d, 1 + 3 * d), **kwargs)

    # remove the top tick-bars from ax_bottom
    ax_bottom.tick_params(top=False)

    # Set x-axis labels
    total_bars = len(model_order)
    ax_bottom.set_xticks(x + bar_width * (total_bars - 1) / 2)
    ax_bottom.set_xticklabels(labels)

    ax[0].set_xticks(x + bar_width * (total_bars - 1) / 2)
    ax[0].set_xticklabels([])
    ax[1].set_xticks(x + bar_width * (total_bars - 1) / 2)
    ax[1].set_xticklabels([])

    ax[0].set_ylim(0, 2.2)
    ax[1].set_ylim(0, 59)
    add_subplotlabels(fig, [ax[0], ax[1], ax_top], ["A", "B", "C"])
    ax_top.set_ylim(80, 450)
    ax_top.set_yticks([150, 300])
    ax_bottom.set_ylim(0, 69)

    for a in ax + [ax_top, ax_bottom]:
        a.set_xlim(-bar_width, len(datasets) - 1 + bar_width * total_bars)

    # resort manually handles for legend
    # handles = [handles[i] for i in [0, 3, 2, 1, 4]]

    ax[0].legend(handles=handles, ncol=2, handlelength=1.5, loc="upper center")
    fig.align_labels()
    fig.tight_layout()

    fig.savefig(
        "TeX/figures/errors.pdf",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )


tmp()
