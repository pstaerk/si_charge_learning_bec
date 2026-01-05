#!/usr/bin/env python3

"""Plot parity plots for APT-predicted atomic polar tensors shown in 2 and S2."""

import matplotlib.pyplot as plt
import ase.io
import mpltex
import numpy as np
from PySnips.plotting import add_subplotlabels

clusters = ase.io.read("data/cluster_eval_valid_summarized.xyz", ":")
bulk = ase.io.read("data/bulk_eval_valid_summarized.xyz", ":")
paired = plt.get_cmap("Paired")

@mpltex.acs_decorator
def plot_apt_parity(models, colors, markers=None, sizes=None, fname=None):
    fig, axes = plt.subplots(2, sharex=True, sharey=True)

    height = fig.get_figheight()
    fig.set_figheight(height * 1.2)
    fig.set_figwidth(3.5)

    # Define models and their properties
    labels = ["Uncoupl.", r"Coupl. $\gamma$", r"Coupl. $\gamma_i$"]
    if markers is None:
        markers = len(labels) * [None]

    if sizes is None:
        sizes = len(labels) * [None]

    datasets = [bulk, clusters]

    for ax, traj in zip(axes, datasets):
        # Collect all labels once (they're the same for all models)
        all_apt_label = []

        sel = list(range(len(traj)))[::100]
        for i in sel:
            all_apt_label.append(traj[i].arrays["apt_label"].reshape(-1, 3, 3))

        all_apt_label = np.concatenate(all_apt_label)

        # Plot each model
        for model, label, color, marker, size in zip(
            models, labels, colors, markers, sizes
        ):
            all_apt_pred = []
            for i in sel:
                all_apt_pred.append(
                    traj[i].arrays[f"{model}_apt_pred"].reshape(-1, 3, 3)
                )
            all_apt_pred = np.concatenate(all_apt_pred)

            ax.scatter(
                np.diagonal(all_apt_label, axis1=1, axis2=2),
                np.diagonal(all_apt_pred, axis1=1, axis2=2),
                color=color,
                s=size,
                marker=marker,
                label=label,
            )

        # Add diagonal line
        min_val = min(all_apt_label.min(), ax.get_ylim()[0])
        max_val = max(all_apt_label.max(), ax.get_ylim()[1])
        ax.plot([min_val, max_val], [min_val, max_val], ls="--", c="gray")

        ax.set_ylabel(r"$Z^\mathrm{pred}_{\alpha, \alpha}$ [e]")
        ax.set_xlim(-1.8, 1.8)
        ax.set_ylim(-1.8, 1.8)

    axes[0].legend()
    add_subplotlabels(fig, axes, labels=["A", "B"], loc="upper left")
    axes[-1].set_xlabel(r"$Z^\mathrm{DFT}_{\alpha, \alpha}$ [e]")

    fig.tight_layout(h_pad=0.0)
    fig.align_labels()
    if fname is not None:
        fig.savefig(
            fname,
            dpi=300,
            transparent=True,
            bbox_inches="tight",
        )

plot_apt_parity(
    models=["physical_uncoup", "physical_coup_global", "physical_coup_local"],
    colors=[paired(1), paired(5), paired(3)],
    markers=["o", ".", "."],
    # sizes=[5, 5, 2],
    fname="TeX/figures/parity.pdf",
)
plot_apt_parity(
    models=["lorem_uncoup", "lorem_coup_global", "lorem_coup_local"],
    colors=[paired(0), paired(4), paired(2)],
    markers=["o", ".", "."],
    fname="TeX/figures/parity_lorem.pdf",
)
