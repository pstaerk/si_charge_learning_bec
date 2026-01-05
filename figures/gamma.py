#!/usr/bin/env python3

"""Plot screening violin plots for gamma_i shown in Figure 3 and S3 of the manuscript."""

import numpy as np
import matplotlib.pyplot as plt
import mpltex
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from PySnips.plotting import add_subplotlabels
import ase.io


def add_colorbar_only(ax, vlim=(None, None), cmap="magma", label=None):
    ax.set_axis_off()

    norm = Normalize(vmin=vlim[0], vmax=vlim[1])
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    cbar = plt.colorbar(sm, ax=ax, fraction=0.15, pad=0.04)
    if label is not None:
        cbar.set_label(label)


@mpltex.acs_decorator
def plot_screening(data, ylabel=None, vlim=(None, None), fname=None):
    fig, ax = plt.subplots(
        2,
        sharex=True,
        sharey=True,
    )
    fig.set_figheight(1.2 * fig.get_figheight())
    fig.set_figwidth(3.5)

    add_colorbar_only(ax[0], vlim=vlim, label=ylabel)

    cluster_O_dict = data["oxygen_dict"]
    cluster_H_dict = data["hydrogen_dict"]

    if "bulk_oxygen_mean" in data.keys():
        ax[1].axhline(y=np.abs(data["bulk_oxygen_mean"]), c="C0", ls=":")
    if "bulk_hydrogen_mean" in data.keys():
        ax[1].axhline(y=np.abs(data["bulk_hydrogen_mean"]), c="C1", ls=":")
    if "bulk_screening_mean" in data.keys():
        ax[1].axhline(y=np.abs(data["bulk_screening_mean"]), c="gray", ls=":")

    O_data = [np.abs(values) for values in cluster_O_dict.values()]
    H_data = [np.abs(values) for values in cluster_H_dict.values()]

    positions = list(cluster_O_dict.keys())
    positions[-1] = 13  # Adjust last position for better spacing from

    v_O = ax[1].violinplot(O_data, positions=positions, showextrema=False)
    v_H = ax[1].violinplot(H_data, positions=positions, showextrema=False)

    ax[1].set_xlabel(r"Cluster Size ($N_{mol}$)")
    if ylabel is not None:
        ax[1].set_ylabel(ylabel)

    ax[1].legend(
        [v_O["bodies"][0], v_H["bodies"][0]],
        ["Oxygen", "Hydrogen"],
        frameon=True,
        edgecolor="None",
    )
    # ax.set_ylim(0.45, 0.95)
    xtick_labels = list(cluster_O_dict.keys())
    ax[1].set_xticks(positions, xtick_labels)

    add_subplotlabels(fig, ax, labels=["", "B"], loc="upper right")

    fig.tight_layout(h_pad=0.0)
    fig.align_labels()

    if fname is not None:
        fig.savefig(fname, dpi=300, transparent=True, bbox_inches="tight")


clusters = ase.io.read("data/cluster_eval_valid_summarized.xyz", ":")

species = np.hstack([f.get_chemical_symbols() for f in clusters])
clustersize = np.hstack([len(f) * [len(f) // 3] for f in clusters])
screen_bulk = np.hstack([abs(f.arrays["lorem_coup_local_gamma"]) for f in clusters])
screen_physical = np.hstack(
    [abs(f.arrays["physical_coup_local_gamma"]) for f in clusters]
)

data = {"oxygen_dict": {}, "hydrogen_dict": {}}
for spec, size, val in zip(species, clustersize, screen_bulk):
    if spec == "O":
        if size not in data["oxygen_dict"]:
            data["oxygen_dict"][size] = []
        data["oxygen_dict"][size].append(val)
    elif spec == "H":
        if size not in data["hydrogen_dict"]:
            data["hydrogen_dict"][size] = []
        data["hydrogen_dict"][size].append(val)

data["oxygen_dict"] = {k: np.array(v) for k, v in data["oxygen_dict"].items()}
data["hydrogen_dict"] = {k: np.array(v) for k, v in data["hydrogen_dict"].items()}
data["bulk_oxygen_mean"] = np.mean(screen_bulk[species == "O"])
data["bulk_hydrogen_mean"] = np.mean(screen_bulk[species == "H"])


data_physical = {"oxygen_dict": {}, "hydrogen_dict": {}}
for spec, size, val in zip(species, clustersize, screen_physical):
    if spec == "O":
        if size not in data_physical["oxygen_dict"]:
            data_physical["oxygen_dict"][size] = []
        data_physical["oxygen_dict"][size].append(val)
    elif spec == "H":
        if size not in data_physical["hydrogen_dict"]:
            data_physical["hydrogen_dict"][size] = []
        data_physical["hydrogen_dict"][size].append(val)

data_physical["oxygen_dict"] = {
    k: np.array(v) for k, v in data_physical["oxygen_dict"].items()
}
data_physical["hydrogen_dict"] = {
    k: np.array(v) for k, v in data_physical["hydrogen_dict"].items()
}
data_physical["bulk_screening_mean"] = screen_physical.mean()

plot_screening(
    data_physical,
    ylabel=r"$\gamma_i$",
    vlim=(0.9, 1.3),
    fname="figures/screening_violin_plot_physical.pdf",
)

plot_screening(
    data,
    ylabel=r"$\left | \gamma_i \right |$",
    vlim=(0.4, 1.0),
    fname="figures/screening_violin_plot.pdf",
)
