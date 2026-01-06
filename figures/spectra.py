#!/usr/bin/env python3

"""Plot dielectric spectra shown in Figure 5 of the manuscript."""

import numpy as np
import matplotlib.pyplot as plt
import mpltex
from PySnips.plotting import add_subplotlabels


# add second axis with wavenumbers in cm^-1
def thz_to_wavenumber(x):
    return x * 33.35641


def wavenumber_to_thz(x):
    return x / 33.35641


def process_spectrum(spectra_dict, fmin=1e-3, fmax=2e2, num=300):
    spectra_dict = dict(spectra_dict)

    nu_raw = spectra_dict["nu_raw"]
    susc_imag_raw = spectra_dict["susc_imag_raw"]
    dsusc_imag_raw = spectra_dict["dsusc_imag_raw"]

    # bin data on geometric grid
    bins = np.geomspace(fmin, fmax, num=num)
    nu_binned = 0.5 * (bins[1:] + bins[:-1])

    susc_binned, _ = np.histogram(nu_raw, bins=bins, weights=susc_imag_raw)
    dsusc_binned, _ = np.histogram(nu_raw, bins=bins, weights=dsusc_imag_raw)

    norm, _ = np.histogram(nu_raw, bins=bins)
    with np.errstate(divide="ignore", invalid="ignore"):
        susc_binned /= norm
        dsusc_binned /= norm

    # remove nan values from empty bins
    valid = ~np.isnan(susc_binned)
    nu_binned = nu_binned[valid]
    susc_binned = susc_binned[valid]
    dsusc_binned = dsusc_binned[valid]

    spectra_dict["nu"] = nu_binned
    spectra_dict["susc_imag"] = susc_binned
    spectra_dict["dsusc_imag"] = dsusc_binned

    return spectra_dict


bulk_exp = np.loadtxt("data/combined_exp_susc_300K.txt")

physical_model = False

if physical_model:
    path = "data/spectra_physical/spectrum_physical_"
    fname_suffix = "_physical"
else:
    path = "data/spectra_lorem/spectrum_lorem_"
    fname_suffix = "_lorem"

bulk_uncoupled = dict(np.load(f"{path}uncoupled_bulk.npz", allow_pickle=True))
book_uncoupled = dict(np.load(f"{path}uncoupled_book.npz", allow_pickle=True))
cage_uncoupled = dict(np.load(f"{path}uncoupled_cage.npz", allow_pickle=True))

bulk_global = dict(np.load(f"{path}global_bulk.npz", allow_pickle=True))
book_global = dict(np.load(f"{path}global_book.npz", allow_pickle=True))
cage_global = dict(np.load(f"{path}global_cage.npz", allow_pickle=True))

bulk_local = dict(np.load(f"{path}local_bulk.npz", allow_pickle=True))
book_local = dict(np.load(f"{path}local_book.npz", allow_pickle=True))
cage_local = dict(np.load(f"{path}local_cage.npz", allow_pickle=True))


@mpltex.acs_decorator
def plot_scrum():
    fig, ax = plt.subplots(
        3,
        sharex=True,
        sharey=True,
    )

    fig.set_figheight(1.8 * fig.get_figheight())
    fig.set_figwidth(3.5)

    labels = [r"Uncoupl.", r"Coupl. $\gamma$", r"Coupl. $\gamma_i$"]
    for i, spectra in enumerate([bulk_uncoupled, bulk_global, bulk_local]):
        ax[0].plot(spectra["nu"], spectra["susc_imag"], label=labels[i])

    for i, spectra in enumerate([book_uncoupled, book_global, book_local]):
        ax[1].plot(spectra["nu"], spectra["mean_susc_imag"], label=labels[i])

    for i, spectra in enumerate([cage_uncoupled, cage_global, cage_local]):
        ax[2].plot(spectra["nu"], spectra["mean_susc_imag"], label=labels[i])

    ax[0].plot(bulk_exp[:, 0], bulk_exp[:, 3], "k", ls="--", label="Experiment")
    ax[-1].set_xlabel(r"$\omega$ [Thz]")
    secax = ax[0].secondary_xaxis(
        "top", functions=(thz_to_wavenumber, wavenumber_to_thz)
    )

    secax.set_xlabel(r"$\omega$ [cm$^{-1}$]")

    for a in ax:
        a.set(
            ylabel=r"$\chi^{\prime \prime}$",
            yscale="log",
            xscale="log",
        )

    ax[0].tick_params(top=False, labeltop=False, bottom=False, labelbottom=False)
    ax[-1].tick_params(top=False, labeltop=False, bottom=True, labelbottom=True)

    ax[0].legend(loc="lower right", ncol=2)

    ax[2].set_ylim(4e-6, 2e1)
    ax[2].set_xlim(1e0, bulk_exp[:, 0].max())

    add_subplotlabels(fig, ax, labels=["A", "B", "C"])

    fig.tight_layout(h_pad=0.0)
    fig.align_labels()

    fig.savefig(
        f"figures/dielectric_spectrum{fname_suffix}.pdf",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )


plot_scrum()
