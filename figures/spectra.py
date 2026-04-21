#!/usr/bin/env python3

"""Plot dielectric spectra shown in Figure 5 of the manuscript."""

import numpy as np
import matplotlib.pyplot as plt
import mpltex
from scipy.constants import c
from PySnips.plotting import add_subplotlabels

# 1 THz = 1e12 Hz; wavenumber (cm⁻¹) = freq (Hz) / c (cm/s)
_THZ_TO_WAVENUMBER = 1e12 / (c * 100)  # ≈ 33.356 cm⁻¹/THz


# add second axis with wavenumbers in cm^-1
def thz_to_wavenumber(x):
    return x * _THZ_TO_WAVENUMBER


def wavenumber_to_thz(x):
    return x / _THZ_TO_WAVENUMBER


def _trapezoid(y, x, xmin=None, xmax=None):
    mask = np.ones(len(x), dtype=bool)
    if xmin is not None:
        mask &= x >= xmin
    if xmax is not None:
        mask &= x <= xmax
    return np.trapezoid(y[mask], x[mask])


@mpltex.acs_decorator
def plot_scrum():
    fig, ax = plt.subplots(3, sharex=True, sharey=True)

    fig.set_figheight(1.8 * fig.get_figheight())
    fig.set_figwidth(3.5)

    ax[2].set_ylim(0, 1.75)

    xmin = 20
    xmax = 150
    ax[2].set_xlim(xmin, xmax)

    labels = [r"Uncoupl.", r"Coupl. $\gamma$", r"Coupl. $\gamma_i$", r"Uncoupl. fixed"]
    bulk_spectra = [bulk_uncoupled, bulk_global, bulk_local, bulk_fixed]
    sim_integrals = np.zeros(len(bulk_spectra))
    for i, spectra in enumerate(bulk_spectra):
        ax[0].plot(spectra["nu"], spectra["susc_imag"], label=labels[i])
        sim_integrals[i] = _trapezoid(spectra["susc_imag"], spectra["nu"], xmin, xmax)

    exp_integral = _trapezoid(bulk_exp[:, 3], bulk_exp[:, 0], xmin, xmax)
    exp_scale = np.mean(sim_integrals) / exp_integral
    ax[0].plot(bulk_exp[:, 0], exp_scale * bulk_exp[:, 3], "k", ls="--", label="Experiment")

    for i, spectra in enumerate([book_uncoupled, book_global, book_local]):
        ax[1].plot(spectra["nu"], spectra["susc_imag"], label=labels[i])

    for i, spectra in enumerate([cage_uncoupled, cage_global, cage_local]):
        ax[2].plot(spectra["nu"], spectra["susc_imag"], label=labels[i])


    ax[-1].set_xlabel(r"$\omega$ [Thz]")
    secax = ax[0].secondary_xaxis(
        "top", functions=(thz_to_wavenumber, wavenumber_to_thz)
    )

    secax.set_xlabel(r"$\omega$ [cm$^{-1}$]")

    for a in ax:
        a.set_ylabel(r"$\chi^{\prime \prime}$")

    ax[0].tick_params(top=False, labeltop=False, bottom=False, labelbottom=False)
    ax[-1].tick_params(top=False, labeltop=False, bottom=True, labelbottom=True)

    ax[0].legend(loc="upper center", ncol=1, handlelength=1.5, bbox_to_anchor=(0.415, 1))

    add_subplotlabels(fig, ax, labels=["A", "B", "C"])

    fig.tight_layout(h_pad=0.0)
    fig.align_labels()

    fig.savefig(
        f"figures/dielectric_spectrum{fname_suffix}.pdf",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )

    plt.show()

if __name__ == "__main__":

    bulk_exp = np.loadtxt("data/combined_exp_susc_300K.txt")

    for physical_model in [True]:

        if physical_model:
            path = "data/spectra_physical/spectrum_physical_"
            fname_suffix = "_physical"
        else:
            path = "data/spectra_lorem/spectrum_lorem_"
            fname_suffix = "_lorem"

        bulk_uncoupled = dict(np.load(f"{path}uncoupled_bulk.npz", allow_pickle=True))
        book_uncoupled = dict(np.load(f"{path}uncoupled_book.npz", allow_pickle=True))
        cage_uncoupled = dict(np.load(f"{path}uncoupled_cage.npz", allow_pickle=True))

        bulk_fixed = dict(np.load(f"{path}uncoupled_fixed_bulk.npz", allow_pickle=True))

        bulk_global = dict(np.load(f"{path}global_bulk.npz", allow_pickle=True))
        book_global = dict(np.load(f"{path}global_book.npz", allow_pickle=True))
        cage_global = dict(np.load(f"{path}global_cage.npz", allow_pickle=True))

        bulk_local = dict(np.load(f"{path}local_bulk.npz", allow_pickle=True))
        book_local = dict(np.load(f"{path}local_book.npz", allow_pickle=True))
        cage_local = dict(np.load(f"{path}local_cage.npz", allow_pickle=True))

        plot_scrum()
