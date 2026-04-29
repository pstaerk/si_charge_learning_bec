#!/usr/bin/env python3

"""Plot dielectric spectra shown in Figure 5 of the manuscript."""

import numpy as np
import matplotlib.pyplot as plt
import mpltex
from scipy.constants import c
from PySnips.plotting import add_subplotlabels

# 1 THz = 1e12 Hz; wavenumber (cm⁻¹) = freq (Hz) / c (cm/s)
_THZ_TO_WAVENUMBER = 1e12 / (c * 100)  # ≈ 33.356 cm⁻¹/THz

# dummy spectra
_NAN_DICT = {"nu": np.nan, "susc_imag": np.nan}


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
    fig, ax = plt.subplots(3, 2, sharey="row")

    fig.set_figheight(1.8 * fig.get_figheight())
    #fig.set_figwidth(7.0)

    xmin_left, xmax_left = 20, 65
    xmin_right, xmax_right = 85, 125

    for row in range(3):
        ax[row, 0].set_xlim(xmin_left, xmax_left)
        ax[row, 1].set_xlim(xmin_right, xmax_right)
        ax[row, 0].minorticks_on()
        ax[row, 1].minorticks_on()
        if row > 0:
            # ax[row, 0].set_yscale("log")
            # ax[row, 1].set_yscale("log")
            ax[row, 0].set_ylim(-0.01, 0.17)
        # hide redundant y-axis on right panels
        ax[row, 1].tick_params(labelleft=False)

    # BULK
    labels = [r"Uncoupl.", r"Coupl. $\gamma$", r"Coupl. $\gamma_i$", r"fixed"]
    bulk_spectra = [bulk_uncoupled, bulk_global, bulk_local, bulk_fixed]
    sim_integrals = np.zeros(len(bulk_spectra))
    for i, spectra in enumerate(bulk_spectra):
        for col in range(2):
            ax[0, col].plot(spectra["nu"], spectra["susc_imag"])
        sim_integrals[i] = _trapezoid(
            spectra["susc_imag"], spectra["nu"], xmin_left, xmax_right
        )

    exp_integral = _trapezoid(bulk_exp[:, 3], bulk_exp[:, 0], xmin_left, xmax_right)
    exp_scale = np.mean(sim_integrals) / exp_integral
    for col in range(2):
        ax[0, col].plot(
            bulk_exp[:, 0], exp_scale * bulk_exp[:, 3], "k", ls="--", label="Experiment"
        )

    ax[0, 0].set_ylim(-.2,1.85)
    ax[0, 1].set_ylim(-0.2,1.85)

    # BOOK
    for i, spectra in enumerate([book_uncoupled, book_global, book_local, book_fixed]):
        for col in range(2):
            ax[1, col].plot(spectra["nu"], spectra["susc_imag"], label=labels[i])

    # CAGE
    for i, spectra in enumerate([cage_uncoupled, cage_global, cage_local, cage_fixed]):
        for col in range(2):
            ax[2, col].plot(spectra["nu"], spectra["susc_imag"], label=labels[i])

    for col in range(2):
        ax[-1, col].set_xlabel(r"$\omega$ [THz]")

    secax_left = ax[0, 0].secondary_xaxis(
        "top", functions=(thz_to_wavenumber, wavenumber_to_thz)
    )
    secax_left.set_xlabel(r"$\omega$ [cm$^{-1}$]")
    secax_right = ax[0, 1].secondary_xaxis(
        "top", functions=(thz_to_wavenumber, wavenumber_to_thz)
    )
    secax_right.set_xlabel(r"$\omega$ [cm$^{-1}$]")

    for row in range(3):
        ax[row, 0].set_ylabel(r"$\chi^{\prime \prime}$")

    for row in range(2):
        for col in range(2):
            ax[row, col].tick_params(bottom=False, labelbottom=False)
    for col in range(2):
        ax[-1, col].tick_params(bottom=True, labelbottom=True)

    ax[0,1].legend()

    _legend_order = [3, 1, 2, 0]  # fixed, Coupl. γ, Coupl. γ_i, Uncoupl.
    handles, lbls = ax[1,1].get_legend_handles_labels()
    ax[1, 1].legend([handles[i] for i in _legend_order], [lbls[i] for i in _legend_order])

    add_subplotlabels(fig, ax.flatten(), labels=["A", "B", "C", "D", "E", "F"])

    fig.tight_layout(h_pad=0.0, w_pad=0.0)
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

        if physical_model:
            bulk_fixed = dict(np.load(f"{path}uncoupled_fixed_bulk.npz", allow_pickle=True))
            book_fixed = dict(np.load(f"{path}uncoupled_fixed_book.npz", allow_pickle=True))
            cage_fixed = dict(np.load(f"{path}uncoupled_fixed_cage.npz", allow_pickle=True))

            book_fixed["susc_imag"] /= 3000
            cage_fixed["susc_imag"] /= 3000
        else:
            bulk_fixed = _NAN_DICT

        bulk_global = dict(np.load(f"{path}global_bulk.npz", allow_pickle=True))
        book_global = dict(np.load(f"{path}global_book.npz", allow_pickle=True))
        cage_global = dict(np.load(f"{path}global_cage.npz", allow_pickle=True))

        bulk_local = dict(np.load(f"{path}local_bulk.npz", allow_pickle=True))
        book_local = dict(np.load(f"{path}local_book.npz", allow_pickle=True))
        cage_local = dict(np.load(f"{path}local_cage.npz", allow_pickle=True))

        plot_scrum()
