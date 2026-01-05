#!/usr/bin/env python3

"""Plot timing benchmark shown in Figure S3 of the manuscript."""

import matplotlib.pyplot as plt
import mpltex
import numpy as np

paired = plt.get_cmap("Paired")
timings = np.loadtxt("data/timings_model1.txt")
timings2 = np.loadtxt("data/timings_model2.txt")


@mpltex.acs_decorator
def benchmark():
    fig, ax = plt.subplots()

    ax.loglog(
        timings2[:, 0],
        timings2[:, 1],
        "s-",
        label=r"Coupl. $\gamma_i$ (LOREM)",
        color=paired(2),
    )

    ax.loglog(
        timings[:, 0],
        timings[:, 1],
        "o-",
        label=r"Uncoupled (LOREM)",
        color=paired(0),
    )

    ax.set_xlabel("Number of atoms - $N$")
    ax.set_ylabel("Time per step [ms]")

    n_ref = np.array([1e3, 8e3])
    time_ref_n3_2 = n_ref ** (3 / 2)  # Ewald sum scaling
    ax.plot(n_ref, time_ref_n3_2 * 7e-5, "k:", label=r"$N^{3/2}$")
    ax.legend()

    ax.set_ylim(0.6, 400)

    fig.tight_layout()
    fig.savefig(
        "TeX/figures/timing_benchmark.pdf",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )

benchmark()
