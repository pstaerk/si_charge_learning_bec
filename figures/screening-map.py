#!/usr/bin/env python3

"""Create chemiscope input files for clusters and bulk."""

import chemiscope
import ase.io
import numpy as np

clusters = ase.io.read("data/cluster_eval_valid_summarized.xyz", ":")
bulk = ase.io.read("data/bulk_eval_valid_summarized.xyz", ":")


# bulk
screen_lorem_bulk = np.hstack([abs(f.arrays["lorem_coup_local_gamma"]) for f in bulk])
screen_physical_bulk = np.hstack(
    [abs(f.arrays["physical_coup_local_gamma"]) for f in bulk]
)


pseudo_static_charges_lorem_bulk = np.hstack(
    [f.arrays["lorem_coup_local_charges"] for f in bulk]
)
pseudo_static_charges_physical_bulk = np.hstack(
    [f.arrays["physical_coup_local_charges"] for f in bulk]
)

# clusters
screen_lorem_clusters = np.hstack(
    [abs(f.arrays["lorem_coup_local_gamma"]) for f in clusters]
)
screen_physical_clusters = np.hstack(
    [abs(f.arrays["physical_coup_local_gamma"]) for f in clusters]
)


pseudo_static_charges_lorem_clusters = np.hstack(
    [f.arrays["lorem_coup_local_charges"] for f in clusters]
)
pseudo_static_charges_physical_clusters = np.hstack(
    [f.arrays["physical_coup_local_charges"] for f in clusters]
)

settings = {
    "map": {
        "color": {
            "property": "pseudo static q physical",
            "palette": "seismic",
            "min": -0.4,
            "max": 0.4,
        },
    },
    "structure": [
        {
            "unitCell": False,
            "color": {
                "property": "𝛄_i physical",
                "palette": "magma",
                "min": 0.9,
                "max": 1.3,
            },
            "environments": {"activated": False},
        }
    ],
}

chemiscope.write_input(
    "figures/clusters_val.chemiscope.json.gz",
    frames=clusters,
    properties={
        "𝛄_i physical": screen_physical_clusters,
        "|𝛄_i| lorem": screen_lorem_clusters,
        "pseudo static q physical": pseudo_static_charges_physical_clusters,
        "pseudo static q lorem": pseudo_static_charges_lorem_clusters,
    },
    settings=settings,
    environments=chemiscope.all_atomic_environments(clusters),
)
