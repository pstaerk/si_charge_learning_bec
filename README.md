# Simultaneous Learning of Static and Dynamic Charges - Auxiliary Code Repository

[![arXiv](https://img.shields.io/badge/arXiv-2601.03656-B31B1B.svg)](https://arxiv.org/abs/2601.03656)
[![Materials
Cloud](https://img.shields.io/badge/Materials%20Cloud-10.24435/materialscloud:fs--8h-AACAFB.svg)](https://doi.org/10.24435/materialscloud:fs-8h)

This repository includes all codes and input files used for the work "Simultaneous Learning of Static and Dynamic Charges" available at
https://arxiv.org/abs/2601.03656.

The dataset used for training and validation is available at https://doi.org/10.24435/materialscloud:fs-8h.

## Installation Instructions

To train and evaluate the models as described in the main paper, you need to install the required dependencies listed in the `requirements.txt` file.

It is recommended to use a virtual environment (e.g., `venv` or `conda`) to manage the dependencies. Consider replacing to a specific version of the JAX version (specifically CUDA, if wanted)

```
pip install -r requirements.txt
```

## Repository Contents

- `checkpoints`: Model checkpoints of the models, trained on the full training set.
- `dft`: Contains script to create cp2k based DFT calculations for the structures, with examples of how to apply external fields.
- `evaluation`: Contains scripts for testing the models on the validation set.
- `figures`: Scripts to generate all figures presented in the main paper and the supplementary information.
- `ir_spectra`: Scripts to calculate and plot the IR spectra from the MD runs.
- `md_runs`: Definitions and scripts of the MD runs.
- `model_definitions`: Definitions of models, including model specific batching and ase calculators
- `start_structs`: Starting structures of the md runs.
- `training_runs`: Contains configuration and training scripts that for generating the checkpoints.

## Naming Conventions

The following suffixes specify which "type" of model is referred to, in accordance with the main paper:

- `uncoupled`: Model where BECs are predicted independently from learned static charges
- `coupl_local`: Model where BECs are calculated from learned static charges with a local coupling factor `gamma`
- `global`: Model where BECs are calculated (a posteriori) from learned static charges, assuming a global screening value
