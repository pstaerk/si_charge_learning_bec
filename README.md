## Installation Instructions
It is probably best to use a virtual environment (tested with Python 3.12).

Other than that, it should be as simple as running (possibly needs replacement of the specific JAX version (specifically CUDA, if wanted))
```
pip install -r requirements.txt
```

## Folder Contents

 - `checkpoints`: Model checkpoints of the models, trained on the full training set.
 - `md_runs`: Definitions and scripts of the MD runs.
 - `model_definitions`: Definitions of models, including model specific batching and ase calculators
 - `start_structs`: Starting structures of the md runs.
 - `evaluation`: Contains scripts for testing the models on the validation set.

## Naming Conventions

The following suffixes specify which "type" of model is referred to, in accordance with the main paper:

 - `uncoupled`: Model where BECs are predicted independently from learned static charges
 - `coupl_local`: Model where BECs are calculated from learned static charges with a local coupling factor `gamma`
 - `global`: Model where BECs are calculated (a posteriori) from learned static charges, assuming a global screening value
