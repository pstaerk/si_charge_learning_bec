## Installation Instructions
It is probably best to use a virtual environment (tested with Python 3.12).

Other than that, it should be as simple as, possibly needs replacement of the specific JAX version (specifically CUDA)
```
pip install -r requirements.txt
```

## Folder Contents

 - `checkpoints`: Model checkpoints of the models, trained on the full training set.
 - `md_runs`: Definitions and scripts of the MD runs.
 - `model_definitions`: Definitions of models, including model specific batching and ase calculators
 - `start_structs`: Starting structures of the md runs.

