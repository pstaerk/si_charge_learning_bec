## Installation Instructions

DFT calculations were performed with CP2K. To set up CP2K, please refer to the
[CP2K installation instructions](https://www.cp2k.org/howto:install_cp2k).

It is probably best to use a virtual environment (production ran with
Python 3.12.12).

You also need to build a version of pycp2k from the current cp2k input. See 
[pycp2k manual installation](https://github.com/SINGROUP/pycp2k?tab=readme-ov-file#manual)
for instructions. (production ran with CP2K version 2025.1):

The script `cp2k_run.ipynb` does all the DFT calculations needed for the
reference data, including external field calculations. It also includes
functions to extract the APTs from the calculated forces and writing to as well
as reading them from file.

Please adjust the Poisson solver depending on your system and needs.

The directory `conf_0` contains an example CP2K input file. In there `field_x`
is an example of how to set up an external electric field calculation in CP2K.
Both are outputs produced by the script.