from ase.io import read, write
from pathlib import Path
from ase.md.langevin import Langevin
from ase.md.andersen import Andersen
from ase.md.bussi import Bussi
from ase import units
from ase.io.trajectory import Trajectory
import nglview as nv
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

import numpy as np
import jax
import jax.numpy as jnp

from ase.calculators.abc import GetPropertiesMixin
from ase.calculators.calculator import PropertyNotImplementedError, compare_atoms
from glp import atoms_to_system
from glp.graph import system_to_graph
from glp.neighborlist import neighbor_list

from marathon import comms
from marathon.data import Batch
# from marathon.fields import Batch as fBatch

from marathon.fields import get_batch
from ase import Atoms
import matplotlib.pyplot as plt

from rich.progress import Progress, track

from calculator2 import Calculator

atoms_start = read('/work/pstaerk_m3/pyiron/ml_nacl/datasets_ml/cheng_water_vapor_combined/training_data.xyz', index='6')

model_dir = Path(f"./run/checkpoints/R2_E+F+A/")

field_strength = 0.1
# So that the sample code does not cry:
atoms_start.info['electric_field'] = np.array([field_strength, 0.0, 0.0])
calc = Calculator.from_checkpoint(model_dir,
                                       field=[field_strength, 0.0, 0.0])

# use dummy apts for sample to work
atoms_start.arrays['apt'] = np.zeros((len(atoms_start), 3, 3))
atoms_start.arrays['apt_charge'] = np.zeros((len(atoms_start)))

atoms = atoms_start.copy()
atoms.calc = calc
atoms.pbc = True

MaxwellBoltzmannDistribution(atoms, temperature_K=300)
dyn = Bussi(atoms, 0.5*units.fs, 300, 100*units.fs)
# dyn = Andersen(atoms, 0.5*units.fs, 300, 0.002)

workdir = f'/work/pstaerk_new/uncoupled_lr_combined_water/1ns_1ps_interface/'
traj = Trajectory(f'{workdir}/traj.traj', 'w', atoms,
                  properties=['energy', 'forces', 'charges', 'apt'])
N_per_step = 2000
dyn.attach(traj.write, interval=N_per_step)


def write_apts(atoms, path='apts.txt'):
    apts = atoms.calc.results['apt']
    # apts_charge = atoms.calc.results['apt_charge']
    charge = atoms.calc.results['charges']

    with open(path, 'a') as f:
        f.write(f"{len(atoms)}\n")
        f.write('# APTs\n')
        for i in range(len(atoms)):
            f.write(f"{i} {apts[i][0][0]} {apts[i][0][1]} {apts[i][0][2]} {charge[i]}\n")
            f.write(f"{i} {apts[i][1][0]} {apts[i][1][1]} {apts[i][1][2]} {charge[i]}\n")
            f.write(f"{i} {apts[i][2][0]} {apts[i][2][1]} {apts[i][2][2]} {charge[i]}\n")


# backup apt.txt and make new apt.txt
backup_path = f'{workdir}/apts.txt'
if Path(backup_path).exists():
    Path(backup_path).rename(f'{workdir}/apts_backup.txt')
with open(backup_path, 'w') as f:
    f.write('')

N_steps = 1000

# Manually start and stop the Progress instance
for _ in track(range(N_steps), description='[red] Running simulation'):
    dyn.run(N_per_step)
    write_apts(dyn.atoms, path=f'{workdir}/apts.txt')

traj.close()
