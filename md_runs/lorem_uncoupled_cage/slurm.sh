#!/bin/bash
#SBATCH --job-name=i_bk_mul
#SBATCH --array=0-12
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/md_seed_%a.out
#SBATCH --error=logs/md_seed_%a.err

eval "$(conda shell.bash hook)"
# Activate your environment
# conda activate myrto
conda activate marathon

mkdir -p logs

# Define specific seeds
SEEDS=(42 123 456 789 1024 10 11 12 13 14 15 16)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

BASE_OUT_DIR="/work/pstaerk/md_runs_ml/lorem_uncoupled_cage_com"
OUT_DIR="${BASE_OUT_DIR}/seed_${SEED}"

echo "Running simulation with seed ${SEED}"
echo "Output directory: ${OUT_DIR}"

python md.py --seed ${SEED} --out-dir ${OUT_DIR}
