#!/bin/sh
#
#SBATCH --job-name="run_catch_characteristics"
#SBATCH --partition=compute
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=10G
#SBATCH --account=research-ceg-wm

module load miniconda3

# Set conda env:
unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"

# Activate conda, run job, deactivate conda
conda activate sr_env

python run_catch_characteristics.py > ~/outputs_log/output.log

