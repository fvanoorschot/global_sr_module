#!/bin/sh
#
#SBATCH --job-name="run_grid_to_catchments-nearest"
#SBATCH --partition=compute
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=100G
#SBATCH --account=research-ceg-wm

python run_grid_to_catchments-nearest.py > ~/outputs_log/output.log