#!/bin/sh
#
#SBATCH --job-name="run_snow_module"
#SBATCH --partition=compute
#SBATCH --time=18:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=10G
#SBATCH --account=research-ceg-wm

python run_snow_module.py > ~/outputs_log/output.log

