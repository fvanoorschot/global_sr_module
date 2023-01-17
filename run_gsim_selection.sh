#!/bin/sh
#
#SBATCH --job-name="run_gsim_selection"
#SBATCH --partition=compute
#SBATCH --time=05:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=10G
#SBATCH --account=research-ceg-wm

python run_gsim_selection.py > ~/outputs_log/output.log

