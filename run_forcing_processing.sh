#!/bin/sh
#
#SBATCH --job-name="run_forcing_processing"
#SBATCH --partition=compute
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=10G
#SBATCH --account=research-ceg-wm

python run_forcing_processing.py > ~/outputs_log/output.log

