#!/bin/sh
#
#SBATCH --job-name="run_sr_calculation"
#SBATCH --partition=compute
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=20G
#SBATCH --account=research-ceg-wm

python run_sr_calculation.py > ~/outputs_log/output.log

