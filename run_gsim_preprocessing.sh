#!/bin/sh
#
#SBATCH --job-name="run_gsim_preprocessing"
#SBATCH --partition=compute
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --account=research-ceg-wm

python run_gsim_preprocessing.py > ~/outputs_log/output.log

