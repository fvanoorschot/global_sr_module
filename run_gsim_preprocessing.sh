#!/bin/sh
#
#SBATCH --job-name="run_gsim_preprocessing.py"
#SBATCH --partition=compute
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --account=research-ceg-wm

module load python
srun python run_gsim_preprocessing.py > ~/output.log

