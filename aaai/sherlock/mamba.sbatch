#!/bin/bash
#SBATCH -J dynadojo_mamba # A single job name for the array 
#SBATCH -N 1                # All cores on one machine (--nodes=)
#SBATCH -n 1                # Cores (--ntasks=<number>) 
#SBATCH --partition=gpu     # GPU Partition 
#SBATCH --gpus=1            # Num of GPUs
#SBATCH --cpus-per-task=1   # Num of CPUs 
#SBATCH --mem=4G            # Memory (4Gb)
#SBATCH -t 0-2:30           # Maximum execution time (D-HH:MM)
#SBATCH --array=1-500%100   # maps to SLURM_ARRAY_TASK_ID below

module load cuda/12.1.1
module load python/3.9.0
export PYTHONPATH=$GROUP_HOME/dynadojo/mamba/lib/python3.9/site-packages:$PYTHONPATH
export PYTHONPATH=$HOME/.local/lib/python3.9/site-packages:$PYTHONPATH

# COMMAND LINE GOES HERE