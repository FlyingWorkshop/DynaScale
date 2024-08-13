#!/bin/bash
#SBATCH -J dynadojo_gilpin  # A single job name for the array
#SBATCH -N 1                # All cores on one machine (--nodes=)
#SBATCH -n 1                # Cores (--ntasks=<number>) 
#SBATCH --partition=dev     # Partition
#SBATCH --cpus-per-task=1   # Num of CPUs 
#SBATCH --mem=1G            # Memory (4Gb)
#SBATCH -t 0-0:10           # Maximum execution time (D-HH:MM)
#SBATCH -o ${SCRATCH}/generate_gilpin/logs/out/%A_%a.out
#SBATCH -e ${SCRATCH}/generate_gilpin/logs/err/%A_%a.err
#SBATCH --array=1-3%3    # Iterations [start i]-[end i]%[concurrently running];  maps to $SLURM_ARRAY_TASK_ID below

module load python/3.9.0
export PYTHONPATH=$HOME/.local/lib/python3.9/site-packages:$PYTHONPATH
cd $HOME/dynadojo
python3 -m generate_gilpin \
            --iteration=$SLURM_ARRAY_TASK_ID \
            --output_dir=$SCRATCH/generate_gilpin