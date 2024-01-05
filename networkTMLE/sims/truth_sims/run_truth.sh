#!/bin/bash

# SBATCH -p general
# SBATCH -N 1
# SBATCH -n 1
# SBATCH -c 4
# SBATCH --mem=7g
# SBATCH -t 11-00:00
# SBATCH --array 1001,1011,2001,2011,2101,2111,4001,4011,5001,5011,5101,5111,6001,6011,7001,7011,7101,7111

python3 -u truth_statin.py $SLURM_ARRAY_TASK_ID