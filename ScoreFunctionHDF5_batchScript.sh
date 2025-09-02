#!/bin/bash -l
#SBATCH -N 1    
#SBATCH -t 00:30:00
#SBATCH -q debug
#SBATCH -J ScoreFuncs
#SBATCH -C cpu
#SBATCH -L SCRATCH,cfs
#SBATCH --ntasks 128
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --image=balewski/ubu20-neuron8:v5

# Input arguments
PREDICT_PATH=$1
ACTUAL_PATH=$2
ERROR_CSV=$3
STIM_INDEX=$4


srun -n1 shifter python3 -m pip install --user fastdtw
# Main scoring command (runs on all tasks)
srun -n1 shifter python3 -u ScoreFunctionsHDF5.py \
    --path1 "$PREDICT_PATH" \
    --path2 "$ACTUAL_PATH" \
    --csv_name "$ERROR_CSV" \
    --stim_index "$STIM_INDEX"
