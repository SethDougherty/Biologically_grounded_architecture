#!/bin/bash -l
#SBATCH -N 8
#SBATCH -t 13:00:00
#SBATCH -q regular
#SBATCH -J DL4N_M1_csv
#SBATCH -L SCRATCH,cfs
#SBATCH -C cpu
#SBATCH --output logs/%A_%a
#SBATCH --image=balewski/ubu20-neuron8:v5
#SBATCH --array 1-1

module unload craype-hugepages2M

# Output directory
OUT_DIR=/pscratch/sd/s/sdough/testRun/
model='M1_TTPC_NA_HH'

# Cells info
CELLS_FILE='excitatorycells.csv'
START_CELL=0
NCELLS=2
END_CELL=$((${START_CELL}+${NCELLS}))

export THREADS_PER_NODE=128
export HDF5_USE_FILE_LOCKING=FALSE

# SLURM info
arrIdx=${SLURM_ARRAY_TASK_ID}
jobId=${SLURM_ARRAY_JOB_ID}_${arrIdx}
RUNDIR=${OUT_DIR}/runs2/${jobId}
mkdir -p $RUNDIR

# Script arguments
mType=$1
eType=$2
i_cell=$3
param_csv=$4 

cell_name="${mType}${eType}${i_cell}"
mkdir -p $RUNDIR/$cell_name/
chmod a+rx $RUNDIR/$cell_name/

cp /pscratch/sd/s/sdough/Neuron_Latest_Pipeline/DL4neurons2/M1_sbatch_submit.sh $RUNDIR
chmod a+rx $RUNDIR
chmod a+rx $RUNDIR/*

echo "Done making outdirs at" `date`

stimfile="stims/5k50kInterChaoticB.csv"

# Build output file name
FILENAME="${model}-v3"
OUT_DIR_RUN=$RUNDIR/$cell_name
FILE_NAME=${FILENAME}-\{NODEID\}-c${i_cell}.h5
OUTFILE=$OUT_DIR_RUN/$FILE_NAME

# Build arguments for run.py
args="--outfile $OUTFILE \
      --stim-file $stimfile \
      --model $model \
      --m-type $mType \
      --e-type $eType \
      --cell-i $i_cell \
      --linear-params-inds 12 17 18 \
      --param-file $param_csv \
      --dt 0.1 \
      --stim-dc-offset 0 \
      --stim-multiplier 1"

echo "Running NEURON with args: $args"

# Run NEURON safely with SLURM
cd /pscratch/sd/s/sdough/Neuron_Latest_Pipeline/DL4neurons2
srun --input none -k -n $((${SLURM_NNODES}*${THREADS_PER_NODE})) \
     --ntasks-per-node ${THREADS_PER_NODE} \
     shifter python3 -u /pscratch/sd/s/sdough/Neuron_Latest_Pipeline/DL4neurons2/run.py $args

echo "All runs finished. Files at: $RUNDIR/$cell_name"
chmod a+r $RUNDIR/$cell_name/*.h5
