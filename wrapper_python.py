import subprocess
import os
import time

# ----------------- Config -----------------
MAX_EPOCHS = 2
CONFIG_PATH = "/pscratch/sd/s/sdough/Neuron_Latest_Pipeline/MLNeuronInverter/MultuStim.hpar.yaml"
OUTPUT_ROOT = "/pscratch/sd/s/sdough/tmp_neuInv/training"

# NEURON Config
mType = "L5_TTPC1"
eType = "cADpyr"
i_cell = 0
argsFileActual = "/pscratch/sd/s/sdough/tmp_neuInv/training/baseline_params.csv"

# Setup directories
os.makedirs(os.path.join(OUTPUT_ROOT, "predictions"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_ROOT, "errors"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_ROOT, "checkpoints"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_ROOT, "state"), exist_ok=True)

def submit_and_wait(cmd):
    """Submit sbatch command, return job_id, and wait for completion"""
    result = subprocess.run(cmd, stdout=subprocess.PIPE, universal_newlines=True, shell=True)
    the_result1 = str(result.stdout)
    the_array = the_result1.splitlines()
    job_id = str((the_array[len(the_array)-2]))
    print(f"Submitted Job ID: {job_id}")

    # Poll SLURM until job finishes
    # while True:
    #     sq_result = subprocess.run(['squeue', '--format', '%T', '--noheader', '-j', job_id],
    #                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    #     status = sq_result.stdout.decode().strip()
    #     if status == "":
    #         print(f"Job {job_id} finished.")
    #         break
    #     time.sleep(10)  # check every 10 seconds
    return job_id

# ----------------- Baseline Simulation -----------------
actual_cmd = f"sbatch --parsable M1_sbatch_MultiParamFile.sh {mType} {eType} {i_cell} {argsFileActual}"
actual_job = submit_and_wait(actual_cmd)
actual_path = f"/pscratch/sd/s/sdough/testRun/runs2/{actual_job}_1/{mType}{eType}{i_cell}/"
# actual_path = f"/pscratch/sd/s/sdough/testRun//runs2/42021029_1/L5_TTPC1cADpyr0/"

# checkpoint option /pscratch/sd/s/sdough/tmp_neuInv/bbp3/L5_TTPC1cADpyr0/38920355/out/

prev_job = ""

# ----------------- Training Loop -----------------
for epoch in range(1, MAX_EPOCHS + 1):
    print(f"\n=== Epoch {epoch} ===")

    # Phase 1: Forward pass
    infer_out = os.path.join(OUTPUT_ROOT, f"predictions/epoch_{epoch}_preds.csv")
    infer_cmd = f"sbatch --parsable {f'--dependency=afterok:{prev_job}' if prev_job else ''} inference_job.slr /pscratch/sd/s/sdough/tmp_neuInv/bbp3/L5_TTPC1cADpyr0/38920355/out {OUTPUT_ROOT}/checkpoints {infer_out} L5_TTPC1cADpyr0"
    infer_job = submit_and_wait(infer_cmd)

    # Phase 2: NEURON Pipeline (Predicted Simulation)
    error_csv = os.path.join(OUTPUT_ROOT, f"errors/epoch_{epoch}_errors.csv")
    pred_cmd = f"sbatch --parsable --dependency=afterok:{infer_job} M1_sbatch_MultiParamFile.sh {mType} {eType} {i_cell} {infer_out}"
    pred_job = submit_and_wait(pred_cmd)
    predict_path = f"/pscratch/sd/s/sdough/testRun/runs2/{pred_job}_1/{mType}{eType}{i_cell}/"

    # Phase 2b: Scoring (depends on predicted + actual)
    score_cmd = f"sbatch --parsable --dependency=afterok:{pred_job}:{actual_job} ScoreFunctionHDF5_batchScript.sh {predict_path} {actual_path} {error_csv} 0"
    # score_cmd = f"sbatch --parsable --dependency=afterok:{pred_job} ScoreFunctionHDF5_batchScript.sh {predict_path} {actual_path} {error_csv} 0"
    score_job = submit_and_wait(score_cmd)

    # Phase 3: Backward pass (depends on scoring)
    update_cmd = f"sbatch --parsable --dependency=afterok:{score_job}:{infer_job} weight_updater.slr /pscratch/sd/s/sdough/tmp_neuInv/bbp3/L5_TTPC1cADpyr0/38636709/out {OUTPUT_ROOT}/checkpoints {error_csv} {OUTPUT_ROOT}/state"
    update_job = submit_and_wait(update_cmd)

    prev_job = update_job

print("\n=== Training loop submitted and completed! ===")
