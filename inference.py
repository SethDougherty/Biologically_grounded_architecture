#!/usr/bin/env python3
"""
Minimal Neuron ML inference with checkpoint support
- Loads baseline model and optional checkpoint
- Runs inference on test dataset
- Saves predicted parameters as CSV
"""

import sys
sys.path.append('/pscratch/sd/s/sdough/Neuron_Latest_Pipeline/MLNeuronInverter')

import os
import time
import torch
import argparse
import numpy as np
import pandas as pd

from toolbox.Util_IOfunc import read_yaml
from toolbox.Dataloader_H5 import get_data_loader
sys.path.append('/pscratch/sd/s/sdough/Neuron_Latest_Pipeline/MLNeuronInverter/toolbox')
from Model import MyModel

#-----------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelPath', required=True, help="Path to trained model folder")
    parser.add_argument('--checkpoint_dir', default=None, help="Directory containing checkpoints (optional)")
    parser.add_argument('--dom', default='test', help="Dataset domain to infer on")
    parser.add_argument('--outPath', required=True, help="Output CSV path")
    parser.add_argument('--numSamples', type=int, default=None, help="Limit number of samples for inference")
    parser.add_argument('--stimsSelect', nargs='+', type=int, default=None, help="List of stim indices to select")
    parser.add_argument('--testStimsSelect', nargs='+', type=int, default=None, help="List of validation stim indices to select")
    parser.add_argument('--cellName', type=str, default=None, help="Alternative cell name")
    args = parser.parse_args()
    return args

#-----------------------------
def find_latest_checkpoint(checkpoint_dir):
    ckpts = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")]
    if not ckpts:
        return None
    def epoch_num(f): return int(f.split("_")[1].split(".")[0])
    latest_ckpt = max(ckpts, key=epoch_num)
    return os.path.join(checkpoint_dir, latest_ckpt)

#-----------------------------
def load_model(trainMD, modelPath, checkpoint_dir=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelF = os.path.join(modelPath, trainMD['train_params']['blank_model'])
    model = torch.load(modelF)
    model = torch.nn.DataParallel(model)

    if checkpoint_dir is not None:
        ckpt_path = find_latest_checkpoint(checkpoint_dir)
        if ckpt_path is not None:
            state = torch.load(ckpt_path, map_location=device)
            stateD = state["model_state"]
            if 'module' not in list(stateD.keys())[0]:
                stateD = {'module.' + k: stateD[k] for k in stateD}
            model.load_state_dict(stateD)
            print(f"Loaded checkpoint {ckpt_path}")
        else:
            print("No checkpoint found, using baseline model")
    return model.to(device)

# def load_model(trainMD, modelPath, checkpoint_dir=None):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Load hyperparameters
#     hpar = trainMD['train_params']['model']
#     # Instantiate model using original class
#     model = MyModel(hpar)

#     # Load baseline model if needed (optional)
#     base_model_file = os.path.join(modelPath, trainMD['train_params']['blank_model'])
#     if os.path.exists(base_model_file):
#         state = torch.load(base_model_file, map_location=device)
#         if isinstance(state, dict) and 'model_state' in state:
#             stateD = state['model_state']
#             if 'module' not in list(stateD.keys())[0]:
#                 stateD = {'module.' + k: stateD[k] for k in stateD}
#             model.load_state_dict(stateD)
#         else:
#             model.load_state_dict(state)
#         print(f"Loaded baseline model from {base_model_file}")

#     # Load checkpoint if provided
#     if checkpoint_dir is not None:
#         ckpt_path = find_latest_checkpoint(checkpoint_dir)
#         if ckpt_path is not None:
#             state = torch.load(ckpt_path, map_location=device)
#             stateD = state["model_state"] if 'model_state' in state else state
#             if 'module' not in list(stateD.keys())[0]:
#                 stateD = {'module.' + k: stateD[k] for k in stateD}
#             model.load_state_dict(stateD)
#             print(f"Loaded checkpoint {ckpt_path}")
#         else:
#             print("No checkpoint found, using baseline model")

#     model = torch.nn.DataParallel(model)
#     return model.to(device)

#-----------------------------
def model_infer(model, test_loader, trainMD):
    device = next(model.parameters()).device
    model.eval()
    criterion = torch.nn.MSELoss().to(device)
    num_samp = len(test_loader.dataset)
    outputSize = trainMD['train_params']['model']['outputSize']
    inputShape = trainMD['train_params']['model']['inputShape']
    inputSize = [num_samp, inputShape[0], inputShape[1]]

    Uall = np.zeros([num_samp, outputSize], dtype=np.float32)
    Zall = np.zeros([num_samp, outputSize], dtype=np.float32)

    nEve = 0
    nStep = 0
    with torch.no_grad():
        for data, target in test_loader:
            data_dev, target_dev = data.to(device), target.to(device)
            output_dev = model(data_dev)
            output = output_dev.cpu().numpy()
            target_np = target.cpu().numpy()
            nEve2 = nEve + target.shape[0]
            Uall[nEve:nEve2, :] = target_np
            Zall[nEve:nEve2, :] = output
            nEve = nEve2
            nStep += 1
    return Uall, Zall

#-----------------------------
if __name__ == '__main__':
    args = parse_args()

    sumF = os.path.join(args.modelPath, 'sum_train.yaml')
    trainMD = read_yaml(sumF)
    parMD = trainMD['train_params']
    inpMD = trainMD['input_meta']

    if args.cellName is not None:
        parMD['cell_name'] = args.cellName
    if args.numSamples is not None:
        parMD['data_conf']['max_glob_samples_per_epoch'] = args.numSamples
    if args.testStimsSelect is not None:
        parMD['data_conf']['valid_stims_select'] = args.testStimsSelect
    if args.stimsSelect is not None:
        parMD['data_conf']['stims_select'] = args.stimsSelect

    parMD['world_size'] = 1
    test_loader = get_data_loader(parMD, args.dom, verb=1)

    model = load_model(trainMD, args.modelPath, checkpoint_dir=args.checkpoint_dir)

    startT = time.time()
    trueU, recoU = model_infer(model, test_loader, trainMD)
    elapsed = time.time() - startT
    print(f"Inference done, samples={trueU.shape[0]}, elapsed={elapsed/60:.2f} min")

    param_names = inpMD.get('parName', [f'p{i}' for i in range(recoU.shape[1])])
    phys_ranges = inpMD['phys_par_range']  # list of [lb, ub, unit]

    scaled_preds = []
    for i, (lb, ub, _) in enumerate(phys_ranges):
        # map from [-1,1] → [lb, ub]
        scaled = lb + (recoU[:, i] + 1) * 0.5 * (ub - lb)
        scaled_preds.append(scaled)

    scaled_preds = np.stack(scaled_preds, axis=1)

    # Save CSV
    # param_names = inpMD.get('parName', [f'p{i}' for i in range(recoU.shape[1])])
    pred_df = pd.DataFrame(scaled_preds)
    pred_df.to_csv(args.outPath, index=False, header=False)
    print(f"Saved predictions to {args.outPath}")
