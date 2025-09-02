#!/usr/bin/env python3
import sys
sys.path.append('/pscratch/sd/s/sdough/Neuron_Latest_Pipeline/MLNeuronInverter')

import torch
import pandas as pd
import os
import logging
from toolbox.Util_IOfunc import read_yaml

class WeightUpdater:
    def __init__(self, model_dir, checkpoint_dir=None):
        """
        model_dir: directory containing blank_model.pt (baseline model)
        checkpoint_dir: directory containing previous checkpoints
        """
        self.model_dir = model_dir
        self.checkpoint_dir = checkpoint_dir
        self.current_epoch = 0

        # Load training metadata (to know blank model name)
        sumF = os.path.join(self.model_dir, "sum_train.yaml")
        trainMD = read_yaml(sumF)
        blank_model_name = trainMD['train_params']['blank_model']
        blank_model_path = os.path.join(self.model_dir, blank_model_name)

        if not os.path.exists(blank_model_path):
            raise FileNotFoundError(f"Blank model not found at {blank_model_path}")

        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load blank model
        self.model = torch.load(blank_model_path, map_location=self.device)
        self.model = torch.nn.DataParallel(self.model)

        # Optimizer (needed for checkpoint consistency)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        # Load latest checkpoint if available
        if self.checkpoint_dir is not None and os.path.exists(self.checkpoint_dir):
            ckpts = [f for f in os.listdir(self.checkpoint_dir) if f.endswith(".pt")]
            if ckpts:
                def epoch_num(f): return int(f.split("_")[1].split(".")[0])
                latest_ckpt = max(ckpts, key=epoch_num)
                ckpt_path = os.path.join(self.checkpoint_dir, latest_ckpt)

                state = torch.load(ckpt_path, map_location=self.device)
                stateD = state["model_state"]

                # Fix missing 'module.' prefix if necessary
                if 'module' not in list(stateD.keys())[0]:
                    stateD = {'module.' + k: stateD[k] for k in stateD}

                self.model.load_state_dict(stateD)
                self.optimizer.load_state_dict(state["optimizer_state"])
                self.current_epoch = state.get("epoch", 0)

                logging.info(f"Loaded checkpoint {latest_ckpt} at epoch {self.current_epoch}")
            else:
                logging.info("No checkpoint found, using blank model")

    def update_from_error(self, error_csv_path, update_factor=0.01, previous_error=None):
        """
        Non-differentiable weight update rule:
        Adjust weights proportionally to error magnitude
        """
        error_df = pd.read_csv(error_csv_path)

        # Combine error metrics
        s_error = error_df["Score_Error"].mean()
        dtw_error = error_df["DTWD_Error"].mean()
        composite_error = 0.7 * s_error + 0.3 * dtw_error

        # Update weights
        with torch.no_grad():
            for param in self.model.parameters():
                if previous_error is not None:
                    direction = -1 if composite_error < previous_error else 1
                else:
                    direction = -1
                base_update = direction * update_factor
                random_component = 0.2 * torch.randn_like(param.data)
                structural_component = param.data.abs() * 0.1
                update = base_update * (structural_component + random_component)
                max_update = 0.05 * param.data.abs().max()
                param.data += torch.clamp(update, -max_update, max_update)

        return composite_error

    def save_checkpoint(self, checkpoint_dir):
        """Save updated model"""
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{self.current_epoch}.pt")
        torch.save({
            'epoch': self.current_epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict()
        }, checkpoint_path)
        logging.info(f"Saved checkpoint for epoch {self.current_epoch}")

    def load_epoch_state(self, state_dir):
        """Track training progress across runs (optional)"""
        state_file = os.path.join(state_dir, "training_state.pt")
        if os.path.exists(state_file):
            state = torch.load(state_file)
            self.current_epoch = state['epoch']
            return True
        return False

    def save_epoch_state(self, state_dir):
        state_file = os.path.join(state_dir, "training_state.pt")
        torch.save({'epoch': self.current_epoch}, state_file)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, help="Directory containing blank_model.pt")
    parser.add_argument("--checkpoint_dir", required=True, help="Directory for checkpoints")
    parser.add_argument("--error_csv", required=True, help="NEURON error CSV")
    parser.add_argument("--state_dir", required=True, help="Directory for training state")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Initialize updater
    updater = WeightUpdater(args.model_dir, args.checkpoint_dir)
    updater.load_epoch_state(args.state_dir)

    # Apply weight update
    current_error = updater.update_from_error(args.error_csv)
    logging.info(f"Epoch {updater.current_epoch} updated with error: {current_error}")

    # Increment epoch
    updater.current_epoch += 1

    # Save checkpoint and epoch state
    updater.save_checkpoint(args.checkpoint_dir)
    updater.save_epoch_state(args.state_dir)


if __name__ == "__main__":
    main()
