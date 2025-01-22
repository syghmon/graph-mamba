import json
import os
import argparse
from src.train_and_eval import train_and_eval
from src.utils.config import command_line_parser
import pandas as pd
import wandb

def get_default_config():
    return {
        "device": "cpu",
        "model":"gamba",
        "regularization":"simple",
        "readout": "add",
        "data":"PROTEINS",
        "epochs":10,
        "batch_size":32,
        "pe":"gnn",
        "laplacePE":0,
        "RW_length":0,
        "hidden_channel":64,
        "layers":4,
        "heads":4,
        "dropout":0.2,
        "seed":0,
        "verbose": False,
        "sample_transform": False,
        "add_virtual_node":True,
        "init_nodefeatures_dim": 128,
        "init_nodefeatures_strategy": "random",
        "wandb": False,
        "weight_decay": 0.01,
        "learning_rate": 1e-4,
        "scheduler": "None",
        "scheduler_patience": 16,
        "simon_gaa": False,
        "num_virtual_tokens": 4,
        "token_aggregation": "mean",
        "use_mamba": True,
        "patience": 20
    }

def run_experiments(config_files, num_trials=3):
    all_results = []

    # Initialize wandb once before the loop
    wandb_run = None
    if get_default_config()["wandb"]:
        wandb_run = wandb.init(
            entity="astinky",
            project="DL_Project",
            config=args.__dict__,
            name=args.name
        )

    try:
        for config_file in config_files:
            results = []  # Results for current config
            # Load the configuration
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            # Run multiple trials for each config
            for trial in range(num_trials):
                # Start with default config and update with file-specific settings
                full_config = get_default_config()
                full_config.update(config)
                
                # Update seed for each trial to ensure different random initializations
                full_config["seed"] = trial

                # Convert config to argparse.Namespace
                args = argparse.Namespace(**full_config)

                # Run training and evaluation
                train_loss, train_acc, val_loss, val_acc, test_loss, test_acc = train_and_eval(args)

                # Collect results
                results.append({
                    "Model": config["name"],
                    "Trial": trial + 1,
                    "Train_Loss": float(train_loss),
                    "Train_Accuracy": float(train_acc),
                    "Val_Loss": float(val_loss),
                    "Val_Accuracy": float(val_acc),
                    "Test_Loss": float(test_loss),
                    "Test_Accuracy": float(test_acc)
                })

            # Create a DataFrame for current config results
            df = pd.DataFrame(results)
            
            # Calculate statistics
            stats = df.groupby('Model').agg({
                'Train_Loss': ['mean', 'std'],
                'Train_Accuracy': ['mean', 'std'],
                'Val_Loss': ['mean', 'std'],
                'Val_Accuracy': ['mean', 'std'],
                'Test_Loss': ['mean', 'std'],
                'Test_Accuracy': ['mean', 'std']
            }).round(4)
            
            # Create formatted stats
            formatted_stats = pd.DataFrame(index=stats.index)
            for metric in ['Train_Loss', 'Train_Accuracy', 'Val_Loss', 'Val_Accuracy', 'Test_Loss', 'Test_Accuracy']:
                formatted_stats[metric] = (
                    stats[metric]['mean'].astype(str) + ' ± ' + 
                    stats[metric]['std'].astype(str)
                )
            
            # Print results for current config
            print(f"\nResults for {config['name']}:")
            print(df.to_markdown(index=False))
            print("\nStatistics:")
            print(formatted_stats.to_markdown(floatfmt='.4f'))

            # Save results for current config
            output_dir = "results"
            os.makedirs(output_dir, exist_ok=True)
            
            df.to_csv(os.path.join(output_dir, f"{config['name']}_individual_runs.csv"), index=False)
            formatted_stats.to_csv(os.path.join(output_dir, f"{config['name']}_summary_statistics.csv"))

    finally:
        if wandb_run is not None and wandb_run.state != 'finished':
            wandb_run.finish()

if __name__ == "__main__":
    # List of configuration files
    config_files = [
        "data/configs/archived_configs/stats_test.json",
        #"data/configs/sample_config2.json",
        #"data/configs/sample_config.json",
        #"data/configs/sample_config2.json",
        #"data/configs/sample_config3.json",
        #"data/configs/sample_config4.json"
    ]

    run_experiments(config_files, num_trials=3) 