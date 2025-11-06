"""
Experiments Module

This module contains wrapper functions for running complete training experiments.
"""

import os
import time
import data_setup as ds
import engine as train_module
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import json
import torch


def run_experiment(data_path,
                   model,
                   optimizer,
                   loss_fn,
                   transform,
                   device,
                   n_samples=None,
                   batch_size=32,
                   epochs=5,
                   split=0.8):
    """
    Run a complete training experiment.
    
    Args:
        data_path: Path to source dataset
        model: PyTorch model
        optimizer: Optimizer
        loss_fn: Loss function
        transform: Data transforms
        device: Device to train on
        n_samples: Max samples per class (None = use all)
        batch_size: Batch size
        epochs: Number of epochs
        split: Train/test split ratio
    
    Returns:
        results: Dictionary with training metrics and time
    """
    train_dir = "../experiments/runs/train_temp"
    test_dir = "../experiments/runs/test_temp"
    
    ds.create_train_test_split(
        source_dir=data_path,
        train_dir=train_dir,
        test_dir=test_dir,
        split=split,
        max_train_samples=n_samples,
        reset_dirs=True
    )
    
    train_dataloader, test_dataloader, class_names = ds.create_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        transform=transform,
        batch_size=batch_size
    )
    
    print(f"Classes: {class_names}")
    
    start_time = time.time()
    
    results = train_module.train(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer,
        device=device,
        loss_fn=loss_fn,
        epochs=epochs
    )
    
    end_time = time.time()
    training_time = end_time - start_time
    
    results["training_time"] = training_time
    print(f"\nTraining time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    
    return results






def run_experiment_2(config,
                   model,
                   optimizer,
                   loss_fn,
                   transform,
                   device,
                   save_results=False,
                   save_model=False):
    """
    Run a complete training experiment with optional saving.
    
    Args:
        config: Dictionary with experiment configuration:
                - experiment_name: Name for the experiment
                - data_path: Path to source dataset
                - n_samples: Max samples per class (None = use all)
                - batch_size: Batch size
                - epochs: Number of epochs
                - split: Train/test split ratio
        model: PyTorch model
        optimizer: Optimizer
        loss_fn: Loss function
        transform: Data transforms
        device: Device to train on
        save_results: If True, save config/results and use tensorboard
        save_model: If True, also save model weights (requires save_results=True)
    
    Returns:
        results: Dictionary with training metrics and time
    """
    # Extract config
    experiment_name = config.get("experiment_name", "unnamed_experiment")
    data_path = config["data_path"]
    n_samples = config.get("n_samples", None)
    batch_size = config.get("batch_size", 32)
    epochs = config.get("epochs", 5)
    split = config.get("split", 0.8)
    
    # Setup tensorboard if saving
    writer = None
    if save_results:
        log_dir = f"../experiments/tensorboard/{experiment_name}"
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)
    
    # Create train/test split
    train_dir = "../experiments/runs/train_temp"
    test_dir = "../experiments/runs/test_temp"
    
    ds.create_train_test_split(
        source_dir=data_path,
        train_dir=train_dir,
        test_dir=test_dir,
        split=split,
        max_train_samples=n_samples,
        reset_dirs=True
    )
    
    train_dataloader, test_dataloader, class_names = ds.create_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        transform=transform,
        batch_size=batch_size
    )
    
    print(f"Classes: {class_names}")
    
    # Train
    start_time = time.time()
    
    results = train_module.train_tensorboard(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer,
        device=device,
        loss_fn=loss_fn,
        epochs=epochs,
        writer=writer  # None if not saving, SummaryWriter if saving
    )
    
    end_time = time.time()
    training_time = end_time - start_time
    results["training_time"] = training_time
    
    print(f"\nTraining time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    
    # Save results if requested
    if save_results:
        results_dir = f"../experiments/results/{experiment_name}"
        os.makedirs(results_dir, exist_ok=True)
        
        # Save config
        with open(f"{results_dir}/config.json", "w") as f:
            json.dump(config, f, indent=4)
        
        # Save results
        with open(f"{results_dir}/results.json", "w") as f:
            json.dump(results, f, indent=4)
        
        print(f"Saved config and results to {results_dir}")
        
        # Save model if requested
        if save_model:
            model_path = f"{results_dir}/model.pth"
            torch.save(model.state_dict(), model_path)
            print(f"Saved model to {model_path}")
        
        # Add training curves to tensorboard
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        epochs_range = range(len(results['train_loss']))
        
        ax1.plot(epochs_range, results['train_loss'], label='train_loss')
        ax1.plot(epochs_range, results['test_loss'], label='test_loss')
        ax1.set_title('Loss')
        ax1.set_xlabel('Epochs')
        ax1.legend()
        ax1.set_ylim(bottom=0)

        
        ax2.plot(epochs_range, results['train_acc'], label='train_accuracy')
        ax2.plot(epochs_range, results['test_acc'], label='test_accuracy')
        ax2.set_title('Accuracy')
        ax2.set_xlabel('Epochs')
        ax2.legend()
        ax2.set_ylim(bottom=0)
        
        writer.add_figure('Training Curves', fig)
        plt.close(fig)
        writer.close()
    
    return results