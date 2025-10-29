"""
Experiments Module

This module contains wrapper functions for running complete training experiments.
"""

import os
import time
import data_setup as ds
import engine as train_module


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
