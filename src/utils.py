"""
Utility Functions Module

This module contains helper functions for:
- Setting random seeds for reproducibility
- Plotting training curves
"""

import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List
import os


def set_seed(seed=42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value (default: 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def plot_loss_curves(results: Dict[str, List[float]]):
    """
    Plot training curves of a results dictionary.
    
    Args:
        results: Dictionary containing lists of values, e.g.
                 {"train_loss": [...],
                  "train_acc": [...],
                  "test_loss": [...],
                  "test_acc": [...]}
    """
    # Get the loss values
    loss = results['train_loss']
    test_loss = results['test_loss']
    
    # Get the accuracy values
    accuracy = results['train_acc']
    test_accuracy = results['test_acc']
    
    epochs = range(len(results['train_loss']))
    
    # Setup a plot
    plt.figure(figsize=(15, 7))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='train_loss')
    plt.plot(epochs, test_loss, label='test_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label='train_accuracy')
    plt.plot(epochs, test_accuracy, label='test_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    
    plt.show()



def save_model(model, save_path):
    """
    Save model state dict.
    
    Args:
        model: PyTorch model
        save_path: Path to save the model (e.g., 'model.pth')
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")