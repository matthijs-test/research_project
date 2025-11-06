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
    loss = results['train_loss']
    test_loss = results['test_loss']
    
    accuracy = results['train_acc']
    test_accuracy = results['test_acc']
    
    epochs = range(len(results['train_loss']))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Plot loss
    ax1.plot(epochs, loss, label='train_loss')
    ax1.plot(epochs, test_loss, label='test_loss')
    ax1.set_title('Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_xlim(0, len(epochs) - 1)
    ax1.set_ylim(bottom=0)
    ax1.legend()
    ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax1.grid(True, alpha=0.3)
    # Plot accuracy
    ax2.plot(epochs, accuracy, label='train_accuracy')
    ax2.plot(epochs, test_accuracy, label='test_accuracy')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_xlim(0, len(epochs) - 1)
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
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


