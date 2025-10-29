"""
Data Setup Module

This module contains functions for:
- Creating train/test splits from ImageFolder datasets
- Creating PyTorch DataLoaders
"""

import os
import shutil
import random
from torchvision import datasets
from torch.utils.data import DataLoader


def create_train_test_split(source_dir, train_dir, test_dir, split=0.8, 
                            max_train_samples=None, reset_dirs=False):
    """
    Create train/test split from an ImageFolder-format dataset.
    
    Args:
        source_dir: Path to source ImageFolder dataset (with class subdirectories)
        train_dir: Path to output training directory
        test_dir: Path to output testing directory
        split: Train/test split ratio (default: 0.8 = 80% train, 20% test)
        max_train_samples: Maximum number of training samples per class (None = use all)
        reset_dirs: If True, delete and recreate train/test directories
    
    Returns:
        None (creates train/test directories with copied files)
    """
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    if reset_dirs:
        for d in (train_dir, test_dir):
            if os.path.exists(d):
                shutil.rmtree(d)
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    classes = sorted([d for d in os.listdir(source_dir) 
                     if os.path.isdir(os.path.join(source_dir, d))])
    
    for cls in classes:
        os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
        os.makedirs(os.path.join(test_dir, cls), exist_ok=True)
        
        cls_path = os.path.join(source_dir, cls)
        files = os.listdir(cls_path)
        random.shuffle(files)
        
        if max_train_samples:
            total_samples = int(max_train_samples / split)
            files = files[:total_samples]
        
        split_idx = int(len(files) * split)
        train_files = files[:split_idx]
        test_files = files[split_idx:]
        
        for f in train_files:
            shutil.copy(os.path.join(cls_path, f), os.path.join(train_dir, cls, f))
        
        for f in test_files:
            shutil.copy(os.path.join(cls_path, f), os.path.join(test_dir, cls, f))
        
        print(f"{cls}: {len(train_files)} train, {len(test_files)} test")


def create_dataloaders(train_dir, test_dir, transform, batch_size):
    """
    Create PyTorch DataLoaders from train/test ImageFolder directories.
    
    Args:
        train_dir: Path to training ImageFolder directory
        test_dir: Path to testing ImageFolder directory
        transform: torchvision transforms to apply
        batch_size: Batch size for DataLoader
    
    Returns:
        train_dataloader: DataLoader for training data
        test_dataloader: DataLoader for testing data
        class_names: List of class names
    """
    train_data = datasets.ImageFolder(train_dir, transform=transform)
    test_data = datasets.ImageFolder(test_dir, transform=transform)
    
    class_names = train_data.classes
    
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    return train_dataloader, test_dataloader, class_names
