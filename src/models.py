"""
Models Module

This module contains functions for:
- Creating pre-trained Vision Transformer (ViT) models
- Creating pre-trained ResNet models
- Getting appropriate transforms for each model
"""

import torch
import torch.nn as nn
import torchvision


def get_vit_transforms():
    """
    Get the default transforms for Vision Transformer (ViT-B/16).
    
    Returns:
        Transform pipeline for ViT
    """
    weights = torchvision.models.ViT_B_16_Weights.DEFAULT
    return weights.transforms()


def create_vit_model(num_classes, device, freeze_backbone=True, unfreeze_patch_embed=False):
    """
    Create a Vision Transformer (ViT-B/16) model for transfer learning.
    
    Args:
        num_classes: Number of output classes
        device: Device to load model on ('cuda' or 'cpu')
        freeze_backbone: If True, freeze all layers except the head
        unfreeze_patch_embed: If True, also unfreeze the patch embedding layer
    
    Returns:
        model: ViT model ready for training
    """
    weights = torchvision.models.ViT_B_16_Weights.DEFAULT
    model = torchvision.models.vit_b_16(weights=weights).to(device)
    
    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False
    
    # Replace classification head
    model.heads = nn.Linear(in_features=768, out_features=num_classes).to(device)
    
    if unfreeze_patch_embed:
        for param in model.conv_proj.parameters():
            param.requires_grad = True
    
    return model


def get_resnet_transforms():
    """
    Get the default transforms for ResNet50.
    
    Returns:
        Transform pipeline for ResNet
    """
    weights = torchvision.models.ResNet50_Weights.DEFAULT
    return weights.transforms()


def create_resnet_model(num_classes, device, freeze_backbone=True):
    """
    Create a ResNet50 model for transfer learning.
    
    Args:
        num_classes: Number of output classes
        device: Device to load model on ('cuda' or 'cpu')
        freeze_backbone: If True, freeze all layers except the final FC layer
    
    Returns:
        model: ResNet50 model ready for training
    """
    weights = torchvision.models.ResNet50_Weights.DEFAULT
    model = torchvision.models.resnet50(weights=weights).to(device)
    
    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False
    
    # Replace final fully connected layer
    model.fc = nn.Linear(in_features=2048, out_features=num_classes).to(device)
    
    return model
