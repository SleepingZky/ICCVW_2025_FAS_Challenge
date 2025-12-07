import os
import time
import shutil
import torch
import random
import numpy as np
import argparse


def load_checkpoint(model, checkpoint_path, device='cuda'):
    """
    Load checkpoint and restore training state
    
    Args:
        model: Trainer model instance
        checkpoint_path: Path to the checkpoint file
        device: Device to load the checkpoint on
    
    Returns:
        tuple: (start_epoch, start_step) or (0, 0) if loading fails
    """
    if not os.path.exists(checkpoint_path):
        print(f"Warning: Checkpoint {checkpoint_path} does not exist. Starting from scratch.")
        return 0, 0
    
    try:
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Print checkpoint structure for debugging
        print("Checkpoint keys:", list(checkpoint.keys()))
        
        # Load model state
        if 'model' in checkpoint:
            model.model.load_state_dict(checkpoint['model'])
            print("✓ Model state loaded successfully")
        else:
            print("Warning: No 'model' key found in checkpoint")
            return 0, 0
        
        # Load optimizer state
        if 'optimizer' in checkpoint and hasattr(model, 'optimizer'):
            try:
                model.optimizer.load_state_dict(checkpoint['optimizer'])
                print("✓ Optimizer state loaded successfully")
            except Exception as e:
                print(f"Warning: Failed to load optimizer state: {e}")
        
        # Load scheduler state
        if 'scheduler' in checkpoint and hasattr(model, 'scheduler'):
            try:
                model.scheduler.load_state_dict(checkpoint['scheduler'])
                print("✓ Scheduler state loaded successfully")
            except Exception as e:
                print(f"Warning: Failed to load scheduler state: {e}")
        
        # Load training progress
        start_epoch = checkpoint.get('epoch', 0)
        start_step = checkpoint.get('total_steps', 0)
        
        # Restore model's total_steps
        if hasattr(model, 'total_steps'):
            model.total_steps = start_step
            print(f"✓ Restored training progress: epoch {start_epoch}, step {start_step}")
        
        # Load other training states if available
        if 'current_step' in checkpoint and hasattr(model, 'current_step'):
            model.current_step = checkpoint['current_step']
            print(f"✓ Restored gradient accumulation step: {model.current_step}")
        
        return start_epoch, start_step
        
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("Starting training from scratch...")
        return 0, 0

def save_checkpoint(model, epoch, filepath, include_optimizer=True):
    """
    Enhanced save function that includes optimizer and training state
    
    Args:
        model: Trainer model instance
        epoch: Current epoch
        filepath: Path to save the checkpoint
        include_optimizer: Whether to include optimizer state
    """
    checkpoint = {
        'model': model.model.state_dict(),
        'epoch': epoch,
        'total_steps': model.total_steps if hasattr(model, 'total_steps') else 0,
    }
    
    # Include optimizer state for resuming
    if include_optimizer and hasattr(model, 'optimizer'):
        checkpoint['optimizer'] = model.optimizer.state_dict()
    
    # Include scheduler state for resuming
    if include_optimizer and hasattr(model, 'scheduler'):
        checkpoint['scheduler'] = model.scheduler.state_dict()
    
    # Include gradient accumulation state
    if hasattr(model, 'current_step'):
        checkpoint['current_step'] = model.current_step
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save checkpoint
    torch.save(checkpoint, filepath)
    print(f"✓ Checkpoint saved to: {filepath}")