import os
import time
import csv
from data import create_dataloader
from networks.trainer import Trainer
from options import TrainOptions
from validate import validate_model
import shutil
import torch
import random
import numpy as np
import argparse
from utils import *
import json

def set_seed(seed):
    """
    Fix random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    

def initialize_csv_file(csv_path):
    """Initialize CSV file with headers if it doesn't exist"""
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['total_steps', 'val_acer', 'test_acer', 'timestamp'])
        print(f"Created CSV file: {csv_path}")
    else:
        print(f"CSV file already exists: {csv_path}")

def log_to_csv(csv_path, total_steps, val_acer, test_acer):
    """Log validation results to CSV file"""
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([total_steps, val_acer, test_acer, timestamp])
    print(f"Logged to CSV: steps={total_steps}, val_acer={val_acer:.4f}, test_acer={test_acer:.4f}")

if __name__ == '__main__':
    seed = 42
    set_seed(seed)
    opt = TrainOptions().parse()
    results_dir = os.path.join(opt.name, opt.checkpoints_dir)
    
    # save_training_config(opt, results_dir, seed)
    # Set device
    device = torch.device(f'cuda:{opt.gpu_ids[0]}' if opt.gpu_ids else 'cpu')
    opt.device = device
    
    # Create model
    model = Trainer(opt)
    
    # Initialize total_steps if not present
    if not hasattr(model, 'total_steps'):
        model.total_steps = 0
    
    # Load checkpoint if specified
    start_epoch = 0
    start_step = 0
    if opt.resume:
        start_epoch, start_step = load_checkpoint(
            model, 
            opt.resume, 
            device=device
        )
        
        # If resume_epoch_only, reset step counter but keep model weights
        if opt.resume_epoch_only:
            print("Resume mode: epoch only - resetting step counter")
            start_step = 0
            model.total_steps = 0
    
    # Create data loader
    data_loader = create_dataloader(opt)
    
    # Setup result files and logging
    
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize CSV file for logging results
    results_csv_path = os.path.join(results_dir, "results.csv")
    initialize_csv_file(results_csv_path)

    
    # Training setup
    start_time = time.time()
    best_val_acer = float('inf')
    best_test_acer = None
    
    print(f"Length of data loader: {len(data_loader)}")
    
    if opt.resume:
        print(f"Resuming training from epoch {start_epoch}, step {start_step}")
    else:
        print("Starting training from scratch")
    
    
    epoch = 0
    count = 0
    
    for i, data in enumerate(data_loader):
        model.set_input(data)
        model.optimize_parameters()
            
        # Logging every 100 steps
        if model.total_steps % 100 == 0:
            current_loss = model.loss.item() if hasattr(model.loss, 'item') else model.loss
            print("Train loss: {:.6f} at step: {}".format(current_loss, model.total_steps))
            
            # Time per step
            elapsed_time = time.time() - start_time
            avg_time_per_step = elapsed_time / max(1, model.total_steps - start_step)
            print("Avg time per step: {:.4f}s".format(avg_time_per_step))

        if model.total_steps >= 1000 and model.total_steps % 1000 ==0:
            # Run validation
            print(f"\nRunning validation at step {model.total_steps}...")
            model.eval()
            
            val_metric = validate_model(
                model=model.model,
                device=device,
                step_or_epoch=model.total_steps,
                results_dir=results_dir,
                val_protocol_file='lists/Protocol-val_abs.txt',
                save_predictions_flag=False,
                crop_size=getattr(opt, 'cropSize', 224)
            )
            val_acer, val_threshold = val_metric.ACER, val_metric.threshold
    
            print(f"Validation ACER: {val_acer:.4f}")
            
            # Save checkpoint with optimizer state
            checkpoint_path = os.path.join(results_dir, 
                                            f'checkpoint_iters_{model.total_steps}_{val_acer}.pth')
            save_checkpoint(model, epoch, checkpoint_path, include_optimizer=False)
            
            model.train()
            print("Validation completed, resuming training...\n")
            
        model.total_steps += 1
