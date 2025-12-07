import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import logging
from easydict import EasyDict
from sklearn.metrics import roc_curve, confusion_matrix

def compute_eer(y_trues, y_preds, threshold=None):
    """Compute Equal Error Rate and ACER metric"""
    metrics = EasyDict()
    
    if len(y_trues) == 0 or len(y_preds) == 0:
        return None
    
    # Calculate FPR, TPR, thresholds
    fpr, tpr, thresholds = roc_curve(y_trues, y_preds)
    fnr = 1 - tpr
    
    # Find EER threshold
    eer_threshold_index = np.nanargmin(np.absolute((fpr - fnr))) 
    if threshold is None:
        threshold = thresholds[eer_threshold_index]
    
    # Compute confusion matrix
    prediction = (np.array(y_preds) >= threshold).astype(int)
    cm = confusion_matrix(y_trues, prediction, labels=[0, 1])
    
    if cm.shape == (2, 2):
        TN, FP, FN, TP = cm.ravel()
    else:
        TN, FP, FN, TP = 0, 0, 0, 0
    
    # Compute ACER
    APCER = float(FP / (TN + FP)) if (TN + FP) > 0 else 0
    BPCER = float(FN / (FN + TP)) if (FN + TP) > 0 else 0
    metrics.ACER = (APCER + BPCER) / 2
    metrics.threshold = threshold
    
    return metrics

def load_protocol_data(protocol_file):
    """Load image paths and labels from protocol file"""
    if not os.path.exists(protocol_file):
        logging.error(f"Protocol file not found: {protocol_file}")
        return [], []
    
    image_paths = []
    labels = []
    
    with open(protocol_file, 'r') as f:
        lines = f.read().splitlines()
    
    for line in lines:
        if line.strip():
            parts = line.strip().split(' ')
            if len(parts) >= 2:
                image_path = parts[0]
                label = 0 if parts[1].startswith('0') else 1
                image_paths.append(image_path)
                labels.append(label)
    
    return image_paths, labels

class ValidationDataset(Dataset):
    """Simplified Dataset for validation - assumes all images are valid"""
    
    def __init__(self, image_paths, labels, crop_size=224):
        self.image_paths = image_paths
        self.labels = labels
        self.crop_size = int(crop_size)
        
        # Fixed transform
        self.transform = transforms.Compose([
            transforms.CenterCrop((self.crop_size, self.crop_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                               std=[0.26862954, 0.26130258, 0.27577711])
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def _center_crop_with_padding(self, img):
        """
        Center crop with padding for images smaller than crop_size
        """
        width, height = img.size
        
        # If image is smaller than crop_size in any dimension, pad it first
        if width < self.crop_size or height < self.crop_size:
            # Calculate padding needed
            pad_width = max(0, self.crop_size - width)
            pad_height = max(0, self.crop_size - height)
            
            # Pad the image (left, top, right, bottom)
            padding = (
                pad_width // 2,  # left
                pad_height // 2,  # top
                pad_width - pad_width // 2,  # right
                pad_height - pad_height // 2   # bottom
            )
            
            # Use reflection padding to maintain image content
            img = transforms.functional.pad(img, padding, fill=0, padding_mode='reflect')
        
        # Now perform center crop
        return transforms.functional.center_crop(img, self.crop_size)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        basename = os.path.basename(img_path)
        
        # Load and preprocess image (assuming all images are valid)
        image = Image.open(img_path).convert('RGB')
        image_tensor = self.transform(image)
        
        return image_tensor, basename, label

def run_inference(model, dataset, device, batch_size=32, num_workers=4):
    """Run batch inference using DataLoader and return predictions with basenames"""
    model.eval()
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False,  # Important: don't shuffle to maintain order
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False,
        drop_last=False
    )
    
    predictions = []
    basenames = []
    labels = []
    
    with torch.no_grad():
        for batch_images, batch_basenames, batch_labels in tqdm(dataloader, desc="Running inference"):
            batch_images = batch_images.to(device)
            
            # Run inference
            outputs = model(batch_images, return_feature=False)
            
            # Convert to probabilities
            probs = torch.sigmoid(outputs.squeeze()).cpu().numpy()
            
            # Handle single image case
            if probs.ndim == 0:
                probs = [float(probs)]
            else:
                probs = probs.tolist()
            
            # Store results
            predictions.extend(probs)
            basenames.extend(batch_basenames)
            labels.extend(batch_labels.tolist())
    
    model.train()
    
    return predictions, basenames, labels


def validate_model(model, device, step_or_epoch, results_dir, 
                  val_protocol_file="lists/Protocol-val_abs.txt",
                  save_predictions_flag=True, crop_size=224, batch_size=64, num_workers=8):
    """
    Main validation function using DataLoader
    
    Args:
        model: The model to validate
        device: Device to run inference on (torch.device or str)
        step_or_epoch: Training step or epoch number
        results_dir: Directory to save results
        val_protocol_file: Path to validation protocol file
        save_predictions_flag: Whether to save predictions
        crop_size: Image crop size (integer)
        batch_size: Batch size for inference
        num_workers: Number of worker processes for data loading
    
    Returns:
        tuple: (val_metrics, test_metrics)
    """
    
    # Validate inputs
    if not isinstance(crop_size, (int, float)):
        raise ValueError(f"crop_size must be a number, got {type(crop_size)}: {crop_size}")
    
    crop_size = int(crop_size)  # Ensure it's an integer
    
    # Ensure device is a proper device object
    if isinstance(device, str):
        device = torch.device(device)
    elif not isinstance(device, torch.device):
        raise ValueError(f"device must be a torch.device or string, got {type(device)}: {device}")
    
    # Load validation and test data
    val_paths, val_labels = load_protocol_data(val_protocol_file)
    
    if not val_paths:
        logging.error("Failed to load protocol data")
        return None
    
    # Create datasets
    val_dataset = ValidationDataset(val_paths, val_labels, crop_size)
    
    # Run inference on validation set
    val_predictions, val_basenames, val_labels_returned = run_inference(
        model=model, 
        dataset=val_dataset, 
        device=device,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    # Compute validation metrics
    val_metrics = compute_eer(val_labels_returned, val_predictions)
    if val_metrics is None:
        logging.error("Failed to compute validation metrics")
        return None, None
    
    logging.info(f"Validation ACER: {val_metrics.ACER:.4f}")
    
    
    return val_metrics


if __name__ == '__main__':
    # Test the validation functions
    import argparse