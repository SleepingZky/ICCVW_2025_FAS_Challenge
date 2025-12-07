import torch
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler
from .custom_transforms import *
from .datasets import CustomBatchSampler, RealFakeDataset, custom_collate_fn

def get_bal_sampler(dataset):
    targets = []
    for d in dataset.datasets:
        targets.extend(d.targets)

    ratio = np.bincount(targets)
    w = 1. / torch.tensor(ratio, dtype=torch.float)
    sample_weights = w[targets]
    sampler = WeightedRandomSampler(weights=sample_weights,
                                    num_samples=len(sample_weights))
    return sampler

import random

def worker_init_fn(worker_id):
    """Initialize each worker process with a unique but deterministic seed"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_dataloader(opt, preprocess=None, return_dataset=False, worker_init_fn=None):
    shuffle = not opt.serial_batches if (opt.isTrain and not opt.class_bal) else False

    # Parse comma-separated VAE model paths to get the number of models
    vae_rec_paths = [path.strip() for path in opt.vae_rec_data_path.split(',')]
    fake_num = len(vae_rec_paths)
    
    batch_size = opt.batch_size // 6
    dataset = RealFakeDataset(opt)
    if '2b' in opt.arch:
        dataset.transform = preprocess
    sampler = get_bal_sampler(dataset) if opt.class_bal else None
    if return_dataset:
        return dataset
    
    # If no worker_init_fn provided, use the default deterministic one
    if worker_init_fn is None:
        def default_worker_init_fn(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)
        worker_init_fn = default_worker_init_fn
    
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=opt.num_threads,
        pin_memory=True,
        drop_last=opt.isTrain,
        collate_fn=custom_collate_fn,
        worker_init_fn=worker_init_fn  # Add this line
    )
    return data_loader

# def create_dataloader(opt, preprocess=None, return_dataset=False):
#     shuffle = not opt.serial_batches if (opt.isTrain and not opt.class_bal) else False

#     # Parse comma-separated VAE model paths to get the number of models
#     vae_rec_paths = [path.strip() for path in opt.vae_rec_data_path.split(',')]
#     fake_num = len(vae_rec_paths)
    
#     batch_size = opt.batch_size // 6
#     dataset = RealFakeDataset(opt)
#     if '2b' in opt.arch:
#         dataset.transform = preprocess
#     sampler = get_bal_sampler(dataset) if opt.class_bal else None
#     print(len(dataset))
#     if return_dataset:
#         return dataset
    
#     data_loader = torch.utils.data.DataLoader(dataset,
#                                             batch_size=batch_size,
#                                             shuffle=shuffle if sampler is None else False,
#                                             sampler=sampler,
#                                             num_workers=opt.num_threads,
#                                             pin_memory=True,
#                                             drop_last=opt.isTrain,
#                                             collate_fn=custom_collate_fn,)
#     return data_loader