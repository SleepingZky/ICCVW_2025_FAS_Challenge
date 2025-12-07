try:
    from .ICCV_RealFake_dataset import RealFakeDataset, RealFakeDataset_Val
except Exception:
    from ICCV_RealFake_dataset import RealFakeDataset, RealFakeDataset_Val

import torch.utils.data as data
from pdb import set_trace as st
import torch
import numpy as np
import random

def create_dataloader(args, split):
    kwargs = getattr(args.dataset, args.dataset.name)
    kwargs.update(eval(f'args.{split}.dataset'))
    
    if split == 'train':
        dataset = eval(args.dataset.name)(args, **kwargs)
    else:
        dataset = RealFakeDataset_Val(args, split=split, **kwargs)

    sampler = None
    if args.distributed:
        sampler = data.distributed.DistributedSampler(dataset)

    shuffle = True if sampler is None and split == 'train' else False
    batch_size = getattr(args, split).batch_size // 6

    if split == 'train':
        drop = True
    else:
        drop = False

    dataloader = data.DataLoader(dataset,
                                 batch_size=batch_size,
                                 shuffle=shuffle,
                                 sampler=sampler,
                                 num_workers=args.num_workers,
                                 pin_memory=False,
                                 drop_last=drop)
    return dataloader

def set_seed(SEED):
    if SEED:
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    import argparse
    from omegaconf import OmegaConf
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='./config/ICCV_Workshop/train.yaml')
    parser.add_argument('--distributed', type=int, default=0)
    args = parser.parse_args()
    args.local_rank = 0
    set_seed(1234)

    local_config = OmegaConf.load(args.config)
    for k, v in local_config.items():
        setattr(args, k, v)
    # args = vars(args)

    train_ = create_dataloader(args, split='train')
    for i, datas in enumerate(train_):
        for k,v in datas.items():
            print(v.shape) if type(v) is torch.Tensor else len(v)
            if type(v) is torch.Tensor:
                print(v.mean((1,2,3)))
            else:
                if isinstance(v, list):
                    for sub_v in v:
                        if type(sub_v) is torch.Tensor:
                            print(sub_v.mean((1,2,3)))
        break