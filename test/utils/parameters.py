import json
import argparse
from omegaconf import OmegaConf
import torch


def get_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='configs/train.yaml')
    parser.add_argument('--distributed', type=int, default=1)
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--sync-bn', action='store_true', default=False)
    parser.add_argument('--debug', action='store_true', default=False)
    args = parser.parse_args()

    _C = OmegaConf.load(args.config)
    _C.merge_with(vars(args))

    if _C.debug:
        _C.train.epochs = 2

    return _C


if __name__ == '__main__':
    args = get_parameters()
    print(args)
