import os
import argparse
import numpy as np
import random
from omegaconf import OmegaConf
from tqdm import tqdm
from sklearn import metrics
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from glob import glob
from pdb import set_trace as st
import torch
import cv2
from tqdm import tqdm

import models
from datasets import create_dataloader, RealFakeDataset
from utils import *

import warnings
warnings.filterwarnings("ignore")

def model_forward(inputs, model):
    output_1 = model(inputs)
    probs_1 = torch.softmax(output_1, dim=1)[:,1]
    prediction_1 = output_1.argmax(dim=1)
    return prediction_1, probs_1

def test(args, model, test_dataloader, filename, device):
    model.eval()
    y_preds = []
    img_paths = []
    results=[]
    for i, datas in enumerate(tqdm(test_dataloader)):
        with torch.no_grad():
            images = datas[0].to(device)
            predictions, probs, outputs = model_forward(images, model)
            
            # multi-head must specific one head
            if args.head is not None:
                y_preds.extend(probs[args.head-1])
                results.extend(outputs[args.head-1])
            # single-head 
            else:
                y_preds.extend(probs[0])
                results.extend(outputs[0])

            if args.test.record_results:
                img_paths.extend(datas[2])
    
    y_preds = torch.stack(y_preds).data.cpu().numpy()
    results = torch.stack(results).data.cpu().numpy()

    with open(filename, 'w') as f:
        for img_path, y_pred in zip(img_paths, y_preds):
            f.writelines(f'{img_path} {y_pred}\n')

def main():
    # set configs
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='configs/ICCV_Workshop/test.yaml')
    parser.add_argument('--ckpt', type=str, default='')
    parser.add_argument('--pred_file', type=str, default='results.txt')
    parser.add_argument('--distributed', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--local_rank', type=int, default=0)

    args = parser.parse_args()
    local_config = OmegaConf.load(args.config)
    for k, v in local_config.items():
        setattr(args, k, v)

    # set enviroment
    os.environ['TORCH_HOME'] = args.torch_home
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # define the output dir
    if args.ckpt:
        args.output_dir = os.path.dirname(args.ckpt)
    
    text_list = [['natural face','simulated mask']]

    # load model
    model = models.__dict__[args.model.name](args.model.params)
    if args.ckpt.endswith('.tar'):
        checkpoint = torch.load(args.ckpt, map_location='cpu')['state_dict']
        model.load_state_dict(checkpoint)
        print("load tar model successful!")
    elif args.ckpt.endswith('.pth'):
        model.load_model(args.ckpt)
    else:
        print(f"ERROR! ckpt {model_path} format error!")

    model = model.to(device)

    model.get_text_features(text_list)

    # dataloader
    val_dataloader = create_dataloader(args, split='val')
    test_dataloader = create_dataloader(args, split='test')

    val_probs = []
    val_path = []

    test_probs = []
    test_path = []

    model.eval()
    with torch.no_grad():
        for i,datas in enumerate(tqdm(val_dataloader)):
            [images, targets, img_path, images2] = datas
            batch_size = images.shape[0]

            images = images.to(device)
            images2 = images2.to(device)
            targets_1 = targets.to(device)
            # model forward
            probs_1 = model(images, images2)
            val_probs.extend(probs_1)
            val_path.extend(img_path)

        for i,datas in enumerate(tqdm(test_dataloader)):
            [images, targets, img_path, images2] = datas
            batch_size = images.shape[0]

            images = images.to(device)
            images2 = images2.to(device)
            targets_1 = targets.to(device)
            # model forward
            probs_1 = model(images, images2)
            
            test_probs.extend(probs_1)
            test_path.extend(img_path)


    result_list_path = args.output_dir+'/'+args.ckpt.split('/')[-1].split('.')[0]+'.txt'
    with open(result_list_path,'w') as f:
        ind = 0
        for i,j in zip(val_path, val_probs):
            _,folder,file = i.rsplit('/',2)
            index = int(file.split('.')[0])
            if ind == index:
                f.write(f'{os.path.join(folder,file)} {j}\n')
                ind+=1
        ind = 0
        for i,j in zip(test_path, test_probs):
            _,folder,file = i.rsplit('/',2)
            index = int(file.split('.')[0])
            if ind == index:
                f.write(f'{os.path.join(folder,file)} {j}\n')
                ind+=1

    print(f"{args.ckpt.split('/')[-1].split('.')[0]+'.txt'}")

if __name__ == '__main__':
    main()
