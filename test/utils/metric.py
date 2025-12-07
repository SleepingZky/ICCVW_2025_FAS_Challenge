import numpy as np
from easydict import EasyDict
from sklearn.metrics import roc_curve, auc, confusion_matrix
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from pdb import set_trace as st
import os


def find_best_threshold(y_trues, y_preds):
    print("Finding best threshold...")
    best_thre = 0.5
    best_metrics = None
    candidate_thres = list(np.unique(np.sort(y_preds)))
    for thre in candidate_thres:
        metrics = cal_metrics(y_trues, y_preds, threshold=thre)
        if best_metrics is None:
            best_metrics = metrics
            best_thre = thre
        elif metrics.ACER < best_metrics.ACER:
            best_metrics = metrics
            best_thre = thre
    print(f"Best threshold is {best_thre}")
    return best_thre, best_metrics

def compute_eer(y_trues, y_preds, threshold=None):
    metrics = EasyDict()
    print("Finding threshold via EER...")
    # 计算 FPR, TPR, thresholds
    fpr, tpr, thresholds = roc_curve(y_trues, y_preds)
    fnr = 1 - tpr
    # 找到 FPR 和 FNR 最接近的点
    eer_threshold_index = np.nanargmin(np.absolute((fpr-fnr))) 
    metrics.EER = (fpr[eer_threshold_index] + fnr[eer_threshold_index]) / 2
    if threshold is None:
        threshold = thresholds[eer_threshold_index]

    metrics.AUC = auc(fpr, tpr)
    prediction = (np.array(y_preds) >= threshold).astype(int)
    TN, FP, FN, TP = confusion_matrix(y_trues, prediction, labels=[0, 1]).ravel()

    metrics.ACC = (TP + TN) / len(y_trues)
    APCER = float(FP / (TN + FP))
    BPCER = float(FN / (FN + TP))
    metrics.ACER = (APCER + BPCER) / 2
    metrics.threshold = threshold
    return metrics  


def cal_metrics(y_trues, y_preds, threshold=0.5):
    metrics = EasyDict()
    
    fpr, tpr, thresholds = roc_curve(y_trues, y_preds)
    metrics.AUC = auc(fpr, tpr)
    
    metrics.EER = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    metrics.Thre = float(interp1d(fpr, thresholds)(metrics.EER))

    if threshold == 'best':
        _, best_metrics = find_best_threshold(y_trues, y_preds)
        return best_metrics

    elif threshold == 'auto':
        threshold = metrics.Thre

    prediction = (np.array(y_preds) > threshold).astype(int)
    
    TN, FP, FN, TP = confusion_matrix(y_trues, prediction, labels=[0, 1]).ravel()
    
    metrics.ACC = (TP + TN) / len(y_trues)
    metrics.TP_rate = float(TP / (TP + FN))
    metrics.TN_rate = float(TN / (TN + FP))
    metrics.APCER = float(FP / (TN + FP))
    metrics.BPCER = float(FN / (FN + TP))
    metrics.ACER = (metrics.APCER + metrics.BPCER) / 2

    for k in metrics.keys():
        metrics[k] *= 100

    return metrics


if __name__ == '__main__':
    import argparse
    from tqdm import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--probs', type=str, default='./checkpoint-2.txt')
    parser.add_argument('-t1', '--labels_val', type=str, default='./Protocol-val.txt')
    parser.add_argument('-t2', '--labels_test', type=str, default='./Protocol-test.txt')
    parser.add_argument('-f', '--flag', type=int, default=0)
    args = parser.parse_args()
    flag = args.flag
    y_probs = [line.split(' ') for line in open(args.probs).read().splitlines()]
    y_probs_val = [float(i[1]) for i in y_probs if 'val' in i[0]]
    y_probs_test = [float(i[1]) for i in y_probs if 'test' in i[0]]

    y_trues_str_val = [str(line.split(' ')[1]) for line in open(args.labels_val).read().splitlines()]
    y_trues_val = []
    for i in y_trues_str_val:
        if i[0] == '0':
            y_trues_val.append(0)
        else:
            y_trues_val.append(1)

    y_trues_str_test = [str(line.split(' ')[1]) for line in open(args.labels_test).read().splitlines()]
    y_trues_test = []

    for i in y_trues_str_test:
        if i[0] == '0':
            y_trues_test.append(0)
        else:
            y_trues_test.append(1)

    
    if flag == 1:
        # 只测试生成
        y_probs_test = [1]*2331+y_probs_test[2331:]
    elif flag == 2:
        y_probs_test = y_probs_test[0:2331]+[1]*(8605-2330)+y_probs_test[8606:]

    assert len(y_probs_test) == len(y_trues_test)
    assert len(y_probs_val) == len(y_trues_val)


    print(f'Total val numbers: {len(y_probs_val)}')
    print(f'Total test numbers: {len(y_probs_test)}')
    metrics = compute_eer(y_trues_val, y_probs_val)
    print(metrics)
    threshold = metrics.threshold

    metrics = compute_eer(y_trues_test, y_probs_test, threshold)
    print(metrics)

    error_file_folder = args.probs.rsplit('.',1)[0]
    if not os.path.exists(error_file_folder):
        os.makedirs(error_file_folder)
    val_folder = './phase1/'
    test_folder = './phase2/'
    y_path_val = [f'{val_folder}{i[0]}' for i in y_probs if 'val' in i[0]]
    y_path_test = [f'{test_folder}{i[0]}' for i in y_probs if 'test' in i[0]]
    
    with open(os.path.join(error_file_folder,'val_real_error.txt'),'w') as f:
        for path,probs,labels in zip(y_path_val,y_probs_val,y_trues_val):
            if labels == 0:
                if probs >= threshold:
                    f.write(f'{path} {probs}\n')
    
    with open(os.path.join(error_file_folder,'val_attack_error.txt'),'w') as f:
        for path,probs,labels in zip(y_path_val,y_probs_val,y_trues_val):
            if labels == 1:
                if probs < threshold:
                    f.write(f'{path} {probs}\n')

    with open(os.path.join(error_file_folder,'test_real_error.txt'),'w') as f:
        for path,probs,labels in zip(y_path_test,y_probs_test,y_trues_test):
            if labels == 0:
                if probs >= threshold:
                    f.write(f'{path} {probs}\n')
    
    with open(os.path.join(error_file_folder,'test_attack_error.txt'),'w') as f:
        for path,probs,labels in zip(y_path_test,y_probs_test,y_trues_test):
            if labels == 1:
                if probs < threshold:
                    f.write(f'{path} {probs}\n')

            

