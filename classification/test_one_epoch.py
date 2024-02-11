from copy import copy
from matplotlib import pyplot as plt

import numpy as np
import torch
import tqdm
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             recall_score, roc_auc_score)
from utils import *


def test(model, test_loader, args, fold, threshold=0.5, device='cuda'):
    model.eval()

    test_predict_bi = []
    test_predict = []
    test_label = []
    test_img = []
    test_path = []

    bar = tqdm.tqdm(test_loader, total=len(test_loader), postfix={'exp' : args.experiments_name, 'state' : 'test', 'fold' : fold})

    for i, (data, label, path) in enumerate(bar):
        ## load data
        data, label = data.to(device), label.to(device)

        ## predict
        with torch.no_grad():
            predict = model(data)

        ## loss
        predict = torch.squeeze(predict, dim=-1)
        
        binary_predict = copy(predict)
        binary_predict = binary_predict.detach().cpu()
        binary_predict[binary_predict > threshold] = 1
        binary_predict[binary_predict <= threshold] = 0

        test_predict.append(predict.cpu().numpy())
        test_predict_bi.append(binary_predict.cpu().numpy())
        test_label.append(label.cpu().numpy())
        test_img.append(data.cpu().numpy())
        test_path.append(path)

    test_predict = np.concatenate(test_predict)
    test_predict_bi = np.concatenate(test_predict_bi)
    test_label = np.concatenate(test_label)
    test_img = np.concatenate(test_img)
    test_path = np.concatenate(test_path)

    test_acc = accuracy_score(test_label, test_predict_bi)
    test_recall = recall_score(test_label,test_predict_bi, zero_division=1)
    test_confuse = confusion_matrix(test_label, test_predict_bi, labels=[1, 0])
    test_spc =  test_confuse[1, 1]/(test_confuse[1, 0] + test_confuse[1, 1])

    try:
        test_auc = roc_auc_score(test_label, test_predict)
    except ValueError:
        test_auc = 0
        pass

    return test_acc, test_recall, test_spc, test_auc, test_confuse, test_predict_bi, test_label, test_predict, test_path
