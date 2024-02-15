import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
from copy import copy
import tqdm
from utils import *

def val(model, test_loader, args, criterion, fold, epoch, threshold=0.5, device='cuda'):
    model.eval()

    total_loss = []
    test_predict_bi = []
    test_predict = []
    test_label = []
    test_auc = []
    test_img = []

    bar = tqdm.tqdm(test_loader, total=len(test_loader), postfix={'exp' : args.note, 'state' : 'validating', 'epoch' : epoch, 'fold' : fold})

    for i, (data, label, _) in enumerate(bar):
        ## load data
        data, label = data.to(device), label.to(device)

        ## predict
        with torch.no_grad():
            predict = model(data)

        ## loss
        predict = torch.squeeze(predict, dim=-1)
        loss = criterion(predict, label)
        
        binary_predict = copy(predict)
        binary_predict = binary_predict.detach().cpu()
        binary_predict[binary_predict > threshold] = 1
        binary_predict[binary_predict <= threshold] = 0

        total_loss.append(loss.cpu().item())
        test_predict.append(predict.cpu().numpy())
        test_predict_bi.append(binary_predict.cpu().numpy())
        test_label.append(label.cpu().numpy())
        test_img.append(data.cpu().numpy())

    test_predict_bi = np.concatenate(test_predict_bi)
    test_predict = np.concatenate(test_predict)
    test_label = np.concatenate(test_label)
    test_img = np.concatenate(test_img)

    test_acc = accuracy_score(test_label, test_predict_bi)
    test_f1 = f1_score(test_label, test_predict_bi, zero_division=1)
    test_pre = precision_score(test_label, test_predict_bi, zero_division=1)
    test_recall = recall_score(test_label,test_predict_bi, zero_division=1)
    try:
        test_auc = roc_auc_score(test_label, test_predict)
    except ValueError:
        test_auc = 0
        pass

    return np.mean(total_loss, axis=0), test_acc, test_f1, test_pre, test_recall, test_auc

