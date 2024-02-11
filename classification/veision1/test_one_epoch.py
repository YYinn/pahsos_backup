from copy import copy
from matplotlib import pyplot as plt

import numpy as np
import torch
import tqdm
from sklearn.metrics import (accuracy_score, auc, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from utils import *


def test(model, test_loader, args, fold, opt_thresh=-1, mod='test_val', threshold=0.5, device='cuda'): #mod = test_val / test_test
    model.eval()

    test_predict_bi = []
    test_predict = []
    test_label = []
    test_auc = []
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

    test_predict_bi = np.concatenate(test_predict_bi)
    test_predict = np.concatenate(test_predict)
    test_label = np.concatenate(test_label)
    test_img = np.concatenate(test_img)
    test_path = np.concatenate(test_path)

    test_acc = accuracy_score(test_label, test_predict_bi)
    test_f1 = f1_score(test_label, test_predict_bi, zero_division=1)
    test_pre = precision_score(test_label, test_predict_bi, zero_division=1)
    test_recall = recall_score(test_label,test_predict_bi, zero_division=1)
    test_confuse = confusion_matrix(test_label, test_predict_bi, labels=[1, 0])
    test_spc =  test_confuse[1, 1]/(test_confuse[1, 0] + test_confuse[1, 1])
    try:
        test_auc = roc_auc_score(test_label, test_predict)
    except ValueError:
        test_auc = 0
        pass
    
    fpr, tpr, threshold = roc_curve(test_label, test_predict)
    if mod == 'test_val':
        '''
        val 用于计算最优block阈值
        '''
        optimal_thresh, point = Find_Optimal_Cutoff(tpr, fpr, threshold)

        opt_bi = copy(test_predict)
        opt_bi[opt_bi > optimal_thresh] = 1
        opt_bi[opt_bi <= optimal_thresh] = 0

        opt_acc = accuracy_score(test_label, opt_bi)

        return test_acc, test_f1, test_pre, test_recall, test_auc, \
                        test_predict_bi, test_label, test_predict, test_path, fpr, tpr, opt_acc, optimal_thresh, point

    else:
        '''
        test 使用val计算的最优阈值计算opt指标
        '''
        opt_bi = copy(test_predict)
        opt_bi[opt_bi > opt_thresh] = 1
        opt_bi[opt_bi <= opt_thresh] = 0

        opt_f1 = f1_score(test_label, opt_bi, zero_division=1)
        opt_pre = precision_score(test_label, opt_bi, zero_division=1)
        opt_recall = recall_score(test_label, opt_bi, zero_division=1)
        opt_confuse = confusion_matrix(test_label, opt_bi, labels=[1,0])
        opt_spc =  opt_confuse[1, 1]/(opt_confuse[1, 0] + opt_confuse[1, 1])

        opt_acc = accuracy_score(test_label, opt_bi)

        return test_acc, test_f1, test_pre, test_recall, test_spc, test_confuse, test_auc, \
                        test_predict_bi, test_label, test_predict, test_path, opt_acc, opt_f1, opt_pre, opt_recall, opt_spc, opt_confuse



    # ######################### bag result ########################################
    # patient_total_pred = []
    # patient_total_label = []
    # total_patient_name = []
    # patient_total_acc_bi = []
    # indd = 0
    # patient_count = 0
    # while indd < len(test_label):
        
    #     patient_name = test_path[indd].split('/')[-5]
    #     if len(total_patient_name) == 0 or patient_name != total_patient_name[patient_count-1]:
    #         total_patient_name.append(patient_name)
    #         patient_total_label.append(test_label[indd])
    #         block_num = 12
    #         patient_pred = 0
    #         patient_count += 1

    #     for block_index_perpatient in range(block_num):
    #         patient_pred = patient_pred + test_predict[indd]
    #         indd += 1

    #     patient_total_pred.append(patient_pred/12.0)

    #     if patient_pred/12.0 > 0.5:
    #         patient_total_acc_bi.append(1)
    #     else:
    #         patient_total_acc_bi.append(0)
        
    # test_total_auc = roc_auc_score(patient_total_label, patient_total_pred)
    # test_total_acc = accuracy_score(patient_total_label, patient_total_acc_bi)
    # print('total_test_patient', len(total_patient_name), 'test_total_auc', test_total_auc, 'test_total_acc', test_total_acc)
     