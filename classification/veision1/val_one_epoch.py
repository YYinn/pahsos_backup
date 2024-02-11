import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
from copy import copy
import tqdm
from utils import *

def val(model, test_loader, args, criterion, fold, epoch, threshold=0.5, device='cuda'):
    model.eval()

    acc = []
    f1score = []
    precision = []
    recall = []
    total_loss = []
    test_predict_bi = []
    test_predict = []
    test_label = []
    test_fea = []
    test_auc = []
    test_img = []
    test_path = []

    bar = tqdm.tqdm(test_loader, total=len(test_loader), postfix={'exp' : args.note, 'state' : 'validating', 'epoch' : epoch, 'fold' : fold})

    for i, (data, label, path) in enumerate(bar):
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
        test_path.append(path)

    # print('test', test_label, test_predict, test_predict_bi)
    test_predict_bi = np.concatenate(test_predict_bi)
    test_predict = np.concatenate(test_predict)
    test_label = np.concatenate(test_label)
    test_img = np.concatenate(test_img)
    test_path = np.concatenate(test_path)

    test_acc = accuracy_score(test_label, test_predict_bi)
    test_f1 = f1_score(test_label, test_predict_bi, zero_division=1)
    test_pre = precision_score(test_label, test_predict_bi, zero_division=1)
    test_recall = recall_score(test_label,test_predict_bi, zero_division=1)
    try:
        test_auc = roc_auc_score(test_label, test_predict)
    except ValueError:
        test_auc = 0
        pass

    return np.mean(total_loss, axis=0), test_acc, test_f1, test_pre, test_recall, test_auc, \
                    test_predict_bi, test_label, test_predict#, test_total_auc, test_total_acc, patient_total_label, patient_total_pred


    ######################### bag result ########################################

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
    #         if patient_name == 'XU-ZHONGSHENG':
    #             block_num = 10
    #         elif patient_name in ('TIAN-GUIFANG', 'ZHOU-AIMEI', 'ZHUANG-MUGEN'):
    #             block_num = 11
    #         else:
    #             block_num = 12
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
    
