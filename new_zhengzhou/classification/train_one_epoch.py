import os
from copy import copy

import numpy as np
import torch
import torch.nn as nn
import tqdm
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from utils import *


def train(model, train_loader, args, optimizer, criterion, fold, epoch, threshold=0.5, device = 'cuda'):
    model.train()

    total_loss = []
    total_acc = []
    total_f1 = []
    total_pre = []
    total_recall = []

    total_label = []
    total_predict = []
    total_predict_binary = []
    total_auc = []

    bar = tqdm.tqdm(train_loader, total=len(train_loader), postfix={'exp' : f'{args.note}{args.block_size}', 'state' : 'training', 'epoch' : epoch, 'fold' : fold})

    for batch_id, (data, label, _) in enumerate(bar):
        ## load data
        data, label = data.to(device), label.to(device)

        ## predict
        optimizer.zero_grad()
        predict = model(data)
        
        ## loss 
        predict = torch.squeeze(predict, dim=-1)
        loss = criterion(predict, label)
        loss.backward()
        optimizer.step()
        
        ## binary predict
        binary_predict = copy(predict)
        binary_predict = binary_predict.detach().cpu()
        binary_predict[binary_predict > threshold] = 1
        binary_predict[binary_predict <= threshold] = 0

        total_loss.append(loss.cpu().item())
        total_label.append(label.cpu().numpy())
        total_predict.append(predict.detach().cpu())
        total_predict_binary.append(binary_predict.numpy())

    total_label = np.concatenate(total_label)
    total_predict = np.concatenate(total_predict)
    total_predict_binary = np.concatenate(total_predict_binary)
    
    total_acc = accuracy_score(total_label, total_predict_binary)
    total_f1 = f1_score(total_label, total_predict_binary, zero_division=1)
    total_pre = precision_score(total_label, total_predict_binary, zero_division=1)
    total_recall = recall_score(total_label, total_predict_binary, zero_division=1)

    try:
        total_auc = roc_auc_score(total_label, total_predict)
    except ValueError:
        total_auc = 0
        pass

    return np.mean(total_loss, axis=0), total_acc, total_f1, total_pre, total_recall, total_auc
