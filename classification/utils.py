import argparse
import os
import random

import albumentations as A
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
import torch
from albumentations.pytorch import ToTensorV2
from matplotlib import pyplot as plt
from scipy.ndimage import zoom


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

img_transform_train = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    ToTensorV2()
])


img_transform_valid = A.Compose([
    ToTensorV2()
])

class BCEFocalLoss(torch.nn.Module):
    def __init__(self, alpha=10, reduction='elementwise_mean'): 
        super().__init__()
        # self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, _input, target):
        pt = _input
        alpha = self.alpha
        # loss = - alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - (1 - alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)
        loss = - alpha ** (1 - pt)  * target * torch.log(pt) -  (1 - target) * torch.log(1 - pt)

        if self.reduction == 'elementwise_mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss

def one_hot(label):
    label = np.array(label)
    labels0 = np.zeros((label.shape[0]))
    labels1 = np.zeros((label.shape[0]))
    labels0[label==0] = 1
    labels1[label==1] = 1
    return torch.tensor(np.vstack((labels0, labels1)).T)


def save_checkpoint(model, fold, epoch, model_save_path, best_result=0, indicator='acc'):
    state_dict = model.state_dict()
    best_result = round(best_result, 4)
    if indicator == "latest":
        save_dict = {
            'fold': fold,
            'epoch': epoch,
            'state_dict': state_dict
            }
        filename = f'{model_save_path}/f{fold}_{indicator}_model.pt'
    else:
        if indicator == "acc":
            save_dict = {
                    'fold': fold,
                    'epoch': epoch,
                    'best_auc': best_result,
                    'state_dict': state_dict
                    }
        elif indicator == "auc":
            save_dict = {
                'fold': fold,
                'epoch': epoch,
                'best_auc': best_result,
                'state_dict': state_dict
                }
        filename = f'{model_save_path}/f{fold}_{indicator}_model.pt'

    torch.save(save_dict, filename)

def load_checkpoint(args, model, fold):
    checkpoint_path = f'{args.log_dir}/{args.experiments_name}/checkpoint/'
    pretrained_dir = os.path.join(checkpoint_path, f'f{fold}_{args.evaluation}_model.pt')

    checkpoint = torch.load(pretrained_dir)

    if 'state_dict' in checkpoint:
        model_dict = torch.load(pretrained_dir)["state_dict"]
        best_epoch = torch.load(pretrained_dir)["epoch"]
        model.load_state_dict(model_dict)
    else:
        print('checkpoint error!')

    return model, best_epoch

def save_test_result(test_log_dir, total_test_rec, total_test_spc, total_test_auc, total_test_acc, \
                     total_test_tp, total_test_fp, total_test_fn, total_test_tn, total_test_pred, test_label, test_path):
    '''
    保存每个fold test sample的预测概率值至 {test_log_dir}test_block_pred.csv
    0.5二值化， 保存每个fold 的各项指标至 {test_log_dir}test_thresh0.5_block_result.csv
    '''
    #####################################
    ## saving pred probability into csv 
    #####################################
    type = []
    name = []
    block_folder = []
    block = []
    for i in range(test_path.shape[0]):
        type.append(test_path[i].split('/')[-4])
        name.append(test_path[i].split('/')[-3])
        block_folder.append(test_path[i].split('/')[-2])
        block.append(test_path[i].split('/')[-1])
    type = np.array(type)
    name = np.array(name)
    block_folder = np.array(block_folder)
    block = np.array(block)

    pred_info = np.stack((test_label, type, name, block_folder, block, total_test_pred[0], total_test_pred[1], total_test_pred[2], total_test_pred[3], total_test_pred[4]), axis=1)
    df1 = pd.DataFrame(data=pred_info, columns=['label', 'type', 'name', 'block_fold', 'block', 'pred0', 'pred1', 'pred2', 'pred3', 'pred4'])
    df1.to_csv(f'{test_log_dir}test_block_pred.csv',index=False)

    ######################
    ## save fold info 
    ######################
    total_test_spc = np.array(total_test_spc)
    total_test_rec = np.array(total_test_rec)
    total_test_auc = np.array(total_test_auc)
    total_test_acc = np.array(total_test_acc)

    fold_pred_info_normalthresh = np.stack((total_test_auc, total_test_acc, total_test_rec, total_test_spc, total_test_tp, total_test_fp, total_test_fn, total_test_tn), axis=0)
    df2 = pd.DataFrame(data=fold_pred_info_normalthresh, columns=['indicator', 'fold0', 'fold1', 'fold2', 'fold3', 'fold4'])
    df2.to_csv(f'{test_log_dir}test_thresh0.5_block_result.csv', index=False)

def save_val_result(test_log_dir, total_test_val_info):
    '''
    保存每个fold test sample的预测概率值至 {test_log_dir}val_pred.csv
    '''
    total_info = np.vstack((total_test_val_info[0], total_test_val_info[1], total_test_val_info[2], total_test_val_info[3], total_test_val_info[4]))
    df1 = pd.DataFrame(data=total_info, columns=['label', 'fold', 'type', 'name', 'block_fold', 'block', 'pred'])
    df1.to_csv(f'{test_log_dir}val_pred.csv',index=False)


def opt_auc_save(test_log_dir, fold, label, predict, curve_auc):

    fpr, tpr, _ = roc_curve(label, predict)
    ## opt AUC curve
    line_width = 1  # 曲线的宽度

    plt.figure(figsize=(8, 5))  # 图的大小
    plt.plot(fpr, tpr, lw=line_width, label=f'AUC = {round(curve_auc, 4)}', color='red')
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.savefig(f'{test_log_dir}blocklevel_ROC_fold{fold}.jpg', dpi=256)# bbox_inches='tight', pad_inches=0, 


def Find_Optimal_Cutoff(TPR, FPR, threshold):
    y = TPR - FPR
    Youden_index = np.argmax(y)  # Only the first occurrence is returned.
    optimal_threshold = threshold[Youden_index]
    point = [FPR[Youden_index], TPR[Youden_index]]
    return optimal_threshold, point


def resample_to_size(npy_image, target_size, order=1):
    source_size = npy_image.shape
    scale = np.array(target_size) / source_size
    target_npy_image = zoom(npy_image, scale, order=order)
    return target_npy_image

def norm_img(image, window_width1=150, window_level1=45):  #liver : 窗宽：180--250HU，窗位：35--55HU
    window_width = window_width1 #窗宽
    window_level = window_level1 #窗位
    minWindow = window_level - window_width *0.5
    image= ((image-minWindow)/window_width)
    image[image<0] = 0
    image[image>1] = 1

    return image