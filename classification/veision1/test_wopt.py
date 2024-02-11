import argparse
import datetime
import logging
import os
import random
import sys
import warnings

import numpy as np
import pandas as pd
import torch
import wandb
from dataloader import *
from models.mymodel import *
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from train_one_epoch import train
from val_one_epoch import val
from test_one_epoch import test
from utils import *

warnings.filterwarnings('ignore')

############################
######## set config ########
############################
parser = argparse.ArgumentParser(description=' ')
# parser.add_argument('--lr', default=3e-5, type=float, help='learning rate')
# parser.add_argument('--num_epochs', default=100, type=int, help='training epoch')
parser.add_argument('--batch_size', default=16, type=int, help='batch size ')
parser.add_argument('--fold', type=int, default=5, help="fold")
parser.add_argument('--seed', type=int, default=147, help='seed')
# parser.add_argument('--optim', type=str, default='SGD', help='use SGD or Adam to optimize')

parser.add_argument('--block_size', type=int, default=128, help='input_size for ori img cropping')
parser.add_argument('--input_size', type=int, default=64, help='input_size for model (after resize block_size data), fix for specific model')
parser.add_argument('--root_path', type=str, default='/mnt/ExtData/pahsos/Data/preprocessed', help='original data root')
parser.add_argument('--json_path', type=str, default='/mnt/ExtData/pahsos/Data/data_split.json', help='json path for data and label loading')
parser.add_argument('--input_type', type=int, default=1, help='1 = 12 block intput | 2 = whole img')
parser.add_argument('--masked', type=bool, default=True, help='use masked img or not')
parser.add_argument('--samples_per_patient', type=int, default=12, help='sampling x blocks for one person') 

parser.add_argument('--log_dir', type=str, default='/mnt/ExtData/pahsos/classification/log', help='original data root')
parser.add_argument('--note', type=str, default='_', help='add on log dir')

parser.add_argument('--experiments_name', type=str, default='s12_block128_maskedTrue_es50_2023-02-05T09:42:27')
parser.add_argument('--evaluation', type=str, default='acc', help='auc or acc or latest')
args = parser.parse_args()

## log
time = datetime.datetime.now().isoformat()[:19]
if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)

test_log_dir = f'{args.log_dir}/{args.experiments_name}/test_new/'
if not os.path.exists(test_log_dir):
    os.makedirs(test_log_dir)

# logging.info('=================================================')
# logging.info('=================== new test ====================')
# logging.info('=================================================')
logging.basicConfig(level=logging.INFO,
                    handlers=[logging.FileHandler(f'{test_log_dir}test.log'), logging.StreamHandler(sys.stdout)])
logging.info(str(args))

## seed
seed_torch(seed=args.seed)

## data_list 
ori_patient_list, ori_label_list = get_data_list(args, key='train')
test_patient_list, test_label_list = get_data_list(args, key='test')

test_block_list = []
test_block_label_list = []
for l in range(0, len(test_patient_list)):
    patient_dir = os.path.join(test_patient_list[l], f'block{args.block_size}_masked{args.masked}/')
    for k in range(1, args.samples_per_patient+1):
        if os.path.exists(os.path.join(patient_dir, f'{k}.npy')):
            test_block_list.append(os.path.join(patient_dir, f'{k}.npy'))
            test_block_label_list.append(test_label_list[l])
        else:
            print(f'🔴 Sampling error: {patient_dir}{k}.npy not exist!')


## kfold, load 12 blocks
sfk = StratifiedKFold(n_splits=args.fold, shuffle=True, random_state=args.seed) 

test_val = []
test_test = []

total_test_val_info = []

total_test_auc = ['auc']
total_test_acc = ['acc']
total_test_rec = ['rec']
total_test_pre = ['pre']
total_test_f1 = ['f1']
total_test_spc = ['spc']

total_test_tp = ['tp']
total_test_fp = ['fp']
total_test_fn = ['fn']
total_test_tn = ['tn']

total_test_pred = []
total_opt_thresh = ['opt_thresh']


total_opt_acc = ['opt_acc']
total_opt_f1 = ['opt_f1']
total_opt_pre = ['opt_pre']
total_opt_rec = ['opt_rec']
total_opt_point = ['opt_point']
total_opt_spc = ['opt_spc']

total_opt_tp = ['opt_tp']
total_opt_fp = ['opt_fp']
total_opt_fn = ['opt_fn']
total_opt_tn = ['opt_tn']

for fold, (train_idx, val_idx) in enumerate(sfk.split(ori_patient_list, ori_label_list)):

    logging.info(f'👉 FOLD {fold}')
    ###############################################
    ##------ Step 1. split&load blocks list -------
    ###############################################
    valid_patient_list = ori_patient_list[val_idx]
    valid_label_list = ori_label_list[val_idx]

    valid_block_list = []
    valid_block_label_list = []

    if args.input_type == 1:
        print('=================== loading data ======================')
        logging.info(f'Loading {args.samples_per_patient} blocks for each person')

        # val
        for l in range(0, len(valid_patient_list)):
            patient_dir = os.path.join(valid_patient_list[l], f'block{args.block_size}_masked{args.masked}/')

            for k in range(1, args.samples_per_patient+1):
                if os.path.exists(os.path.join(patient_dir, f'{k}.npy')):
                    valid_block_list.append(os.path.join(patient_dir, f'{k}.npy'))
                    valid_block_label_list.append(valid_label_list[l])
                else:
                    print(f'🔴 Sampling error: {patient_dir}{k}.npy not exist!')

    seed_torch(seed=args.seed)

    ###############################################
    ##------------ Step 2. load data --------------
    ###############################################
    valid_ds = datareader(valid_block_list, valid_block_label_list, input_size = args.input_size, transform = 'valid')
    valid_data = DataLoader(valid_ds, batch_size=1, shuffle=False, worker_init_fn=seed_torch(seed=args.seed))          

    test_ds = datareader(test_block_list, test_block_label_list, input_size = args.input_size, transform = 'valid')
    test_data = DataLoader(test_ds, batch_size=1, shuffle=False, worker_init_fn=seed_torch(seed=args.seed))      


    ###############################################
    #---------- Step 3. initial model -------------
    ###############################################
    # model
    model = mymodel()
    model, epoch = load_checkpoint(args, model, fold)
    if torch.cuda.is_available():
        model.cuda()

    ###############################################
    #---------- Step 4. train and val -------------
    ###############################################

    print('=================== start validate ======================')
    val_acc, val_f1, val_pre, val_rec, val_auc, val_pred_bi, val_label, val_pred, val_path, fpr, tpr, opt_acc, optimal_thresh, point \
                                = test(model, valid_data, args, fold, 'test_val')
    logging.info(f"-- test_val -- best {args.evaluation} in fold {fold} epoch {epoch} -- accuracy : {val_acc:5.4f} | f1 : {val_f1:5.4f} | pre : {val_pre:5.4f} | recall : {val_rec:5.4f} | auc : {val_auc:5.4f}")

    type = []
    name = []
    block_folder = []
    block = []
    for i in range(len(val_path)):
        type.append(val_path[i].split('/')[-4])
        name.append(val_path[i].split('/')[-3])
        block_folder.append(val_path[i].split('/')[-2])
        block.append(val_path[i].split('/')[-1])
    type = np.array(type)
    name = np.array(name)
    block_folder = np.array(block_folder)
    block = np.array(block)

    fold_idx = np.ones((name.shape)) * fold
    
    val_info = np.stack((val_label, fold_idx, type, name, block_folder, block, val_pred), axis=1)
    total_test_val_info.append(val_info)

    print('=================== start test ======================')
    # test_acc, test_f1, test_pre, test_rec, test_auc, test_pred_bi, test_label, test_pred, test_path, opt_acc, opt_f1, opt_pre, opt_rec\
    #                             = test(model, test_data, args, fold, optimal_thresh, 'test_test')
    test_acc, test_f1, test_pre, test_rec, test_spc, test_confuse, test_auc, \
                        test_predict_bi, test_label, test_pred, test_path, opt_acc, opt_f1, opt_pre, opt_rec, opt_spc, opt_confuse = test(model, test_data, args, fold, optimal_thresh, 'test_test')

    opt_auc_save(test_log_dir, fold, test_label, test_pred, test_auc)
    
    # 0.5
    total_test_pred.append(test_pred)
    total_test_auc.append(test_auc)

    total_test_pre.append(test_pre)
    total_test_f1.append(test_f1)
    total_test_rec.append(test_rec)
    total_test_acc.append(test_acc)
    total_test_spc.append(test_spc)

    total_test_tp.append(test_confuse[0, 0])
    total_test_fn.append(test_confuse[0, 1])
    total_test_fp.append(test_confuse[1, 0])
    total_test_tn.append(test_confuse[1, 1])

    # opt
    total_opt_thresh.append(optimal_thresh)
    total_opt_point.append(point)

    total_opt_acc.append(opt_acc)
    total_opt_f1.append(opt_f1)
    total_opt_pre.append(opt_pre)
    total_opt_rec.append(opt_rec)
    total_opt_spc.append(opt_spc)

    total_opt_tp.append(opt_confuse[0, 0])
    total_opt_fp.append(opt_confuse[0, 1])
    total_opt_fn.append(opt_confuse[1, 0])
    total_opt_tn.append(test_confuse[1, 1])

    logging.info(f"-- test -- best {args.evaluation} fold {fold} -- accuracy : {test_acc:5.4f} | f1 : {test_f1:5.4f} | pre : {test_pre:5.4f} | recall : {test_rec:5.4f} | auc : {test_auc:5.4f}")

save_val_result(test_log_dir, total_test_val_info)


save_test_result(test_log_dir, total_test_pre, total_test_f1, total_test_rec, total_test_spc, total_test_auc, \
            total_test_acc, total_test_tp, total_test_fp, total_test_fn, total_test_tn, total_test_pred, test_label, test_path, \
            total_opt_acc, total_opt_f1, total_opt_pre, total_opt_rec, total_opt_spc, \
            total_opt_tp, total_opt_fp, total_opt_fn, total_opt_tn, total_opt_thresh, total_opt_point)
