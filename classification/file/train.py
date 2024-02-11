import argparse
import datetime
import logging
import os
import random
import sys
import warnings

import numpy as np
import torch
import wandb
from dataloader import *
from models.mymodel import *
from sklearn.model_selection import StratifiedKFold
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from train_one_epoch import train
from val_one_epoch import val
from utils import *

warnings.filterwarnings('ignore')

############################
######## set config ########
############################
parser = argparse.ArgumentParser(description=' ')
parser.add_argument('--lr', default=3e-5, type=float, help='learning rate')
parser.add_argument('--num_epochs', default=200, type=int, help='training epoch')
parser.add_argument('--batch_size', default=16, type=int, help='batch size ')
parser.add_argument('--fold', type=int, default=5, help="fold")
parser.add_argument('--seed', type=int, default=147, help='seed')
parser.add_argument('--optim', type=str, default='SGD', help='use SGD or Adam to optimize')
parser.add_argument('--earlystop', type=int, default=20, help='how many epochs we want to wait after the last time \
                                                                the validation loss decreased before breaking the training loop')

parser.add_argument('--block_size', type=int, default=128, help='input_size for ori img cropping')
parser.add_argument('--input_size', type=int, default=64, help='input_size for model (after resize block_size data), fix for specific model')
parser.add_argument('--root_path', type=str, default='/mnt/ExtData/pahsos/Data/preprocessed', help='original data root')
parser.add_argument('--json_path', type=str, default='/mnt/ExtData/pahsos/Data/data_split.json', help='json path for data and label loading')
parser.add_argument('--samples_per_patient', type=int, default=12, help='use how many blocks per patient')

parser.add_argument('--log_dir', type=str, default='./log', help='original data root')
parser.add_argument('--note', type=str, default='new', help='add on log dir')

parser.add_argument('--checkpoint', type=str, default=None, help='loading checkpoint to continue training')
parser.add_argument('--start_fold', type=int, default=0, help='loading checkpoint and resume training from fold x')
parser.add_argument('--start_epoch', type=int, default=0, help='loading checkpoint and resume training from epoch x')
args = parser.parse_args()


## log
time = datetime.datetime.now().isoformat()[:19]
if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)

log_dir = f'{args.log_dir}/block{args.block_size}_{args.note}_{time}/'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

model_save_path = os.path.join(log_dir, 'checkpoint')
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

wandb.init(project='PA-HSOS', name=f'{args.note}_block{args.block_size}_{time}')
logging.basicConfig(level=logging.INFO,
                    handlers=[logging.FileHandler(f'{log_dir}train.log'), logging.StreamHandler(sys.stdout)])
logging.info(str(args))

## seed
seed_torch(seed=args.seed)

## data_list 
ori_patient_list, ori_label_list = get_data_list(args, key='train')

## kfold, load 12 blocks
sfk = StratifiedKFold(n_splits=args.fold, shuffle=True, random_state=args.seed) 

for fold, (train_idx, val_idx) in enumerate(sfk.split(ori_patient_list, ori_label_list)):
    logging.info(f'👉 FOLD {fold}')
    ###############################################
    ##------ Step 1. split&load blocks list -------
    ###############################################
    train_patient_list = ori_patient_list[train_idx]
    train_label_list = ori_label_list[train_idx]
    valid_patient_list = ori_patient_list[val_idx]
    valid_label_list = ori_label_list[val_idx]

    train_block_list = []
    train_block_label_list = []
    valid_block_list = []
    valid_block_label_list = []

    print('=================== loading data ======================')
    logging.info(f'Loading {args.samples_per_patient} blocks for each person')

    for l in range(0, len(train_patient_list)):
        patient_dir = os.path.join(train_patient_list[l], f'block{args.block_size}_maskedTrue_new/')

        for k in range(1, args.samples_per_patient+1):
            if os.path.exists(os.path.join(patient_dir, f'{k}.npy')):
                train_block_list.append(os.path.join(patient_dir, f'{k}.npy'))
                train_block_label_list.append(train_label_list[l])
            else:
                print(f'🔴 Sampling error: {patient_dir}{k}.npy not exist!')


    for l in range(0, len(valid_patient_list)):
        patient_dir = os.path.join(valid_patient_list[l], f'block{args.block_size}_maskedTrue_new/')

        for k in range(1, args.samples_per_patient+1):
            if os.path.exists(os.path.join(patient_dir, f'{k}.npy')):
                valid_block_list.append(os.path.join(patient_dir, f'{k}.npy'))
                valid_block_label_list.append(valid_label_list[l])
            else:
                print(f'🔴 Sampling error: {patient_dir}{k}.npy not exist!')

    print(f'train/valid list', train_patient_list.shape, train_label_list.shape, valid_patient_list.shape, valid_label_list.shape)
    
    seed_torch(seed=args.seed)


    ###############################################
    ##------------ Step 2. load data --------------
    ###############################################
    train_ds = datareader(train_block_list, train_block_label_list, input_size = args.input_size, transform = 'train')
    valid_ds = datareader(valid_block_list, valid_block_label_list, input_size = args.input_size, transform = 'valid')

    ## print train valid dataset size
    print("train total", len(train_ds), " valid total", len(valid_ds))
    cnt = 0
    for i, (img, label, path) in enumerate(train_ds):
        if label == 0:
            cnt += 1
    print("[train] 0 label num:", cnt/len(train_ds), " 1 label num:", 1-cnt/len(train_ds))
    cnt = 0
    for i, (img, label, path) in enumerate(valid_ds):
        if label == 0:
            cnt += 1
    print("[val] 0 label num:", cnt/len(valid_ds), " 1 label num:", 1-cnt/len(valid_ds))

    ## split into batch
    train_data = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, worker_init_fn=seed_torch(seed=args.seed))
    valid_data = DataLoader(valid_ds, batch_size=1, shuffle=False, worker_init_fn=seed_torch(seed=args.seed))              
    print("train total (batch)", len(train_data), " val total  (batch)", len(valid_data))


    ###############################################
    #----Step 3. initial model and optimizer-------
    ###############################################
    model = mymodel()

    criterion = BCEFocalLoss()
    
    if args.checkpoint is not None:
        model_dict = torch.load(args.checkpoint)
        model.load_state_dict(model_dict)
        start_epoch = args.start_epoch
    if torch.cuda.is_available():
        model.cuda()

    # optimizer
    if args.optim == 'SGD':
        optimizer = SGD(model.parameters(), lr=args.lr, weight_decay=0.0001, momentum=0.9)
    elif args.optim == 'Adam':
        optimizer = Adam(model.parameters(), lr=args.lr, eps=1e-8, weight_decay=1e-6)

    if fold < args.start_fold: # for continue training from a checkpoint
        continue

    ###############################################
    #---------- Step 4. train and val -------------
    ###############################################
    best_acc = 0
    best_acc_epoch = 0
    best_auc = 0
    best_acc_epoch = 0
    
    min_val_loss = -1
    step = 0

    start_epoch = 0
    print('=================== start training ======================')
    for epoch in range(start_epoch, args.num_epochs):
        loss, acc, f1, pre, rec, auc = train(model, train_data, args, optimizer, criterion, fold, epoch)
        logging.info(f'🟢 fold {fold} epoch {epoch}')
        logging.info(f'-- train -- loss : {loss:5.4f} | accuracy : {acc:5.4f} | f1 : {f1:5.4f} | pre : {pre:5.4f} | recall : {rec:5.4f} | auc : {auc:5.4f}')

        val_loss, val_acc, val_f1, val_pre, val_rec, val_auc = val(model, valid_data, args, criterion, fold, epoch)
        logging.info(f"-- val --  val_loss : {val_loss:5.4f}, accuracy : {val_acc:5.4f} | f1 : {val_f1:5.4f} | pre : {val_pre:5.4f} | recall : {val_rec:5.4f} | auc : {val_auc:5.4f}")
        

        wandb.log({f"train/fold{fold}": {"train_loss": loss, "train_acc" : acc, "train_f1": f1, "train_pre": pre, "train_rec": rec, "train_auc": auc}}, step=epoch)
        wandb.log({f"val/fold{fold}": {"val_loss": val_loss, "val_acc" : val_acc, "val_f1": val_f1, "val_pre": val_pre, "val_rec": val_rec, "val_auc": val_auc}}, step=epoch)

        if min_val_loss > val_loss or min_val_loss == -1:
            min_val_loss = val_loss
            step = 0
        else:
            step += 1
                
        if step >= args.earlystop:
            logging.info('Early stop ! ')
            break
        else:
            logging.info(f'Early stop process : {step} / {args.earlystop}')

        if(val_acc > best_acc):
            best_acc = val_acc
            best_acc_epoch = epoch
            save_checkpoint(model, fold, epoch, model_save_path, best_acc, indicator='acc')
            logging.info(f'✅ get better accuracy {best_acc}')

        if(val_auc > best_auc):
            best_auc = val_auc
            best_auc_epoch = epoch
            save_checkpoint(model, fold, epoch, model_save_path, best_auc, indicator='auc')
            logging.info(f'✅ get better auc {best_auc}')

        logging.info(f"== best accuracy : {best_acc} in fold {fold} epoch {best_acc_epoch}")
        logging.info(f"== best AUC : {best_auc} in fold {fold} epoch {best_acc_epoch}")

        save_checkpoint(model, fold, epoch, model_save_path, 0, indicator='latest')

wandb.finish()