# calculate test time for each fold
import argparse
import os
import time
import warnings

import torch
from dataloader import *
from models.mymodel import *
from test_one_epoch import test
from torch.utils.data import DataLoader
from utils import *

warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser(description=' ')
parser.add_argument('--batch_size', default=16, type=int, help='batch size ')
parser.add_argument('--fold', type=int, default=5, help="fold")
parser.add_argument('--seed', type=int, default=147, help='seed')

parser.add_argument('--block_size', type=int, default=128, help='input_size for ori img cropping')
parser.add_argument('--input_size', type=int, default=64, help='input_size for model (after resize block_size data), fix for specific model')
parser.add_argument('--root_path', type=str, default='/mnt/ExtData/pahsos/Data/preprocessed', help='original data root')
parser.add_argument('--json_path', type=str, default='/mnt/ExtData/pahsos/Data/data_split.json', help='json path for data and label loading')
parser.add_argument('--samples_per_patient', type=int, default=12, help='sampling x blocks for one person') 

parser.add_argument('--log_dir', type=str, default='/mnt/ExtData/pahsos/classification/log', help='original data root')
parser.add_argument('--note', type=str, default='_', help='add on log dir')

parser.add_argument('--experiments_name', type=str, default='s12_block128_maskedTrue_es50_2023-02-05T09:42:27')
parser.add_argument('--evaluation', type=str, default='auc', help='auc or acc or latest')
args = parser.parse_args()

## seed
seed_torch(seed=args.seed)

## data_list 
test_patient_list, test_label_list = get_data_list(args, key='test')

test_block_list = []
test_block_label_list = []
for l in range(0, len(test_patient_list)):
    patient_dir = os.path.join(test_patient_list[l], f'block{args.block_size}_maskedTrue_new/')
    for k in range(1, args.samples_per_patient+1):
        if os.path.exists(os.path.join(patient_dir, f'{k}.npy')):
            test_block_list.append(os.path.join(patient_dir, f'{k}.npy'))
            test_block_label_list.append(test_label_list[l])
        else:
            print(f'🔴 Sampling error: {patient_dir}{k}.npy not exist!')

for fold in range(args.fold):

    start_time = time.time()
    seed_torch(seed=args.seed)
    test_ds = datareader(test_block_list, test_block_label_list, input_size = args.input_size, transform = 'valid')
    test_data = DataLoader(test_ds, batch_size=1, shuffle=False, worker_init_fn=seed_torch(seed=args.seed))      

    model = mymodel()
    model, epoch = load_checkpoint(args, model, fold)
    if torch.cuda.is_available():
        model.cuda()

    test_acc, test_rec, test_spc, test_auc, test_confuse, test_predict_bi, test_label, test_predict, test_path = test(model, test_data, args, fold)
    end_time = time.time()

    print(fold, end_time - start_time)