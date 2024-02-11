import json
import numpy as np
import os
import glob
from torch.utils.data import Dataset, DataLoader

from utils import *
import cv2
import torch



def get_data_list(args, key):
    '''
    key: train or test
    '''
    data_root = args.root_path
    json_path = args.json_path

    patinet_list = [] #patient folders, 为了让一个病人的所有block不同时存在于训练集和测试集合。
    label_list = []
    pos = 0
    neg = 0
    aug_pos = 0

    with open(json_path) as f:
        json_data = json.load(f)
    json_data = json_data[key]

    for d in json_data:
        patient_name = d['patient_name']
        type = d['type']
        label = d['label']

        # ori
        if label == 1:
            pos += 1
        else:
            neg += 1

        img_path = os.path.join(data_root, f'{type}/{patient_name}')
        patinet_list.append(img_path)
        label_list.append(label)      

    print(f'{key} samples: {len(patinet_list)} pos {pos} neg {neg}')

    patinet_list = np.array(patinet_list)
    label_list = np.array(label_list)
    
    return patinet_list, label_list


class datareader(Dataset):
    def __init__(self, img_list, label_list, input_size, transform=None):
        self.img_list = img_list 
        self.label_list = label_list
        self.input_size = input_size
        if transform == 'train':
            self.transform = img_transform_train
        elif transform == 'valid':
            self.transform = img_transform_valid
        
    def __getitem__(self, item):
        # load image
        ori_img = np.load(self.img_list[item])
        
        img = resample_to_size(ori_img, (self.input_size, self.input_size, self.input_size))
        img = norm_img(img).astype(np.float32)

        img = self.transform(image = img)['image']
        
        img = img.unsqueeze(dim=0)

        label = self.label_list[item]
        label = torch.tensor(label)

        return img, label.float(), self.img_list[item]

    def __len__(self):
        return len(self.label_list)
