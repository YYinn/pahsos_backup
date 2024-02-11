# 1. Data preprocess 
## 1.1 Data folders preprocess
Part 1 in ./preprocess.ipynb
Simplify original data.

- original data : /mnt/ExtData/pahsos/Data/raw_data (img : .dcm, seg : .nii.gz). msequences in different folders 

- simple data : /mnt/ExtData/pahsos/Data/simple_data . only contains specific folders for each patient (img : .dcm, seg : .nii.gz)

## 1.2 Data prerpocess for auto segmentation
Part 2 in ./preprocess.ipynb

## 1.3 Data prerpocess for classification
Part 3 in ./preprocess.ipynb

save into : /mnt/ExtData/pahsos/Data/preprocessed 

- img & resampled img & mask & resampled mask : .npy
- blocked_img : $/block{block_size}_masked{maskedornot}/{idx}.npy

# 2. Split dataset for train/val/test
./split_data.ipynb

Saving in ./Data/data_split.csv(for manually checking) and ./Data/data_split.json(for training and testing). 

# 3. Liver Segmentation

Segment liver as a mask for preprocess for classification (1.3) (the crop blocks for each patient only contains liver)

- pretrained model : /mnt/ExtData/pahsos/segmentation/nnUNet_trained_model

- ori data in required name format : /mnt/ExtData/pahsos/segmentation/nnUNet_raw_data_base/nnUNet_raw_data/Task029_LITS/imagesTs. (corresponding name in ipynb files or /mnt/ExtData/pahsos/mask_name.csv)

- use /mnt/ExtData/pahsos/segmentation/Task029_LiverTumorSegmentationChallenge.py to create json file

- output : /mnt/ExtData/pahsos/segmentation/output

```
1. install

    https://github.com/MIC-DKFZ/nnUNet#run-inference
    pip install nnunet

2. 查看可用的pretrained网络

    nnUNet_print_available_pretrained_models

3. 下载，得到./nnUNet_trained_model

    nnUNet_download_pretrained_model Task029_LiTS
    或者直接下载：https://zenodo.org/record/4003545/files/Task029_LITS.zip?download=1

4. export environment variables 

    export nnUNet_raw_data_base="/mnt/ExtData/pahsos/segmentation/nnUNet_raw_data_base"

    export nnUNet_preprocessed="/mnt/ExtData/pahsos/segmentation/nnUNet_preprocessed"
    
    export RESULTS_FOLDER="/mnt/ExtData/pahsos/segmentation/nnUNet_trained_model"

5. check pretrained model info 

    nnUNet_print_pretrained_model_info Task029_LiTS

6. inference 
    nnUNet_predict -i /./segmentation/nnUNet_raw_data_base/nnUNet_raw_data/Task029_LITS/imagesTs -o ./segmentation/output -t 29 -m 3d_fullres

```
result save in ./segmentation/output



# 4. Pahsos classification
```
python train.py --block_size [block_size]
```


# 5. Model inference
```
python test.py --experiments_name [experiment name] --block_size [block_size]

python test.py --experiments_name block64_new_2023-03-08T21:35:22 --block_size 64 --evaluation acc
python test.py --experiments_name block96_new_2023-03-09T15:40:03 --block_size 96 --evaluation acc
python test.py --experiments_name block128_new_2023-03-10T14:09:58 --block_size 128 --evaluation acc
python test.py --experiments_name block160_new_2023-03-10T23:36:42 --block_size 160 --evaluation acc




```

# 5. Result evalation
./result_val.ipynb : evaluate each fold of models 
./visualize.ipybn : visualization of block/patient level ROC for each fold
./result_val_ensemble.ipynb : evaluate ensemble model of 5 folds 





