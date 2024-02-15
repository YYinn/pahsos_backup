#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from collections import OrderedDict
import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p, subfiles, join
from multiprocessing import Pool
import numpy as np
from nnunet.configuration import default_num_threads
from scipy.ndimage import label
import os
import json

def export_segmentations(indir, outdir):
    niftis = subfiles(indir, suffix='nii.gz', join=False)
    for n in niftis:
        identifier = str(n.split("_")[-1][:-7])
        outfname = join(outdir, "test-segmentation-%s.nii" % identifier)
        img = sitk.ReadImage(join(indir, n))
        sitk.WriteImage(img, outfname)


def export_segmentations_postprocess(indir, outdir):
    maybe_mkdir_p(outdir)
    niftis = subfiles(indir, suffix='nii.gz', join=False)
    for n in niftis:
        print("\n", n)
        identifier = str(n.split("_")[-1][:-7])
        outfname = join(outdir, "test-segmentation-%s.nii" % identifier)
        img = sitk.ReadImage(join(indir, n))
        img_npy = sitk.GetArrayFromImage(img)
        lmap, num_objects = label((img_npy > 0).astype(int))
        sizes = []
        for o in range(1, num_objects + 1):
            sizes.append((lmap == o).sum())
        mx = np.argmax(sizes) + 1
        print(sizes)
        img_npy[lmap != mx] = 0
        img_new = sitk.GetImageFromArray(img_npy)
        img_new.CopyInformation(img)
        sitk.WriteImage(img_new, outfname)

def load_save_test(args):
    data_file = args
    pat_id = data_file.split("/")[-1]
    # pat_id = pat_id[: pat_id.rfind("_")]
    pat_id = pat_id[: pat_id.rfind(".nii")]

    return pat_id


if __name__ == "__main__":
    output_folder = "/mnt/ExtData/pahsos/segmentation/nnUNet_raw_data_base/nnUNet_raw_data/Task029_LITS"
    img_dir_te = join(output_folder, "imagesTs")
    maybe_mkdir_p(img_dir_te)

    nii_files_ts = subfiles(img_dir_te, True, "test", "nii.gz", True)
    print(nii_files_ts)

    p = Pool(default_num_threads)

    test_ids = p.map(load_save_test, nii_files_ts)
    p.close()
    p.join()

    json_dict = OrderedDict()
    json_dict['name'] = "LITS"
    json_dict['description'] = "LITS"
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = "see challenge website"
    json_dict['licence'] = "see challenge website"
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "CT"
    }

    json_dict['labels'] = {
        "0": "background",
        "1": "liver",
    }

    json_dict['numTraining'] = 0 #len(train_ids)
    json_dict['numTest'] = len(test_ids)
    json_dict['training'] = []#[{'image': "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i} for i in train_ids]
    json_dict['test'] = ["./imagesTs/%s.nii.gz" % i for i in test_ids]

    print(os.path.join(output_folder, "dataset.json"))
    with open(os.path.join(output_folder, "dataset.json"), 'w') as f:
        json.dump(json_dict, f, indent=4, sort_keys=True)