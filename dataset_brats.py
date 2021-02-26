import os
import numpy as np
from glob import glob
from skimage import io
import random
#torch
import torch
from torch.utils.data import Dataset

IMG_MEAN = np.array((0.0860377,  0.1216296,  0.07420689, 0.09033176), dtype=np.float32)

def mat2img(slices):
    tmin = np.amin(slices)
    tmax = np.amax(slices)
    diff = tmax -tmin
    if (diff == 0):
        return slices
    else:
        return np.uint8(255 * (slices - tmin) / (diff))


class dataset_brats19(Dataset):
    def __init__(self, args=None, transform=None, isTrain=True):
        self.isTrain = isTrain
        if self.isTrain:
            self.data_dir = glob(args.data_root+'/cv_train/**')
        else:
            self.data_dir = glob(args.data_root+'/cv_valid/**')

        self.args = args

    def __len__(self):
        return len(self.data_dir)

    def __getitem__(self, index):
        modalities = np.zeros((5, 155, 240, 240))
        flair = glob(self.data_dir[index] + '/*_flair.nii.gz')
        t1 = glob(self.data_dir[index] + '/*_t1.nii.gz')
        t1c = glob(self.data_dir[index] + '/*_t1ce.nii.gz')
        t2 = glob(self.data_dir[index] + '/*_t2.nii.gz')
        gt = glob(self.data_dir[index] + '/*seg.nii.gz')
        scans = [flair[0], t1[0], t1c[0], t2[0], gt[0]]
        for scan_idx in range(4):
            modalities[scan_idx] = mat2img(io.imread(scans[scan_idx], plugin='simpleitk').astype(float)) / 255 - \
                                   IMG_MEAN[scan_idx] #(5, 155, 240, 240)
        modalities[4] = io.imread(scans[4], plugin='simpleitk').astype(float)

        _img = modalities[0:4, self.args.init_z:self.args.init_z + self.args.crop_z, self.args.init_xy:self.args.init_xy
                + self.args.crop_xy,self.args.init_xy:self.args.init_xy + self.args.crop_xy]
        
        _target = modalities[4, self.args.init_z:self.args.init_z + self.args.crop_z, self.args.init_xy:self.args.init_xy
            + self.args.crop_xy,self.args.init_xy:self.args.init_xy + self.args.crop_xy]
        if self.isTrain:
            hflip = random.random() < 0.5
            if hflip:
                _img = _img[:, ::-1, :, :]
                _target = _target[::-1, :, :]

        _target[_target==4] = 3

        _img = torch.from_numpy(np.array(_img)).float()
        _target = torch.from_numpy(np.array(_target)).long()
        return _img, _target, os.path.basename(self.data_dir[index])
