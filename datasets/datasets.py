import glob
import os
import matplotlib.pyplot as plt
#from skimage import transform
import cv2
import torch
from torch.utils.data import Dataset


class DriveDataset(Dataset):
    def __init__(self, filepath, mode='train', transform=None):
        self.filepath = os.path.join(filepath, mode)
        self.imgs = sorted(glob.glob(os.path.join(self.filepath, 'origin/*.tif')))
        self.masks = sorted(glob.glob(os.path.join(self.filepath, 'groundtruth/*.tif')))
        self.file_list = [os.path.basename(f).split('_')[0] for f in self.imgs]
        self.img_dim = (256, 256)
        self.transform = transform

    def __getitem__(self, idx):
        img = plt.imread(self.imgs[idx])
        img = cv2.resize(img, self.img_dim)
        img = torch.from_numpy(img).type(torch.float32)
        img = img.permute(2, 0, 1)
        mask = plt.imread(self.masks[idx])
        mask = cv2.resize(mask, self.img_dim)
        mask[mask>0] = 1
        mask = torch.from_numpy(mask).type(torch.long)

        return {'image': img, 'mask': mask, 'filename': self.file_list[idx]}

    def __len__(self):
        return len(self.file_list)