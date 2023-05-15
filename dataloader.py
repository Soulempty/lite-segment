import numpy as np
import torch
from PIL import Image
import random
import os
from augmentation import *
import scipy.misc as m
from torchvision.transforms import ToTensor, ToPILImage
from torchvision.transforms import Normalize,Resize
from torch.utils.data import Dataset

class CFDataset(Dataset):
    def __init__(self, root_dir=None, augmentations=None, split="train"):
        self.split = split
        self.files = []
        self.augmentations = augmentations

        self.image_dir = os.path.join(root_dir, 'images', self.split)
        self.label_dir = os.path.join(root_dir, 'labels', self.split)

        self.files = os.listdir(self.image_dir)

        if not self.files:
            raise Exception("No files for split=[%s] found in %s" % (split, self.image_dir))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        result = dict(label = [])
        result['img_path'] = os.path.join(self.image_dir,self.files[index])
        result['img'] = Image.open(result['img_path']).convert('RGB')
        lbl_path = os.path.join(self.label_dir,os.path.splitext(self.files[index])[0]+'.png')
        if os.path.exists(lbl_path):
            result['label'] = Image.open(lbl_path).convert('L')   
            label = np.array(result['label'])
            if 248 in np.unique(label) or 249 in np.unique(label) or 255 in np.unique(label):
                label = np.where(label>248,1,0)
            result['label'] = Image.fromarray(label).convert('L')
        if self.augmentations:
            result = self.augmentations(result)
        return result
    