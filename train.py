import os
import torch
import cv2
import argparse
import logging
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau,StepLR,MultiStepLR,CosineAnnealingLR,PolynomialLR
from model import FCN,DDRNet_23,BiSeNetV2,PPLiteSeg
from utils import train
from loss import NLLLoss,CrossEntropy
from dataloader import CFDataset
from augmentation import Compose,RandomCrop,RandomDistort,RandomFlip,Resize,Normalize,ToTensor

MODEL = {'fcn':FCN,'ddrnet23':DDRNet_23,'bisnetv2':BiSeNetV2,'ppliteseg':PPLiteSeg}

logging.basicConfig(level = logging.INFO)
logger = logging.getLogger('Segmentation')

def get_loader(args):
    train_transforms = Compose([Resize(args.base_size),
                                RandomCrop(args.crop_size),
                                RandomFlip(),
                                RandomDistort(brightness_range=0.4,contrast_range=0.4,saturation_range=0.4),
                                ToTensor(),
                                Normalize()])
    val_transforms = Compose([Resize(scale=args.base_size),
                              ToTensor(),
                              Normalize()])
    train_dataset=CFDataset(args.data_path,augmentations=train_transforms,split='train')
    val_dataset=CFDataset(args.data_path,augmentations=val_transforms,split='val')
    return train_dataset,val_dataset

def main(args):
    train_dataset,val_dataset = get_loader(args)
    num_epochs = args.num_epochs
    class_names = args.class_names
    iters_per_epoch = len(train_dataset)//args.batch_size
    iters = num_epochs*iters_per_epoch

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MODEL[args.model_name](len(class_names)).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,momentum=0.9,weight_decay=5e-4) 
    lr_scheduler = PolynomialLR(optimizer, iters, power=0.9)
    criterion = CrossEntropy(sample='ohem',ignore_label=255)
    
    
    train(model,
          train_dataset,
          val_dataset,
          optimizer,
          lr_scheduler,
          save_dir=args.work_dir, 
          iters=iters,
          batch_size=args.batch_size,
          save_interval=iters_per_epoch,
          logger=logger,
          log_iters=20,
          num_workers=args.num_workers,
          criterion=criterion,
          device=device,
          class_names=class_names,
          visualizer=None,
          keep_checkpoint_max=5)

if __name__ == "__main__":
    # class_names = ["background","Biaoji","Bianyuanbaidian","Huahen","Lvpao","Bengbian","Baidian","Heidian","Baiban","Xianyichang",
    #             "Yiwu","Zangwu","Lvbuqi","Duankai","Huashang","Cashang","Huanqie","Kailie","Xianyichang2","Lvbuping"]
    class_names = ['background','cls-1','cls-2','cls-3','cls-4',]
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='../../dataset/steel', help='data path')
    parser.add_argument('--model_name', type=str, default='ppliteseg', choices=['ppliteseg','fcn','ddrnet23','ddrnet23_slim','bisnetv2'], help='select model.')
    parser.add_argument('--work_dir', default='work_dirs/checkpoint/steel-ppliteseg', help='data path')
    parser.add_argument('--num_epochs', type=int, default=60, help='epoch number')
    parser.add_argument('--class_names', type=list, default=class_names, help='class names')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate') 
    parser.add_argument('--base_size',default=(1600,256), help='base_size') 
    parser.add_argument('--crop_size',default=(800,128), help='crop_size')  
    parser.add_argument('--num_workers', type=int, default=4, help='num workers')
    parser.add_argument('--batch_size', type=int, default=4, help='training batch size')
    args = parser.parse_args()
    main(args)

    