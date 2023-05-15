import os
import cv2
import torch
import numpy as np
import torch.nn.functional as F
import time
import argparse
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from torch.autograd import Variable
from collections import OrderedDict
from model import FCN,DDRNet_23,PPLiteSeg,BiSeNetV2
from dataloader import CFDataset
from utils import evaluate
from torch.utils.data import DataLoader
from augmentation import Compose,RandomCrop,RandomFlip,Resize,Normalize,ToTensor

MODEL = {'fcn':FCN,'ddrnet23':DDRNet_23,'bisnetv2':BiSeNetV2,'ppliteseg':PPLiteSeg}


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
color_palette = [ 
        (0,0,0), (0,0,255), (156,102,102), (0, 0, 142), (0, 0, 230), (106, 0, 228),
        (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
        (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30),
        (165, 42, 42), (255, 77, 255), (0, 226, 252), (182, 182, 255),
        (0, 82, 0)]

class Inference(object):

    def __init__(self, model_path,mode_name,data_path,save_path,size=[864,864],class_names=["background","cls_1","cls_2","cls_3","cls_4"],eval=False,merge=False):
        self.seg_model =MODEL[mode_name](len(class_names))
        self.__load_weight(self.seg_model, model_path)
        self.seg_model = self.seg_model.cuda()
        self.data_path = data_path
        self.save_path = save_path
        self.eval = eval
        self.size = size
        self.merge = merge
        self.class_names = class_names

    def infer(self):
        os.makedirs(self.save_path,exist_ok=True)
        dataset, data_loader = self.get_loader()
        self.seg_model.eval()
        if self.eval:
            evaluate(self.seg_model,dataset,class_names=self.class_names,device=device)
        duration = 0
        step = 0
        for item in tqdm(data_loader):
            img_path = item['img_path'][0]
            torch.cuda.synchronize()
            st = time.time()
            image = cv2.imread(img_path)
            img = item['img'].to(device)
            output = self.seg_model(img)[0]
            if step>0:
                duration += ed-st
            mask = torch.argmax(output,dim=1)[0]
            result = mask.detach().cpu().numpy().astype(np.uint8)
            torch.cuda.synchronize()
            ed = time.time()
            basename = os.path.splitext(os.path.basename(img_path))[0]+'.png'
            if self.merge:
                basename = os.path.basename(img_path)
                result = self.pre_to_img(image,result)
            cv2.imwrite(os.path.join(self.save_path,basename),result)
            step += 1
        print(f"time using {duration/(step-1)} s for each image.")
            
    def get_loader(self):
        data_transforms = Compose([Resize(scale=self.size),
                               ToTensor(),
                               Normalize()])
        data_dataset=CFDataset(self.data_path,augmentations=data_transforms,split='val')
        data_loader = DataLoader(data_dataset, batch_size=1, shuffle=False, num_workers=4)
        return data_dataset,data_loader
 
    @staticmethod
    def __load_weight(seg_model, model_path, is_local=True):
        print("loading pre-trained weight")
        weight = torch.load(model_path, map_location=lambda storage, loc: storage)

        if is_local:
            seg_model.load_state_dict(weight)
        else:
            new_state_dict = OrderedDict()
            for k, v in weight.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            seg_model.load_state_dict(new_state_dict)

    @staticmethod
    def pre_to_img(img,mask,alpha=0.5):
        h,w = img.shape[:2]
        mask = cv2.resize(mask,(w,h),cv2.INTER_NEAREST)
        ids = np.unique(mask)
        for id_ in ids:
            if id_ == 0:
                continue
            img[mask==id_] = np.array([color_palette[id_]])*alpha + img[mask==id_]*(1-alpha) 
        return img
    
if __name__ =="__main__":  
    class_names = ["background","Biaoji","Bianyuanbaidian","Huahen","Lvpao","Bengbian","Baidian","Heidian","Baiban","Xianyichang",
                "Yiwu","Zangwu","Lvbuqi","Duankai","Huashang","Cashang","Huanqie","Kailie","Xianyichang2","Lvbuping"]
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='work_dirs/checkpoint/wafer-ppliteseg/best.pth', help='model path')
    parser.add_argument('--data_path', default='../../dataset/wafer', help='data path')
    parser.add_argument('--save_path', default='work_dirs/checkpoint/wafer-ppliteseg/result', help='result save path')
    parser.add_argument('--model_name', type=str, default='ppliteseg', choices=['ppliteseg','fcn','ddrnet23','ddrnet23_slim','bisnetv2'], help='select model.')
    parser.add_argument('--size', default=(864,864), help='base_size') 

    args = parser.parse_args()
    infer = Inference(args.model_path,
                      args.model_name,
                      args.data_path,
                      args.save_path,
                      args.size,
                      class_names=class_names,
                      eval=False,merge=False)
    infer.infer()

