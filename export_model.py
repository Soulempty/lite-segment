import os
import cv2
import tqdm
import torch
import onnx
import numpy as np
import onnxruntime
import torchvision.models as models
from model import FCN,DDRNet_23,DDRNet_23_slim,BiSeNetV2,PPLiteSeg,FactSeg

MODEL = {'fcn':FCN,'DDRNET23':DDRNet_23,'ddrnet23_slim':DDRNet_23_slim,'bisnetv2':BiSeNetV2,'ppliteseg':PPLiteSeg,'factseg':FactSeg}

PALETTE = [ 
        (0,0,0), (0,0,255), (156,102,102), (0, 0, 142), (0, 0, 230), (106, 0, 228),
        (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
        (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30),
        (165, 42, 42), (255, 77, 255), (0, 226, 252), (182, 182, 255),
        (0, 82, 0)]

class ONNXInfer(object):
    def __init__(self,model_name='factseg',num_classes=20,checkpoint_path='work_dirs/checkpoint/wafer/best.pth',data_path='',input_size=[1,3,864,864],save_path=None):
        super().__init__()
        self.model = MODEL[model_name]
        self.num_classes = num_classes
        self.checkpoint_path = checkpoint_path
        self.data_path = data_path
        self.onnx_path = os.path.join(os.path.dirname(checkpoint_path),f'{model_name}.onnx')
        
        self.input_size = input_size
        self.save_path = save_path
        os.makedirs(save_path,exist_ok=True)
        self.init_param()
    
    def init_param(self):
        self.device = torch.device("cpu")
        providers = ['CPUExecutionProvider']
        if torch.cuda.is_available():
          providers.insert(0, 'CUDAExecutionProvider')
          self.device = torch.device("cuda:0")
        self.providers = providers
        
    def run(self):
        ort_session = self.load_model()
        dataset = self.load_data()
        input_name = ort_session.get_inputs()[0].name
        for filename in tqdm.tqdm(dataset):
            img = dataset[filename].astype(np.float32)
            image = cv2.imread(filename)
            ort_inputs = {input_name: img}
            ort_outs = ort_session.run(None, ort_inputs)
            pred = ort_outs[0].argmax(axis=1)[0].astype(np.uint8)
            save_p = os.path.join(self.save_path,os.path.basename(filename))
            result = self.pre_to_img(image,pred)
            cv2.imwrite(save_p,result)

    @staticmethod
    def pre_to_img(img,mask,alpha=0.5):
        h,w = img.shape[:2]
        mask = cv2.resize(mask,(w,h),cv2.INTER_NEAREST)
        ids = np.unique(mask)
        for id_ in ids:
            if id_ == 0:
                continue
            img[mask==id_] = np.array([PALETTE[id_]])*alpha + img[mask==id_]*(1-alpha) 
        return img

 
    def load_data(self):
        dataset = dict()
        def get_data(filename):
            if os.path.isfile(filename):
                bgr_img = cv2.imread(filename)
                rgb_img = cv2.cvtColor(bgr_img,cv2.COLOR_BGR2RGB)
                rs_img = cv2.resize(rgb_img,self.input_size[2:],interpolation=cv2.INTER_LINEAR)
                img = rs_img/255.0
                img = img.transpose(2, 0, 1)
                img = np.expand_dims(img, axis=0)
                return filename,img
        if os.path.isdir(self.data_path):
            dataset = dict(get_data(filename.path) for filename in os.scandir(self.data_path))
        else:
            filename,data = get_data(self.data_path)
            dataset = {filename:data}
        return dataset

    def export_model(self):
        if os.path.exists(self.onnx_path):
            print(f"The onnx model {self.onnx_path} has alreadly exist.")
            return
        input = torch.randn(size=self.input_size, device=self.device)
        model = self.model(self.num_classes).to(self.device)
        weight = torch.load(self.checkpoint_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(weight)
        model.eval()
        input_names = [ "input" ] 
        output_names = [ "output" ]
        torch.onnx.export(model, 
                  input, 
                  self.onnx_path, 
                  verbose=True, 
                  input_names=input_names, 
                  output_names=output_names)
    def load_model(self):
        ort_session = onnxruntime.InferenceSession(self.onnx_path,providers=self.providers)
        return ort_session
        
if __name__ =="__main__":
    ort = ONNXInfer(data_path='../../dataset/wafer/images/val',save_path='work_dirs/checkpoint/wafer/onnx')
    ort.export_model()
    ort.run()

