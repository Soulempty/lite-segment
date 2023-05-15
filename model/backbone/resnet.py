import torch.nn as nn
from torchvision import models

class ResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT) 
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x)))) 
        layer1 = self.layer1(x)       # 1/4
        layer2 = self.layer2(layer1)  # 1/8
        layer3 = self.layer3(layer2)  # 1/16
        layer4 = self.layer4(layer3)  # 1/32
        return layer1,layer2,layer3,layer4
    
class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT) 
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x)))) 
        layer1 = self.layer1(x)       # 1/4
        layer2 = self.layer2(layer1)  # 1/8
        layer3 = self.layer3(layer2)  # 1/16
        layer4 = self.layer4(layer3)  # 1/32
        return layer1,layer2,layer3,layer4