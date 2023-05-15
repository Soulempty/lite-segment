import torch
import torch.nn as nn

class Conv3x3(nn.Module):
    def __init__(self,in_channel,out_channel,stride=1,padding=1,dilation=1):
        super(Conv3x3,self).__init__()
        self.conv=nn.Sequential(nn.Conv2d(in_channel,out_channel,3,stride,padding,dilation,bias=False),
                                nn.BatchNorm2d(out_channel),
                                nn.ReLU(inplace=True))
    def forward(self,x):
        x = self.conv(x)
        return x

class DWConv3x3(nn.Module):
    def __init__(self, in_channels,out_channels,stride=1,padding=1,dilation=1):
        super(DWConv3x3, self).__init__()
        self.conv1 = nn.Conv2d(in_channels,in_channels,3,stride,padding,dilation,groups=in_channels,bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1,padding=0,bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
 
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(x)))
        return out

def init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

class FCNNet(nn.Module):
    def __init__(self,n_class=2):
        super(FCNNet,self).__init__()
        self.conv1 = Conv3x3(3,32,2)  
        self.dwconv1 = DWConv3x3(32,64)   
        self.dwconv2 = DWConv3x3(64,128,2)  
        self.dwconv3 = DWConv3x3(128,128)  
        self.dwconv4 = DWConv3x3(128,256)  
        self.dwconv5 = DWConv3x3(256,256)  
        self.dwconv6 = DWConv3x3(256,512)  
        self.dwconv7 = DWConv3x3(512,512)  
        self.dwconv8 = DWConv3x3(512,512,2) 
        self.dwconv9 = DWConv3x3(512,512,padding=2,dilation=2)  
        self.dwconv10 = DWConv3x3(512,512,padding=3,dilation=3)  
     
        self.conv2 = nn.Conv2d(512,16,1,1)  
        self.deconv1 = nn.ConvTranspose2d(16,16,kernel_size=(4,4),stride=(2,2),padding=(1,1),bias=False)  
        self.conv3 = Conv3x3(16,16)  
        self.deconv2 = nn.ConvTranspose2d(16,16,kernel_size=(2,2),stride=(2,2),bias=False)  
        self.conv4 = Conv3x3(16,16) 
        self.deconv5 = nn.ConvTranspose2d(16,16,kernel_size=(4,4),stride=(2,2),padding=(1,1),bias=False)  
        self.conv5 = nn.Conv2d(16,n_class,1,1)
        init_weight(self)
    def forward(self,x):
        x = self.conv1(x)
        x = self.dwconv1(x)
        x = self.dwconv2(x)
        x = self.dwconv3(x)
        x = self.dwconv4(x)
        x = self.dwconv5(x)
        x = self.dwconv6(x)
        x = self.dwconv7(x)
        x = self.dwconv8(x)
        x = self.dwconv9(x) 
        x = self.dwconv10(x)  

        x = self.conv2(x)
        x = self.deconv1(x)
        x = self.conv3(x)
        x = self.deconv2(x)
        x = self.conv4(x)
        out = self.deconv5(x)
        out = self.conv5(out)
        return [out]

if __name__ == "__main__":
    model = FCNNet()
    test_data = torch.rand(1, 3, 512, 512)
    print(model)
    test_outputs = model(test_data)
    print(test_outputs.size())
