# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 20:59:45 2019

@author: HD

Reference: 
    https://github.com/pytorch/vision/tree/master/torchvision
    https://github.com/hujie-frank/SENet

"""

from cifar100_dataset import *
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torchvision.models import ResNet

class SEblock(nn.Module):
    def __init__(self,channel,reduction=16):
        super(SEblock,self).__init__()
        self.avg_pool=nn.AdaptiveAvgPool2d(1)
        self.fc=nn.Sequential(
                nn.Linear(channel, channel // reduction, bias =False),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel, bias =False),
                nn.Sigmoid()
                )
    def forward(self,x):
        b,c,_,_=x.size()
        y=self.avg_pool(x).view(b,c)
        y=self.fc(y).view(b,c,1,1)
        return x*y.expand_as(x)

class ResNetBlock(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64,reduction=16, norm_layer=None):
        super(ResNetBlock, self).__init__()
        
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv2d(inplanes,width,kernel_size=1)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width,kernel_size=3, padding=1, stride=stride,  groups=groups)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * self.expansion,kernel_size=1)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEblock(planes*4, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.se(x)

        if self.downsample is not None:
            identity = self.downsample(identity)

        x += identity
        x = self.relu(x)

        return x

# According to ResNet50 layers constructure
# 
# 50layers =  3*3layers, 4*3layers, 6*3layers, 3*3layers
        
model=ResNet(ResNetBlock,[3,4,6,3],num_classes=1000)
model.avgpool==nn.AdaptiveAvgPool2d(1)

if torch.cuda.is_available():
    print("GPU is runing Now!")
    model.cuda()

criterion = nn.CrossEntropyLoss()     
optimizer = optim.SGD(model.parameters(),weight_decay=0.0001,lr=0.1,momentum=0.9)

for epoch in range(300):
    running_loss =0.0
    for i,data in enumerate(trainloader,0):
        inputs,labels=data
        inputs,labels=Variable(inputs),Variable(labels)
        
        optimizer.zero_grad()
        if torch.cuda.is_available():
            inputs,labels=inputs.cuda(),labels.cuda()                 
        
        outputs=model(inputs)
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()        
        if i % 10 == 9:    
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000)) 
            running_loss = 0.0 

print('Finish Training')
