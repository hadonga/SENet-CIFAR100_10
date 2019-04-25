# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 19:39:42 2019
from pytorch tutorial 
@author: HD
"""
#input dataset CIFAR100


import torch
import torchvision
import torchvision.transforms as transforms

#transform to tensor meanwhile normalize -- from pytorch tutorial 

transform= transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

trainset = torchvision.datasets.CIFAR100(root='./data',train=True,
                                         download=True,transform=transform)

trainloader=torch.utils.data.DataLoader(trainset,batch_size=512,
                                        shuffle=True,num_workers=0)


testset =torchvision.datasets.CIFAR100(root='./data',train=False,
                                       download=True,transform=transform)

testloader=torch.utils.data.DataLoader(testset,batch_size=512,
                                       shuffle=False,num_workers=0)
#print torch.cuda.is_available()