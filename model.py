import torch
import torch.nn as nn
import h5py
import numpy as np


NUM_POINTS = 2048
NUM_CUTS = 32
SIZE_SUB = 16
SIZE_TOP = 16
SIZE_IMG = SIZE_SUB*SIZE_SUB


class Inception(nn.Module):
    def __init__(self, size_in, size_out):
        super().__init__()
        self.size_in, self.size_out = size_in, size_out
        
        self.cv1 = nn.Conv2d(size_in, size_out, 1, padding = 'same')
        self.cv2 = nn.conv2d(size_in, size_out, 3, padding = 'same')
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x0 = self.cv1(x)
        x0 = self.relu(x0)
        
        x1 = self.cv2(x)
        x1 = self.relu(x1)
        
        x = torch.cat((x0,x1))
        return x   
    
    
    

class MultiSacleUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.inception_1 = Inception(3,64)
        self.inception_2 = Inception(128,128)
        self.inception_3 = Inception(512,128)
        self.inception_4 = Inception(384,64)
        
        self.maxPool = nn.MaxPool2d(SIZE_SUB, padding = 'valid')
        self.maxPool = nn.MaxPool2d(SIZE_TOP, padding = 'valid')

        self.upSample = nn.Upsample(size = SIZE_SUB)
        self.upSample = nn.Upsample(size = SIZE_TOP)
        
        self.fc1 = nn.Linear(256,256)
        self.fc2 = nn.Linear(128,50)
        
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, x):
        
        x0 = self.inception_1(x)
        x1 = self.maxPool(x0)
        
        x1 = self.inception_2(x1)
        x2 = self.maxPool(x1)
        
        xg = x2
        xg = self.fc1(xg)
        y2 = xg
        
        y1 = self.upSample(y2)
        y1 = torch.cat((x1, y1), dim=1)
        y1 = self.inception_3(y1)
        
        
        y0 = self.upSample(y1)
        y0 = torch.cat((x0, y0), dim=1)
        y0 = self.inception_4(y0)
        
        
        outputs = self.fc2(y0)
        
        
        return  outputs
    
