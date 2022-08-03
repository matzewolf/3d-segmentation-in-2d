import torch
import torch.nn as nn

NUM_POINTS = 2048
NUM_CUTS = 32
INPUT_L = 256
PARTS = 50
SIZE_SUB = 16
SIZE_TOP = 16
SIZE_IMG = SIZE_SUB*SIZE_SUB

"""
#from their repo
def build_mask(inputs):
    print("input shape",inputs.shape)
    mask_mat = torch.detach(torch.abs(inputs)).view(-1, INPUT_L, INPUT_L, PARTS)
    print("mask_mat shape",mask_mat.shape)
    mask = torch.detach(torch.sign(torch.detach(torch.sum(mask_mat, dim=-1))))
    print("mask shape",mask.shape)
    return mask
"""
def build_mask(inputs):
    input_for_mask = inputs.view(-1, INPUT_L, INPUT_L, 3)
    mask_abs = torch.detach(torch.abs(inputs))
    mask_sum = torch.detach(torch.sum(mask_abs, axis=1))
    mask_sign = torch.detach(torch.sign(mask_sum))
    mask_sign = mask_sign.unsqueeze(-1)
    tiled_mask = torch.tile(mask_sign, (1, 1, 1,50))       
    not_mask = 1 - mask_sign
    return tiled_mask, not_mask

class Inception(nn.Module):
    def __init__(self, size_in, size_out):
        super().__init__()
        self.size_in, self.size_out = size_in, size_out
        self.cv1 = nn.Conv2d(size_in, size_out, 1, padding='same')
        self.cv2 = nn.Conv2d(size_in, size_out, 3, padding='same')
        self.relu = nn.ReLU()

    def forward(self, x):
        # first conv path
        x0 = self.cv1(x)
        x0 = self.relu(x0)
        # second conv path
        x1 = self.cv2(x)
        x1 = self.relu(x1)
        # concatenate different conv paths
        x = torch.cat((x0, x1), axis=1)
        return x

class MultiSacleUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.inception_1 = Inception(3, 64)
        self.inception_2 = Inception(128, 128)
        self.inception_3 = Inception(512, 128)
        self.inception_4 = Inception(384, 64)

        self.maxPool_1 = nn.MaxPool2d(SIZE_SUB, padding=(1, 1))
        self.maxPool_2 = nn.MaxPool2d(SIZE_TOP, padding=(1, 1))

        self.upSample_1 = nn.Upsample(size=SIZE_SUB)
        self.upSample_2 = nn.Upsample(size=INPUT_L)

        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(128, 50)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=3)

    def forward(self, x):
        # masking
        mask,not_mask = build_mask(x)
        # mask = mask[:,None]        
        # tiled_mask = torch.tile(mask.reshape((1,INPUT_L,INPUT_L,1)), (1, 1, 1,50))       
        # the UNET encoder
        x0 = self.inception_1(x)
        x1 = self.maxPool_1(x0)
        x1 = self.inception_2(x1)
        x2 = self.maxPool_2(x1)
        # the UNET bottle-neck
        xg = x2
        # print("x2 shape.",x2.shape)
        xg = torch.permute(xg,(0,2,3,1))
        # print("x2 shape.",xg.shape)
        xg = self.fc1(xg)
        # print("x2 shape.",xg.shape)
        xg = torch.permute(xg,(0,3,1,2))
        # print("x2 shape.",xg.shape)
        y2 = xg
        # the UNET decoder
        y1 = self.upSample_1(y2)
        y1 = torch.cat((x1, y1), dim=1)
        y1 = self.inception_3(y1)
        y0 = self.upSample_2(y1)
        y0 = torch.cat((x0, y0), dim=1)
        y0 = self.inception_4(y0)
        # the last feed forward
        y0 = torch.permute(y0,(0,2,3,1))
        y0 = y0.view(-1, INPUT_L, INPUT_L, 128)
        outputs = self.fc2(y0)
        # applying masking 
        outputs = torch.mul(outputs,mask)
        # print(outputs.shape)
        # print(not_mask.shape)
        outputs = torch.cat([outputs,not_mask], dim=-1)
        outputs = torch.permute(outputs,(0,-1,1,2))
        return outputs