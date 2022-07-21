import torch
import torch.nn as nn

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
        self.cv2 = nn.Conv2d(size_in, size_out, 3, padding = 'same')
        self.relu = nn.ReLU()

    def forward(self, x):
        x0 = self.cv1(x)
        x0 = self.relu(x0)

        x1 = self.cv2(x)
        x1 = self.relu(x1)
        # print("inside inception")
        # print("X0: ",x0.shape)
        # print("X1: ",x1.shape)
        x = torch.cat((x0, x1), axis = 1)
        # print("X-cat: ",x.shape)
        return x

class MultiSacleUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.inception_1 = Inception(3, 64)
        self.inception_2 = Inception(128, 128)
        self.inception_3 = Inception(512, 128)
        self.inception_4 = Inception(384, 64)

        self.maxPool_1 = nn.MaxPool2d(SIZE_SUB, padding = (1,1))
        self.maxPool_2 = nn.MaxPool2d(SIZE_TOP, padding = (1,1))

        self.upSample_1 = nn.Upsample(size = SIZE_SUB)
        self.upSample_2 = nn.Upsample(size = 256)

        self.fc1 = nn.Linear(256,256)
        self.fc2 = nn.Linear(128,50)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=3)

    def forward(self, x):
        # print("input to forward pass",x.shape)
        x0 = self.inception_1(x)
        # print("after inception 1",x0.shape)
        x1 = self.maxPool_1(x0)
        # print("after maxpool 1",x1.shape)

        x1 = self.inception_2(x1)
        # print("after inception 2",x1.shape)
        x2 = self.maxPool_2(x1)
        # print("after maxpool 2",x2.shape)


        xg = x2
        # xg = np.transpose(xg,(0,2,3,1))
        xg = xg.view(-1,1,1,256)
        # print("input to linear layer", xg.shape)
        xg = self.fc1(xg)
        xg = xg.view(-1,256,1,1)
        # print("output of linear layer", xg.shape)
        y2 = xg

        y1 = self.upSample_1(y2)
        # print("after upsample 1",y1.shape)
        y1 = torch.cat((x1, y1), dim=1)
        # print("after cat 1",y1.shape)
        y1 = self.inception_3(y1)
        # print("after inception 3",y1.shape)


        y0 = self.upSample_2(y1)
        # print("after upsample 2",y0.shape)
        y0 = torch.cat((x0, y0), dim=1)
        # print("after cat 2",y0.shape)
        y0 = self.inception_4(y0)
        # print("after inception 4",y0.shape)

        y0 = y0.view(-1,256,256,128)
        outputs = self.fc2(y0)
        outputs = outputs.view(-1,256,256,50)
        # print("after last conv 1x1", outputs.shape)

        outputs = self.softmax(outputs)
        # print("after softmax", outputs.shape)

        mask_abs = torch.abs(outputs)
        mask_sum = torch.sum(mask_abs, axis = -1)
        mask_sign = torch.sign(mask_sum)
        single_mask = torch.unsqueeze(mask_sign, dim = -1)
        mask = torch.tile(single_mask, (1,1,50))
        not_mask = 1 - single_mask

        masked_output = torch.multiply(outputs, mask)
        concat_outputs = torch.concat([masked_output, not_mask], axis = -1)



        return  concat_outputs
