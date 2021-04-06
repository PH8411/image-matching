import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_, zeros_
from superpoint.models.unet_parts import *
from superpoint.models.model_utils import flattenDetection
import numpy as np

class SuperPoint(torch.nn.Module):
    """ Pytorch definition of SuperPoint Network. """
    def __init__(self, descriptor_length=256):
        super(SuperPoint, self).__init__()
        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256
        det_h = 65
        d1 = descriptor_length
        self.inc = inconv(1, c1)
        self.down1 = down(c1, c2)
        self.down2 = down(c2, c3)
        self.down3 = down(c3, c4)
        self.relu = torch.nn.ReLU(inplace=True)
        self.convPa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.bnPa = nn.BatchNorm2d(c5)
        self.convPb = torch.nn.Conv2d(c5, det_h, kernel_size=1, stride=1, padding=0)
        self.bnPb = nn.BatchNorm2d(det_h)
        # Descriptor Head.
        self.convDa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.bnDa = nn.BatchNorm2d(c5)
        self.convDb = torch.nn.Conv2d(c5, d1, kernel_size=1, stride=1, padding=0)
        self.bnDb = nn.BatchNorm2d(d1)
        self.output = None

    def forward(self, x):
        """ Forward pass that jointly computes unprocessed point and descriptor
        tensors.
        Input
          x: Image pytorch tensor shaped N x 1 x patch_size x patch_size.
        Output
          semi: Output point pytorch tensor shaped N x 65 x H/8 x W/8.
          desc: Output descriptor pytorch tensor shaped N x 256 x H/8 x W/8.
        """
        # Let's stick to this version: first BN, then relu
        x1 = self.inc(x)#(batch,64,120,160)
        x2 = self.down1(x1)#(batch,64,60,80)
        x3 = self.down2(x2)#(batch,128,30,40)
        x4 = self.down3(x3)#(batch,128,15,30)

        # Detector Head.
        cPa = self.relu(self.bnPa(self.convPa(x4)))#(batch,256,15,30)
        semi = self.bnPb(self.convPb(cPa))#(batch,65,15,30)
        # Descriptor Head.
        cDa = self.relu(self.bnDa(self.convDa(x4)))#(batch,256,15,30)
        desc = self.bnDb(self.convDb(cDa))#(batch,256,15,30)

        dn = torch.norm(desc, p=2, dim=1) # Compute the norm:(batch,15,30)
        desc = desc.div(torch.unsqueeze(dn, 1)) # Divide by norm to normalize:(batch,256,15,30)
        output = {'semi': semi, 'desc': desc}
        self.output = output

        return output