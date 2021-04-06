import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_, zeros_
from superpoint.models.unet_parts import *
import numpy as np

def simple_nms(scores, nms_radius: int):
    """ Fast Non-maximum suppression to remove nearby points """
    assert(nms_radius >= 0)

    def max_pool(x):
        return torch.nn.functional.max_pool2d(
            x, kernel_size=nms_radius*2+1, stride=1, padding=nms_radius)

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)


def remove_borders(keypoints, scores, border: int, height: int, width: int):
    """ Removes keypoints too close to the border """
    mask_h = (keypoints[:, 0] >= border) & (keypoints[:, 0] < (height - border))
    mask_w = (keypoints[:, 1] >= border) & (keypoints[:, 1] < (width - border))
    mask = mask_h & mask_w
    return keypoints[mask], scores[mask]


def top_k_keypoints(keypoints, scores, k: int):
    if k >= len(keypoints):
        return keypoints, scores
    scores, indices = torch.topk(scores, k, dim=0)
    return keypoints[indices], scores


def sample_descriptors(keypoints, descriptors, s: int = 8):
    """ Interpolate descriptors at keypoint locations """
    b, c, h, w = descriptors.shape
    keypoints = keypoints - s / 2 + 0.5
    keypoints /= torch.tensor([(w*s - s/2 - 0.5), (h*s - s/2 - 0.5)],
                              ).to(keypoints)[None]
    keypoints = keypoints*2 - 1  # normalize to (-1, 1)
    args = {'align_corners': True} if int(torch.__version__[2]) > 2 else {}
    descriptors = torch.nn.functional.grid_sample(
        descriptors, keypoints.view(b, 1, -1, 2), mode='bilinear', **args)
    descriptors = torch.nn.functional.normalize(
        descriptors.reshape(b, c, -1), p=2, dim=1)
    return descriptors

# from models.SubpixelNet import SubpixelNet
class SuperPoint(torch.nn.Module):
    """ Pytorch definition of SuperPoint Network. """
    default_config = {
        'descriptor_dim': 256,
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': -1,
        'remove_borders': 4,
    }
    def __init__(self, config):
        super(SuperPoint, self).__init__()
        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256
        self.config = {**self.default_config, **config}
        det_h = 65
        d1=self.config['descriptor_dim']
        self.inc = inconv(1, c1)
        self.down1 = down(c1, c2)
        self.down2 = down(c2, c3)
        self.down3 = down(c3, c4)
        self.relu = torch.nn.ReLU(inplace=True)
        # Detector Head.
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

        if self.config['weights']:
            checkpoints=torch.load(self.config['weights'])
            pretrained_dict = checkpoints['model_state_dict']
            #多卡训练单卡加载，带module时
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in pretrained_dict.items():
                if "module" in k:
                    name = k[7:] ## 多gpu 训练带moudule默认参数名字,预训练删除
                    new_state_dict[name] = v 
                else:
                    new_state_dict[k] = v 
            self.load_state_dict(new_state_dict)
            print("Loaded SuperPoint model")


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

        scores = torch.nn.functional.softmax(semi, 1)[:, :-1]#(1,64,64,64)
        b, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h*8, w*8)
        scores = simple_nms(scores, self.config['nms_radius'])

        # Extract keypoints
        keypoints = [
            torch.nonzero(s > self.config['keypoint_threshold'])
            for s in scores]
        scores = [s[tuple(k.t())] for s, k in zip(scores, keypoints)]

        # Discard keypoints near the image borders
        keypoints, scores = list(zip(*[
            remove_borders(k, s, self.config['remove_borders'], h*8, w*8)
            for k, s in zip(keypoints, scores)]))

        # Keep the k keypoints with highest score
        if self.config['max_keypoints'] >= 0:
            keypoints, scores = list(zip(*[
                top_k_keypoints(k, s, self.config['max_keypoints'])
                for k, s in zip(keypoints, scores)]))
        # Convert (h, w) to (x, y)
        keypoints = [torch.flip(k, [1]).float() for k in keypoints]

        # Extract descriptors
        descriptors = [sample_descriptors(k[None], d[None], 8)[0]
                       for k, d in zip(keypoints, desc)]

        return {
            'keypoints': keypoints,
            'scores': scores,
            'descriptors': descriptors,
        }


if __name__ == "__main__":
  weights_path="superpoint/models/weights/superPointNet_120000.pth.tar"
  input=torch.randn(3,1,240,320)
  net=SuperPointNet_gauss2()
  checkpoint = torch.load(weights_path,map_location=lambda storage, loc: storage)
  net.load_state_dict(checkpoint['model_state_dict'])
  net.eval()
  output=net(input)


