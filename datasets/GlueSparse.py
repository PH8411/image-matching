import os
import cv2
import torch
import numpy as np
from scipy.spatial.distance import cdist
from torch.utils.data import Dataset
from superpoint.models.superpoint_test import SuperPoint
from utils.utils import frame2tensor

class GlueSparse(Dataset):
    """Sparse correspondences dataset."""

    def __init__(self,train_path,sp_config,resize,device):
        self.device=device
        self.resize=resize
        self.files=[]
        self.files=[train_path+'/'+ f for f in os.listdir(train_path)]
        self.superpoint = SuperPoint(sp_config).to(device)
        self.superpoint.eval()
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file_name=self.files[index]
        image=cv2.imread(file_name,0)
        image=cv2.resize(image,(self.resize[0],self.resize[1]))
        width,height=image.shape[:2]
        corners=np.array([[0,0],[0,height],[width,0],[width,height]],dtype=np.float32)
        warp=np.random.randint(-100,100,size=(4,2)).astype(np.float32)
        M = cv2.getPerspectiveTransform(corners, corners + warp)
        warped = cv2.warpPerspective(src=image, M=M, dsize=(image.shape[1], image.shape[0]))
       
        # extract keypoints of the image pair using SuperPoint
        image_tensor = frame2tensor(image, self.device)
        warped_tensor = frame2tensor(warped, self.device)

        pred1 = self.superpoint(image_tensor)
        pred2 = self.superpoint(warped_tensor)

        #keypoints,descriptors,scores
        kp1_np=pred1["keypoints"][0].cpu().detach().numpy()#(636,2)
        kp2_np=pred2["keypoints"][0].cpu().detach().numpy()#(666,2)

        descs1=pred1["descriptors"][0].cpu().detach().numpy().transpose()#(636,256)
        descs2=pred2["descriptors"][0].cpu().detach().numpy().transpose()#(666,256)

        scores1_np=pred1["scores"][0].cpu().detach().numpy()#(636,)
        scores2_np=pred2["scores"][0].cpu().detach().numpy()#(666,)

        # skip this image pair if no keypoints detected in image
        if len(kp1_np) < 1 or len(kp2_np) < 1:
            return{
                'keypoints0': torch.zeros([0, 0, 2], dtype=torch.double),
                'keypoints1': torch.zeros([0, 0, 2], dtype=torch.double),
                'descriptors0': torch.zeros([0, 2], dtype=torch.double),
                'descriptors1': torch.zeros([0, 2], dtype=torch.double),
                'image0': image,
                'image1': warped,
                'file_name': file_name
            } 

        # obtain the matching matrix of the image pair
        kp1_projected = cv2.perspectiveTransform(kp1_np.reshape((1, -1, 2)), M)[0, :, :] 
        dists = cdist(kp1_projected, kp2_np)#(636,666)

        min1 = np.argmin(dists, axis=0) # 在axis=0方向上找最小的值并返回索引
        min2 = np.argmin(dists, axis=1) # 在axis=1方向上找最小的值并返回索引

        min1v = np.min(dists, axis=1)
        min1f = min2[min1v < 3]

        xx = np.where(min2[min1] == np.arange(min1.shape[0]))[0]#最佳匹配点，两个方向最近的匹配点
        matches = np.intersect1d(min1f, xx)#34

        missing1 = np.setdiff1d(np.arange(kp1_np.shape[0]), min1[matches])#返回两个数组的差集，非配对点
        missing2 = np.setdiff1d(np.arange(kp2_np.shape[0]), matches)#返回两个数组的差集，非配对点

        MN = np.concatenate([min1[matches][np.newaxis, :], matches[np.newaxis, :]])#正确匹配的点对应
        MN2 = np.concatenate([missing1[np.newaxis, :], (len(kp2_np)) * np.ones((1, len(missing1)), dtype=np.int64)])#没有匹配点的dustbin 列
        MN3 = np.concatenate([(len(kp1_np)) * np.ones((1, len(missing2)), dtype=np.int64), missing2[np.newaxis, :]])#没有匹配点的dustbin 行
        all_matches = np.concatenate([MN, MN2, MN3], axis=1)

        kp1_np = kp1_np.reshape((1, -1, 2))#(1,636,2)
        kp2_np = kp2_np.reshape((1, -1, 2))#(1,666,2)
        descs1 = np.transpose(descs1)#(256,636)
        descs2 = np.transpose(descs2)#(256,666)

        image = torch.from_numpy(image/255.).double()[None].to(self.device)
        warped = torch.from_numpy(warped/255.).double()[None].to(self.device)

        return{
            'keypoints0': list(kp1_np),
            'keypoints1': list(kp2_np),
            'descriptors0': list(descs1),
            'descriptors1': list(descs2),
            'scores0': list(scores1_np),
            'scores1': list(scores2_np),
            'image0': image,
            'image1': warped,
            'matches':MN,
            'all_matches': list(all_matches),
            'file_name': file_name
        } 