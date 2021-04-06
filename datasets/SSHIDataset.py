import os
import cv2
import numpy as np
from torch.utils.data import Dataset

class SSHIDataset(Dataset):
    def __init__(self,source_dir,template_path,resize_scale):
        self.source_list=os.listdir(source_dir)
        self.source_dir = []
        self.source_dir += [source_dir + f for f in self.source_list]
        self.template_path=template_path
        self.resize_scale=resize_scale
    
    def __getitem__(self, index):
        filename=self.source_list[index]
        source_path=self.source_dir[index]
        source_original = cv2.imread(source_path, cv2.IMREAD_GRAYSCALE)
        template_original = cv2.imread(self.template_path,cv2.IMREAD_GRAYSCALE)

        if self.resize_scale is not None:
            source_image = cv2.resize(source_original,(int(self.resize_scale*source_original.shape[1]),int(self.resize_scale*source_original.shape[0])))
            template_image = cv2.resize(template_original,(int(self.resize_scale*template_original.shape[1]),int(self.resize_scale*template_original.shape[0])))
        else:
            source_image = source_original
            template_image = template_original
        source_original=source_original[None]/255   
        source_image=source_image[None]/255                
        template_image = template_image[None]/255
        return source_original,source_image,template_image,filename
    
    def __len__(self):
        return len(self.source_list)

