import os
import cv2
import torch
import argparse
import numpy as np
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from datasets.SSHIDataset import SSHIDataset
from superpoint.models.superpoint_test import SuperPoint
from utils.utils import (make_plot_matches, frame2tensor)

MIN_MATCH_COUNT = 4

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SuperPoint_flann test',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--img_dir', type=str, default='datasets/Amazon/',help='path to source image directory')
    parser.add_argument('--Result_dir', type=str, default='Results/Camera/superpoint_allss_descriptor_128',help='Directory where to write matching Results ')
    parser.add_argument('--resize_scale', type=float, default=0.25,help='resize scale;height,weight=scale*height,scale*weight')
    parser.add_argument('--match_viz', default=True, help='Whether write the match result or not')

    parser.add_argument('--weights_path', type=str, default='superpoint/models/weights/superPointNet_allss_descriptor_128.pth.tar',help='pretrain model path')
    parser.add_argument('--descriptor_dim', type=int, default=128, help='the dimension of descriptor')
    parser.add_argument('--max_keypoints', type=int, default=1200, help='Maximum number of keypoints detected by Superpoint'' (\'-1\' keeps all keypoints)')
    parser.add_argument('--keypoint_threshold', type=float, default=0.005,help='SuperPoint keypoint detector confidence threshold')
    parser.add_argument('--nms_radius', type=int, default=4,help='SuperPoint Non Maximum Suppression (NMS) radius'' (Must be positive)')
    opt = parser.parse_args()
    print(opt)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_grad_enabled(False)

    config = {
        'superpoint': {
            'weights':opt.weights_path,
            'descriptor_dim': opt.descriptor_dim,
            'nms_radius': opt.nms_radius,
            'keypoint_threshold': opt.keypoint_threshold,
            'max_keypoints': opt.max_keypoints},
             }

    # data path
    source_dir=opt.img_dir+'source1/'
    template_dir=opt.img_dir+'template1/'
    template_img_name=os.listdir(template_dir)[0]
    template_img_path=template_dir+template_img_name

    # load eval data
    test_dataset=SSHIDataset(source_dir,template_img_path,opt.resize_scale)
    test_loader=DataLoader(test_dataset,batch_size=1,shuffle=False,num_workers=1)

    superpoint = SuperPoint(config.get('superpoint', {}))
    superpoint.to(device).eval()

    for iter,(source_original,source_image,template_image,filename) in enumerate(test_loader):

        image1_tensor = source_image.float().to(device) 
        image2_tensor = template_image.float().to(device)

        pred1 = superpoint(image1_tensor)
        pred2 = superpoint(image2_tensor)

        KeyP1=pred1["keypoints"][0].cpu().detach().numpy()
        KeyP2=pred2["keypoints"][0].cpu().detach().numpy()

        Desc1=pred1["descriptors"][0].cpu().detach().numpy().transpose()
        Desc2=pred2["descriptors"][0].cpu().detach().numpy().transpose()

        FlANN_INDEX_KDTREE=1
        index_params=dict(algorithm=FlANN_INDEX_KDTREE,trees=5)
        search_params=dict(checks=50)
        flann=cv2.FlannBasedMatcher(index_params,search_params)
        knn_matches=flann.knnMatch(Desc1,Desc2,k=2)

        good_matches=[]
        for m,n in knn_matches:
            if m.distance<0.7*n.distance:
                good_matches.append([m])
        
        if len(good_matches)>MIN_MATCH_COUNT:
            src_pts = np.float32([KeyP1[m[0].queryIdx] for m in good_matches])
            dst_pts = np.float32([KeyP2[m[0].trainIdx] for m in good_matches])
            match_dist = np.float32([m[0].distance for m in good_matches])
            Matrix, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC,ransacReprojThreshold=7) 

            #output the transformed images           
            if opt.resize_scale is not None:
                Matrix[:,2]=Matrix[:,2]/opt.resize_scale
            source_original=source_original.squeeze().cpu().numpy()*255
            Transform=cv2.warpAffine(source_original,Matrix,(source_original.shape[1],source_original.shape[0]))
            Transform_dir=os.path.join(opt.Result_dir,'transformed/')
            os.makedirs(Transform_dir,exist_ok=True)
            cv2.imwrite(Transform_dir+'trans_{}'.format(filename[0]),Transform)

            #output the matching images
            RansacMask = (mask==1).ravel().tolist()
            src_ransac_pts=src_pts[RansacMask]
            dst_ransac_pts=dst_pts[RansacMask]
            match_ransac_dist=match_dist[RansacMask]
            if match_ransac_dist.max() > 1:
                best, worst = 0, Desc1.shape[1] * 2  # estimated range
            else:
                best, worst = 0, 1

            # 1: for best match, 0: for worst match
            match_scores = match_ransac_dist / worst
            match_scores[match_scores > 1] = 1
            match_scores[match_scores < 0] = 0
            match_scores = 1 - match_scores

            image01=source_image.squeeze().cpu().numpy()*255
            image02=template_image.squeeze().cpu().numpy()*255
            image1=np.repeat(image01[...,np.newaxis],3,2)
            image2=np.repeat(image02[...,np.newaxis],3,2)

            img = make_plot_matches(image1, image2, src_ransac_pts, dst_ransac_pts, match_scores, layout='lr')
            Match_dir=os.path.join(opt.Result_dir,"Match/")
            os.makedirs(Match_dir,exist_ok=True)
            cv2.imwrite(Match_dir+"match_{}".format(filename[0]),img)