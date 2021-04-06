import os
import cv2
import torch
import time
import argparse
from pathlib import Path
import matplotlib.cm as cm
from torch.utils.data import DataLoader
from datasets.SSHIDataset import SSHIDataset
from superglue.models.matching_test import Matching
from superglue.models.utils import (make_matching_plot_fast, frame2tensor)

torch.set_grad_enabled(False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SuperPoint + SuperGlue registration test',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--exper_name', type=str, default='superpoint_glue_descriptor',help='path to source image directory')
    parser.add_argument('--img_dir', type=str, default='datasets/Amazon/',help='path to source image directory')
    parser.add_argument('--Result_dir', type=str, default='Results/Amazon/',help='Directory where to write matching Results ')
    parser.add_argument('--resize_scale', type=float, default=0.125,help='resize scale;height,weight=scale*height,scale*weight')
    parser.add_argument('--match_viz', default=True, help='Whether write the match result or not')
    parser.add_argument('--show_keypoints', default=True,help='Show the detected keypoints')
    parser.add_argument('--descriptor_dim',type=int, default=128,help='The dimension of feature descriptor')

    #superpoint hyper parameter
    parser.add_argument('--superpoint_weights',type=str,default="superpoint/models/weights/superPointNet_allss_descriptor_128.pth.tar") 
    parser.add_argument('--keypoint_threshold', type=float, default=0.005,help='SuperPoint keypoint detector confidence threshold')
    parser.add_argument('--nms_radius', type=int, default=4,help='SuperPoint Non Maximum Suppression (NMS) radius'' (Must be positive)')
    parser.add_argument('--max_keypoints', type=int, default=-1, help='Maximum number of keypoints detected by Superpoint'' (\'-1\' keeps all keypoints)')  
   
    #superglue hyper parameter
    parser.add_argument('--superglue_weights', type=str, default='superglue/models/weights/SuperGlue_allss_descriptor_128.pth',help='SuperGlue weights')
    parser.add_argument('--keypoint_encoder', default=[32, 64, 128],help='The dimension of keypoint encoder')
    parser.add_argument('--sinkhorn_iterations', type=int, default=30,help='Number of Sinkhorn iterations performed by SuperGlue')    
    parser.add_argument('--match_threshold', type=float, default=0.1,help='SuperGlue match threshold')

    opt = parser.parse_args()
    print(opt)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    config = {
        'superpoint': {
            'weights':opt.superpoint_weights,
            'descriptor_dim':opt.descriptor_dim,
            'nms_radius': opt.nms_radius,
            'keypoint_threshold': opt.keypoint_threshold,
            'max_keypoints': opt.max_keypoints
        },
        'superglue': {
            'weights': opt.superglue_weights,
            'descriptor_dim':opt.descriptor_dim,
            'keypoint_encoder':opt.keypoint_encoder,
            'sinkhorn_iterations': opt.sinkhorn_iterations,
            'match_threshold': opt.match_threshold,
        }
    }

    # data path
    source_dir=opt.img_dir+'source1/'
    template_dir=opt.img_dir+'template1/'
    template_img_name=os.listdir(template_dir)[0]
    template_img_path=template_dir+template_img_name

    # load eval data
    test_dataset=SSHIDataset(source_dir,template_img_path,opt.resize_scale)
    test_loader=DataLoader(test_dataset,batch_size=1,shuffle=False,num_workers=1)

    # model
    matching = Matching(config).eval().to(device)

    #eval
    for iter,(source_original,source_image,template_image,filename) in enumerate(test_loader):

        source_tensor=source_image.float().to(device) 
        template_tensor=template_image.float().to(device) 
        #registration and compute time
        start = time.perf_counter()
        pred = matching({'image0': source_tensor,'image1':template_tensor})
        kpts0 = pred['keypoints0'][0].cpu().numpy()
        kpts1 = pred['keypoints1'][0].cpu().numpy()
        matches = pred['matches0'][0].cpu().numpy()
        confidence = pred['matching_scores0'][0].cpu().numpy()
        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]     
        if len(mkpts0)>3:
            # Matrix,mask=cv2.estimateAffine2D(mkpts0,mkpts1,method=cv2.RANSAC,ransacReprojThreshold=7)
            Matrix,mask=cv2.estimateAffinePartial2D(mkpts0,mkpts1,method=cv2.RANSAC,ransacReprojThreshold=7)
            if opt.resize_scale is not None:
                Matrix[:,2]=Matrix[:,2]/opt.resize_scale
            flag=(mask>0).ravel().tolist()
            mkpts0,mkpts1=mkpts0[flag],mkpts1[flag]
        end = time.perf_counter()
        elapsed = end-start
        print("Time used:",elapsed)

        source_image=source_image.squeeze().cpu().numpy()*255
        template_image=template_image.squeeze().cpu().numpy()*255

        source_original=source_original.squeeze().cpu().numpy()*255
        Transform=cv2.warpAffine(source_original,Matrix,(source_original.shape[1],source_original.shape[0]))

        #output the Results
        if not os.path.exists(opt.Result_dir):
            os.makedirs(opt.Result_dir)

        #Output the Transform Results
        exper_name=opt.exper_name+"_"+str(opt.exper_name)
        Transform_dir=os.path.join(opt.Result_dir,opt.exper_name,'Transform/')
        if not os.path.exists(Transform_dir):
            os.makedirs(Transform_dir)
        cv2.imwrite(Transform_dir+'trans_{}'.format(filename[0]),Transform)

        #Output the Match Results
        color = cm.jet(confidence[valid])
        text = [
            'SuperGlue',
            'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
            'Matches: {}'.format(len(mkpts0))
        ]

        k_thresh = matching.superpoint.config['keypoint_threshold']
        m_thresh = matching.superglue.config['match_threshold']
        small_text = [
            'Keypoint Threshold: {:.4f}'.format(k_thresh),
            'Match Threshold: {:.2f}'.format(m_thresh),
            ' '
        ]

        out = make_matching_plot_fast(
            source_image, template_image, kpts0, kpts1, mkpts0, mkpts1, color, text,
            path=None, show_keypoints=opt.show_keypoints, small_text=small_text)  

        Match_dir=os.path.join(opt.Result_dir,opt.exper_name,'Match/')
        if not os.path.exists(Match_dir):
            os.makedirs(Match_dir)

        out_file = str(Path(Match_dir, filename[0]))
        print('\nWriting image to {}'.format(out_file))
        cv2.imwrite(out_file, out)
