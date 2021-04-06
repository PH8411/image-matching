import os
import cv2
import time
import argparse
import numpy as np
from Traditional.registration import SIFT_REGIS,ORB_REGIS

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Traditional Registration',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--Method', type=str, default='SIFT',help='The method of feature based registration')
    parser.add_argument('--img_dir', type=str, default='datasets/Amazon/',help='path to source image directory')
    parser.add_argument('--Result_dir', type=str, default='Results/Amazon/',help='Directory where to write matching Results ')
    parser.add_argument('--resize_scale', type=float, default=0.5,help='resize scale;height,weight=scale*height,scale*weight')
    parser.add_argument('--match_viz', default=True, help='Whether write the match result or not')
    opt = parser.parse_args()

    #path for source and template img
    source_dir=opt.img_dir+'source1/'
    template_dir=opt.img_dir+'template1/'
    source_dir_List=os.listdir(source_dir)
    template_img_name=os.listdir(template_dir)[0]
    template_img=cv2.imread(template_dir+template_img_name)

    for i in source_dir_List:
        source_img_name=os.path.join(source_dir+i)
        source_img=cv2.imread(source_img_name)
       
        #registration and compute time
        start = time.perf_counter()
        if opt.Method=='SIFT':
            Matrix,match_img=SIFT_REGIS(source_img,template_img,opt.resize_scale,opt.match_viz)
        if opt.Method=='ORB':
            Matrix,match_img=ORB_REGIS(source_img,template_img,opt.resize_scale,opt.match_viz)

        if opt.resize_scale is not None:
            Matrix[:,2]=Matrix[:,2]/opt.resize_scale
        end = time.perf_counter()
        elapsed = end-start
        print("Time used:",elapsed)

        #output the matching results
        if not os.path.exists(opt.Result_dir):
            os.makedirs(opt.Result_dir)

        Transform_dir=os.path.join(opt.Result_dir,opt.Method+'/Transform1/')
        if not os.path.exists(Transform_dir):
            os.makedirs(Transform_dir)

        Match_dir=os.path.join(opt.Result_dir,opt.Method+'/Match1/')
        if not os.path.exists(Match_dir):
            os.makedirs(Match_dir)
        
        Transform=cv2.warpAffine(source_img,Matrix,(template_img.shape[1],template_img.shape[0]))   
        cv2.imwrite(Transform_dir+'trans_{}'.format(i),Transform)
        
        if opt.match_viz:
            cv2.imwrite(Match_dir+'match_{}'.format(i),match_img)

