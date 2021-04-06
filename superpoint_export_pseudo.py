import os
import torch
import argparse
import time
import cv2
import yaml
import logging
from pathlib import Path
import numpy as np
from imageio import imread
from tqdm import tqdm
import torch.utils.data as data
from datasets.ALLSS import ALLSS
import torchvision.transforms as transforms
from utils.utils import combine_heatmap,draw_keypoints
from superpoint.models.model_wrap import SuperPointFrontend_torch, PointTracker

if __name__ == "__main__":
     # add parser
    parser = argparse.ArgumentParser() 
    parser.add_argument("--command", type=str,default='export_detector_homoAdapt')
    parser.add_argument("--config", type=str,default='superpoint/configs/magicpoint_allss_export.yaml')
    parser.add_argument("--exper_name", type=str,default='magicpoint_synth_homoAdapt_allss_50_[640,480]')
    parser.add_argument("--export_task", type=str,default='train',help="export mode: train or val")
    parser.add_argument("--save_output", type=str,default='Results/ALLSS',help="export mode: train or val")
    parser.add_argument("--eval", action="store_true",default=False,help="turn on eval mode")
    parser.add_argument("--outputImg", action="store_true",default=True, help="output image for visualization")
    parser.add_argument("--debug", action="store_true", default=False, help="turn on debuging mode")
    args = parser.parse_args()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.set_default_tensor_type(torch.FloatTensor)
    logging.basicConfig(format="[%(asctime)s %(levelname)s] %(message)s",datefmt="%m/%d/%Y %H:%M:%S",level=logging.INFO,)

    with open(args.config, "r") as f:
        config = yaml.load(f)
    print("check config!! ", config)

    # data loading
    test_set = ALLSS(export=True,task=args.export_task,**config['data'])
    test_loader = data.DataLoader(test_set, batch_size=1, shuffle=False)

    fe = SuperPointFrontend_torch(
        config=config,
        weights_path=config["pretrained"],
        nms_dist=config["model"]["nms"],
        conf_thresh=config["model"]["detection_threshold"],
        nn_thresh=0.7,
        cuda=False,
        device=device,
    )
    print("==> Successfully loaded pre-trained network.")
    fe.net_parallel()
    tracker = PointTracker(max_length=5, nn_thresh=fe.nn_thresh)
        
    count = 0
    for i, sample in tqdm(enumerate(test_loader)):
        img, mask_2D = sample["image"], sample["valid_mask"]
        img = img.transpose(0, 1)
        img_2D = sample["image_2D"].numpy().squeeze()
        mask_2D = mask_2D.transpose(0, 1)

        inv_homographies, homographies = (
            sample["homographies"],
            sample["inv_homographies"],
        )
        img, mask_2D, homographies, inv_homographies = (
            img.to(device),
            mask_2D.to(device),
            homographies.to(device),
            inv_homographies.to(device),
        )
        # sample = test_set[i]
        name = sample["name"][0]
        logging.info(f"name: {name}")

        # pass through network
        heatmap = fe.run(img, onlyHeatmap=True, train=False)#(100,1,240,320)
        outputs = combine_heatmap(heatmap, inv_homographies, mask_2D, device=device)
        pts = fe.getPtsFromHeatmap(outputs.detach().cpu().squeeze())  # (x,y, prob)

        # subpixel prediction
        if config["model"]["subpixel"]["enable"]:
            fe.heatmap = outputs  # tensor [batch, 1, H, W]
            print("outputs: ", outputs.shape)
            print("pts: ", pts.shape)
            pts = fe.soft_argmax_points([pts])
            pts = pts[0]

        ## top K points
        pts = pts.transpose()
        print("total points: ", pts.shape)
        print("pts: ", pts[:5])

        top_k=config["model"]["top_k"]
        if top_k:
            if pts.shape[0] > top_k:
                pts = pts[:top_k, :]
                print("topK filter: ", pts.shape)

        ## save keypoints
        pred = {}
        pred.update({"pts": pts})

        ## - make directories
        filename = str(name)
        save_output=os.path.join(args.save_output,args.exper_name,args.export_task)
        os.makedirs(save_output,exist_ok=True)
        path = Path(save_output, "{}.npz".format(filename))
        np.savez_compressed(path, **pred)

        ## output images for visualization labels
        output_images = args.outputImg
        if output_images:
            img_pts = draw_keypoints(img_2D * 255, pts.transpose(),s=1)
            f = save_output+'/'+filename + ".png"
            cv2.imwrite(str(f), img_pts)
        count += 1

    print("output pseudo ground truth: ", count)
