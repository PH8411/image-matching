import os
import torch
import random
import argparse
import numpy as np

import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
import matplotlib.cm as cm
import torch.multiprocessing
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from datasets.GlueSparse import GlueSparse
from superglue.models.superglue_train import SuperGlue
from utils.utils import make_matching_plot

#model introduction and path
parser = argparse.ArgumentParser(description='Image pair matching and pose evaluation with SuperGlue',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--image_path', type=str, default='datasets/ALLSS/', help='Path to the directory of training imgs.')
parser.add_argument('--Result_dir', type=str, default='Results/ALLSS/superglue_descriptor_128',help='Path to the directory')

#base parameter
parser.add_argument('--epoch', type=int, default=200,help='Number of epoches')
parser.add_argument('--batch_size', type=int, default=1,help='batch_size')
parser.add_argument('--learning_rate', type=int, default=0.0001,help='Learning rate')
parser.add_argument('--shuffle', default=True ,help='Shuffle ordering of pairs before processing')
parser.add_argument('--show_keypoints', default=True ,help='Plot the keypoints in addition to the matches')
parser.add_argument('--viz_extension', type=str, default='png',choices=['png','pdf'],help='visualization file extension.Use pdf for highest-quality')

#model hyper parameter 
parser.add_argument('--superpoint_weights',type=str,default="superpoint/models/weights/superPointNet_allss_descriptor_128.pth.tar") 
parser.add_argument('--descriptor_dim',type=int, default=128,help='The dimension of feature descriptor')
parser.add_argument('--keypoint_encoder', default=[32,64,128],help='The dimension of keypoint encoder')
parser.add_argument('--max_keypoints', type=int, default=1200,help='Maximum number of keypoints detected by Superpoint'' (\'-1\' keeps all keypoints)')
parser.add_argument('--keypoint_threshold', type=float, default=0.005,help='SuperPoint keypoint detector confidence threshold')
parser.add_argument('--nms_radius', type=int, default=4,help='SuperPoint Non Maximum Suppression (NMS) radius'' (Must be positive)')

parser.add_argument('--sinkhorn_iterations', type=int, default=30,help='Number of Sinkhorn iterations performed by SuperGlue')
parser.add_argument('--match_threshold', type=float, default=0.2,help='SuperGlue match threshold')
parser.add_argument('--resize',default=[640,480],help='The size of image')

#pretrain or checkpoint model
parser.add_argument('--pretrain_weights', type=str, default='',help='SuperGlue official weights')
parser.add_argument('--checkpoints_dir', type=str, default='checkpoints/', help='models saved here')
parser.add_argument('--checkpoints_name', type=str, default='', help='checkpoint model name')
parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints')
parser.add_argument('--resume', default=False, help='if specified, the model start from checkpoints')


if __name__ == '__main__':
    opt = parser.parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(opt) 

    config = {
        'superpoint':{
                'weights':opt.superpoint_weights,
                'descriptor_dim': opt.descriptor_dim,      
                'nms_radius': opt.nms_radius,
                'keypoint_threshold': opt.keypoint_threshold,
                'max_keypoints': opt.max_keypoints,
        },
        'superglue': {
            'descriptor_dim': opt.descriptor_dim,
            'keypoint_encoder': opt.keypoint_encoder,
            'sinkhorn_iterations': opt.sinkhorn_iterations,
            'match_threshold': opt.match_threshold,
        }
    }

    # load training data
    train_path= os.path.join(opt.image_path,'train')
    train_set = GlueSparse(train_path,config.get('superpoint', {}),opt.resize,device)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, shuffle=True, batch_size=opt.batch_size, drop_last=True)

    start_epoch=0
    writer =SummaryWriter(opt.Result_dir+'/logdir')
    superglue = SuperGlue(config.get('superglue', {})).to(device)
    optimizer = torch.optim.Adam(superglue.parameters(), lr=opt.learning_rate)

    if opt.pretrain_weights in ['indoor','outdoor']:
        path_pretrain = 'superglue/models/weights/superglue_{}.pth'.format(opt.pretrain_weights)
        pretrain = torch.load(path_pretrain)
        superglue.load_state_dict(pretrain)
        print('Loaded SuperGlue official model (\"{}\" weights)'.format(opt.pretrain_weights))

    if opt.resume:
        path_checkpoints= os.path.join(opt.Result_dir,checkpoints_dir,opt.checkpoints_name)
        checkpoint= torch.load(path_checkpoints)
        superglue.load_state_dict(checkpoint['net'])
        start_epoch=checkpoint['epoch']
        print('Loaded checkpoint model (\"{}\" weights)'.format(opt.checkpoints_name))
    
    
    # store viz results
    eval_output_dir = Path(opt.Result_dir,'match')
    eval_output_dir.mkdir(exist_ok=True, parents=True)
    print('Will write visualization images to','directory \"{}\"'.format(eval_output_dir))

    # start training
    for epoch in range(start_epoch+1, opt.epoch+1):
        epoch_loss = 0
        mean_loss = []
       
        for i, pred in enumerate(train_loader):            
            for k in pred:
                if k != 'file_name' and k!='image0' and k!='image1':
                    if type(pred[k]) == torch.Tensor:
                        pred[k] = Variable(pred[k].cuda()).type(torch.cuda.FloatTensor)
                    else:
                        pred[k] = Variable(torch.stack(pred[k]).cuda())

            superglue.train()              
            data = superglue(pred)
            for k, v in pred.items():
                pred[k] = v[0]
            pred = {**pred, **data}

            if pred['skip_train'] == True: # image has no keypoint
                continue
            
            # process loss
            Loss = pred['loss']
            epoch_loss += Loss.item()
            mean_loss.append(Loss)
            optimizer.zero_grad()  
            Loss.backward()
            optimizer.step()

            # for every 50 images, print progress and visualize the matches
            if (i+1) % 5 == 0:
                mean_loss_item = torch.mean(torch.stack(mean_loss)).item()
                writer.add_scalar('Mean_Loss',mean_loss_item,len(train_loader)*(epoch-1)+i+1)
                print ('Epoch [{}/{}], Step [{}/{}], Mean Loss: {:.4f}' 
                    .format(epoch, opt.epoch, i+1, len(train_loader), mean_loss_item))
                mean_loss = []

                ### eval ###
                # Visualize the matches.
                superglue.eval()
                image0, image1 = pred['image0'].cpu().numpy()[0]*255., pred['image1'].cpu().numpy()[0]*255.
                kpts0, kpts1 = pred['keypoints0'].cpu().numpy()[0], pred['keypoints1'].cpu().numpy()[0]
                matches, conf = pred['matches0'].cpu().detach().numpy(), pred['matching_scores0'].cpu().detach().numpy()
                valid = matches > -1
                mkpts0 = kpts0[valid]
                mkpts1 = kpts1[matches[valid]]
                mconf = conf[valid]
                viz_path = eval_output_dir / '{}_matches.{}'.format(str(i), opt.viz_extension)
                color = cm.jet(mconf)
                stem = pred['file_name']
                text = []

                make_matching_plot(
                    image0, image1, kpts0, kpts1, mkpts0, mkpts1, color,
                    text, viz_path, stem, stem, opt.show_keypoints,
                    True, False, 'Matches')

        # save checkpoint when an epoch finishes
        epoch_loss /= len(train_loader)
        writer.add_scalar("epoch_loss",epoch_loss,epoch)
        model_out_path = opt.Result_dir+"/checkpoints"
        os.makedirs(model_out_path,exist_ok=True)
        model_checkpoint = model_out_path+"/SuperGlue_epoch_{}.pth".format(epoch)
       
        checkpoint = {'epoch': epoch,'net': superglue.state_dict()}
        torch.save(checkpoint, model_checkpoint)
        print("Epoch [{}/{}] done. Epoch Loss {}. Checkpoint saved to {}".format(epoch, opt.epoch, epoch_loss, model_out_path)) 
    
    writer.close()