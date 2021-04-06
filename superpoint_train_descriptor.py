import argparse
import yaml
import os
import logging
import torch
import torch.optim
import torch.utils.data as data
from tensorboardX import SummaryWriter
from datasets.ALLSS import ALLSS
from superpoint.Train_model_heatmap import Train_model_heatmap

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument("--config", type=str,default='superpoint/configs/superpoint_allss_train_heatmap.yaml')
    parser.add_argument("--exper_name", type=str,default='superpoint_allss_descriptor_128')
    parser.add_argument("--output_dir",type=str,default='Results/ALLSS/')
    args=parser.parse_args()

    device=torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    logging.info('Start training on {}'.format(device))

    #output path for saving training record
    output_dir=os.path.join(args.output_dir,args.exper_name)
    checkpoints_dir=os.path.join(output_dir,'checkpoints')
    os.makedirs(output_dir,exist_ok=True)
    os.makedirs(checkpoints_dir,exist_ok=True)
    writer=SummaryWriter(output_dir)

    #load configs and save configs
    with open(args.config, 'r') as f:
        config = yaml.load(f)
    with open(os.path.join(output_dir, 'config.yml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    train_set = ALLSS(task='train',**config['data'])
    val_set = ALLSS(task='val',**config['data'])
    train_loader = data.DataLoader(train_set, batch_size=config['model']['batch_size'], shuffle=False)
    val_loader = data.DataLoader(val_set, batch_size=config['model']['eval_batch_size'], shuffle=False)

    train_agent = Train_model_heatmap(config, save_path=checkpoints_dir, device=device)
    train_agent.writer = writer
    train_agent.train_loader = train_loader
    train_agent.val_loader = val_loader
    train_agent.loadModel()
    train_agent.dataParallel()

    try:
        train_agent.train()
    except KeyboardInterrupt:
        print ("press ctrl + c, save model!")
        train_agent.saveModel()
        pass

