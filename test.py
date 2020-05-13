#coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import os
import time
import argparse

from cp_dataset import CPDataset
from networks import GMM, UnetGenerator, load_checkpoint
from visualization import board_add_images, save_images

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default = "GMM")
    parser.add_argument("--stage", default = "GMM")
    parser.add_argument("--gpu_ids", default = "")
    parser.add_argument("--dataroot", default = "data")
    parser.add_argument("--datamode", default = "train")
    parser.add_argument("--data_list", default = "train_pairs.txt")
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')
    parser.add_argument("--fine_width", type=int, default = 192)
    parser.add_argument("--fine_height", type=int, default = 256)
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('-b', '--batch-size', type=int, default=4)
    parser.add_argument("--radius", type=int, default = 5)
    parser.add_argument("--grid_size", type=int, default = 5)
    parser.add_argument('--tensorboard_dir', type=str, default='tensorboard', help='save tensorboard infos')
    parser.add_argument('--result_dir', type=str, default='test', help='save result infos')
    parser.add_argument('--checkpoint', type=str, default='', help='model checkpoint for test')
    parser.add_argument("--display_count", type=int, default = 1)
    
    opt = parser.parse_args()
    return opt

def test_gmm(opt, test_loader, model, board):
    # load model
    model.cuda()
    model.eval()

    # make dirs
    base_name = os.path.basename(opt.checkpoint)
    warp_cloth_dir = os.path.join(opt.result_dir, opt.name, base_name, 'warp_cloth')
    os.makedirs(warp_cloth_dir, exist_ok=True)
    warp_mask_dir = os.path.join(opt.result_dir, opt.name, base_name, 'warp_mask')
    os.makedirs(warp_mask_dir, exist_ok=True)

    # test loop
    for step, inputs in enumerate(iter(test_loader)):
        iter_start_time = time.time()
        
        c_names = inputs['c_name']
        im = inputs['image'].cuda()
        im_pose = inputs['pose_image'].cuda()
        im_h = inputs['head'].cuda()
        shape = inputs['shape'].cuda()
        agnostic = inputs['agnostic'].cuda()
        c = inputs['cloth'].cuda()
        cm = inputs['cloth_mask'].cuda()
        im_c =  inputs['parse_cloth'].cuda()
        im_g = inputs['grid_image'].cuda()
            
        grid, theta = model(agnostic, c)
        warped_cloth = F.grid_sample(c, grid, padding_mode='border')
        warped_mask = F.grid_sample(cm, grid, padding_mode='zeros')
        warped_grid = F.grid_sample(im_g, grid, padding_mode='zeros')

        visuals = [ [im_h, shape, im_pose], 
                   [c, warped_cloth, im_c], 
                   [warped_grid, (warped_cloth+im)*0.5, im]]
        
        save_images(warped_cloth, c_names, warp_cloth_dir) 
        save_images(warped_mask*2-1, c_names, warp_mask_dir) 

        if (step+1) % opt.display_count == 0:
            board_add_images(board, 'combine', visuals, step+1)
            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f' % (step+1, t), flush=True)
        
def test_tom(opt, test_loader, model, board):
    # load model
    model.cuda()
    model.eval()
    
    # make dirs
    base_name = os.path.basename(opt.checkpoint)
    try_on_dir = os.path.join(opt.result_dir, opt.name, base_name, 'try_on')
    os.makedirs(try_on_dir, exist_ok=True)
    
    # test loop
    for step, inputs in enumerate(test_loader.data_loader):
        iter_start_time = time.time()
        
        im_names = inputs['im_name']
        im = inputs['image'].cuda()
        im_pose = inputs['pose_image']
        im_h = inputs['head']
        shape = inputs['shape']

        agnostic = inputs['agnostic'].cuda()
        c = inputs['cloth'].cuda()
        cm = inputs['cloth_mask'].cuda()
        
        outputs = model(torch.cat([agnostic, c],1))
        p_rendered, m_composite = torch.split(outputs, 3,1)
        p_rendered = torch.tanh(p_rendered)
        m_composite = torch.sigmoid(m_composite)
        p_tryon = c * m_composite + p_rendered * (1 - m_composite)

        visuals = [ [im_h, shape, im_pose], 
                   [c, 2*cm-1, m_composite], 
                   [p_rendered, p_tryon, im]]
            
        save_images(p_tryon, im_names, try_on_dir) 
        if (step+1) % opt.display_count == 0:
            board_add_images(board, 'combine', visuals, step+1)
            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f' % (step+1, t), flush=True)

if __name__ == "__main__":
    opt = get_opt()
    print(opt)
    print("Start to test stage: %s, named: %s!" % (opt.stage, opt.name))

    # set gpu(s)
    if not opt.gpu_ids == "":
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
   
    # create dataset & dataloader
    test_dataset = CP3Dataset(opt)
    if opt.shuffle :
        test_sampler = None
    else:
        test_sampler = sampler.RandomSampler(test_dataset)
    test_loader = DataLoader(
                test_dataset, batch_size=opt.batch_size, shuffle=opt.shuffle,
                num_workers=opt.workers, pin_memory=True, sampler=test_sampler)

    # visualization
    os.makedirs(opt.tensorboard_dir, exist_ok=True)
    board = SummaryWriter(log_dir = os.path.join(opt.tensorboard_dir, opt.name))
   
    # create model & train
    if opt.stage == 'GMM':
        print('Dataset size: %05d!' % (len(test_dataset)), flush=True)
        model = GMM(opt)
        load_checkpoint(model, opt.checkpoint)
        with torch.no_grad():
            test_gmm(opt, train_loader, model, board)
    elif opt.stage == 'TOM':
        print('Dataset size: %05d!' % (len(test_dataset)), flush=True)
        model = UnetGenerator(25, 4, 6, ngf=64, norm_layer=nn.InstanceNorm2d)
        load_checkpoint(model, opt.checkpoint)
        with torch.no_grad():
            test_tom(opt, train_loader, model, board)
    else:
        raise NotImplementedError('Model [%s] is not implemented' % opt.stage)
  
    print('Finished test %s, named: %s!' % (opt.stage, opt.name))
