#coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import sampler, DataLoader
from torch.utils.tensorboard import SummaryWriter

import os
import time
import argparse

from cp_dataset import CPDataset
from networks import GMM, UnetGenerator, VGGLoss, load_checkpoint, save_checkpoint
from visualization import board_add_images

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default = "GMM")
    parser.add_argument("--stage", default = "GMM")
    parser.add_argument("--gpu_ids", default = "")
    parser.add_argument("--multi_gpu", action='store_true', help='use multi gpu')
    parser.add_argument("--dataroot", default = "/data/zhaofuwei/cp-vton")
    parser.add_argument("--data_list", default = "train_pairs.txt")
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')
    parser.add_argument("--fine_width", type=int, default = 192)
    parser.add_argument("--fine_height", type=int, default = 256)
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('-b', '--batch-size', type=int, default=4)
    parser.add_argument("--radius", type=int, default = 5)
    parser.add_argument("--grid_size", type=int, default = 5)
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
    parser.add_argument("--keep_step", type=int, default = 100000)
    parser.add_argument("--decay_step", type=int, default = 100000)
    parser.add_argument('--tensorboard_dir', type=str, default='/data/zhaofuwei/cp-vton/train/tensorboard', help='save tensorboard infos')
    parser.add_argument('--checkpoint_dir', type=str, default='/data/zhaofuwei/cp-vton/train/checkpoints', help='save checkpoint infos')
    parser.add_argument('--checkpoint', type=str, default='', help='model checkpoint for initialization')
    parser.add_argument("--display_count", type=int, default = 20)
    parser.add_argument("--save_count", type=int, default = 100)

    opt = parser.parse_args()
    return opt

def train_gmm(opt, train_loader, model, board):
    # load model
    if opt.multi_gpu:
        model = nn.DataParallel(model)
    model.cuda()
    model.train()

    # criterion
    criterionL1 = nn.L1Loss()
    
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda step: 1.0 -
            max(0, step - opt.keep_step) / float(opt.decay_step + 1))
    
    # train log
    if not opt.checkpoint == '':
        train_log = open(os.path.join(opt.checkpoint_dir, opt.name, 'train_log.txt'), 'a')
    else:
        os.makedirs(os.path.join(opt.checkpoint_dir, opt.name), exist_ok=True)
        train_log = open(os.path.join(opt.checkpoint_dir, opt.name, 'train_log.txt'), 'w')
        train_log.write('='*30 + ' Training Option ' + '='*30 + '\n')
        train_log.write(str(opt) + '\n\n')
        train_log.write('='*30 + ' Network Architecture ' + '='*30 + '\n')
        print(str(model) + '\n', file=train_log)
        train_log.write('='*30 + ' Training Log ' + '='*30 + '\n')
    
    # train loop
    checkpoint_step = 0
    if not opt.checkpoint == '':
        checkpoint_step += int(opt.checkpoint.split('/')[-1][5:11])
    for step in range(checkpoint_step, opt.keep_step + opt.decay_step):
        iter_start_time = time.time()

        dl_iter = iter(train_loader)
        inputs = dl_iter.next()
            
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
        warped_cloth = F.grid_sample(c, grid, padding_mode='border', align_corners=True)
        warped_mask = F.grid_sample(cm, grid, padding_mode='zeros', align_corners=True)
        warped_grid = F.grid_sample(im_g, grid, padding_mode='zeros', align_corners=True)

        visuals = [ [im_h, shape, im_pose], 
                   [c, warped_cloth, im_c], 
                   [warped_grid, (warped_cloth+im)*0.5, im]]
        
        loss = criterionL1(warped_cloth, im_c)    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
            
        if (step+1) % opt.display_count == 0:
            board_add_images(board, 'combine', visuals, step+1)
            board.add_scalar('metric', loss.item(), step+1)
            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f, loss: %4f' % (step+1, t, loss.item()), flush=True)
            train_log.write('step: %8d, time: %.3f, loss: %.4f' 
                    % (step+1, t, loss.item()) + '\n')

        if (step+1) % opt.save_count == 0:
            save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'step_%06d.pth' % (step+1)))

def train_tom(opt, train_loader, model, board):
    # load model
    if opt.multi_gpu:
        model = nn.DataParallel(model)
    model.cuda()
    model.train()
    
    # criterion
    criterionL1 = nn.L1Loss()
    criterionVGG = VGGLoss()
    criterionMask = nn.L1Loss()
    
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda step: 1.0 -
            max(0, step - opt.keep_step) / float(opt.decay_step + 1))

    # train log
    if not opt.checkpoint == '':
        train_log = open(os.path.join(opt.checkpoint_dir, opt.name, 'train_log.txt'), 'a')
    else:
        os.makedirs(os.path.join(opt.checkpoint_dir, opt.name), exist_ok=True)
        train_log = open(os.path.join(opt.checkpoint_dir, opt.name, 'train_log.txt'), 'w')
        train_log.write('='*30 + ' Training Option ' + '='*30 + '\n')
        train_log.write(str(opt) + '\n\n')
        train_log.write('='*30 + ' Network Architecture ' + '='*30 + '\n')
        print(str(model) + '\n', file=train_log)
        train_log.write('='*30 + ' Training Log ' + '='*30 + '\n')

    # train loop
    checkpoint_step = 0
    if not opt.checkpoint == '':
        checkpoint_step += int(opt.checkpoint.split('/')[-1][5:11])
    for step in range(checkpoint_step, opt.keep_step + opt.decay_step):
        iter_start_time = time.time()

        dl_iter = iter(train_loader)
        inputs = dl_iter.next()
            
        im = inputs['image'].cuda()
        im_pose = inputs['pose_image']
        im_h = inputs['head']
        shape = inputs['shape']

        agnostic = inputs['agnostic'].cuda()
        c = inputs['cloth'].cuda()
        cm = inputs['cloth_mask'].cuda()
        
        outputs = model(torch.cat([agnostic, c],1))
        p_rendered, m_composite = torch.split(outputs, 3,1)
        p_rendered = F.tanh(p_rendered)
        m_composite = F.sigmoid(m_composite)
        p_tryon = c * m_composite+ p_rendered * (1 - m_composite)

        visuals = [ [im_h, shape, im_pose], 
                   [c, cm*2-1, m_composite*2-1], 
                   [p_rendered, p_tryon, im]]
            
        loss_l1 = criterionL1(p_tryon, im)
        loss_vgg = criterionVGG(p_tryon, im)
        loss_mask = criterionMask(m_composite, cm)
        loss = loss_l1 + loss_vgg + loss_mask
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
            
        if (step+1) % opt.display_count == 0:
            board_add_images(board, 'combine', visuals, step+1)
            board.add_scalar('metric', loss.item(), step+1)
            board.add_scalar('L1', loss_l1.item(), step+1)
            board.add_scalar('VGG', loss_vgg.item(), step+1)
            board.add_scalar('MaskL1', loss_mask.item(), step+1)
            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f, loss: %.4f, l1: %.4f, vgg: %.4f, mask: %.4f' 
                    % (step+1, t, loss.item(), loss_l1.item(), 
                    loss_vgg.item(), loss_mask.item()), flush=True)
            train_log.write('step: %8d, time: %.3f, loss: %.4f, l1: %.4f, vgg: %.4f, mask: %.4f' 
                    % (step+1, t, loss.item(), loss_l1.item(), loss_vgg.item(), loss_mask.item()) + '\n')

        if (step+1) % opt.save_count == 0:
            save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'step_%06d.pth' % (step+1)))

if __name__ == "__main__":
    opt = get_opt()
    print(opt)
    print("Start to train stage: %s, named: %s!" % (opt.stage, opt.name))
    
    # set gpu(s)
    if not opt.gpu_ids == "":
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids

    # create dataset & dataloader
    train_dataset = CPDataset(opt)
    if opt.shuffle :
        train_sampler = None
    else:
        train_sampler = sampler.RandomSampler(train_dataset)
    train_loader = DataLoader(
                train_dataset, batch_size=opt.batch_size, shuffle=opt.shuffle,
                num_workers=opt.workers, pin_memory=True, sampler=train_sampler)

    # visualization
    if not os.path.exists(opt.tensorboard_dir):
        os.makedirs(opt.tensorboard_dir)
    board = SummaryWriter(log_dir = os.path.join(opt.tensorboard_dir, opt.name))
   
    # create model & train & save the final checkpoint
    if opt.stage == 'GMM':
        model = GMM(opt)
        if not opt.checkpoint =='' and os.path.exists(opt.checkpoint):
            load_checkpoint(model, opt.checkpoint)
        train_gmm(opt, train_loader, model, board)
        save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'gmm_final.pth'))
    elif opt.stage == 'TOM':
        model = UnetGenerator(25, 4, 6, ngf=64, norm_layer=nn.InstanceNorm2d)
        if not opt.checkpoint =='' and os.path.exists(opt.checkpoint):
            load_checkpoint(model, opt.checkpoint)
        train_tom(opt, train_loader, model, board)
        save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'tom_final.pth'))
    else:
        raise NotImplementedError('Model [%s] is not implemented' % opt.stage)
        
  
    print('Finished training %s, nameed: %s!' % (opt.stage, opt.name))