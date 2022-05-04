import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision import utils as vutils

import random
from tqdm import tqdm

from fastgan.models import weights_init, Discriminator, Generator
from fastgan.operation import copy_G_params, load_params, get_dir
from fastgan.operation import ImageFolder, InfiniteSamplerWrapper
from fastgan.diffaug import DiffAugment
policy = 'color,translation'
from fastgan import lpips
percept = lpips.PerceptualLoss(model='net-lin', net='vgg', use_gpu=True)

import json
import dataloaders
from utils import get_downsampling_scheme
import numpy as np

#torch.backends.cudnn.benchmark = True

def crop_image_by_part(image, part):
    hw = image.shape[2]//2
    if part==0:
        return image[:,:,:hw,:hw]
    if part==1:
        return image[:,:,:hw,hw:]
    if part==2:
        return image[:,:,hw:,:hw]
    if part==3:
        return image[:,:,hw:,hw:]

def train_d(net, data, label="real"):
    """Train function of discriminator"""
    if label=="real":
        part = random.randint(0, 3)
        pred, [rec_all, rec_small, rec_part] = net(data, label, part=part)
        err = F.relu(  torch.rand_like(pred) * 0.2 + 0.8 -  pred).mean() + \
            percept( rec_all, F.interpolate(data, rec_all.shape[2]) ).sum() +\
            percept( rec_small, F.interpolate(data, rec_small.shape[2]) ).sum() +\
            percept( rec_part, F.interpolate(crop_image_by_part(data, part), rec_part.shape[2]) ).sum()
        err.backward()
        return pred.mean().item(), rec_all, rec_small, rec_part
    else:
        pred = net(data, label)
        err = F.relu( torch.rand_like(pred) * 0.2 + 0.8 + pred).mean()
        err.backward()
        return pred.mean().item()

if __name__ == "__main__":
    
    config = 'config.json'
    with open(config, 'r') as f:
        args = json.load(f)
        
    saved_model_folder, saved_image_folder = get_dir(args)
    
    device = torch.device("cpu")
    if args['use_cuda']:
        device = torch.device("cuda:0")

    transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    print('Loading dataset')
    if 'lmdb' in args['path']:
        from operation import MultiResolutionDataset
        dataset = MultiResolutionDataset(args['path'], transform, 1024)
        dataloader = iter(DataLoader(
            dataset, batch_size=args['batch_size'], shuffle=False,
            sampler=InfiniteSamplerWrapper(dataset),
            num_workers=args['dataloader_workers'], pin_memory=True))
    else:
        files = dataloaders.get_files(args['path'], ('.png', '.jpg', '.jpeg'))
        dataloader = dataloaders.BasicDataset(
            X=files, initial_transform=None, transform=transform,
            measure_time=True, batch_size=args['batch_size'],
            convert_to_rgb=False)
    print('Dataset loaded')
    print('Dataset shape:', (len(dataloader), *dataloader.shape))
    
    sizes = get_downsampling_scheme(args['im_size'], args['min_img_size'])
    
    print('initialization')
    #from model_s import Generator, Discriminator
    
    print('Building generator')
    netG = Generator(sizes=sizes[::-1], ngf=args['ngf'], nz=args['nz'])
    print('Initializing generator')
    netG.apply(weights_init)

    print('Testing generator')
    fixed_noise = torch.FloatTensor(4, args['nz']).normal_(0, 1)
    out = netG.forward(fixed_noise)

    print('\nBuilding discriminator')
    netD = Discriminator(sizes=sizes, ndf=args['ndf'])
    print('Initializing discriminator')
    netD.apply(weights_init)
    
    print('Testing discriminator')
    fixed_noise = torch.FloatTensor(4, 3, *args['im_size']).normal_(0, 1)
    out = netD.forward(fixed_noise, label='real', part=np.random.randint(0, 3))
    
    netG.to(device)
    netD.to(device)
    print('initialization complete')
    
    avg_param_G = copy_G_params(netG)

    fixed_noise = torch.FloatTensor(8, args['nz']).normal_(0, 1).to(device)
    
    optimizerG = optim.Adam(netG.parameters(), lr=args['nlr'], betas=(args['nbeta1'], 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=args['nlr'], betas=(args['nbeta1'], 0.999))
    
    if args['ckpt'] != 'None':
        ckpt = torch.load(args['ckpt'])
        netG.load_state_dict(ckpt['g'])
        netD.load_state_dict(ckpt['d'])
        avg_param_G = ckpt['g_ema']
        optimizerG.load_state_dict(ckpt['opt_g'])
        optimizerD.load_state_dict(ckpt['opt_d'])
        current_iteration = int(args['ckpt'].split('_')[-1].split('.')[0])
        del ckpt
    else:
        current_iteration = 0
        
    if args['multi_gpu']:
        netG = nn.DataParallel(netG.to(device))
        netD = nn.DataParallel(netD.to(device))

    
    for iteration in tqdm(range(current_iteration, args['iter']+1)):
        real_image = next(dataloader)
        real_image = real_image.to(device)
        current_batch_size = real_image.size(0)
        noise = torch.Tensor(current_batch_size, args['nz']).normal_(0, 1).to(device)

        fake_images = netG(noise)

        real_image = DiffAugment(real_image, policy=policy)
        fake_images = [DiffAugment(fake, policy=policy) for fake in fake_images]
        
        ## 2. train Discriminator
        netD.zero_grad()

        err_dr, rec_img_all, rec_img_small, rec_img_part = train_d(netD, real_image, label="real")
        train_d(netD, [fi.detach() for fi in fake_images], label="fake")
        optimizerD.step()
        
        ## 3. train Generator
        netG.zero_grad()
        pred_g = netD(fake_images, "fake")
        err_g = -pred_g.mean()

        err_g.backward()
        optimizerG.step()

        for p, avg_p in zip(netG.parameters(), avg_param_G):
            avg_p.mul_(0.999).add_(0.001 * p.data)

        if iteration % 100 == 0:
            print("GAN: loss d: %.5f    loss g: %.5f"%(err_dr, -err_g.item()))

        if iteration % (args['save_interval']*10) == 0:
            backup_para = copy_G_params(netG)
            load_params(netG, avg_param_G)
            with torch.no_grad():
                vutils.save_image(netG(fixed_noise)[0].add(1).mul(0.5), saved_image_folder+'/%d.jpg'%iteration, nrow=4)
                vutils.save_image( torch.cat([
                        F.interpolate(real_image, 128), 
                        rec_img_all, rec_img_small,
                        rec_img_part]).add(1).mul(0.5), saved_image_folder+'/rec_%d.jpg'%iteration )
            load_params(netG, backup_para)

        if iteration % (args['save_interval']*50) == 0 or iteration == args['iter']:
            backup_para = copy_G_params(netG)
            load_params(netG, avg_param_G)
            torch.save({'g':netG.state_dict(),'d':netD.state_dict()}, saved_model_folder+'/%d.pth'%iteration)
            load_params(netG, backup_para)
            torch.save({'g':netG.state_dict(),
                        'd':netD.state_dict(),
                        'g_ema': avg_param_G,
                        'opt_g': optimizerG.state_dict(),
                        'opt_d': optimizerD.state_dict()}, saved_model_folder+'/all_%d.pth'%iteration)
