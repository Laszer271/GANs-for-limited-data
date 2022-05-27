import os
import torch
import torchvision
from PIL import Image
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import numpy as np
import dataloaders
import pokegan
import matplotlib.pyplot as plt
import time
from pokegan.aegan import AEGAN
from utils import get_downsampling_scheme
import wandb
import json
import shutil
import re

def visualize(images, nrows=6):
    viz = torchvision.utils.make_grid(images, normalize=False, nrow=nrows, padding=2)
    viz = viz.numpy().transpose((1,2,0))
    viz = np.array(viz * 255, dtype=np.uint8)
    
    return viz

def gen_to_wandb(GAN, vec):
    images = GAN.generate_samples(vec).detach().cpu()
    images.add_(1).mul_(0.5)
    images = torch.clip(images, 0, 1)
    viz = visualize(images)
    img = wandb.Image(viz)
    return img
    

if __name__ == '__main__':
    
    cnf_path = os.path.join('pokegan', 'configs')
    configs = ['iconset.json', 'margonem.json', 'pokemon_pixelart.json',
               'profantasy.json', 'dnd.json',
               'pokemon_artwork.json']
    temp_path = 'results'
    
    for config in configs:
        print('\n', '='*50)
        network_name = re.sub('\..*', '', config) + '_network'

        config = os.path.join(cnf_path, config)
        with open(config, 'r') as f:
            config = json.load(f)
        print('TRAINING ON', config['job_type'].upper())

        print('INITIATING WANDB')
        wandb.init(project=config['project'], entity=config['entity'], config=config,
                   group=config['group'], job_type=config['job_type'])
        
        print('\nDOWNSAMPLING SCHEME:')
        SIZES = get_downsampling_scheme(config['image_size'], min_size=config['min_img_size'])
        
        network_checkpoints = os.path.join(temp_path, 'checkpoints')
        
        os.makedirs(network_checkpoints, exist_ok=True)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        files = dataloaders.get_files(config['input_path'], ('.png', '.jpg', '.jpeg'))
        
        transform = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.RandomAffine(
                    degrees=0,
                    translate=(config['translate_factor'] / config['image_size'][0],
                               config['translate_factor'] / config['image_size'][1]),
                    fill=config['fill_color']),
                torchvision.transforms.ColorJitter(hue=config['clr_jit_hue']),
                torchvision.transforms.RandomHorizontalFlip(p=0.5),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
        
        # 0 is fully transparent, 255 is fully non
        
        print('\nLOADING DATASET')
        dataset = dataloaders.BasicDataset(
            files, torchvision.transforms.ToTensor(), transform,
            measure_time=True, batch_size=config['batch_size'], convert_to_rgb=False)
        
        print('LOADED DATASET SHAPE:', dataset.get_shape())
    
        print('SAVING USED CONFIG')
        dump_config_path = os.path.join(temp_path, 'used_config.json')
        with open(dump_config_path, 'w') as f:
            json.dump(config, f)
        wandb.save(dump_config_path)
            
        test_images = dataset[0]
        i = 1
        while test_images.shape[0] < 36:
            test_images = torch.cat((test_images, dataset[i]), 0)
            i += 1
        test_images = test_images[:36]
        
        print('SAVING AUGMENTATION PREVIEW')        
        example = visualize(test_images)
        image = Image.fromarray(example)
        image.save(os.path.join(temp_path, 'test_aug.png'))
        wandb.save('test_aug.png')
        
        test_images = test_images.cuda()
    
        noise_fn = lambda x: torch.randn((x, config['latent_dim']), device=device)
        test_noise = noise_fn(36)
        print('BUILDING AEGAN MODEL')
        gan = AEGAN(
            latent_dim=config['latent_dim'],
            sizes=SIZES,
            noise_fn=noise_fn,
            dataloader=dataset,
            device=device,
            batch_size=config['batch_size'],
            )
        
        start = time.time()
        
        imgs_per_step = config['total_kimg'] * 1000 // config['n_steps']
        batches_per_step = imgs_per_step // config['batch_size']
        image_snapshot_ticks = config['image_snapshot_ticks']
        network_snapshot_ticks = config['network_snapshot_ticks']

        logs = {}
        print('\nTRAINING')
        for i in range(config['n_steps']):            
            new_logs = gan.train_epoch(config['batch_size'], n_batches=batches_per_step)
            s = f'Step {i+1}/{config["n_steps"]}\n'
            for k, v in new_logs.items():
                s += '{}={:0.3f}\t'.format(k, v)
            print(s, end='\n', flush=True)

            logs.update(new_logs)
            wandb.log(logs)
            logs = new_logs

            if i % network_snapshot_ticks == 0:
                ckpt_path = gan.save_model(str(i), network_checkpoints)
                wandb.log_artifact(ckpt_path, name=network_name, type='networks_ckpt')
                
            if i % image_snapshot_ticks == 0:
                img = gen_to_wandb(gan, test_noise)
                logs['generated'] = img
    
                with torch.no_grad():
                    rec_img = gen_to_wandb(gan, gan.encoder(test_images))
                    logs['reconstructed'] = img
                
        print(f'FINISHED TRAINING IN: {time.time() - start:.3f} SECONDS')
        wandb.finish()
        
        shutil.rmtree(temp_path)
        shutil.rmtree('./wandb/')
        
        print('Memory allocated before emptying cache:', torch.cuda.memory_allocated(0))
        del gan
        del dataset
        del test_noise
        del test_images
        del img
        del rec_img
        torch.cuda.empty_cache()
        print('Memory allocated after emptying cache:', torch.cuda.memory_allocated(0))
    