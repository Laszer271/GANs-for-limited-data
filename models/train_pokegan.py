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

def visualize(images, nrows=6):
    viz = torchvision.utils.make_grid(images, normalize=True, nrow=nrows, padding=2)
    viz = viz.numpy().transpose((1,2,0))
    viz = np.array(viz * 255, dtype=np.uint8)
    
    return viz

def save_images(GAN, vec, filename):
    images = GAN.generate_samples(vec).detach().cpu()
    viz = visualize(images)
    img = wandb.Image(viz)
    wandb.log({'generated': img}, commit=False)
    viz = Image.fromarray(viz)
    viz.save(filename)

if __name__ == '__main__':
    
    cnf_path = os.path.join('pokegan', 'configs')
    configs = ['iconset.json', 'margonem.json', 'pokemon_pixelart.json',
               'profantasy.json', 'dnd.json',
               'pokemon_artwork.json']
    
    for config in configs:
        print('\n', '='*50)
        config = os.path.join(cnf_path, config)
        with open(config, 'r') as f:
            config = json.load(f)
        print('TRAINING ON', config['job_type'].upper())

        print('INITIATING WANDB')
        wandb.init(project=config['project'], entity=config['entity'], config=config,
                   group=config['group'], job_type=config['job_type'])
        
        print('\nDOWNSAMPLING SCHEME:')
        SIZES = get_downsampling_scheme(config['image_size'], min_size=config['min_img_size'])
        
        generated_images = os.path.join(config['output_path'], 'generated')
        reconstructed_images = os.path.join(config['output_path'], 'reconstructed')
        network_checkpoints = os.path.join(config['output_path'], 'checkpoints')
        
        os.makedirs(generated_images, exist_ok=True)
        os.makedirs(reconstructed_images, exist_ok=True)
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
    
        print('SAVING AUGMENTATION PREVIEW')        
        example = visualize(dataset[0])
        image = Image.fromarray(example)
        image.save(os.path.join(config['output_path'], 'test_aug.png'))
        print('SAVING USED CONFIG')
        with open(os.path.join(config['output_path'], 'used_config.json'), 'w') as f:
            json.dump(config, f)
            
        test_images = dataset[0]
        i = 1
        while test_images.shape[0] < 36:
            test_images = torch.cat((test_images, dataset[i]), 0)
            i += 1
        test_images = test_images[:36].cuda()
    
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
    
        print('\nTRAINING')
        for i in range(config['epochs']):
            print(f'Epoch: {i+1}/{config["epochs"]}')
            gan.train_epoch(print_frequency=config['print_frequency'])
            if (i + 1) % 10 == 0:
                gan.save_model(str(i), network_checkpoints)
            
            save_images(gan, test_noise,
                os.path.join(generated_images, f"gen_{i}.png"))
    
            with torch.no_grad():
                reconstructed = gan.generator(gan.encoder(test_images)).detach().cpu()
                
            reconstructed = visualize(reconstructed)
            wandb.log({'reconstructed': wandb.Image(reconstructed)}, commit=False)
            reconstructed = Image.fromarray(reconstructed)
            reconstructed.save(os.path.join(reconstructed_images, f"gen_{i}.png"))
            
        #wandb.save(os.path.join(network_checkpoints, 'generator', '*.pt'))
        #wandb.save(os.path.join(network_checkpoints, 'encoder', '*.pt'))
        #wandb.save(os.path.join(network_checkpoints, 'disc_img', '*.pt'))
        #wandb.save(os.path.join(network_checkpoints, 'disc_latent', '*.pt'))
        print(f'FINISHED TRAINING IN: {time.time() - start:.3f} SECONDS')
        wandb.finish()
        
        print('Memory allocated before emptying cache:', torch.cuda.memory_allocated(0))
        del gan
        del reconstructed
        del dataset
        del test_noise
        torch.cuda.empty_cache()
        print('Memory allocated after emptying cache:', torch.cuda.memory_allocated(0))
    