# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Train a GAN using the techniques described in the paper
"Training Generative Adversarial Networks with Limited Data"."""

import os
import click
import re
import json
import tempfile
import torch

from stylegan2_ada import dnnlib

from stylegan2_ada.training import training_loop
from stylegan2_ada.metrics import metric_main
from stylegan2_ada.torch_utils import training_stats
from stylegan2_ada.torch_utils import custom_ops

import wandb

def subprocess_fn(rank, args, temp_dir):
    dnnlib.util.Logger(file_name=os.path.join(args.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    # Init torch.distributed.
    if args.num_gpus > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
        if os.name == 'nt':
            init_method = 'file:///' + init_file.replace('\\', '/')
            torch.distributed.init_process_group(backend='gloo', init_method=init_method, rank=rank, world_size=args.num_gpus)
        else:
            init_method = f'file://{init_file}'
            torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank, world_size=args.num_gpus)

    # Init torch_utils.
    #sync_device = torch.device('cuda', rank) if args.num_gpus > 1 else None
    #training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)
    if rank != 0:
        custom_ops.verbosity = 'none'

    # Execute training loop.
    training_loop.training_loop(rank=rank, **args)

if __name__ == "__main__":
    configs = [
        'stylegan2_ada/configs/config_margonem.json',
        'stylegan2_ada/configs/config_iconset.json',
        'stylegan2_ada/configs/config_pokemon_pixelart.json',
        'stylegan2_ada/configs/config_dnd.json',
        'stylegan2_ada/configs/config_profantasy.json',
        'stylegan2_ada/configs/config_pokemon_artwork.json',
        ]
    
    job_types = ['margonem', 'iconset', 'pokemon_pixelart', 'dnd',
                'profantasy', 'pokemon_artwork']
    project = 'praca_dyplomowa'
    entity = 'laszer'
    group = 'stylegan2_ada'
    
    for config, job_type in zip(configs, job_types):
        print('='*50)
        print('Starting config:', config, '\n')
        dnnlib.util.Logger(should_flush=True)
        training_stats.init_globals()
        
        wandb.init(project=project, entity=entity, #config=config,
                   group=group, job_type=job_types)
        
        with open(config, 'r') as f:
            args = json.load(f)
        args = {k: args[k] if not isinstance(args[k], dict) else dnnlib.EasyDict(**args[k]) for k in args.keys()}
        args = dnnlib.EasyDict(**args)
    
        # Print options.
        print()
        print('Training options:')
        print(json.dumps(args, indent=2))
        print()
        print(f'Output directory:   {args.run_dir}')
        print(f'Training data:      {args.training_set_kwargs.path}')
        print(f'Training duration:  {args.total_kimg} kimg')
        print(f'Number of GPUs:     {args.num_gpus}')
        print(f'Number of images:   {args.training_set_kwargs.max_size}')
        print(f'Image resolution:   {args.training_set_kwargs.res_w} x {args.training_set_kwargs.res_h}')
        print(f'Conditional model:  {args.training_set_kwargs.use_labels}')
        print(f'Dataset x-flips:    {args.training_set_kwargs.xflip}')
        print()
    
        # Create output directory.
        print('Creating output directory...')
        os.makedirs(args.run_dir, exist_ok=True)
        with open(os.path.join(args.run_dir, 'training_options.json'), 'wt') as f:
            json.dump(args, f, indent=2)
    
        # Launch processes.
        print('Launching processes...')
        with tempfile.TemporaryDirectory() as temp_dir:
            subprocess_fn(rank=0, args=args, temp_dir=temp_dir)
            
        wandb.finish()
        torch.cuda.empty_cache()
        print('Completed congfig:', config, '\n')
        
