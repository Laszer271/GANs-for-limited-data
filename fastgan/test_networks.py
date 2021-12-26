from lightweight_gan import cli
import json

ARGS = ('data',
'results_dir',
'models_dir',
'name',
'new',
'load_from',
'image_size',
'optimizer',
'fmap_max',
'transparent',
'greyscale',
'batch_size',
'gradient_accumulate_every',
'num_train_steps',
'learning_rate',
'save_every',
'evaluate_every',
'generate',
'generate_types',
'generate_interpolation',
'aug_test',
'aug_prob',
'aug_types',
'dataset_aug_prob',
'attn_res_layers',
'freq_chan_attn',
'disc_output_size',
'dual_contrast_loss',
'antialias',
'interpolation_num_steps',
'save_frames',
'num_image_tiles',
'num_workers',
'multi_gpus',
'calculate_fid_every',
'calculate_fid_num_images',
'clear_fid_cache',
'seed',
'amp',
'show_progress',)

with open('config.json', 'r') as f:
    d = json.load(f)
    
config = {key: d['general'][key] for key in d['general'] if key in ARGS}
cli.train_from_folder(**config)




