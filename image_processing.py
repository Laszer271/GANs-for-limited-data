import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import pandas as pd

def get_all_files(path, allowed_extensions, required_keywords=None, banned_keywords=None,
                  get_skipped_files=False, get_wrong_extensions=False):
    files = []
    skipped = []
    if not os.path.isdir(path):
        if path.endswith(allowed_extensions): 
            if (required_keywords is None or all([keyword in path for keyword in required_keywords]))\
                and (banned_keywords is None or all([keyword not in path for keyword in banned_keywords])):
                files.append(path)
            elif get_skipped_files:
                skipped.append(path)
        elif get_wrong_extensions:
            skipped.append(path)

    else:
        for f in os.listdir(path):
            if f == '.git':
                continue
            l = get_all_files(os.path.join(path, f), allowed_extensions,
                              required_keywords=required_keywords, 
                              banned_keywords=banned_keywords,
                              get_skipped_files=get_skipped_files,
                              get_wrong_extensions=get_wrong_extensions)
            if (get_skipped_files or get_wrong_extensions):
                files.extend(l[0])
                skipped.extend(l[1])
            elif l:
                files.extend(l)

    if get_skipped_files or get_wrong_extensions:
        return files, skipped
    else:
        return files

def make_preview(images, cols, margin=(8, 12)):
    n_images = images.shape[0]
    height = images.shape[1]
    width = images.shape[2]
    rows = np.ceil(n_images / cols) 
    
    preview_height = int(rows * height + (rows - 1) * margin[1])
    preview_width = (cols * width + (cols - 1) * margin[0])
    arr = np.full((preview_height, preview_width, images.shape[3]), 255, dtype=np.uint8)
    
    for i in range(n_images):
        row = i // cols
        col = i % cols
        
        y = row * (height + margin[1])
        x = col * (width + margin[0])
        arr[y: y+height, x: x+width, :] = images[i]
        
    return arr

def get_images(path):
    if not isinstance(path, (list, tuple)):
        paths = os.listdir(path)
        paths = [os.path.join(path, img_path) for img_path in paths]
    else:
        paths = path
        
    images = []
    for path in paths:
        try:
            images.append(Image.open(path).copy()) # copy to avoid: [Errno 24] Too many open files
        except OSError:
            print('Error trying to open:', path)
    
    converted_images = []
    for img in images:
        if img.mode == 'P':
            img = img.convert(img.palette.mode)
        if img.mode in ('RGB', 'RGBA', 'L', 'LA'):
            converted_images.append(img)
        else:
            raise RuntimeError(f'The image mode ({img.mode}) not supported')
    
    return converted_images

if __name__ == '__main__':
    path = 'sprites-master/sprites/pokemon/versions'
    images_paths1 = get_all_files(path, ('png', 'gif'),
                                  required_keywords=['transparent'],
                                  banned_keywords=['back'])
    path = 'sprites-master/sprites/pokemon'
    images_paths2 = get_all_files(path, ('png', 'gif'),
                                  #required_keywords=['transparent'],
                                  banned_keywords=['versions', 'back'])
    
    background_color = 255
    padding = int((128-96) / 2)
    images = []
    
    dest_path = 'gen_datasets/pokemon_clean/'
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    
    for i, image_path in enumerate(images_paths1 + images_paths2):
        try:
            img = Image.open(image_path)
        except OSError:
            print('Error trying to open:', image_path)
    
        '''
        if img.size != (96, 96):
            continue
        img = img.convert('RGBA')
        img = np.array(img)
        n_pixels = (img[..., -1] == 255).sum()
        all_pixels_cnt = 96 * 96
        if n_pixels != 0 and n_pixels != all_pixels_cnt:
            for channel in range(img.shape[-1]-1): 
                mask = img[..., -1] == 0
                stats = img.mean()
                img[..., channel][mask] = background_color
            img = img[..., :-1]
            img = np.pad(img, ((padding, padding), (padding, padding), (0, 0)),
                         mode='constant', constant_values=background_color)
            images.append(img)
        '''
        images.append(img)
        img.save(os.path.join(dest_path, f'{i}.png'))
    
    '''
    images = np.array(images)
    images = np.unique(images, axis=0)
    np.save('pokemon_dataset.npy', images)
    
    for i, img in enumerate(images):
        img = Image.fromarray(img)
        img.save(f'data/pokemon/{i}.png')
        print(img.getbands())
        
    arr = make_preview(images, 100)
    img = Image.fromarray(arr)
    img.save('preview.png')
    '''