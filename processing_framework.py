import numpy as np
from PIL import Image
import os
from abc import ABC, abstractmethod
import re
#from pathlib import Path
import time
import shutil
import statistics

class Sequencer: 
    def __init__(self, operations):
        if isinstance(operations, (list, tuple)):
            self.seq = operations
        else:
            self.seq = [operations]
            
        self.time_taken = 0
            
    def add_operation(self, op, index=None):
        if index is None:
            self.seq.append(op)
        else:
            self.seq = self.seq[:index] + [op] + self.seq[index:]
            
    def __call__(self, **kwargs):        
        for i, op in enumerate(self.seq, 1):
            start = time.time()
            print(f'Step {i}/{len(self.seq)} ({op}) in progress', flush=True)
            print(kwargs.keys())
            kwargs = op(**kwargs)
            time_taken = time.time() - start
            self.time_taken += time_taken
            print(f'Step {i}/{len(self.seq)} ({op}) completed, time elapsed: {time_taken} seconds\n')
            
        return kwargs
    
class Operation(ABC):
    
    @abstractmethod
    def __call__(self, **kwargs):
        pass
    
    def __str__(self):
        return type(self).__name__
    
    def __repr__(self):
        return self.__str__()

class GetPaths(Operation):
    def __init__(self, allowed_extensions=None, required_keywords=None,
                 banned_keywords=None, recursive=False):
        self.allowed_extensions = tuple(allowed_extensions)
        self.required_keywords = required_keywords
        self.banned_keywords = banned_keywords
        self.recursive = recursive
        
    def __call__(self, *, paths, depth=0, **kwargs):
        files = []
            
        for path in paths:
            if not os.path.isdir(path):
                if self.allowed_extensions is None or path.endswith(self.allowed_extensions):
                    if (self.required_keywords is None or all([keyword in path for keyword in self.required_keywords]))\
                        and (self.banned_keywords is None or all([keyword not in path for keyword in self.banned_keywords])):
                            files.append(path)
            elif self.recursive: # path is dir
                if (self.required_keywords is None or all([keyword in path for keyword in self.required_keywords]))\
                        and (self.banned_keywords is None or all([keyword not in path for keyword in self.banned_keywords])):
                    new_paths = [os.path.join(path, new_path) for new_path in os.listdir(path)]
                    files.extend(self.__call__(paths=new_paths, depth=depth+1)) # recursive
        
        if depth == 0:
            print(f'Found {len(files)} paths')
            kwargs['paths'] = files
            return kwargs
            
        return files
        
class LoadImages(Operation):
    def __init__(self):
        pass
    
    def __call__(self, *, paths, **kwargs):
        images = []
        for i, path in enumerate(paths):
            try:
                images.append(Image.open(path).copy())
            except OSError:
                print(f'Could not open image: {path}')
                del paths[i]
        
        print(f'Loaded {len(images)} images')
        kwargs['images'] = images
        kwargs['paths'] = paths
        return kwargs
    
class ChangeImageMode(Operation):
    def __init__(self, mode=''):
        self.mode = mode
        
    def __call__(self, *, images, **kwargs):
        new_images = [img.convert(self.mode) for img in images]
        n_changed = sum([1 for img, new_img in zip(images, new_images) if img.mode != new_img.mode])
        print(f'Mode of {n_changed} images changed')
        
        kwargs['images'] = new_images
        return kwargs
    
class FilterImages(Operation):
    def __init__(self, mode=None, resolution=None, n_channels=None,
                 ratio=None, filter_paths=False, eps=10e-5):
        self.mode = mode
        self.resolution = resolution
        self.n_channels = n_channels
        self.ratio = ratio
        self.filter_paths = filter_paths
        self.eps = eps
        if resolution is not None and ratio is not None:
            print('Resolution was set so ratio will be disregarded')
            self.ratio = None
        
    def __call__(self, *, images, paths=None, **kwargs):
        original_len = len(images)
        
        if self.resolution == 'most_frequent':
            vals, cnts = np.unique([img.size for img in images], return_counts=True, axis=0)
            res = tuple(vals[np.argmax(cnts)])
            print(res)
        else:
            res = self.resolution
        if self.ratio == 'most_frequent':
            vals, cnts = np.unique([img.size[0] / img.size[1] for img in images], return_counts=True, axis=0)
            ratio = vals[np.argmax(cnts)]
        else:
            ratio = self.ratio
        
        new_images = [img for img in images if (self.mode is None or img.mode == self.mode)\
                      and (res is None or img.size == res)\
                      and (self.n_channels is None or len(img.getbands()) == self.n_channels)\
                      and (ratio is None or np.abs((img.size[0] / img.size[1]) - ratio) <= self.eps)]
          
        print(f'Filtered {original_len - len(new_images)} images out of {original_len}. {len(new_images)} images remaining')
        kwargs['images'] = new_images
        
        print(len(paths))
        print(len(new_images))
        if self.filter_paths:
            new_paths = []
            i = 0
            for image in images:
                if i == len(new_images):
                    break
                if image == new_images[i]:
                    new_paths.append(paths[i])
                    i += 1
            kwargs['paths'] = new_paths
        elif paths is not None:
            kwargs['paths'] = paths
        print(len(kwargs['paths']))
                
        return kwargs
    
class ResizeImages(Operation):
    def __init__(self, resolution, resample='lanczos'):
        self.resolution = resolution
        
        resample = resample.lower()
        if resample == 'nearest':
            self.resample = Image.NEAREST
        elif resample == 'box':
            self.resample = Image.BOX
        elif resample == 'bilinear':
            self.resample = Image.BILINEAR
        elif resample == 'hamming':
            self.resample = Image.HAMMING
        elif resample == 'bicubic':
            self.resample = Image.BICUBIC
        elif resample == 'lanczos':
            self.resample = Image.LANCZOS
        else:
            self.resample = resample
        
    def __call__(self, *, images, **kwargs):
        if self.resolution == 'most_frequent':
            vals, cnts = np.unique([img.size for img in images], return_counts=True, axis=0)
            res = tuple(vals[np.argmax(cnts)])
        elif self.resolution == 'greatest_height':
            res = max([img.size[1] for img in images])
        elif self.resolution == 'greates_width':
            res = max([img.size[0] for img in images])
        elif self.resolution == 'smallest_height':
            res = min([img.size[1] for img in images])
        elif self.resolution == 'smallest_width':
            res = min([img.size[0] for img in images])
        else:
            res = self.resolution
          
        new_images = [img.resize(res, resample=self.resample) for img in images]
        n_changed = sum([1 for img, new_img in zip(images, new_images) if img.size != new_img.size])
        print(f'Resolution of {n_changed} images out of {len(images)} changed')
        
        kwargs['images'] = new_images
        return kwargs
    
class PadImages(Operation):
    def __init__(self, to_resolution=None, padding=None, color=0, transparent_alpha=True):
        self.to_resolution = to_resolution
        self.padding = padding
        self.transparent_alpha = transparent_alpha
        self.color = color
        if to_resolution is None and padding is None:
            print('PadImages does nothing - both argsuments to_resolution and padding are None')
            
    def __call__(self, *, images, **kwargs):
        
        new_images = []
        for img in images:
            res = img.size
            
            if self.to_resolution is not None:
                if self.to_resolution[0] < res[0] or self.to_resolution[1] < res[1]:
                    raise RuntimeError(f'Image resolution was higher than to_resolution {res} > {self.to_resolution}')
                    
                pad_width = self.to_resolution[0] - res[0]
                pad_height = self.to_resolution[1] - res[1]
            else:
                pad_width = self.padding
                pad_height = self.padding
                
            if not isinstance(self.color, tuple):
                bands = img.getbands()
                if self.transparent_alpha and bands[-1] == 'A':
                    color = (self.color, ) * (len(bands) - 1) + (0, )
                else:
                    color = (self.color, ) * len(bands)
            else:
                color = self.color
            
            new_img = Image.new(img.mode, (res[0] + pad_width, res[1] + pad_height), color)
            new_img.paste(img, (pad_width // 2, pad_height // 2))
            new_images.append(new_img)
            
        n_changed = sum([1 for img, new_img in zip(images, new_images) if img.size != new_img.size])
        print(f'Padding added to {n_changed} images out of {len(images)}')
        
        kwargs['images'] = new_images
        return kwargs
    
class ImagesToArrays(Operation):
    def __init__(self, dtype=None, preserve=True):
        self.dtype = dtype
        self.preserve = preserve
    
    def __call__(self, *, images, **kwargs):
        arrays = [np.array(img, self.dtype) for img in images]
        size = sum([arr.nbytes for arr in arrays])
        i = 0
        units = ['bytes', 'KB', 'MB', 'GB', 'TB']
        while size > 1000:
            i += 1
            size /= 1000
        print(f'Converted images to arrays. Size in memory: {size} {units[i]}')
        
        kwargs['arrays'] = arrays
        if self.preserve:
            kwargs['images'] = images
        return kwargs
    
class ArraysToImages(Operation):
    def __init__(self, preserve=True):
        self.preserve = preserve
    
    def __call__(self, *, arrays, **kwargs):
        kwargs['images'] = [Image.fromarray(arr) for arr in arrays]
        if self.preserve:
            kwargs['arrays'] = arrays
        return kwargs
    
class StandarizeBackgroundByAlpha(Operation):
    def __init__(self, color, binarize_alpha=False):
        self.color = color
        self.binarize_alpha = binarize_alpha
        
    def __call__(self, *, arrays, **kwargs):
        new_arrays = []
        for arr in arrays:
            mask = arr[..., -1] == 0
            arr[mask, :-1] = self.color
            if self.binarize_alpha:
                arr[~mask, -1] = 255
            new_arrays.append(arr)
            
        kwargs['arrays'] = new_arrays
        return kwargs
    
class SaveImages(Operation):
    def __init__(self, destination_path, extension='png', use_paths=False,
                 clean_destination_dir=False):
        self.destination_path = destination_path
        self.extension = extension
        self.use_paths = use_paths
        self.clean_destination_dir = clean_destination_dir
        
    def __call__(self, *, images, paths=None, **kwargs):
        print(len(paths), '==', len(images))
        if self.clean_destination_dir:
            if os.path.isdir(self.destination_path):
                shutil.rmtree(self.destination_path)
            
        for i, img in enumerate(images, 1):
            file_name = str(i) + '.' + self.extension
            if self.use_paths:
                path = os.path.join(self.destination_path, paths[i-1])
            else:
                path = self.destination_path
                
            if not os.path.isdir(path):
                os.makedirs(path)
                
            path = os.path.join(path, file_name)
            
            img.save(path)
            
        print(f'{len(images)} images saved')
        if paths is not None:
            kwargs['paths'] = paths
        return kwargs
    
class ShortenPaths(Operation):
    def __init__(self, delete_patterns=None, delete_last_segment=False):
        self.delete_patterns = delete_patterns
        self.delete_last_segment = delete_last_segment

    def __call__(self, *, paths, **kwargs):
        if self.delete_last_segment:
            paths = [re.sub(r'(\\|\/)+[^\\\/]+$', '', path) for path in paths]
        if self.delete_patterns is not None:
            for pattern in self.delete_patterns:
                paths = [re.sub(pattern, '', path) for path in paths]
                
        paths = [re.sub(r'\\+|\/+', '_', path) for path in paths]
                
        kwargs['paths'] = paths
        return kwargs
    
class FilterByColorsNumber(Operation):
    def __init__(self, min_colors=None, max_colors=None):
        self.min_colors = min_colors
        self.max_colors = max_colors
        
    def __call__(self, *, images, **kwargs):
        colors_counts = [statistics.get_number_of_colors(img) for img in images]
        new_images = [img for cnt, img in zip(colors_counts, images) if \
                      (self.min_colors is None or cnt >= self.min_colors) and \
                      (self.max_colors is None or cnt <= self.max_colors)]
        
        print(f'Filtered {len(images) - len(new_images)} images out of {len(images)}. {len(new_images)} images remaining')
            
        kwargs['images'] = new_images
        return kwargs

class GetBoxesFromArrays(Operation):
    def __init__(self, preserve=True):
        self.preserve = preserve
        
    def __call__(self, *, arrays, **kwargs):
        masks = [arr[:, :, :, 3] for arr in arrays] # asserts that arrays are in good format
        
        top_offsets = statistics.get_top_offsets(masks) 
        bottom_offsets = statistics.get_bottom_offsets(masks) 
        left_offsets = statistics.get_left_offset(masks)      
        right_offsets = statistics.get_right_offsets(masks) 
        
        boxes = [(t_o, b_o, l_o, r_o) for t_o, b_o, l_o, r_o in zip(top_offsets, bottom_offsets, left_offsets, right_offsets)]
        kwargs['boxes'] = np.array(boxes)
        if self.preserve:
            kwargs['arrays'] = arrays
        
        return kwargs
        
    
if __name__ == '__main__':
    path = 'raw/pokemon'
    seq = Sequencer([GetPaths(recursive=True,
                              allowed_extensions=['.gif', '.png', '.jpg', '.jpeg'],
                              banned_keywords=['back']),
                     LoadImages(),
                     #ChangeImageMode(),
                     #FilterImages(ratio='most_frequent', mode='RGBA', filter_paths=True),
                     #ResizeImages((96, 96), 'lanczos'),
                     #PadImages(to_resolution=(128, 128), color=255),
                     #ImagesToArrays(),
                     #StandarizeBackgroundByAlpha((255, 255, 255), binarize_alpha=True),
                     #ArraysToImages(),
                     ShortenPaths(delete_last_segment=True),
                     SaveImages('processed/pokemon_all', use_paths=True, clean_destination_dir=True)
                     ])
    data = seq(paths=[path])
    print('\nOverall time:', seq.time_taken)
                