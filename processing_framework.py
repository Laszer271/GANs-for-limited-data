import numpy as np
from PIL import Image
import os
from abc import ABC, abstractmethod
import re
#from pathlib import Path
import time
import shutil
import image_statistics
import json
import matplotlib.pyplot as plt

class Sequencer: 
    def __init__(self, operations=None, verbose=None):
        if operations is None:
            self.seq = []
        elif isinstance(operations, (list, tuple)):
            self.seq = operations
        else:
            self.seq = [operations]
            
        self.verbose = verbose
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
            if self.verbose >= 1:
                if 'arrays' in kwargs:
                    plt.imshow(kwargs['arrays'][0])
                    plt.title('(ARR) Before ' + str(op))
                    plt.show()
            if self.verbose >= 2:
                if 'images' in kwargs:
                    arr = np.array(kwargs['images'][0])
                    if arr.shape[-1] == '4':
                        arr = arr[..., :-1]
                    plt.imshow(arr)
                    plt.title('(IMG) Before ' + str(op))
                    plt.show()
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
        
        if isinstance(allowed_extensions, list):
            self.allowed_extensions = tuple(allowed_extensions)
        else:
            self.allowed_extensions = allowed_extensions
            
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
        if isinstance(self.resolution, list):
            self.resolution = tuple(self.resolution)    
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
        else:
            res = self.resolution
        if self.ratio == 'most_frequent':
            vals, cnts = np.unique([img.size[0] / img.size[1] for img in images], return_counts=True, axis=0)
            ratio = vals[np.argmax(cnts)]
        else:
            ratio = self.ratio
        
        new_images = []
        indices = []
        for i, img in enumerate(images):
            if (self.mode is None or img.mode == self.mode)\
                and (res is None or img.size == res)\
                and (self.n_channels is None or len(img.getbands()) == self.n_channels)\
                and (ratio is None or np.abs((img.size[0] / img.size[1]) - ratio) <= self.eps):
                
                    new_images.append(img)
                    if self.filter_paths:
                        indices.append(i)
          
        print(f'Filtered {original_len - len(new_images)} images out of {original_len}. {len(new_images)} images remaining')
        kwargs['images'] = new_images
        
        if self.filter_paths:
            kwargs['paths'] = list(map(paths.__getitem__, indices))
        elif paths is not None:
            kwargs['paths'] = paths

        return kwargs
    
class FilterImagesByResolution(Operation):
    def __init__(self, min_width=None, min_height=None,
                 max_width=None, max_height=None, filter_paths=False):
        self.min_width = min_width
        self.min_height = min_height
        self.max_width = max_width
        self.max_height = max_height
        self.filter_paths = filter_paths
        
    def __call__(self, *, images, paths=None, **kwargs):
        new_images = []
        indices = []
        for i, img in enumerate(images):
            if (self.min_width is None or img.size[0] >= self.min_width) and \
               (self.min_height is None or img.size[1] >= self.min_height) and \
               (self.max_width is None or img.size[0] <= self.max_width) and \
               (self.max_height is None or img.size[1] <= self.max_height):
                   new_images.append(img)
                   if self.filter_paths:
                       indices.append(i)

        print(f'Filtered {len(images) - len(new_images)} images out of {len(images)}. {len(new_images)} images remaining')
        kwargs['images'] = new_images
        
        if self.filter_paths:
            kwargs['paths'] = list(map(paths.__getitem__, indices))
        elif paths is not None:
            kwargs['paths'] = paths

        return kwargs
    
class ResizeImages(Operation):
    def __init__(self, resolution, resample='lanczos', standarize_background=True):
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
            
        self.standarize_background = standarize_background
        
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
        
        if self.standarize_background:
            kwargs = ImagesToArrays()(**kwargs)
            kwargs = StandarizeBackgroundByAlpha(color=[255, 255, 255], binarize_alpha=True)(**kwargs)
            kwargs = ArraysToImages()(**kwargs)
        
            
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
    def __init__(self, color, binarize_alpha=False, disregard_not_rgba=True):
        self.color = color
        self.binarize_alpha = binarize_alpha
        self.disregard_not_rgba = disregard_not_rgba
        
    def __call__(self, *, arrays, **kwargs):
        new_arrays = []
        for i, arr in enumerate(arrays):
            if arr.shape[-1] == 4:
                mask = arr[..., -1] == 0
                arr[mask, :-1] = self.color
                if self.binarize_alpha:
                    alpha = arr[..., -1:]
                    alpha = alpha / 255
                    original_colors = alpha * arr[..., :-1]
                    background_colors = (1 - alpha) * self.color
                    arr[..., :-1] = (original_colors + background_colors).astype(np.uint8)
                    arr[~mask, -1] = 255
                new_arrays.append(arr)
            elif self.disregard_not_rgba:
                new_arrays.append(arr)
        
        kwargs['arrays'] = new_arrays
        return kwargs
    
class SaveImages(Operation):
    def __init__(self, destination_path, extension='png', use_paths=False,
                 clean_destination_dir=False, concat_path_to_filename=False):
        self.destination_path = destination_path
        self.extension = extension
        self.use_paths = use_paths
        self.clean_destination_dir = clean_destination_dir
        self.concat_path_to_filename = concat_path_to_filename
        
    def __call__(self, *, images, paths=None, **kwargs):
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
            
            if self.concat_path_to_filename:
                file_name = paths[i-1] + '_' + file_name
            path = os.path.join(path, file_name)
            
            img.save(path)
            
        print(f'{len(images)} images saved')
        if paths is not None:
            kwargs['paths'] = paths
        kwargs['images'] = images
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
    def __init__(self, min_colors=None, max_colors=None, filter_paths=False):
        self.min_colors = min_colors
        self.max_colors = max_colors
        self.filter_paths = filter_paths
        
    def __call__(self, *, arrays, paths=None, **kwargs):
        colors_counts = [image_statistics.get_number_of_colors(arr) for arr in arrays]
        
        new_arrays = []
        indices = []
        for i, (arr, cnt) in enumerate(zip(arrays, colors_counts)):
            if (self.min_colors is None or cnt >= self.min_colors) and \
               (self.max_colors is None or cnt <= self.max_colors):
                   new_arrays.append(arr)
                   if self.filter_paths:
                       indices.append(i)              
        
        print(f'Filtered {len(arrays) - len(new_arrays)} images out of {len(arrays)}. {len(new_arrays)} images remaining')
            
        if self.filter_paths:
            kwargs['paths'] = list(map(paths.__getitem__, indices))
        elif paths is not None:
            kwargs['paths'] = paths
    
        kwargs['arrays'] = new_arrays
        return kwargs

class GetBoxesFromArrays(Operation):
    def __init__(self, preserve=True):
        self.preserve = preserve
        
    def __call__(self, *, arrays, **kwargs):
        masks = [arr[:, :, 3] for arr in arrays] # asserts that arrays are in good format
        
        top_coords = image_statistics.get_top_offsets(masks) 
        bottom_offsets = image_statistics.get_bottom_offsets(masks) 
        left_coords = image_statistics.get_left_offsets(masks)      
        right_offsets = image_statistics.get_right_offsets(masks) 
        
        bottom_coords = [arr.shape[0] - offset for arr, offset in zip(arrays, bottom_offsets)]
        right_coords = [arr.shape[1] - offset for arr, offset in zip(arrays, right_offsets)]
        
        boxes = [(t_c, b_c, l_c, r_c) for t_c, b_c, l_c, r_c in zip(top_coords, bottom_coords, left_coords, right_coords)]
        kwargs['boxes'] = np.array(boxes)
        if self.preserve:
            kwargs['arrays'] = arrays
        
        return kwargs
    
class FilterByBoxSize(Operation):
    def __init__(self, min_height=None, min_width=None, max_height=None, max_width=None):
        self.min_height = min_height
        self.min_width = min_width
        self.max_height = max_height
        self.max_width = max_width
        
    def _is_between(self, val, lb, hb):
        if (lb is None or val >= lb) and (hb is None or val <= hb):
            return True
        
    def __call__(self, *, boxes, paths=None, images=None, arrays=None, **kwargs):
        
        indices = []
        for i, box in enumerate(boxes):
            height = box[1] - box[0]
            width = box[-1] - box[-2]
            
            if self._is_between(height, self.min_height, self.max_height) and \
                self._is_between(width, self.min_width, self.max_width):
                    indices.append(i)
        
        if paths is not None:
            kwargs['paths'] = list(map(paths.__getitem__, indices))
        if images is not None:
            kwargs['images'] = list(map(images.__getitem__, indices))
        if paths is not None:
            kwargs['arrays'] = list(map(arrays.__getitem__, indices))
        new_boxes = list(map(boxes.__getitem__, indices))
        kwargs['boxes'] = new_boxes
        print(f'Filtered {len(boxes) - len(new_boxes)} images out of {len(boxes)}. {len(new_boxes)} images remaining')

        return kwargs
            
class CropArraysByBoxes(Operation):
    def __init__(self, preserve=True):
        self.preserve = preserve
        
    def __call__(self, *, boxes, arrays, **kwargs):
        new_arrays = [arr[box[0]: box[1], box[2]: box[3]] for arr, box in zip(arrays, boxes)]
        if self.preserve:
            kwargs['boxes'] = boxes
        kwargs['arrays'] = new_arrays
        
        return kwargs
    
class DropDuplicateArrays(Operation):
    def __init__(self):
        pass
    
    def __call__(self, *, arrays, paths=None, images=None, **kwargs):
        arrays = np.array(arrays)
        new_arrays, indices = np.unique(arrays, return_index=True, axis=0)
        
        if paths is not None:
            kwargs['paths'] = list(map(paths.__getitem__, indices))
        if images is not None:
            kwargs['images'] = list(map(images.__getitem__, indices))
        kwargs['arrays'] = list(map(arrays.__getitem__, indices))
        print(f'Dropped {len(arrays) - len(new_arrays)} images out of {len(arrays)}. {len(new_arrays)} images remaining')
        return kwargs
    
class CropImages(Operation):
    def __init__(self, crop_box, get_positions='first', stride=None, clone_paths=True):
        self.crop_box = crop_box
        self.get_positions = get_positions
        if stride is None:
            self.stride = crop_box
        else:
            self.stride = stride
        self.clone_paths = clone_paths
        
    def __call__(self, *, images, paths=False, **kwargs):
        if self.get_positions == 'first':
            positions = [(0, 0)]
        elif self.get_positions != 'all':
            positions = self.get_positions
        else:
            positions = None
            
        new_images = []
        new_paths = []
        for i, img in enumerate(images):
            res = img.size
            if positions == None:
                x_positions = res[0] // self.stride[0]
                y_positions = res[1] // self.stride[1]
                positions = [(x, y) for x in range(x_positions) for y in range(y_positions)]
                
            for position in positions:
                x_pos = position[0] * self.stride[0]
                y_pos = position[1] * self.stride[1]
                box = (x_pos, y_pos, x_pos+self.crop_box[0], y_pos+self.crop_box[1])
                cropped_img = img.crop(box)
                new_images.append(cropped_img)
                if self.clone_paths:
                    new_paths.append(paths[i])
                    
        if self.clone_paths:
            kwargs['paths'] = new_paths
        elif paths is not None:
            kwargs['paths'] = paths
        kwargs['images'] = new_images
        return kwargs        

class AddBorderToArray(Operation):
    def __init__(self, width, color, alpha_channel_val=None):
        self.width = width
        self.color = color
        self.alpha_channel_val = alpha_channel_val
        
    def __call__(self, *, arrays, **kwargs):
        new_arrays = []
        for i, arr in enumerate(arrays):
            arr_height = arr.shape[0]
            arr_width = arr.shape[1]
            
            if arr.shape[-1] == 4:
                last_channel = 3
            else:
                last_channel = arr.shape[-1]
                
            arr[0: self.width, :, :last_channel] = self.color
            arr[arr_height-self.width: , :, :last_channel] = self.color
            
            arr[:, 0: self.width, :last_channel] = self.color
            arr[:, arr_width-self.width: , :last_channel] = self.color
            
            if self.alpha_channel_val is not None:
                arr[0: self.width, :, last_channel] = self.alpha_channel_val
                arr[arr_height-self.width: , :, last_channel] = self.alpha_channel_val
                arr[:, 0: self.width, last_channel] = self.alpha_channel_val
                arr[:, arr_width-self.width: , last_channel] = self.alpha_channel_val
                
            new_arrays.append(arr)
    
        kwargs['arrays'] = new_arrays
        return kwargs
    
class DropAlpha(Operation):
    def __init__(self):
        pass
        
    def __call__(self, *, images, **kwargs):
        bands = [img.split() for img in images]
        kwargs['images'] = [Image.merge('RGB', bands[:-1]) if len(bands) == 4 else img for img, bands in zip(images, bands)]
        return kwargs
    
if __name__ == '__main__':
    path = 'raw/pokemon'
    seq = Sequencer([
        GetPaths(recursive=True,
                 allowed_extensions=['.gif', '.png', '.jpg', '.jpeg'],
                 banned_keywords=['back']),
        LoadImages(),
        ChangeImageMode(),
        FilterImages(mode='RGBA', filter_paths=True),
        ImagesToArrays(),
        FilterByColorsNumber(min_colors=2, max_colors=32, filter_paths=True),
        StandarizeBackgroundByAlpha((255, 255, 255), binarize_alpha=True),
        ShortenPaths(delete_patterns=[r'raw\/'], delete_last_segment=True),
        GetBoxesFromArrays(),
        FilterByBoxSize(min_height=2, min_width=2, max_height=128, max_width=128),
        CropArraysByBoxes(),
        ArraysToImages(),
        PadImages(to_resolution=(128, 128), color=255),
        ImagesToArrays(),
        DropDuplicateArrays(),
        SaveImages('processed/pokemon_pixelart_final', use_paths=False, clean_destination_dir=True)
        ])
    data = seq(paths=[path])
    print('\nOverall time:', seq.time_taken)
                