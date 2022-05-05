from PIL import Image
import torch
import torchvision
import os
from torch.utils.data import Dataset
import time
import random

def get_files(path, allowed_extensions):
    files = []
    if not os.path.isdir(path):
        if path.endswith(allowed_extensions): 
            files.append(path)
        else:
            print('skipping', path, '- Wrong extension')
    else:
        for f in os.listdir(path):
            if f == '.git':
                continue
            l = get_files(os.path.join(path, f), allowed_extensions)
            files.extend(l)

    return files

def drop_alpha(img):
    img_bands = img.split()
    if len(img_bands) == 4:
        img = Image.merge('RGB', img_bands[:-1])
    return img

def load_and_transform(files, transform, convert_to_rgb=False):
    images = []
    if transform is None:
        transform = lambda x: x
    
    for f in files:
        temp = Image.open(f)
        img = temp.copy()
        images.append(img)
        temp.close()
    
    if convert_to_rgb:
        return [transform(drop_alpha(img)) for img in images]
    return [transform(img) for img in images]

class BasicDataset(Dataset):
    def __init__(self, X, initial_transform=None,
                 transform=None, measure_time=False, batch_size=1, reshuffle=True,
                 convert_to_rgb=False):
        super().__init__()
        self.X = load_and_transform(X, initial_transform, convert_to_rgb)
        
        self.transform = transform
        self.shape = self.transform(self.X[0]).shape
        assert all([self.shape == self.transform(x).shape for x in self.X])
        
        self.measure_time = measure_time
        self.time_taken = 0.0
        self.batch_size = batch_size
        self.length = len(self.X)
        self.reshuffle = reshuffle
        
        self.index = 0
        
        if self.reshuffle:
            random.shuffle(self.X)
            
    def __len__(self):
        return self.length
    
    def get_nr_of_batches(self):
        return len(self) // self.batch_size
    
    def get_shape(self):
        return (len(self), *(self.X[0].shape))
    
    def sample(self, index):
        start_pos = index * self.batch_size
        end_pos = start_pos + self.batch_size
        if end_pos > len(self):
            if self.reshuffle:
                random.shuffle(self.X)
            else:
                self.X = self.X[start_pos:] + self.X[:start_pos]
            raise IndexError

        sampled = self.X[start_pos: start_pos + self.batch_size]
        return sampled
    
    def __getitem__(self, index):
        if self.measure_time:
            start = time.time()
            
        sampled = self.sample(index)
        if self.transform is not None:
            sampled = [self.transform(x) for x in sampled]

        if self.measure_time:
            self.time_taken += time.time() - start
            
        return torch.stack(sampled)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        start_pos = self.index * self.batch_size
        end_pos = start_pos + self.batch_size
        
        if end_pos > len(self):
            if self.reshuffle:
                random.shuffle(self.X)
            else:
                self.X = self.X[start_pos:] + self.X[:start_pos]
            self.index = 0
        
        sampled = self.X[start_pos: start_pos + self.batch_size]
        return sampled
        