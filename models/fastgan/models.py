import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch.nn.functional as F

import random
import numpy as np

import copy

seq = nn.Sequential

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            m.weight.data.normal_(0.0, 0.02)
        except:
            pass
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def conv2d(*args, **kwargs):
    return spectral_norm(nn.Conv2d(*args, **kwargs))

def convTranspose2d(*args, **kwargs):
    return spectral_norm(nn.ConvTranspose2d(*args, **kwargs))

def batchNorm2d(*args, **kwargs):
    return nn.BatchNorm2d(*args, **kwargs)

def linear(*args, **kwargs):
    return spectral_norm(nn.Linear(*args, **kwargs))

class PixelNorm(nn.Module):
    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)

class Reshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.target_shape = shape

    def forward(self, feat):
        batch = feat.shape[0]
        return feat.view(batch, *self.target_shape)        


class GLU(nn.Module):
    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc/2)
        return x[:, :nc] * torch.sigmoid(x[:, nc:])


class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, feat, noise=None):
        if noise is None:
            batch, _, height, width = feat.shape
            noise = torch.randn(batch, 1, height, width).to(feat.device)

        return feat + self.weight * noise


class Swish(nn.Module):
    def forward(self, feat):
        return feat * torch.sigmoid(feat)


class SEBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()

        self.main = nn.Sequential(  nn.AdaptiveAvgPool2d(4), 
                                    conv2d(ch_in, ch_out, 4, 1, 0, bias=False), Swish(),
                                    conv2d(ch_out, ch_out, 1, 1, 0, bias=False), nn.Sigmoid() )

    def forward(self, feat_small, feat_big):
        return feat_big * self.main(feat_small)


class InitLayer(nn.Module):
    def __init__(self, nz, channel, size):
        super().__init__()

        self.init = nn.Sequential(
                        convTranspose2d(nz, channel*2, size, 1, 0, bias=False),
                        batchNorm2d(channel*2), GLU() )

    def forward(self, noise):
        noise = noise.view(noise.shape[0], -1, 1, 1)
        return self.init(noise)


def UpBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv2d(in_planes, out_planes*2, 3, 1, 1, bias=False),
        #convTranspose2d(in_planes, out_planes*2, 4, 2, 1, bias=False),
        batchNorm2d(out_planes*2), GLU())
    return block


def UpBlockComp(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv2d(in_planes, out_planes*2, 3, 1, 1, bias=False),
        #convTranspose2d(in_planes, out_planes*2, 4, 2, 1, bias=False),
        NoiseInjection(),
        batchNorm2d(out_planes*2), GLU(),
        conv2d(out_planes, out_planes*2, 3, 1, 1, bias=False),
        NoiseInjection(),
        batchNorm2d(out_planes*2), GLU()
        )
    return block


class Generator(nn.Module):
    def __init__(self, sizes, ngf=64, nz=100, nc=3):
        super(Generator, self).__init__()

        self.sizes = sizes
        
        self.output_size = sizes[-1]
        possible_n_channels = [2**i for i in range(1, 12)]
        
        nfc = {}
        for s in self.sizes:
            val = 64 * ngf // int(np.round(np.sqrt(np.prod(s))))
            diffs = [np.abs(val - v) for v in possible_n_channels]
            val = possible_n_channels[np.argmin(diffs)]
            nfc[s] = val

        print('nfc:', nfc)

        self.init = InitLayer(nz, channel=nfc[sizes[0]], size=sizes[0])
        
        for i in range(len(self.sizes) - 1):
            n_channels = nfc[sizes[i]]
            next_n_channels = nfc[sizes[i+1]]
            if i % 2 == 0:
                block = UpBlockComp(n_channels, next_n_channels)
            else:
                block = UpBlock(n_channels, next_n_channels)
            
            setattr(self, 'feat_' + str('_'.join(str(i) for i in sizes[i+1])), block)
            
            if i >= len(self.sizes) // 2:
                se_block = SEBlock(nfc[sizes[i-len(self.sizes)//2]], nfc[sizes[i+1]])
                setattr(self, 'se_' + str('_'.join(str(i) for i in sizes[i+1])), se_block)

        self.minor_size = self.sizes[-2]
        self.to_small = conv2d(nfc[self.minor_size], nc,
                               kernel_size=1, stride=1,
                               padding=0, bias=False) 
        self.to_big = conv2d(nfc[self.output_size], nc,
                             kernel_size=3, stride=1,
                             padding=1, bias=False) 
        
        
    def forward(self, input):
                
        feat = self.init(input)
        feats = [feat]
        for i in range(len(self.sizes) - 1):
            size = self.sizes[i+1]
            block_name = 'feat_' + str('_'.join(str(i) for i in size))
            block = getattr(self, block_name)
            feat = block(feat)
            
            if i >= len(self.sizes) // 2:
                block_name = 'se_' + str('_'.join(str(i) for i in size))
                se_block = getattr(self, block_name)
                feat = se_block(feats[i - len(self.sizes) // 2], feat)
            
            feats.append(feat)

        out1 = self.to_big(feats[-1])
        out2 = self.to_small(feats[-2])
        
        return [out1, out2]



class DownBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(DownBlock, self).__init__()

        self.main = nn.Sequential(
            conv2d(in_planes, out_planes, 4, 2, 1, bias=False),
            batchNorm2d(out_planes), nn.LeakyReLU(0.2, inplace=True),
            )

    def forward(self, feat):
        return self.main(feat)


class DownBlockComp(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(DownBlockComp, self).__init__()

        self.main = nn.Sequential(
            conv2d(in_planes, out_planes, 4, 2, 1, bias=False),
            batchNorm2d(out_planes), nn.LeakyReLU(0.2, inplace=True),
            conv2d(out_planes, out_planes, 3, 1, 1, bias=False),
            batchNorm2d(out_planes), nn.LeakyReLU(0.2)
            )

        self.direct = nn.Sequential(
            nn.AvgPool2d(2, 2),
            conv2d(in_planes, out_planes, 1, 1, 0, bias=False),
            batchNorm2d(out_planes), nn.LeakyReLU(0.2))

    def forward(self, feat):
        return (self.main(feat) + self.direct(feat)) / 2


class Discriminator(nn.Module):
    def __init__(self, sizes, ndf=64, nc=3, channels_out=1, final_kernel=(1, 1)):
        super(Discriminator, self).__init__()
        self.ndf = ndf
        self.im_size = sizes[0]
        self.sizes = sizes
        
        self.output_size = sizes[-1]
        possible_n_channels = [2**i for i in range(1, 12)]

        nfc = {}
        for s in self.sizes:
            val = 64 * ndf // int(np.round(np.sqrt(np.prod(s))))
            diffs = [np.abs(val - v) for v in possible_n_channels]
            val = possible_n_channels[np.argmin(diffs)]
            nfc[s] = val
            
        print('nfc:', nfc)

        self.down_from_big = nn.Sequential( 
                                conv2d(nc, nfc[self.im_size], kernel_size=3,
                                       stride=1, padding=1, bias=False),
                                nn.LeakyReLU(0.2, inplace=True))
        
        for i in range(len(self.sizes)-1):
            current_size = self.sizes[i]
            next_size = self.sizes[i+1]
            
            block_name = 'down_to_' + 'x'.join((str(s) for s in next_size))
            block = DownBlockComp(nfc[current_size], nfc[next_size])
            setattr(self, block_name, block)
                        
            if i >= (len(self.sizes)) // 2:
                smaller_size = self.sizes[i - len(self.sizes) // 2]
                se_block = SEBlock(nfc[smaller_size], nfc[next_size])
                block_name = 'se_to_' + 'x'.join(str(i) for i in next_size)
                setattr(self, block_name, se_block)
            
        self.rf_big = conv2d(nfc[self.sizes[-1]], channels_out,
                kernel_size=final_kernel, stride=1, padding=0, bias=False)

        sizes = self.sizes[1:]
        layers_for_small = [conv2d(nc, nfc[sizes[0]], kernel_size=3,
                                   stride=1, padding=1, bias=False), 
                            nn.LeakyReLU(0.2, inplace=True),]
        for i in range(len(sizes)-1):
            current_size = sizes[i]
            next_size = sizes[i+1]
            
            block_name = 'smaller_down_to_' + 'x'.join((str(s) for s in next_size))
            block = DownBlock(nfc[current_size], nfc[next_size])
            layers_for_small.append(block)
                        
        self.down_from_small = nn.Sequential(*layers_for_small)

        self.rf_small = conv2d(nfc[sizes[-1]], channels_out, kernel_size=final_kernel,
                               stride=1, padding=0, bias=False)
        
        print('='*50)
        print('decoder_big:')
        self.decoder_big = SimpleDecoder(copy.deepcopy(nfc),in_res=self.sizes[-1],
                                          out_res=self.sizes[1],
                                          nfc_in=nfc[self.sizes[-1]], nc=nc)
        print('='*50)
        print('decoder_small:')
        self.decoder_small = SimpleDecoder(copy.deepcopy(nfc),in_res=self.sizes[-1],
                                          out_res=self.sizes[1],
                                          nfc_in=nfc[self.sizes[-1]], nc=nc)
        print('='*50)
        print('decoder_part:')
        self.decoder_part = SimpleDecoder(copy.deepcopy(nfc), in_res=self.sizes[-1],
                                          out_res=self.sizes[1],
                                          nfc_in=nfc[self.sizes[-2]], nc=nc)
        
    def forward(self, imgs, label, part=None):
        if type(imgs) is not list:
            imgs = [F.interpolate(imgs, size=self.im_size), F.interpolate(imgs, size=self.sizes[1])]

        feat = self.down_from_big(imgs[0])
        
        feats = [feat]
        
        for i in range(len(self.sizes)-1):
            next_size = self.sizes[i+1]

            block_name = 'down_to_' + 'x'.join((str(s) for s in next_size))
            block = getattr(self, block_name)
            feat = block(feat)
            if i >= (len(self.sizes)) // 2:
                small_index = i - len(self.sizes) // 2
                block_name = 'se_to_' + 'x'.join(str(i) for i in next_size)
                block = getattr(self, block_name)
                feat = block(feats[small_index], feat)
            
            feats.append(feat)
        
        feat_small = self.down_from_small(imgs[1])
        
        rf_0 = self.rf_big(feat).view(-1)
        rf_1 = self.rf_small(feat_small).view(-1)

        if label=='real':    
            rec_img_big = self.decoder_big(feat)
            rec_img_small = self.decoder_small(feat_small)
            
            feat_for_crop = feats[-2]
            crop_size = (feat_for_crop.shape[2] // 2, feat_for_crop.shape[3] // 2)
            
            assert part is not None
            rec_img_part = None
            if part==0:
                crop = feat_for_crop[:,:,:crop_size[0],:crop_size[1]]
            if part==1:
                crop = feat_for_crop[:,:,:crop_size[0],crop_size[1]:]
            if part==2:
                crop = feat_for_crop[:,:,crop_size[0]:,:crop_size[1]]
            if part==3:
                crop = feat_for_crop[:,:,crop_size[0]:,crop_size[1]:]

            rec_img_part = self.decoder_part(crop)
            return torch.cat([rf_0, rf_1]) , [rec_img_big, rec_img_small, rec_img_part]

        return torch.cat([rf_0, rf_1]) 


class SimpleDecoder(nn.Module):
    """docstring for CAN_SimpleDecoder"""
    def __init__(self, nfc, in_res, out_res, nfc_in, nc=3):
        super(SimpleDecoder, self).__init__()

        print('nfc_in:', nfc_in)
        decoder_nfc = {}
        for k, v in nfc.items():
            if k[0] >= in_res[0] and k[1] >= in_res[1] and k[0] <= out_res[0] and k[1] <= out_res[1]:
                decoder_nfc[k] = nfc[k] // 2
        print('decoder nfc:', decoder_nfc)
        
        sizes = sorted(list(decoder_nfc.keys()), key=lambda x: x[0])
        print('sizes:', sizes)

        def upBlock(in_planes, out_planes):
            block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                conv2d(in_planes, out_planes*2, 3, 1, 1, bias=False),
                batchNorm2d(out_planes*2),
                GLU())
            return block

        layers = []
        current_nfc = nfc_in
        for i in range(len(sizes)-1):
            next_nfc = decoder_nfc[sizes[i+1]]
            block = upBlock(current_nfc, next_nfc)
            current_nfc = next_nfc
            layers.append(block)
        
        layers.append(conv2d(current_nfc, nc, 3, 1, 1, bias=False))
        self.main = nn.Sequential(*layers)
        
    def forward(self, input):
        # input shape: c x 4 x 4
        return self.main(input)

from random import randint
def random_crop(image, size):
    h, w = image.shape[2:]
    ch = randint(0, h-size-1)
    cw = randint(0, w-size-1)
    return image[:,:,ch:ch+size,cw:cw+size]

class TextureDiscriminator(nn.Module):
    def __init__(self, ndf=64, nc=3, im_size=512):
        super(TextureDiscriminator, self).__init__()
        self.ndf = ndf
        self.im_size = im_size

        nfc_multi = {4:16, 8:8, 16:8, 32:4, 64:2, 128:1, 256:0.5, 512:0.25, 1024:0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v*ndf)

        self.down_from_small = nn.Sequential( 
                                            conv2d(nc, nfc[256], 4, 2, 1, bias=False), 
                                            nn.LeakyReLU(0.2, inplace=True),
                                            DownBlock(nfc[256],  nfc[128]),
                                            DownBlock(nfc[128],  nfc[64]),
                                            DownBlock(nfc[64],  nfc[32]), )
        self.rf_small = nn.Sequential(
                            conv2d(nfc[16], 1, 4, 1, 0, bias=False))

        self.decoder_small = SimpleDecoder(nfc[32], nc)
        
    def forward(self, img, label):
        img = random_crop(img, size=128)

        feat_small = self.down_from_small(img)
        rf = self.rf_small(feat_small).view(-1)
        
        if label=='real':    
            rec_img_small = self.decoder_small(feat_small)

            return rf, rec_img_small, img

        return rf