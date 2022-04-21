import os
import json
import sys
import time

import torch
from torch import nn
from torch import optim
import torchvision as tv
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import wandb

EPS = 1e-6
ALPHA_RECONSTRUCT_IMAGE = 1
ALPHA_RECONSTRUCT_LATENT = 0.5
ALPHA_DISCRIMINATE_IMAGE = 0.005
ALPHA_DISCRIMINATE_LATENT = 0.1

class Generator(nn.Module):
    """A generator for mapping a latent space to a sample space.

    Input shape: (?, latent_dim)
    Output shape: (?, 3, 96, 96)
    """

    def __init__(self, latent_dim, sizes, max_features=512):
        """Initialize generator.

        Args:
            latent_dim (int): latent dimension ("noise vector")
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.sizes = sizes
        self.max_features = max_features
        self._init_modules()

    def build_colourspace(self, input_dim: int, output_dim: int):
        """Build a small module for selecting colours."""
        colourspace = nn.Sequential(
            nn.Linear(
                input_dim,
                128,
                bias=True),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),

            nn.Linear(
                128,
                64,
                bias=True),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),

            nn.Linear(
                64,
                output_dim,
                bias=True),
            nn.Tanh(),
            )
        return colourspace

    def _init_modules(self):
        """Initialize the modules."""
        projection_widths = [8, 8, 8, 8, 8, 8, 8]
        self.projection_dim = sum(projection_widths) + self.latent_dim
        self.projection = nn.ModuleList()
        for index, i in enumerate(projection_widths):
            self.projection.append(
                nn.Sequential(
                    nn.Linear(
                        self.latent_dim + sum(projection_widths[:index]),
                        i,
                        bias=True,
                        ),
                    nn.BatchNorm1d(8),
                    nn.LeakyReLU(),
                    )
                )
        self.projection_upscaler = nn.Upsample(scale_factor=self.sizes[0])

        self.colourspace_r = self.build_colourspace(self.projection_dim, 16)
        self.colourspace_g = self.build_colourspace(self.projection_dim, 16)
        self.colourspace_b = self.build_colourspace(self.projection_dim, 16)
        self.colourspace_upscaler = nn.Upsample(scale_factor=self.sizes[-1])
        
        min_im_size_flatten = self.sizes[0][0] * self.sizes[0][1]
        
        self.seed = nn.Sequential(
            nn.Linear(
                self.projection_dim,
                self.max_features * min_im_size_flatten,
                bias=True),
            nn.BatchNorm1d(self.max_features * min_im_size_flatten),
            nn.LeakyReLU(),
            )

        self.upscaling = nn.ModuleList()
        self.conv = nn.ModuleList()

        self.upscaling.extend([nn.Upsample(scale_factor=2) for _ in range(len(self.sizes) - 1)])
        self.upscaling.append(nn.Upsample(scale_factor=1))
        
        def build_conv(in_channels=256, out_channels=256, kernel_size=4):
            conv = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=1, 
                    padding='same',
                    bias=True
                    ),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(),
                )
            return conv

        self.conv.append(build_conv(512//4, 512, 3))
        params = [{'in_channels': (256 + self.projection_dim) // 4,
                   'out_channels': 256,
                   'kernel_size': 4} for _ in range(len(self.sizes) - 2)]
        if len(params) > 0:
            params[0]['in_channels'] = (512 + self.projection_dim) // 4
            params[-1]['out_channels'] = 64
            self.conv.extend([build_conv(**p) for p in params])

        self.conv.append(nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding='same',
                bias=True
                ),
            nn.Softmax(dim=1),
            ))

    def forward(self, input_tensor):
        """Forward pass; map latent vectors to samples."""
        last = input_tensor
        for module in self.projection:
            projection = module(last)
            last = torch.cat((last, projection), -1)
        projection = last

        intermediate = self.seed(projection)
        intermediate = intermediate.view((-1, self.max_features, self.sizes[0][0], self.sizes[0][1]))

        projection_2d = projection.view((-1, self.projection_dim, 1, 1))
        projection_2d = self.projection_upscaler(projection_2d)

        for i, (conv, upscaling) in enumerate(zip(self.conv, self.upscaling)):
            if i + 1 != len(self.upscaling):
                if i > 0:
                    intermediate = torch.cat((intermediate, projection_2d), 1)
                intermediate = torch.nn.functional.pixel_shuffle(intermediate, 2)
            intermediate = conv(intermediate)
            projection_2d = upscaling(projection_2d)

        r_space = self.colourspace_r(projection)
        r_space = r_space.view((-1, 16, 1, 1))
        r_space = self.colourspace_upscaler(r_space)
        r_space = intermediate * r_space
        r_space = torch.sum(r_space, dim=1, keepdim=True)

        g_space = self.colourspace_g(projection)
        g_space = g_space.view((-1, 16, 1, 1))
        g_space = self.colourspace_upscaler(g_space)
        g_space = intermediate * g_space
        g_space = torch.sum(g_space, dim=1, keepdim=True)

        b_space = self.colourspace_b(projection)
        b_space = b_space.view((-1, 16, 1, 1))
        b_space = self.colourspace_upscaler(b_space)
        b_space = intermediate * b_space
        b_space = torch.sum(b_space, dim=1, keepdim=True)

        output = torch.cat((r_space, g_space, b_space), dim=1)
        
        return output


class Encoder(nn.Module):
    """An Encoder for encoding images as latent vectors.

    Input shape: (?, 3, 96, 96)
    Output shape: (?, latent_dim)
    """

    def __init__(self, device: str = "cpu", latent_dim: int = 8, sizes=None):
        """Initialize encoder.

        Args:
            device: chich GPU or CPU to use.
            latent_dim: output dimension
        """
        super().__init__()
        self.device = device
        self.latent_dim = latent_dim
        self.sizes = sizes
        self._init_modules()

    def _init_modules(self):
        """Initialize the modules."""
        #print('SIZE:', self.sizes)
        down_channels = [3] + [64 * (2 ** i) for i in range(len(self.sizes) - 1)]
        #print('Encoder\'s DOWN_CHANNELS:', down_channels)
        self.down = nn.ModuleList()
        for i in range(len(down_channels)-1):
            self.down.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=down_channels[i],
                        out_channels=down_channels[i+1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias=True,
                        ),
                    nn.BatchNorm2d(down_channels[i+1]),
                    nn.LeakyReLU(),
                    )
                )

        self.reducer = nn.Sequential(
            nn.Conv2d(
                in_channels=down_channels[-1],
                out_channels=down_channels[-2],
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
                ),
            nn.BatchNorm2d(down_channels[-2]),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2)
            )

        #up_channels = [256, 128, 64, 64, 64]
        #up_channels = [max(i, 64) for i in down_channels[::-1]]
        up_channels = [max(64, down_channels[-2] // (2 ** i)) for i in range(len(self.sizes))]
        #print('Encoder\'s UP_CHANNELS:', up_channels)
        scale_factors = [2 for _ in up_channels[:-2]] + [1]
        self.up = nn.ModuleList()
        for i in range(len(up_channels)-1):
            self.up.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=up_channels[i] + down_channels[-2-i],
                        out_channels=up_channels[i+1],
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=True,
                        ),
                    nn.BatchNorm2d(up_channels[i+1]),
                    nn.LeakyReLU(),
                    nn.Upsample(scale_factor=scale_factors[i]),
                    )
                )

        down_again_channels = [64 for _ in up_channels]
        down_again_channels[0] = 64 + 3
        #down_again_channels = [64+3, 64, 64, 64, 64]
        self.down_again = nn.ModuleList()
        for i in range(len(down_again_channels)-1):
            self.down_again.append(
                nn.Conv2d(
                    in_channels=down_again_channels[i],
                    out_channels=down_again_channels[i+1],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=True,
                    )
                )
            self.down_again.append(nn.BatchNorm2d(down_again_channels[i+1]))
            self.down_again.append(nn.LeakyReLU())

        flattened_size = self.sizes[-1][0] * self.sizes[-1][1]
        self.projection = nn.Sequential(
            nn.Linear(
                down_channels[-1] * flattened_size + down_again_channels[-1] * flattened_size,
                256,
                bias=True,
                ),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),

            nn.Linear(
                256,
                128,
                bias=True,
                ),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),

            nn.Linear(
                128,
                self.latent_dim,
                bias=True,
                ),
            )

    def forward(self, input_tensor):
        """Forward pass; map latent vectors to samples."""
        rv = torch.randn(input_tensor.size(), device=self.device) * 0.02
        augmented_input = input_tensor + rv
        intermediate = augmented_input
        intermediates = [augmented_input]
        for module in self.down:
            intermediate = module(intermediate)
            intermediates.append(intermediate)
        intermediates = intermediates[:-1][::-1]
        flattened_size = self.sizes[-1][0] * self.sizes[-1][1]
        down = intermediate.view(-1, flattened_size * intermediate.shape[1])

        intermediate = self.reducer(intermediate)
        for index, module in enumerate(self.up):
            intermediate = torch.cat((intermediate, intermediates[index]), 1)
            intermediate = module(intermediate)

        intermediate = torch.cat((intermediate, input_tensor), 1)

        for module in self.down_again:
            intermediate = module(intermediate)

        intermediate = intermediate.view(-1, flattened_size * intermediate.shape[1])
        intermediate = torch.cat((down, intermediate), -1)
        projected = self.projection(intermediate)

        return projected


class DiscriminatorImage(nn.Module):
    """A discriminator for discerning real from generated images.

    Input shape: (?, 3, 96, 96)
    Output shape: (?, 1)
    """

    def __init__(self, device="cpu", sizes=None):
        """Initialize the discriminator."""
        super().__init__()
        self.device = device
        self.sizes = sizes
        self._init_modules()

    def _init_modules(self):
        """Initialize the modules."""
        #down_channels = [3, 64, 128, 256, 512]
        down_channels = [3] + [64 * (2 ** i) for i in range(len(self.sizes) - 1)]
        self.down = nn.ModuleList()
        leaky_relu = nn.LeakyReLU()
        for i in range(len(down_channels)-1):
            self.down.append(
                nn.Conv2d(
                    in_channels=down_channels[i],
                    out_channels=down_channels[i+1],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=True,
                    )
                )
            self.down.append(nn.BatchNorm2d(down_channels[i+1]))
            self.down.append(leaky_relu)

        self.classifier = nn.ModuleList()
        self.width = down_channels[-1] * self.sizes[-1][0] * self.sizes[-1][1]
        self.classifier.append(nn.Linear(self.width, 1))
        self.classifier.append(nn.Sigmoid())

    def forward(self, input_tensor):
        """Forward pass; map latent vectors to samples."""
        rv = torch.randn(input_tensor.size(), device=self.device) * 0.02
        intermediate = input_tensor + rv
        for module in self.down:
            intermediate = module(intermediate)
            rv = torch.randn(intermediate.size(), device=self.device) * 0.02 + 1
            intermediate *= rv

        intermediate = intermediate.view(-1, self.width)

        for module in self.classifier:
            intermediate = module(intermediate)

        return intermediate


class DiscriminatorLatent(nn.Module):
    """A discriminator for discerning real from generated vectors.

    Input shape: (?, latent_dim)
    Output shape: (?, 1)
    """

    def __init__(self, latent_dim=8, device="cpu"):
        """Initialize the Discriminator."""
        super().__init__()
        self.latent_dim = latent_dim
        self.device = device
        self._init_modules()

    def _init_modules(self, depth=7, width=8):
        """Initialize the modules."""
        self.pyramid = nn.ModuleList()
        for i in range(depth):
            self.pyramid.append(
                nn.Sequential(
                    nn.Linear(
                        self.latent_dim + width*i,
                        width,
                        bias=True,
                        ),
                    nn.BatchNorm1d(width),
                    nn.LeakyReLU(),
                    )
                )

        self.classifier = nn.ModuleList()
        self.classifier.append(nn.Linear(depth*width + self.latent_dim, 1))
        self.classifier.append(nn.Sigmoid())

    def forward(self, input_tensor):
        """Forward pass; map latent vectors to samples."""
        last = input_tensor
        for module in self.pyramid:
            projection = module(last)
            rv = torch.randn(projection.size(), device=self.device) * 0.02 + 1
            projection *= rv
            last = torch.cat((last, projection), -1)
        for module in self.classifier:
            last = module(last)
        return last


class AEGAN():
    """An Autoencoder Generative Adversarial Network for making pokemon."""

    def __init__(self, *, latent_dim, sizes, noise_fn, dataloader,
                 batch_size=32, device='cpu'):
        """Initialize the AEGAN.

        Args:
            latent_dim: latent-space dimension. Must be divisible by 4.
            noise_fn: function f(num: int) -> pytorch tensor, (latent vectors)
            dataloader: a pytorch dataloader for loading images
            batch_size: training batch size. Must match that of dataloader
            device: cpu or CUDA
        """
        assert latent_dim % 4 == 0
        self.latent_dim = latent_dim
        self.device = device
        self.noise_fn = noise_fn
        self.dataloader = dataloader
        self.batch_size = batch_size

        self.criterion_gen = nn.BCELoss()
        self.criterion_recon_image = nn.L1Loss()
        self.criterion_recon_latent = nn.MSELoss()
        self.target_ones = torch.ones((batch_size, 1), device=device)
        self.target_zeros = torch.zeros((batch_size, 1), device=device)
        
        self.sizes = sizes
        self._init_generator()
        self._init_encoder()
        self._init_dx()
        self._init_dz()
        
    def save_model(self, name, path=None):
        if path is None:
            path = wandb.run.dir
        
        gen_path = os.path.join(path, 'generator')
        enc_path = os.path.join(path, 'encoder')
        disc_img_path = os.path.join(path, 'disc_img')
        disc_latent_path = os.path.join(path, 'disc_latent')
        
        os.makedirs(gen_path, exist_ok=True)
        os.makedirs(enc_path, exist_ok=True)
        os.makedirs(disc_img_path, exist_ok=True)
        os.makedirs(disc_latent_path, exist_ok=True)
    
        torch.save(
            {'model_state_dict': self.generator.state_dict(),
             'optimizer_state_dict': self.optim_g.state_dict()},
            os.path.join(gen_path, f"{name}.pt"))
        
        torch.save(
            {'model_state_dict': self.encoder.state_dict(),
             'optimizer_state_dict': self.optim_e.state_dict()},
            os.path.join(enc_path, f"{name}.pt"))

        torch.save(
            {'model_state_dict': self.discriminator_image.state_dict(),
             'optimizer_state_dict': self.optim_di.state_dict()},
            os.path.join(disc_img_path, f"{name}.pt"))
        
        torch.save(
            {'model_state_dict': self.discriminator_latent.state_dict(),
             'optimizer_state_dict': self.optim_dl.state_dict()},
            os.path.join(disc_latent_path, f"{name}.pt"))
        
    def load_model(self, path, name):
        gen_path = os.path.join(path, 'generator', name)
        enc_path = os.path.join(path, 'encoder', name)
        disc_img_path = os.path.join(path, 'disc_img', name)
        disc_latent_path = os.path.join(path, 'disc_latent', name)
        
        paths = [gen_path, enc_path, disc_img_path, disc_latent_path]
        models = [self.generator, self.encoder,
                  self.discriminator_image, self.discriminator_latent]
        optims = [self.optim_g, self.optim_e, self.optim_di, self.optim_dl]
        
        for p, model, opt in zip(paths, models, optims):
            checkpoint = torch.load(p)
            model.load_state_dict(checkpoint['model_state_dict'])
            opt.load_state_dict(checkpoint['optimizer_state_dict'])

    def _init_generator(self):
        self.generator = Generator(latent_dim=self.latent_dim, sizes=list(reversed(self.sizes)))
        self.generator = self.generator.to(self.device)
        self.optim_g = optim.Adam(self.generator.parameters(),
                                  lr=2e-4, betas=(0.5, 0.999),
                                  weight_decay=1e-8)

    def _init_encoder(self):
        self.encoder = Encoder(latent_dim=self.latent_dim, device=self.device, sizes=self.sizes)
        self.encoder = self.encoder.to(self.device)
        self.optim_e = optim.Adam(self.encoder.parameters(),
                                  lr=2e-4, betas=(0.5, 0.999),
                                  weight_decay=1e-8)

    def _init_dx(self):
        self.discriminator_image = DiscriminatorImage(device=self.device, sizes=self.sizes).to(self.device)
        self.optim_di = optim.Adam(self.discriminator_image.parameters(),
                                   lr=1e-4, betas=(0.5, 0.999),
                                   weight_decay=1e-8)

    def _init_dz(self):
        self.discriminator_latent = DiscriminatorLatent(
            latent_dim=self.latent_dim,
            device=self.device,
            ).to(self.device)
        self.optim_dl = optim.Adam(self.discriminator_latent.parameters(),
                                   lr=1e-4, betas=(0.5, 0.999),
                                   weight_decay=1e-8)


    def generate_samples(self, latent_vec=None, num=None):
        """Sample images from the generator.

        Images are returned as a 4D tensor of values between -1 and 1.
        Dimensions are (number, channels, height, width). Returns the tensor
        on cpu.

        Args:
            latent_vec: A pytorch latent vector or None
            num: The number of samples to generate if latent_vec is None

        If latent_vec and num are None then use self.batch_size
        random latent vectors.
        """
        num = self.batch_size if num is None else num
        latent_vec = self.noise_fn(num) if latent_vec is None else latent_vec
        with torch.no_grad():
            samples = self.generator(latent_vec)
        samples = samples.cpu()  # move images to cpu
        return samples

    def train_step_generators(self, X):
        """Train the generator one step and return the loss."""
        self.generator.zero_grad()
        self.encoder.zero_grad()

        Z = self.noise_fn(self.batch_size)

        X_hat = self.generator(Z)
        Z_hat = self.encoder(X)
        X_tilde = self.generator(Z_hat)
        Z_tilde = self.encoder(X_hat)

        X_hat_confidence = self.discriminator_image(X_hat)
        Z_hat_confidence = self.discriminator_latent(Z_hat)
        X_tilde_confidence = self.discriminator_image(X_tilde)
        Z_tilde_confidence = self.discriminator_latent(Z_tilde)

        X_hat_loss = self.criterion_gen(X_hat_confidence, self.target_ones)
        Z_hat_loss = self.criterion_gen(Z_hat_confidence, self.target_ones)
        X_tilde_loss = self.criterion_gen(X_tilde_confidence, self.target_ones)
        Z_tilde_loss = self.criterion_gen(Z_tilde_confidence, self.target_ones)

        X_recon_loss = self.criterion_recon_image(X_tilde, X) * ALPHA_RECONSTRUCT_IMAGE
        Z_recon_loss = self.criterion_recon_latent(Z_tilde, Z) * ALPHA_RECONSTRUCT_LATENT

        X_loss = (X_hat_loss + X_tilde_loss) / 2 * ALPHA_DISCRIMINATE_IMAGE
        Z_loss = (Z_hat_loss + Z_tilde_loss) / 2 * ALPHA_DISCRIMINATE_LATENT
        loss = X_loss + Z_loss + X_recon_loss + Z_recon_loss

        loss.backward()
        self.optim_e.step()
        self.optim_g.step()

        return X_loss.item(), Z_loss.item(), X_recon_loss.item(), Z_recon_loss.item()

    def train_step_discriminators(self, X):
        """Train the discriminator one step and return the losses."""
        self.discriminator_image.zero_grad()
        self.discriminator_latent.zero_grad()

        Z = self.noise_fn(self.batch_size)

        with torch.no_grad():
            X_hat = self.generator(Z)
            Z_hat = self.encoder(X)
            X_tilde = self.generator(Z_hat)
            Z_tilde = self.encoder(X_hat)

        X_confidence = self.discriminator_image(X)
        X_hat_confidence = self.discriminator_image(X_hat)
        X_tilde_confidence = self.discriminator_image(X_tilde)
        Z_confidence = self.discriminator_latent(Z)
        Z_hat_confidence = self.discriminator_latent(Z_hat)
        Z_tilde_confidence = self.discriminator_latent(Z_tilde)

        X_loss = 2 * self.criterion_gen(X_confidence, self.target_ones)
        X_hat_loss = self.criterion_gen(X_hat_confidence, self.target_zeros)
        X_tilde_loss = self.criterion_gen(X_tilde_confidence, self.target_zeros)
        Z_loss = 2 * self.criterion_gen(Z_confidence, self.target_ones)
        Z_hat_loss = self.criterion_gen(Z_hat_confidence, self.target_zeros)
        Z_tilde_loss = self.criterion_gen(Z_tilde_confidence, self.target_zeros)

        loss_images = (X_loss + X_hat_loss + X_tilde_loss) / 4
        loss_latent = (Z_loss + Z_hat_loss + Z_tilde_loss) / 4
        loss = loss_images + loss_latent

        loss.backward()
        self.optim_di.step()
        self.optim_dl.step()

        return loss_images.item(), loss_latent.item()

    def train_epoch(self, print_frequency=1, max_steps=0):
        """Train both networks for one epoch and return the losses.

        Args:
            print_frequency (int): print stats every `print_frequency` steps.
            max_steps (int): End epoch after `max_steps` steps, or set to 0
                             to do the full epoch.
        """
        ldx, ldz, lgx, lgz, lrx, lrz = 0, 0, 0, 0, 0, 0
        eps = 1e-9
        batch = 0
        for real_samples in self.dataloader:
            real_samples = real_samples.to(self.device)
            ldx_, ldz_ = self.train_step_discriminators(real_samples)
            ldx += ldx_
            ldz += ldz_
            lgx_, lgz_, lrx_, lrz_ = self.train_step_generators(real_samples)
            lgx += lgx_
            lgz += lgz_
            lrx += lrx_
            lrz += lrz_
            if print_frequency and (batch+1) % print_frequency == 0:
                d = {
                    'G': lgx / (eps + (batch+1) * ALPHA_DISCRIMINATE_IMAGE),
                    'E': lgz / (eps + (batch+1) * ALPHA_DISCRIMINATE_LATENT),
                    'Dx': ldx / (eps + (batch+1)),
                    'Dz': ldz / (eps + (batch+1)),
                    'Rx': lrx / (eps + (batch+1) * ALPHA_RECONSTRUCT_IMAGE),
                    'Rz': lrz / (eps + (batch+1) * ALPHA_RECONSTRUCT_LATENT)
                 }
                print(f"{batch+1}/{self.dataloader.get_nr_of_batches()}:"
                      f" G=:{d['G']:.3f},"
                      f" E={d['E']:.3f},"
                      f" Dx={d['Dx']:.3f},"
                      f" Dz={d['Dz']:.3f}",
                      f" Rx={d['Rx']:.3f}",
                      f" Rz={d['Rz']:.3f}",
                      end='\n',
                      flush=True)
                wandb.log(d)
            if max_steps and batch == max_steps:
                break
            batch += 1
        if print_frequency:
            print()
        lgx /= batch
        lgz /= batch
        ldx /= batch
        ldz /= batch
        lrx /= batch
        lrz /= batch
        return lgx, lgz, ldx, ldz, lrx, lrz
