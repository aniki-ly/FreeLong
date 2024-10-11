# adopted from
# https://github.com/openai/guided-diffusion/blob/0ba878e517b276c45d1195eb29f6f5f72659a05b/guided_diffusion/nn.py
# and
# https://github.com/TianxingWu/FreeInit/blob/master/freeinit_utils.py
# and
# https://github.com/AILab-CVC/FreeNoise
# thanks!


import os
import math
import torch

import numpy as np
import torch.nn as nn

from einops import rearrange, repeat
import  torch.fft as fft


#################################################################################
#                                  Long Video Utils                                   #
#################################################################################

def get_views(video_length, window_size=16, stride=4):
    num_blocks_time = (video_length - window_size) // stride + 1
    views = []
    for i in range(num_blocks_time):
        t_start = int(i * stride)
        t_end = t_start + window_size
        views.append((t_start,t_end))
    return views


def generate_weight_sequence(n):
    if n % 2 == 0:
        max_weight = n // 2
        weight_sequence = list(range(1, max_weight + 1, 1)) + list(range(max_weight, 0, -1))
    else:
        max_weight = (n + 1) // 2
        weight_sequence = list(range(1, max_weight, 1)) + [max_weight] + list(range(max_weight - 1, 0, -1))
    return weight_sequence


def gaussian_low_pass_filter(shape, d_s=0.25, d_t=0.25):
    """
    Compute the Gaussian low pass filter mask using vectorized operations, ensuring exact
    calculation to match the old loop-based implementation.

    Args:
        shape: shape of the filter (volume)
        d_s: normalized stop frequency for spatial dimensions (0.0-1.0)
        d_t: normalized stop frequency for temporal dimension (0.0-1.0)
    """
    T, H, W = shape[-3], shape[-2], shape[-1]
    if d_s == 0 or d_t == 0:
        return torch.zeros(shape)

    # Create normalized coordinate grids for T, H, W
    # Generate indices as in the old loop-based method
    t = torch.arange(T).float() * 2 / T - 1
    h = torch.arange(H).float() * 2 / H - 1
    w = torch.arange(W).float() * 2 / W - 1
    
    # Use meshgrid to create 3D grid of coordinates
    grid_t, grid_h, grid_w = torch.meshgrid(t, h, w, indexing='ij')

    # Compute squared distance from the center, adjusted for the frequency cut-offs
    d_square = ((grid_t * (1 / d_t)).pow(2) + (grid_h * (1 / d_s)).pow(2) + (grid_w * (1 / d_s)).pow(2))

    # Compute the Gaussian mask
    mask = torch.exp(-0.5 * d_square)

    # Adjust shape for multiple channels if necessary
    if len(shape) > 3:
        T, C = shape[0], shape[1]
        mask = mask.unsqueeze(0).unsqueeze(0).repeat(T, C, 1, 1, 1)

    return mask


def freq_mix_3d(x, noise, LPF):
    """
    Noise reinitialization.

    Args:
        x: diffused latent
        noise: randomly sampled noise
        LPF: low pass filter
    """
    # FFT
    x_freq = fft.fftn(x, dim=(-3, -2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-3, -2, -1))
    noise_freq = fft.fftn(noise, dim=(-3, -2, -1))
    noise_freq = fft.fftshift(noise_freq, dim=(-3, -2, -1))

    # frequency mix
    HPF = 1 - LPF
    x_freq_low = x_freq * LPF
    noise_freq_high = noise_freq * HPF
    x_freq_mixed = x_freq_low + noise_freq_high # mix in freq domain

    # IFFT
    x_freq_mixed = fft.ifftshift(x_freq_mixed, dim=(-3, -2, -1))
    x_mixed = fft.ifftn(x_freq_mixed, dim=(-3, -2, -1)).real

    return x_mixed