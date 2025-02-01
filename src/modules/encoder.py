"""Adapted from taming transformers: https://github.com/CompVis/taming-transformers"""

import sys
import torch

import torch.nn as nn

from pyprojroot import here as project_root

sys.path.insert(0, str(project_root()))

from src.modules.blocks import ResnetBlock, AttnBlock, Downsample, nonlinearity, Normalize

class Encoder(nn.Module):
  def __init__(self, *, ch, in_channels, z_channels, double_z=True):
    super().__init__()
    self.ch = ch
    self.in_channels = in_channels

    # downsampling
    self.conv_in = torch.nn.Conv2d(in_channels,
                                   self.ch,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)

    # end
    self.norm_out = Normalize(self.ch)
    self.conv_out = torch.nn.Conv2d(self.ch,
                                    2 * z_channels if double_z else z_channels,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1)

  def forward(self, x):
    # assert x.shape[2] == x.shape[3] == self.resolution, "{}, {}, {}".format(x.shape[2], x.shape[3], self.resolution)
    # (batch size, 3, width, height)

    # downsampling
    h = self.conv_in(x)

    # end
    h = self.norm_out(h)
    h = nonlinearity(h)
    h = self.conv_out(h)
    return h


