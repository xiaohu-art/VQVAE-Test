"""Adapted from taming transformers: https://github.com/CompVis/taming-transformers"""
import torch
import sys

import numpy as np
import torch.nn as nn

from pyprojroot import here as project_root

sys.path.insert(0, str(project_root()))

from src.modules.blocks import ResnetBlock, AttnBlock, nonlinearity, Normalize, Upsample


class Decoder(nn.Module):
  def __init__(self, *, ch, out_channels, z_channels, double_z=False):
    super().__init__()
    self.ch = ch

    # z to block_in
    self.conv_in = torch.nn.Conv2d(z_channels,
                                   z_channels,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)

    # end
    self.norm_out = Normalize(z_channels)
    self.conv_out = torch.nn.Conv2d(z_channels,
                                    out_channels,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1)

  def forward(self, z):
    # assert z.shape[1:] == self.z_shape[1:]

    # z to block_in
    h = self.conv_in(z)

    h = self.norm_out(h)
    h = nonlinearity(h)
    h = self.conv_out(h)
    return h
