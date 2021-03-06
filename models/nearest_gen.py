#Simple StyleGAN vs simple DCGAN

import torch
import torch.nn as nn

from functools import reduce
from operator import mul

import models.utils as utils

class Generator(nn.Module):
    def __init__(self, mapping, shapes, z_shape, dropout):
        super(Generator, self).__init__()
        self.z_size = z_shape[0] # 512
        filters = 512

        self.init_shape = (filters, *shapes[0]) # 512x5x5
        # print("self.init_shape",self.init_shape)
        self.preprocess = nn.Sequential(
            nn.Linear(self.z_size, reduce(mul, self.init_shape), bias=False),
            nn.LeakyReLU(True))

        self.blocks = nn.ModuleList()
        in_ch = filters
        for s in shapes[1:-1]:
            out_ch = in_ch // 2
            block = nn.Sequential(
                utils.Resize(s),
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.LeakyReLU(True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(True),
            )
            in_ch = out_ch
            self.blocks.append(block)

        out_ch = len(mapping)
        self.output = nn.Sequential(
            utils.Resize(shapes[-1]),
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=True),
            nn.Softmax2d()
        )

    def forward(self, z):
        x = self.preprocess(z)
        d, h, w = self.init_shape
        x = x.view(-1, d, h, w)
        for b in self.blocks:
            x = b(x)
        x = self.output(x)
        return x
