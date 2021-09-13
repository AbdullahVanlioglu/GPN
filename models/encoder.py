#Simple StyleGAN vs simple DCGAN

import torch
import torch.nn as nn

from functools import reduce
from operator import mul

import models.utils as utils

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        network = nn.Sequential(
            nn.Conv2d(2, 8, 6),
            nn.BatchNorm2d(8),
            nn.ReLU(True),

            nn.Conv2d(8, 16, 3),
            nn.BatchNorm2d(16),
            nn.ReLU(True),

            nn.Conv2d(16, 32, 3),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            )
            
    def forward(self, x):
        print("Encoder in x", x.shape)
        x = self.network(x)
        print("Encoder out x", x.shape)
        return x
