import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import reduce
from operator import mul

import models.utils as utils

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.network = nn.Sequential(
            nn.Conv2d(2, 4, 6, stride = 2),
            nn.BatchNorm2d(4),
            nn.ReLU(True),

            nn.Conv2d(4, 8, 6, stride = 2),
            nn.BatchNorm2d(8),
            nn.ReLU(True),

            nn.Conv2d(8, 16, 6),
            nn.BatchNorm2d(16),
            nn.ReLU(True),

            nn.Conv2d(16, 32, 6),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.Conv2d(32, 32, 6),
            nn.BatchNorm2d(32),
            nn.ReLU(True)
            )

        self.feature = nn.Sequential(
        nn.Linear(128, 256),
        nn.ReLU(True)
        )
        
        self.final_layer = nn.Sequential(
        nn.Linear(256,8),
        nn.Softmax()
        )

        self.optimizer = torch.optim.Adam(list(self.network.parameters())+
        list(self.feature.parameters())+
        list(self.final_layer.parameters()), lr = 1e-3)
        self.loss = F.mse_loss

    def forward(self, x):
        x = torch.Tensor(x).unsqueeze(0)
        x = torch.flatten(self.network(x))
        high_feature = self.feature(x)
        output = self.final_layer(high_feature)
        return high_feature, output