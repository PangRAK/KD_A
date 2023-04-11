import torch.nn as nn
from torch.nn import functional as F

import torch




class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.fc_se = nn.Sequential(
        #     nn.Linear(channel, channel // reduction, bias=False),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(channel // reduction, channel, bias=False),
        #     nn.Sigmoid()
        # )
        # block = 1
        # num_classes = 100 
        # self.fc = nn.Linear(64 * block, num_classes)
        self.prototypes = nn.Linear(64, 200, bias=False)
        self.prototypes2 = nn.Linear(64 * 3, 200, bias=False)

    def forward(self, x):
        # self.prototypes = nn.Linear(output_dim, nmb_prototypes, bias=False)
        # print(x.shape)
        if x.shape[1] == 64: 
            y = self.prototypes(x)
        else:
            y = self.prototypes2(x)

        return y
