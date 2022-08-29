import torch
import torch.nn as nn

from models.conv_block import ConvBlock


class Extractor(nn.Module):
    def __init__(self):
        super(Extractor, self).__init__()

        # group S
        self.conv_1 = ConvBlock(1, 16, p=1)

        # group C
        self.conv_c1 = ConvBlock(16, 16, p=3, d=3)
        self.conv_c2 = ConvBlock(32, 16, p=1)
        self.conv_c3 = ConvBlock(48, 16, p=1)

    def forward(self, x):
        # group S
        x = self.conv_1(x)

        # group C
        c1 = self.conv_c1(x)
        c1 = torch.cat([x, c1], 1)

        c2 = self.conv_c2(c1)
        c2 = torch.cat([c1, c2], 1)

        c3 = self.conv_c3(c2)
        c3 = torch.cat([c2, c3], 1)

        # pass
        return c3
