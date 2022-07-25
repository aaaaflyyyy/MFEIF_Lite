import torch.nn as nn

from models.conv_block import ConvBlock


class Extractor(nn.Module):
    def __init__(self):
        super(Extractor, self).__init__()

        # group S
        self.conv_1 = ConvBlock(1, 8, p=1)

        # group A
        self.conv_a1 = ConvBlock(8, 16, p=1)
        self.conv_a2 = ConvBlock(16, 24, p=1)
        self.conv_a3 = ConvBlock(24, 32, p=1)

        # group B
        self.conv_b1 = ConvBlock(8, 16, k=5, p=2)
        self.conv_b2 = ConvBlock(16, 24, p=1)
        self.conv_b3 = ConvBlock(24, 32, p=1)

        # group C
        self.conv_c1 = ConvBlock(8, 16, k=7, p=3)
        self.conv_c2 = ConvBlock(16, 24, p=1)
        self.conv_c3 = ConvBlock(24, 32, p=1)

    def forward(self, x):
        # group S
        x = self.conv_1(x)

        # group A
        a1 = self.conv_a1(x)
        # print('a1:', a1.shape)
        a2 = self.conv_a2(a1)
        # print('a2:', a2.shape)
        a3 = self.conv_a3(a2)
        # print('a3:', a3.shape)

        # group B
        b1 = self.conv_b1(x)
        # print('b1:', b1.shape)
        b2 = self.conv_b2(b1)
        # print('b2:', b2.shape)
        b3 = self.conv_b3(b2)
        # print('b3:', b3.shape)

        # group C
        c1 = self.conv_c1(x)
        # print('c1:', c1.shape)
        c2 = self.conv_c2(c1)
        # print('c2:', c2.shape)
        c3 = self.conv_c3(c2)
        # print('c3:', c3.shape)
        # assert False
        # final feathers
        w_tp = [0.1, 0.1, 1]
        f = w_tp[0] * a3 + w_tp[1] * b3 + w_tp[2] * c3

        # transform block
        b_2 = a1 + b1 + c1
        b_1 = a2 + b2 + c2

        # pass
        return f, b_1, b_2
