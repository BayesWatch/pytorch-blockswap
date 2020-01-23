# network definition
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# wildcard import for legacy reasons
from .blocks import *

def parse_options(convtype, blocktype):
    # legacy cmdline argument parsing
    if isinstance(convtype, str):
        conv = conv_function(convtype)
    elif isinstance(convtype, list):
        conv = [conv_function(item) for item in convtype]
    else:
        raise NotImplementedError("conv must be a string or list")

    if isinstance(blocktype, str):
        block = block_function(blocktype)
    elif isinstance(blocktype, list):
        block = [block_function(item) for item in blocktype]
    else:
        raise NotImplementedError("conv must be a string or list")
    return conv, block

class WideResNet(nn.Module):
    def __init__(self, depth, widen_factor, conv=Conv, block=BasicBlock, num_classes=10, dropRate=0.0, s=1, convs=[], masked=False, darts=False):
        super(WideResNet, self).__init__()
        self.depth = depth
        self.widen_factor = widen_factor

        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        nChannels = [int(a) for a in nChannels]

        # for indexing conv list
        l = 0

        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) // 6

        assert n % s == 0, 'n mod s must be zero'

        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)

        nb_layers = n
        self.nb_layers = nb_layers
        if len(convs) == 0:
            convs = [conv for i in range(2 * nb_layers * s * 3)]

        self.convs = convs

        # 1st block
        self.block1 = torch.nn.ModuleList()
        for i in range(s):
            self.block1.append(NetworkBlock(nb_layers, nChannels[0] if i == 0 else nChannels[1],
                                            nChannels[1], 1, dropRate, convs[l:l+nb_layers], masked=masked, darts=darts))
        l += nb_layers * s

        # 2nd block
        self.block2 = torch.nn.ModuleList()
        for i in range(s):
            self.block2.append(NetworkBlock(nb_layers, nChannels[1] if i == 0 else nChannels[2],
                                            nChannels[2], 2 if i == 0 else 1, dropRate, convs[l:l+nb_layers], masked=masked, darts=darts))
        l += nb_layers * s

        # 3rd block
        self.block3 = torch.nn.ModuleList()
        for i in range(s):
            self.block3.append(NetworkBlock(nb_layers, nChannels[2] if i == 0 else nChannels[3],
                                            nChannels[3], 2 if i == 0 else 1, dropRate, convs[l:l+nb_layers], masked=masked, darts=darts))
        l += nb_layers * s


        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]
        self.input_spatial_dims = None

        # normal is better than uniform initialisation
        # this should really be in `self.reset_parameters`
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                try:
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                except AttributeError:
                    pass
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _get_convs(self):
        cs = []
        for c in self.convs:
            cs.append(c)
        return cs

    def forward(self, x):
        self.input_spatial_dims = x.size()
        activations = []
        out = self.conv1(x)
        # activations.append(out)

        for sub_block in self.block1:
            out = sub_block(out)
            activations.append(out)

        for sub_block in self.block2:
            out = sub_block(out)
            activations.append(out)

        for sub_block in self.block3:
            out = sub_block(out)
            activations.append(out)

        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out), activations


def test():
    net = WideResNet(40, 2, Conv, BasicBlock)
    x = torch.randn(1, 3, 32, 32)
    y, _ = net(Variable(x))
    print(y.size())


if __name__ == '__main__':
    test()
