'''This is a rewriting of the native resnet definition that comes with Pytorch, to allow it to use our blocks and
 convolutions for imagenet experiments. Annoyingly, the pre-trained models don't use pre-activation blocks.'''

import torch
import torch.nn as nn
import math
import torchvision.models.resnet
import torch.utils.model_zoo as model_zoo
from .blocks import *
import pandas as pd

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50']  # , 'resnet50', 'resnet101','resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def get_channel_list(channels, n):
    new_chans = []

    new_chans.append([channels[0]])

    for c, n_ in zip(channels[1:], n):
        new_chans.append([c] * 3 * n_)

    new_chans = [item for sublist in new_chans for item in sublist]

    print(new_chans)
    return new_chans


class ResNet(nn.Module):

    def __init__(self, conv, block, layers, num_classes=1000, convs=[], masked=False,
                 nChannels=[64, 64, 128, 256, 512]):
        super(ResNet, self).__init__()

        dropRate = 0.0

        # Initial conv then maxpool

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.mpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # for indexing conv list
        l = 0
        i = 0

        if len(convs) == 0:
            convs = [conv for _ in range(sum(layers))]

        self.convs = convs

        # 1st block
        self.block1 = torch.nn.ModuleList()

        self.block1.append(NetworkBlock(layers[0], nChannels[0] if i == 0 else nChannels[1],
                                        nChannels[1], 1, dropRate, convs[l:l + layers[0]], masked=masked))
        l += layers[0]

        # 2nd block
        self.block2 = torch.nn.ModuleList()

        self.block2.append(NetworkBlock(layers[1], nChannels[1] if i == 0 else nChannels[2],
                                        nChannels[2], 2 if i == 0 else 1, dropRate, convs[l:l + layers[1]],
                                        masked=masked))
        l += layers[1]

        # 3rd block
        self.block3 = torch.nn.ModuleList()

        self.block3.append(NetworkBlock(layers[2], nChannels[2] if i == 0 else nChannels[3],
                                        nChannels[3], 2 if i == 0 else 1, dropRate, convs[l:l + layers[2]],
                                        masked=masked))
        l += layers[2]

        # 4rd block
        self.block4 = torch.nn.ModuleList()

        self.block4.append(NetworkBlock(layers[3], nChannels[3] if i == 0 else nChannels[4],
                                        nChannels[4], 2 if i == 0 else 1, dropRate, convs[l:l + layers[3]],
                                        masked=masked))
        l += layers[3]

        # Sanity check

        assert l == sum(layers)

        self.fc = nn.Linear(nChannels[4], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        self.input_spatial_dims = x.size()
        activations = []
        out = self.mpool(self.relu(self.bn1(self.conv1(x))))

        for sub_block in self.block1:
            out = sub_block(out)
            activations.append(out)

        for sub_block in self.block2:
            out = sub_block(out)
            activations.append(out)

        for sub_block in self.block3:
            out = sub_block(out)
            activations.append(out)

        for sub_block in self.block4:
            out = sub_block(out)
            activations.append(out)

        out = F.avg_pool2d(out, 7)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out, activations


def resnet18(pretrained=False, conv=Conv, block=BasicBlock):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(conv, block, [2, 2, 2, 2])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, conv=Conv, block=BasicBlock, masked=False, convs=[Conv,Conv]):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(conv, block, [3, 4, 6, 3],masked=masked,convs=convs)
    print(model)
    if pretrained:
        old_model = torchvision.models.resnet.resnet34(pretrained=False)
        old_model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))

        new_state_dict = model.state_dict()
        old_state_dict = old_model.state_dict()

        # This assumes the sequence of each module in the network is the same in both cases.
        # Ridiculously, batch norm params are stored in a different sequence in the downloaded state dict, so we have to
        # load the old model definition, load in its downloaded state dict to change the order back, then transfer this!

        old_model = torchvision.models.resnet.resnet34(pretrained=False)
        old_model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))

        old_names = [v for v in old_state_dict]
        new_names = [v for v in new_state_dict]

        for i, j in enumerate(old_names):
            new_state_dict[new_names[i]] = old_state_dict[j]

        model.load_state_dict(new_state_dict)

    return model


def resnet50(pretrained=False, conv=Conv, block=BottleBlock, masked=False):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(conv, block, [3, 4, 6, 3], nChannels = [64, 256, 512, 1024, 2048], masked=masked)
    print(model)
    if pretrained:
        old_model = torchvision.models.resnet.resnet50(pretrained=False)
        old_model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))

        new_state_dict = model.state_dict()
        old_state_dict = old_model.state_dict()

        # This assumes the sequence of each module in the network is the same in both cases.
        # Ridiculously, batch norm params are stored in a different sequence in the downloaded state dict, so we have to
        # load the old model definition, load in its downloaded state dict to change the order back, then transfer this!

        old_model = torchvision.models.resnet.resnet50(pretrained=False)
        old_model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))

        old_names = [v for v in old_state_dict]
        new_names = [v for v in new_state_dict]

        for i, j in enumerate(old_names):
            new_state_dict[new_names[i]] = old_state_dict[j]

        model.load_state_dict(new_state_dict)

    return model
