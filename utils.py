import torch
import torch.nn.functional as F
from models import *
from collections import OrderedDict
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
import torch.nn as nn
import os
from copy import deepcopy
import glob
import numpy as np
#### DICTIONARIES FOR CONVERTING BETWEEN STRING AND CLASS

string_to_conv = {
    'Conv' : Conv,
    'DConvA2' : DConvA2,
    'DConvA4' : DConvA4,
    'DConvA8' :  DConvA8,
    'DConvA16' : DConvA16,
    'DConvG16' : DConvG16,
    'DConvG8' :  DConvG8,
    'DConvG4' :  DConvG4,
    'DConvG2' :  DConvG2,
    'DConv' :    DConv,
    'ConvB2' :   ConvB2,
    'ConvB4' :   ConvB4,
    'A2B2' :     A2B2,
    'A4B2' :     A4B2,
    'A8B2' :     A8B2,
    'A16B2' :    A16B2,
    'G16B2' :    G16B2,
    'G8B2' :     G8B2,
    'G8B4':      G8B4,
    'G4B2' :     G4B2,
    'G2B2' :     G2B2,
    'G4B4' :     G4B4,
    'G2B4' :     G2B4,
}

conv_to_string = {
    Conv : 'Conv',
    DConvA2 : 'DConvA2',
    DConvA4 : 'DConvA4',
    DConvA8 :  'DConvA8',
    DConvA16 : 'DConvA16',
    DConvG16 : 'DConvG16',
    DConvG8 :  'DConvG8',
    DConvG4 :  'DConvG4',
    DConvG2 :  'DConvG2',
    DConv :    'DConv',
    ConvB2 :   'ConvB2',
    ConvB4 :   'ConvB4',
    A2B2 :     'A2B2',
    A4B2 :     'A4B2',
    A8B2 :     'A8B2',
    A16B2 :    'A16B2',
    G16B2 :    'G16B2',
    G8B2 :     'G8B2',
    G8B4 :     'G8B4',
    G4B2 :     'G4B2',
    G2B2 :     'G2B2',
    G4B4 :     'G4B4',
    G2B4 :     'G2B4',
}

####

def distillation(y, teacher_scores, labels, T, alpha):
    return F.kl_div(F.log_softmax(y/T, dim=1), F.softmax(teacher_scores/T, dim=1)) * (T*T * 2. * alpha)\
           + F.cross_entropy(y, labels) * (1. - alpha)

def at(x):
    return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))

def at_loss(x, y):
    return (at(x) - at(y)).pow(2).mean()

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def expand_model(model, layers=[]):
    for layer in model.children():
         if len(list(layer.children())) > 0:
             expand_model(layer, layers)
         else:
             layers.append(layer)
    return layers

def get_flops(net, x):
    layers = expand_model(net, [])
    flops = 0
    for layer in layers:
        if isinstance(layer, nn.Conv2d):
            out_h = int((x.size()[2] + 2 * layer.padding[0] - layer.kernel_size[0]) /
                    layer.stride[0] + 1)
            out_w = int((x.size()[3] + 2 * layer.padding[1] - layer.kernel_size[1]) /
                        layer.stride[1] + 1)
            ops = layer.in_channels * layer.out_channels * layer.kernel_size[0] *  \
                    layer.kernel_size[1] * out_h * out_w / layer.groups

            flops += ops

    return flops

def get_no_params(net, verbose=False):

    params = net.state_dict()
    tot= 0
    conv_tot = 0
    for p in params:
        no = params[p].view(-1).__len__()
        if ('num_batches_tracked' not in p) and ('running' not in p) and ('mask' not in p):
            tot += no

        if 'conv' in p:
            conv_tot += no

    if verbose:
        print('Net has %d conv params' % conv_tot)
        print('Net has %d params in total' % tot)
    return tot

def get_cifar_loaders(cifar_loc, batch_size=128, workers=0, cutout=True, n_holes=1, length=16, pin_memory=False):
    num_classes = 10
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if cutout:
        transform_train.transforms.append(Cutout(n_holes=n_holes, length=length))

    transform_validate = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(root=cifar_loc,
                                        train=True, download=False, transform=transform_train)
    valset = torchvision.datasets.CIFAR10(root=cifar_loc,
                                       train=False, download=False, transform=transform_validate)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                              num_workers=workers, pin_memory=pin_memory)
    valloader = torch.utils.data.DataLoader(valset, batch_size=100, shuffle=False,
                                             num_workers=workers, pin_memory=pin_memory)
    return trainloader, valloader


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Pruner:
    def __init__(self, module_name='MaskBlock'):
        self.module_name = module_name
        self.masks = []
        self.prune_history= []

    def go_fish(self, model):
        self._get_fisher(model)
        tot_loss = self.fisher.div(1) + 1e6 * (1 - self.masks) #giga
        return tot_loss

    def _get_fisher(self, model):
        masks=[]
        fisher=[]
        flops=[]

        self._update_flops(model)

        for m in model.modules():
            if m._get_name() == 'MaskBlock' or m._get_name() == 'MaskBottleBlock':
                masks.append(m.mask.detach())
                fisher.append(m.running_fisher.detach())
                flops.append(m.flops_vector)

                m.reset_fisher()

        self.masks = self.concat(masks)
        self.fisher = self.concat(fisher)
        self.flops = self.concat(flops)

    def _get_masks(self, model):
        masks=[]

        for m in model.modules():
            if m._get_name() == 'MaskBlock' or m._get_name() == 'MaskBottleBlock':
                masks.append(m.mask.detach())

        self.masks = self.concat(masks)

    def _update_flops(self, model):
        for m in model.modules():
            if m._get_name() == 'MaskBlock' or m._get_name() == 'MaskBottleBlock':
                m.cost()

    @staticmethod
    def concat(input):
        return torch.cat([item for item in input])


class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img
