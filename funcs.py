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
import pandas as pd

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
    'G4B2' :     G4B2,
    'G2B2' :     G2B2
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
    G4B2 :     'G4B2',
    G2B2 :     'G2B2'
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

def get_imagenet_loaders(imagenet_loc, batch_size=128, workers=12):
    num_classes = 1000
    traindir = os.path.join(imagenet_loc, 'train')
    valdir = os.path.join(imagenet_loc, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    transform_validate = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    trainset = torchvision.datasets.ImageFolder(traindir, transform_train)
    valset = torchvision.datasets.ImageFolder(valdir, transform_validate)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                              num_workers=workers,
                                              pin_memory = True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=100, shuffle=False,
                                             num_workers=workers,
                                             pin_memory=True)
    return trainloader, valloader

def get_cifar_loaders(cifar_loc, batch_size=128, workers=0):
    num_classes = 10
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_validate = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(root=cifar_loc,
                                        train=True, download=False, transform=transform_train)
    valset = torchvision.datasets.CIFAR10(root=cifar_loc,
                                       train=False, download=False, transform=transform_validate)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                              num_workers=workers)
    valloader = torch.utils.data.DataLoader(valset, batch_size=100, shuffle=False,
                                             num_workers=workers)
    return trainloader, valloader

def get_cifar100_loaders(cifar_loc='../cifar100', batch_size=128, workers=0):
    num_classes = 100
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023)),
    ])
    transform_validate = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023)),
    ])
    trainset = torchvision.datasets.CIFAR100(root=cifar_loc,
                                            train=True, download=False, transform=transform_train)
    valset = torchvision.datasets.CIFAR100(root=cifar_loc,
                                           train=False, download=False, transform=transform_validate)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                              num_workers=workers)
    valloader = torch.utils.data.DataLoader(valset, batch_size=100, shuffle=False,
                                             num_workers=workers)
    return trainloader, valloader

def get_imagenet_loaders(imagenet_loc, batch_size=128, workers=12):
    num_classes = 1000
    traindir = os.path.join(imagenet_loc, 'train')
    valdir = os.path.join(imagenet_loc, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    transform_validate = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    trainset = torchvision.datasets.ImageFolder(traindir, transform_train)
    valset = torchvision.datasets.ImageFolder(valdir, transform_validate)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                              num_workers=workers,
                                              pin_memory = True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=100, shuffle=False,
                                             num_workers=workers,
                                             pin_memory=True)
    return trainloader, valloader

def cifar_random_search(save_file):
    df = []
    save_counter = 1
    while(True):
        random_blocks = np.random.choice(list(conv_to_string.keys()), 18)
        net = WideResNet(40,2,Conv,BasicBlock,convs=random_blocks)
        params = get_no_params(net, verbose=False)
        df.append([random_blocks, params])
        df = pd.DataFrame(df, columns=['convs','params'])
        if save_counter > 1:
            df2 = pd.read_csv(savefile)
            df  = pd.concat([df,df2], sort=True)
        df.to_csv(savefile)
        df = []

def imagenet_random_search():
    df = []
    save_counter = 0
    while(True):
        random_blocks =  np.random.choice(list(conv_to_string.keys()), 16)
        net = ResNet(Conv, Block, [3, 4, 6, 3], convs=random_blocks)
        params = get_no_params(net, verbose=False)
        df.append([random_blocks, params])
        if save_counter % 10000 == 0:
            print('saving df ', save_counter)
            df = pd.DataFrame(df, columns=['convs','params'])
            df.to_csv(str('imagenet_models' + str(save_counter) + '.csv'))
            df = []
        save_counter = save_counter + 1

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

def get_convs(model):
    blocks    = []
    for m in model.modules():
        if 'Mask' in m._get_name():
            if isinstance(m.conv1, Conv):
                blocks.append('S')
            else:
                blocks.append('G')
    return blocks

def concat_archs():
    archs = glob.glob("arch/*.csv")

    # get all of our archs in a row
    master = pd.concat([pd.read_csv(arch) for arch in archs])
    master.to_csv('arch/master_arch.csv')

def one_shot_fisher(net, trainloader, n_steps=1, cuda=True):
    params = get_no_params(net, verbose=False)
    criterion = nn.CrossEntropyLoss()
    # switch to train mode
    net.train()
    optimizer  = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005)
    dataiter = iter(trainloader)
    pruner = Pruner()
    data = torch.rand(net.input_spatial_dims)
    if cuda:
        data = data.cuda()
    net(data)
    pruner._get_masks(net)

    NO_STEPS = n_steps # single minibatch
    for i in range(0, NO_STEPS):
        try:
            input, target = dataiter.next()
        except StopIteration:
            dataiter = iter(trainloader)
            input, target = dataiter.next()

        if cuda:
            input, target = input.cuda(), target.cuda()

        # compute output
        output, act = net(input)

        loss = criterion(output, target)
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    fisher_inf = pruner.go_fish(net)
    features = fisher_inf.size()
    fish_market = dict()
    running_index = 0

    block_count = 0
    data = torch.rand(1,16,32,32)
    if cuda:
        data = data.cuda()
    # partition fisher_inf into blocks blocks blocks
    for m in net.modules():
        if m._get_name() == 'MaskBlock' or m._get_name() == 'MaskBottleBlock':
            mask_indices = range(running_index, running_index + len(m.mask))
            fishies = [fisher_inf[j] for j in mask_indices]
            running_index += len(m.mask)

            fish = sum(fishies)
            data  = m(data)
            flops = m.flops * data.size()[2]

            fish = fish / flops

            fish_market[block_count] = fish
            block_count +=1

    return params, fish_market
