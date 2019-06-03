import torch
from funcs import *
from models import *
#import matplotlib.pyplot as plt
import pandas as pd
#import numpy as np
from tqdm import tqdm
from statistics import mean, median
import argparse


parser = argparse.ArgumentParser(description='Student/teacher training')
parser.add_argument('dataset',         type=str, choices=['cifar10', 'cifar100', 'imagenet'], help='Choose between Cifar10/100/imagenet.')
parser.add_argument('--data_loc',      default='/disk/scratch/imagenet',type=str, help='folder containing dataset train and val folders')
parser.add_argument('--batch_size',    default=128, type=int)
parser.add_argument('--workers',       default=0, type=int)
parser.add_argument('--param_goal',    default=800000, type=int)
parser.add_argument('--top_n',         default=1, type=int)
parser.add_argument('--continue_from', default=0, type=int)
parser.add_argument('--generate_random', action='store_true')
parser.add_argument('--save_file',     default='archs/master_arch.csv')
args = parser.parse_args()


def update_model(model, convs):
    convs = str(convs)[1:-1].rstrip().replace(',', '')
    cs = convs.replace('\n', ' ').split()
    prefix = len("'models.blocks.")
    suffix = len("'>")

    r = [w for w in cs if '<class' not in w]
    blocks = [c[prefix:-suffix] for c in r]
    convs  = [string_to_conv[c] for c in blocks]

    for i, c in enumerate(convs):
        model = update_block(i, model, c)

    return model, blocks

def generate_random():
    cifar_random_search(args.save_file)


def rank_at_param_goal(param_goal):
    master = pd.read_csv(args.save_file)

    var = 0.025 * param_goal
    lower_bound = param_goal - var
    upper_bound = param_goal + var
    models = master[master['params'].between(lower_bound, upper_bound, inclusive=True)]

    if args.dataset == 'cifar10':
        data = torch.rand(1,3,32,32).cuda()
        student = WideResNet(40, 2, Conv, BasicBlock, num_classes=10, dropRate=0,masked=True).cuda()
        student(data)
        train, val = get_cifar_loaders(args.data_loc)
    elif args.dataset == 'cifar100':
        data = torch.rand(1,3,32,32).cuda()
        student = WideResNet(40, 2, Conv, BasicBlock, num_classes=100, dropRate=0,masked=True).cuda()
        student(data)
        train, val = get_cifar100_loaders(args.data_loc)
    elif args.dataset == 'imagenet':
        data = torch.rand(1,3,224,224).cuda()
        student = ResNet(Conv, BasicBlock, [3,4,6,3], masked=True).cuda()
        student(data)
        train, val = get_imagenet_loaders(args.data_loc)

    df = []

    for i in tqdm(range(args.continue_from, args.continue_from+100)):
        row = models.iloc[i]
        cs = row['convs'][1:-1].rstrip()
        cs = cs.replace('\n', ' ').split()
        prefix = len("'models.blocks.")
        suffix = len("'>")

        r = [w for w in cs if '<class' not in w]
        convs = [c[prefix:-suffix] for c in r]
        convs = [string_to_conv[c] for c in convs]

        for j, c in enumerate(convs):
            student = update_block(j, student, c)

        params, fish = one_shot_fisher(student, train, 10)
        fish = float(sum(fish.values()))

        df.append([i, convs, row['params'], fish])

    df = pd.DataFrame(df, columns=['index','convs','params','fisher'])

    # concat any old searches
    if args.continue_from > 0:
        og_df = pd.read_csv('results/fish_dict_at_' + str(param_goal) + '.df')
        df = pd.concat([og_df, df],sort=False)

    df.to_csv('results/fish_dict_at_' + str(param_goal) + '.df')

    df = df.sort_values(by=['fisher'],ascending=False)
    for i in range(args.top_n):
        row   = df.iloc[i]
        index = row['index']
        convs = row['convs']
        fish  = row['fisher']

        geno = pd.DataFrame({'index' : [index], 'convs': [convs], 'fisher': [fish]})
        geno.to_csv('genotypes/' + str(param_goal) + '_' + str(index)+'.csv')

if args.generate_random:
    generate_random()
else:
    rank_at_param_goal(args.param_goal)
