import time
from utils import *
from models import *
import pandas as pd
import numpy  as np
import argparse
import random

THE_START = time.time()

parser = argparse.ArgumentParser(description='BlockSwap search')
parser.add_argument('--data_loc', default='/datasets/cifar', type=str,
                    help='folder containing dataset train and val folders')
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--GPU', default='0', type=str)
parser.add_argument('--param_budget', default=800000, type=int)
parser.add_argument('--minibatches', default=1, type=int)
parser.add_argument('--samples', default=1000, type=int)
parser.add_argument('--save_file', default='800K')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def generate_archs():
    # Function for generating architectures

    var = 0.025 * args.param_budget
    lower_bound = args.param_budget - var
    upper_bound = args.param_budget + var

    model = WideResNet(40, 2)
    sizes = []

    # fast parameter count calculation
    for layer in [model.block1, model.block2, model.block3]:
        for block in layer:
            for sub_block in block.layer:
                sizes.append((sub_block.in_channels, sub_block.out_channels))

    p_dict = {}

    opts = list(conv_to_string.keys())  # conv_to_string lives in utils.py and is a dictionary of available convs
    for o in opts:
        params = []
        for (in_, out_) in sizes:
            block = NetworkBlock(1, in_, out_, 1, convs=[o, o])
            params.append(get_no_params(block))
        p_dict[o] = params

    def get_param_tot(convs):
        conv_tot = 0
        for i, c in enumerate(convs):
            conv_tot += p_dict[c][i]

        return conv_tot + 1978  # 1978 is the number of the parameters in uncounted layers at the start/end of the net

    print('Generating random architectures...')
    candidates = []
    while len(candidates) < args.samples:
        convs = np.random.choice(opts, 18)
        param = get_param_tot(convs)
        if param > lower_bound and param < upper_bound:
            candidates.append([conv for conv in convs])

    return candidates


def one_shot_fisher(net, trainloader, n_steps=1):
    # Function for scoring

    params = get_no_params(net, verbose=False)
    criterion = nn.CrossEntropyLoss()
    net.train()
    pruner = Pruner()  # Pruner calculates fisher info
    pruner._get_masks(net)

    NO_STEPS = n_steps  # single minibatch with n_steps = 1
    for i in range(0, NO_STEPS):
        input, target = next(iter(trainloader))

        input, target = input.to(device), target.to(device)
        # compute output
        output, _ = net(input)

        loss = criterion(output, target)
        loss.backward()

    fisher_inf = pruner.go_fish(net)
    fish_market = dict()
    running_index = 0

    block_count = 0

    # partition fisher_inf into blocks blocks blocks
    for m in net.modules():
        if m._get_name() == 'MaskBlock' or m._get_name() == 'MaskBottleBlock':
            mask_indices = range(running_index, running_index + len(m.mask))
            fishies = [abs(fisher_inf[j]) for j in mask_indices]
            running_index += len(m.mask)
            fish = sum(fishies)
            fish_market[block_count] = fish
            block_count += 1

    return params, fish_market


def rank_arch(candidate):
    start = time.time()
    student = WideResNet(40, 2, masked=True, convs=candidate).to(device)
    student.eval()
    create_time.update(time.time() - start)

    mid = time.time()
    data = torch.rand(1, 3, 32, 32).to(device)
    with torch.no_grad():  # Pass a single example through to create masks
        student(data)
    prep_time.update(time.time() - mid)
    mid = time.time()
    params, fish = one_shot_fisher(student, train_data, 1)
    fish = float(sum(fish.values()))
    fish_time.update(time.time() - mid)

    # .acts aren't being removed for some reason, so need to do this manually. This could be done less hackily.
    for m in student.modules():
        try:
            del m.act
        except:
            pass
    total_time.update(time.time() - start)

    return fish


candidates = generate_archs()

scores = []
create_time = AverageMeter()
prep_time = AverageMeter()
fish_time = AverageMeter()
total_time = AverageMeter()

train_data, _ = get_cifar_loaders(args.data_loc, workers=0, pin_memory=False)

for cand_id, candidate in enumerate(candidates):

    scores = np.append(scores, rank_arch(candidate))
    if cand_id % 10 == 0:
        print('Fish: [{0}/{1}]\t'
              'CreateTime {create_time.val:.3f} ({create_time.avg:.3f})\t'
              'PrepTime {prep_time.val:.3f} ({prep_time.avg:.3f})\t'
              'FishTime {fish_time.val:.3f} ({fish_time.avg:.3f})\t'
              'TotalTime {total_time.val:.3f} ({total_time.avg:.3f})'
              .format(cand_id, args.samples, create_time=create_time, prep_time=prep_time, fish_time=fish_time,
                      total_time=total_time))

best = np.argmax(scores)
print(scores)
print(best)

best_geno = geno = [conv_to_string[conv] for conv in candidates[best]]
best = pd.DataFrame({'index': [0], 'convs': [best_geno], 'fisher': [0.]})
best.to_csv('genotypes/' + args.save_file + str('best') + '.csv')

THE_END = time.time()

# Finally, write out
df = pd.DataFrame(columns=['convs', 'fisher'])
for i, geno in enumerate(candidates):
    geno = [conv_to_string[conv] for conv in geno]
    df.loc[i] = [geno] + [scores[i]]
df.to_csv('genotypes/' + args.save_file + str('all') + '.csv')
print('Done.')
print(THE_END - THE_START)
