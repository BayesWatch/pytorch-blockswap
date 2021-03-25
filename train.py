from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import random
import argparse
from tqdm import tqdm
import time
from utils import *

import pandas as pd

os.mkdir("checkpoints/") if not os.path.isdir("checkpoints/") else None

parser = argparse.ArgumentParser(description="Student/teacher training")
parser.add_argument(
    "mode",
    choices=["student", "teacher"],
    type=str,
    help="Learn a teacher or a student",
)
parser.add_argument(
    "--data_loc",
    default="/datasets/cifar",
    type=str,
    help="folder containing cifar train and val folders",
)
parser.add_argument(
    "--workers",
    default=0,
    type=int,
    help="No. of data loading workers. Make this high for imagenet",
)
parser.add_argument("--print_freq", default=10, type=int)
parser.add_argument("--GPU", default="0", type=str, help="GPU to use")
parser.add_argument(
    "--student_checkpoint",
    "-s",
    default="wrn_40_2_student_KT",
    type=str,
    help="checkpoint to save/load student",
)
parser.add_argument(
    "--teacher_checkpoint",
    "-t",
    default="wrn_40_2_T",
    type=str,
    help="checkpoint to load in teacher",
)

# network stuff
parser.add_argument("--wrn_depth", default=40, type=int, help="depth for WRN")
parser.add_argument("--wrn_width", default=2, type=float, help="width for WRN")
parser.add_argument("--diff_shape", action="store_true")
parser.add_argument(
    "--stu_depth", default=40, type=int, help="student depth if different from teacher"
)
parser.add_argument(
    "--stu_width", default=2, type=float, help="student width if different from teacher"
)
parser.add_argument("--from_genotype", default="", type=str, help="Load a template")

# learning stuff
parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
parser.add_argument(
    "--AT_split", default=1, type=int, help="group splitting for AT loss"
)
parser.add_argument("--aux_loss", default="AT", type=str, help="AT or SE loss")
parser.add_argument("--beta", default=1e3, type=float, help="beta for AT")
parser.add_argument(
    "--epochs", default=200, type=int, metavar="N", help="number of total epochs to run"
)
parser.add_argument("--batch_size", default=128, type=int, help="minibatch size")
parser.add_argument("--weightDecay", default=0.0005, type=float)
args = parser.parse_args()

print(args)
os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_teacher(net):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    net.train()

    end = time.time()

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)

        inputs, targets = inputs.to(device), targets.to(device)
        outputs, _ = net(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        err1 = 100.0 - prec1
        err5 = 100.0 - prec5
        losses.update(loss.item(), inputs.size(0))
        top1.update(err1[0], inputs.size(0))
        top5.update(err5[0], inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % args.print_freq == 0:
            print(
                "Epoch: [{0}][{1}/{2}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Error@1 {top1.val:.3f} ({top1.avg:.3f})\t"
                "Error@5 {top5.val:.3f} ({top5.avg:.3f})".format(
                    epoch,
                    batch_idx,
                    len(trainloader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    top1=top1,
                    top5=top5,
                )
            )

    train_losses.append(losses.avg)
    train_errors.append(top1.avg)


def train_student(net, teach):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    net.train()
    teach.eval()

    end = time.time()

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)

        inputs, targets = inputs.to(device), targets.to(device)
        outputs_student, ints_student = net(inputs)
        outputs_teacher, ints_teacher = teach(inputs)

        loss = criterion(outputs_student, targets)

        # Add an attention tranfer loss for each intermediate. Let's assume the default is three (as in the original
        # paper) and adjust the beta term accordingly.
        adjusted_beta = (args.beta * 3) / len(ints_student)
        for i in range(len(ints_student)):
            loss += adjusted_beta * aux_loss(ints_student[i], ints_teacher[i])

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs_student.data, targets.data, topk=(1, 5))
        err1 = 100.0 - prec1
        err5 = 100.0 - prec5
        losses.update(loss.item(), inputs.size(0))
        top1.update(err1[0], inputs.size(0))
        top5.update(err5[0], inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % args.print_freq == 0:
            print(
                "Epoch: [{0}][{1}/{2}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Error@1 {top1.val:.3f} ({top1.avg:.3f})\t"
                "Error@5 {top5.val:.3f} ({top5.avg:.3f})".format(
                    epoch,
                    batch_idx,
                    len(trainloader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    top1=top1,
                    top5=top5,
                )
            )

    train_losses.append(losses.avg)
    train_errors.append(top1.avg)


def validate(net, checkpoint=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    net.eval()
    end = time.time()

    for batch_idx, (inputs, targets) in enumerate(valloader):
        # measure data loading time
        data_time.update(time.time() - end)

        inputs, targets = inputs.to(device), targets.to(device)
        outputs, _ = net(inputs)
        if isinstance(outputs, tuple):
            outputs = outputs[0]

        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        err1 = 100.0 - prec1
        err5 = 100.0 - prec5

        losses.update(loss.item(), inputs.size(0))
        top1.update(err1[0], inputs.size(0))
        top5.update(err5[0], inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % args.print_freq == 0:
            print(
                "Epoch: [{0}][{1}/{2}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Error@1 {top1.val:.3f} ({top1.avg:.3f})\t"
                "Error@5 {top5.val:.3f} ({top5.avg:.3f})".format(
                    epoch,
                    batch_idx,
                    len(valloader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    top1=top1,
                    top5=top5,
                )
            )

    val_losses.append(losses.avg)
    val_errors.append(top1.avg)

    if checkpoint:
        state = {
            "net": net.state_dict(),
            "epoch": epoch,
            "width": args.wrn_width,
            "depth": args.wrn_depth,
            "convs": net.convs,
            "train_losses": train_losses,
            "train_errors": train_errors,
            "val_losses": val_losses,
            "val_errors": val_errors,
        }
        torch.save(state, "checkpoints/%s.t7" % checkpoint)


if __name__ == "__main__":

    aux_loss = at_loss

    print(vars(args))
    val_losses = []
    train_losses = []
    val_errors = []
    train_errors = []

    best_acc = 0
    start_epoch = 0

    # Data and loaders
    print("==> Preparing data..")
    num_classes = 10
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    transform_train.transforms.append(Cutout(n_holes=1, length=16))

    transform_validate = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    trainset = torchvision.datasets.CIFAR10(
        root=args.data_loc, train=True, download=False, transform=transform_train
    )
    valset = torchvision.datasets.CIFAR10(
        root=args.data_loc, train=False, download=False, transform=transform_validate
    )

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=False,
    )
    valloader = torch.utils.data.DataLoader(
        valset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
    )
    criterion = nn.CrossEntropyLoss()

    def load_network(loc):
        net_checkpoint = torch.load(loc)
        start_epoch = net_checkpoint["epoch"]
        net = WideResNet(
            args.wrn_depth, args.wrn_width, num_classes=num_classes, dropRate=0
        ).to(device)
        net.load_state_dict(net_checkpoint["net"])
        return net, start_epoch

    # if first training the teacher network alone
    if args.mode == "teacher":

        print("Mode Teacher: Making a teacher network from scratch and training it...")
        teach = WideResNet(
            args.wrn_depth, args.wrn_width, num_classes=num_classes, dropRate=0
        ).to(device)

        get_no_params(teach)
        optimizer = optim.SGD(
            teach.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weightDecay
        )

        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, args.epochs, eta_min=1e-10
        )

        for epoch in tqdm(range(start_epoch, args.epochs)):
            print("Teacher Epoch %d:" % epoch)
            print("Learning rate is %s" % [v["lr"] for v in optimizer.param_groups][0])
            train_teacher(teach)
            scheduler.step()
            validate(teach, args.teacher_checkpoint)

    #  if you already have the teacher and you want to train a student
    elif args.mode == "student":
        print(
            "Mode Student: First, load a teacher network and convert for (optional) attention transfer"
        )
        teach, _ = load_network("checkpoints/%s.t7" % args.teacher_checkpoint)
        teach_optim = optim.SGD(
            teach.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weightDecay
        )

        convs = []

        candidate = pd.read_csv(args.from_genotype)["convs"]
        candidate = candidate[0][2:-2].split("', '")
        student = WideResNet(
            args.wrn_depth,
            args.wrn_width,
            masked=True,
            convs=[string_to_conv[c] for c in candidate],
        ).to(device)

        # Very important to explicitly say we require no gradients for the teacher network
        for param in teach.parameters():
            param.requires_grad = False
        epoch = 0
        validate(teach)
        val_losses, val_errors = (
            [],
            [],
        )  # or we'd save the teacher's error as the first entry

        print(args.student_checkpoint)
        optimizer = optim.SGD(
            student.parameters(),
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.weightDecay,
        )
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, args.epochs, eta_min=1e-10
        )

        for epoch in tqdm(range(start_epoch, args.epochs)):
            print("Student Epoch %d:" % epoch)
            print("Learning rate is %s" % [v["lr"] for v in optimizer.param_groups][0])
            train_student(student, teach)
            validate(student, args.student_checkpoint)
            scheduler.step()
