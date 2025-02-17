import sys
import time
import torch
import utils
from .impl import iterative_unlearn

sys.path.append(".")
from imagenet import get_x_y_from_data_dict

import argparse
import os
import pdb
import pickle
import random
import shutil
import time
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

import arg_parser
from trainer import train, validate
from utils import *
from utils import NormalizeByChannelMeanStd

best_sa = 0


@iterative_unlearn
def ideal(data_loaders, model, criterion, optimizer, epoch, args):
    forget_loader = data_loaders["forget"]

    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu))
        device = torch.device(f"cuda:{int(args.gpu)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(device)
    initalization = None

    os.makedirs(args.save_dir, exist_ok=True)
    if args.seed:
        setup_seed(args.seed)

    # prepare dataset
    if args.dataset == "cifar10" or args.dataset == "cifar100":
        model, _, val_loader, test_loader, _ = setup_model_dataset(args)
    else:
        model, _, val_loader, test_loader = setup_model_dataset(args)
    train_loader = data_loaders["retain"]
    retain_loader_iter = enumerate(train_loader)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    decreasing_lr = list(map(int, args.decreasing_lr.split(",")))

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=decreasing_lr, gamma=0.1
    )  # 0.1 is fixed

    all_result = {}
    all_result["train_ta"] = []
    all_result["test_ta"] = []
    all_result["val_ta"] = []

    start_epoch = 0
    state = 0

    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        print(optimizer.state_dict()["param_groups"][0]["lr"])
        acc = train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        tacc = validate(val_loader, model, criterion, args)
        # evaluate on test set
        test_tacc = validate(test_loader, model, criterion, args)

        scheduler.step()

        all_result["train_ta"].append(acc)
        all_result["val_ta"].append(tacc)
        all_result["test_ta"].append(test_tacc)

        print("one epoch duration:{}".format(time.time() - start_time))

    print("Performance on the test data set")
    test_tacc = validate(test_loader, model, criterion, args)
    if len(all_result["val_ta"]) != 0:
        val_pick_best_epoch = np.argmax(np.array(all_result["val_ta"]))
        print(
            "* best SA = {}, Epoch = {}".format(
                all_result["test_ta"][val_pick_best_epoch], val_pick_best_epoch + 1
            )
        )

    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()

    model.eval()

    start = time.time()    
    if args.imagenet_arch:
        for i, data in enumerate(forget_loader):
            image, target = get_x_y_from_data_dict(data, device)
            if epoch < args.warmup:
                utils.warmup_lr(
                    epoch, i + 1, optimizer, one_epoch_step=len(forget_loader), args=args
                )

            # compute loss and grad on the retain
            _, data = next(retain_loader_iter)
            image, target = get_x_y_from_data_dict(data, device)

            output_clean = model(image)

            loss = criterion(output_clean, target)
            
            # measure accuracy and record loss
            output = output_clean.float()
            loss = loss.float()
            prec1 = utils.accuracy(output.data, target)[0]
            losses.update(loss.item(), image.size(0))
            top1.update(prec1.item(), image.size(0))

            if (i + 1) % args.print_freq == 0:
                end = time.time()
                print(
                    "Epoch: [{0}][{1}/{2}]\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Accuracy {top1.val:.3f} ({top1.avg:.3f})\t"
                    "Time {3:.2f}".format(
                        epoch, i, len(forget_loader), end - start, loss=losses, top1=top1
                    )
                )
                start = time.time()
    else:
        for i, (image, target) in enumerate(forget_loader):
            if epoch < args.warmup:
                utils.warmup_lr(
                    epoch, i + 1, optimizer, one_epoch_step=len(forget_loader), args=args
                )

            # compute loss and grad on the retain
            _, data = next(retain_loader_iter)
            image, target = data[0].to(device), data[1].to(device)

            output_clean = model(image)

            loss = criterion(output_clean, target)

            # measure accuracy and record loss
            output = output_clean.float()
            loss = loss.float()
            prec1 = utils.accuracy(output.data, target)[0]
            losses.update(loss.item(), image.size(0))
            top1.update(prec1.item(), image.size(0))

            if (i + 1) % args.print_freq == 0:
                end = time.time()
                print(
                    "Epoch: [{0}][{1}/{2}]\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Accuracy {top1.val:.3f} ({top1.avg:.3f})\t"
                    "Time {3:.2f}".format(
                        epoch, i, len(forget_loader), end - start, loss=losses, top1=top1
                    )
                )
                start = time.time()

    print("train_accuracy {top1.avg:.3f}".format(top1=top1))

    return top1.avg