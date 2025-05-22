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
    # device
    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu))
        device = torch.device(f"cuda:{int(args.gpu)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    os.makedirs(args.save_dir, exist_ok=True)
    if args.seed:
        setup_seed(args.seed)

    model.to(device)
    train_loader = data_loaders["retain"]

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
    
    # train for one epoch
    train(train_loader, model, criterion, optimizer, epoch, args)

    scheduler.step()

    return None