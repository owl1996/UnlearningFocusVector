import sys
import time
import torch
import utils
from .impl import iterative_unlearn

sys.path.append(".")
from imagenet import get_x_y_from_data_dict

@iterative_unlearn
def nothing(data_loaders, model, criterion, optimizer, epoch, args):
    return None