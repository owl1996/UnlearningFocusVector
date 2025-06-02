import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import utils
import torch
from models import *

def relativeUA(model, loader, args, device):
    if args.imagenet_arch:
        ideal = model_dict[args.arch](num_classes=args.num_classes, imagenet=True)
    else:
        ideal = model_dict[args.arch](num_classes=args.num_classes)
    ideal.to(device)

    checkpoint_path = "ideal_" + str(args.num_indexes_to_replace) + "_" + str(args.class_to_replace) + "_" + str(args.dataset) + "_" + str(args.arch) + "_" + str(args.seed)
    checkpoint = torch.load("./results/" + str(args.dataset) + "/" + checkpoint_path + "model.pth.tar", map_location=device, weights_only = True)
    ideal.load_state_dict(checkpoint)
    ideal.eval()

    correct = 0
    len_loader = len(loader.dataset)
    correct_predicted = 0
    correct_ideal_predicted = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            ideal_outputs = ideal(inputs)
            _, predicted = torch.max(outputs.data, 1)
            # print("predicted: ", predicted)
            _, ideal_predicted = torch.max(ideal_outputs.data, 1)
            # print("ideal_predicted: ", ideal_predicted)
            # print("targets: ", targets)
            correct += (predicted == ideal_predicted).sum().item()
            correct_predicted += (predicted == targets).sum().item()
            correct_ideal_predicted += (ideal_predicted == targets).sum().item()

    # result = 100 * correct / len_loader
    # print("ideal UA : ", 100 * correct_ideal_predicted / len_loader)
    # result = 100 * (correct_predicted - correct_ideal_predicted) / len_loader
    result = {
        "Fid" : 100 * (correct) / len_loader,
        "rUA" : 100 * (correct_predicted - correct_ideal_predicted) / len_loader
    }
    return result