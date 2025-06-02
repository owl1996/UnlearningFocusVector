import copy
import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data

import arg_parser
import evaluation
import utils
from trainer import validate
from time import time
import mlflow
import unlearn

def main():
    global args
    args = arg_parser.parse_args()

    args_u_e = args.unlearn_epochs
    args_u = args.unlearn
    args_ulr = args.unlearn_lr

    args.unlearn = "ideal"
    args.unlearn_epochs = args.epochs
    args.unlearn_lr = args.lr 

    model_tag = "ideal" + "_" + str(args.num_indexes_to_replace) + "_" + str(args.class_to_replace) + "_" + str(args.dataset) + "_" + str(args.arch) + "_" + str(args.seed)
    if model_tag + "checkpoint.pth.tar" in os.listdir(args.save_dir):
        args.unlearn_epochs = args_u_e
        args.unlearn = args_u
        print(f"The ideal {model_tag} model has already been trained, skip")
        return

    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu))
        device = torch.device(f"cuda:{int(args.gpu)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    os.makedirs(args.save_dir, exist_ok=True)
    if args.seed:
        utils.setup_seed(args.seed)
    seed = args.seed
    # prepare dataset
    (
        model,
        train_loader_full,
        val_loader,
        test_loader,
        marked_loader,
    ) = utils.setup_model_dataset(args)
    model.to(device)


    def replace_loader_dataset(
        dataset, batch_size=args.batch_size, seed=1, shuffle=True
    ):
        utils.setup_seed(seed)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=True,
            shuffle=shuffle,
        )

    forget_dataset = copy.deepcopy(marked_loader.dataset)
    if args.dataset == "svhn":
        try:
            marked = forget_dataset.targets < 0
        except AttributeError:
            marked = forget_dataset.labels < 0
        forget_dataset.data = forget_dataset.data[marked]
        try:
            forget_dataset.targets = -forget_dataset.targets[marked] - 1
        except AttributeError:
            forget_dataset.labels = -forget_dataset.labels[marked] - 1
        forget_loader = replace_loader_dataset(forget_dataset, seed=seed, shuffle=True)

        retain_dataset = copy.deepcopy(marked_loader.dataset)
        try:
            marked = retain_dataset.targets >= 0
        except AttributeError:
            marked = retain_dataset.labels >= 0
        retain_dataset.data = retain_dataset.data[marked]
        try:
            retain_dataset.targets = retain_dataset.targets[marked]
        except AttributeError:
            retain_dataset.labels = retain_dataset.labels[marked]
        retain_loader = replace_loader_dataset(retain_dataset, seed=seed, shuffle=True)
        assert len(forget_dataset) + len(retain_dataset) == len(
            train_loader_full.dataset
        )
    else:
        try:
            marked = forget_dataset.targets < 0
            forget_dataset.data = forget_dataset.data[marked]
            forget_dataset.targets = -forget_dataset.targets[marked] - 1
            forget_loader = replace_loader_dataset(
                forget_dataset, seed=seed, shuffle=True
            )
            retain_dataset = copy.deepcopy(marked_loader.dataset)
            marked = retain_dataset.targets >= 0
            retain_dataset.data = retain_dataset.data[marked]
            retain_dataset.targets = retain_dataset.targets[marked]
            retain_loader = replace_loader_dataset(
                retain_dataset, seed=seed, shuffle=True
            )
            assert len(forget_dataset) + len(retain_dataset) == len(
                train_loader_full.dataset
            )
        except AttributeError:
            marked = forget_dataset.targets < 0
            forget_dataset.imgs = forget_dataset.imgs[marked]
            forget_dataset.targets = -forget_dataset.targets[marked] - 1
            forget_loader = replace_loader_dataset(
                forget_dataset, seed=seed, shuffle=True
            )
            retain_dataset = copy.deepcopy(marked_loader.dataset)
            marked = retain_dataset.targets >= 0
            retain_dataset.imgs = retain_dataset.imgs[marked]
            retain_dataset.targets = retain_dataset.targets[marked]
            retain_loader = replace_loader_dataset(
                retain_dataset, seed=seed, shuffle=True
            )
            assert len(forget_dataset) + len(retain_dataset) == len(
                train_loader_full.dataset
            )

    unlearn_data_loaders = OrderedDict(
        retain=retain_loader, forget=forget_loader, val=val_loader, test=test_loader
    )
    
    criterion = nn.CrossEntropyLoss()
    
    unlearn_method = unlearn.get_unlearn_method("ideal")
    unlearn_method(unlearn_data_loaders, model, criterion, args)

    unlearn.save_unlearn_checkpoint(model, None, args)
    torch.save(
        model.state_dict(),
        os.path.join(args.save_dir, model_tag + "model.pth.tar"),
    )

    args.unlearn_epochs = args_u_e
    args.unlearn = args_u
    args.unlearn_lr = args_ulr

if __name__ == "__main__":
    main()
