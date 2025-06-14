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

import unlearn

import main_baseline
import main_ideal

def main():
    args = arg_parser.parse_args()

    main_baseline.main()
    main_ideal.main()

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
    
    print("retain size :", len(unlearn_data_loaders["retain"].dataset))
    print("forget size :", len(unlearn_data_loaders["forget"].dataset))
    print("val size :", len(unlearn_data_loaders["val"].dataset))
    print("test size :", len(unlearn_data_loaders["test"].dataset))

    # print the classes in forget dataset
    print("forget dataset classes: ", set(forget_dataset.targets.tolist()))
    print("retain dataset classes: ", set(retain_dataset.targets.tolist()))

    criterion = nn.CrossEntropyLoss()
    

    evaluation_result = None

    if args.resume:
        checkpoint = unlearn.load_unlearn_checkpoint(model, device, args)

    if args.resume and checkpoint is not None:
        model, evaluation_result = checkpoint
    else:
        checkpoint = torch.load(args.mask, map_location=device, weights_only = False)
        if "state_dict" in checkpoint.keys():
            checkpoint = checkpoint["state_dict"]

        if (
            args.unlearn != "retrain"
            and args.unlearn != "retrain_sam"
            and args.unlearn != "retrain_ls"
        ):
            model.load_state_dict(checkpoint, strict=False)
            model.to(device)

        unlearn_method = unlearn.get_unlearn_method(args.unlearn)

        unlearn_method(unlearn_data_loaders, model, criterion, args)
        unlearn.save_unlearn_checkpoint(model, None, args)

if __name__ == "__main__":
    main()