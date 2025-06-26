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

def parse_model_path(model_path):
    args_model = {}
    split_path = model_path.split('_')
    print(split_path)
    if split_path[0] == 'ideal':
        # e.g. ideal_1900_0_cifar10_vgg16_bn_3model.pth.tar
        # e.g. ideal_1900_0_cifar10_resnet18_3model.pth.tar
        args_model['unlearn'] = split_path[0]
        args_model['num_indexes_to_replace'] = split_path[1]
        args_model['class_to_replace'] = split_path[2]
        args_model['dataset'] = split_path[3]
        if split_path[4] == 'vgg16':
            args_model['arch'] = split_path[4] + '_bn'
            args_model['seed'] = split_path[6][0]
        else:
            args_model['arch'] = split_path[4]
            args_model['seed'] = split_path[5][0]
    else:
        print('Future Error : Not an ideal model !')
    return args_model

def evaluate(model_path):
    args = arg_parser.parse_args()
    args_model = parse_model_path(model_path)
    # modify args
    # ...
    args.unlearn = args_model['unlearn']
    args.num_indexes_to_replace = int(args_model['num_indexes_to_replace'])
    args.class_to_replace = int(args_model['class_to_replace'])
    args.dataset = args_model['dataset']
    args.arch = args_model['arch']
    args.seed = int(args_model['seed'])
    # ...

    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu))
        device = torch.device(f"cuda:{int(args.gpu)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

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

    criterion = nn.CrossEntropyLoss()

    # --------------------------
    #
    # Eval the ideal model ...
    #
    # --------------------------
    print('ideal model : ', model_path)
    checkpoint = torch.load(f'results/{args.dataset}/{model_path}', map_location=device, weights_only = False)
    model.load_state_dict(checkpoint, strict=False)
    model.to(device)

    mlflow.start_run()
    mlflow.log_param("seed", args.seed)
    mlflow.log_param("save_dir", args.save_dir)
    mlflow.log_param("model", model_path)
    mlflow.log_param("unlearn", 'ideal')
    mlflow.log_param("num_indexes_to_replace", args.num_indexes_to_replace)
    mlflow.log_param("unlearn_epochs", 0)
    mlflow.log_param("unlearn_lr", args.unlearn_lr) 
    mlflow.log_param("beta", args.beta)
    mlflow.log_param("quantile", args.quantile)
    mlflow.log_param("num_indexes_to_replace", args.num_indexes_to_replace)
    mlflow.log_param("class_to_replace", args.class_to_replace)
    mlflow.log_param("arch", args.arch) 
    mlflow.log_param("dataset", args.dataset)
    mlflow.log_metric("RTE", 0.)

    for name, loader in unlearn_data_loaders.items():
        utils.dataset_convert_to_test(loader.dataset, args)
        val_acc = validate(loader, model, criterion, args)
        mlflow.log_metric(name, val_acc)
        print(name, val_acc)
    
    MIA_trainer_loader = torch.utils.data.DataLoader(
                torch.utils.data.Subset(unlearn_data_loaders["retain"].dataset, list(range(len(unlearn_data_loaders["test"].dataset)))), batch_size=args.batch_size, shuffle=False
            )
    MIA_classifiers = evaluation.SVC_classifiers(MIA_trainer_loader, unlearn_data_loaders["test"], model)

    eval = evaluation.SVC_predict(MIA_classifiers, forget_loader, model)
    
    for key, val in eval.items():
        mlflow.log_metric("MIA_" + key, val)
        print(key, val)

    result = evaluation.relativeUA(model, forget_loader, args, device)
    print(result)

    mlflow.log_metric("relativeUA", result["rUA"])
    mlflow.log_metric("Fid", result["Fid"])

    mlflow.end_run()
    
    # --------------------------
    #
    # Now eval the initial model 
    #
    # --------------------------
    imodel, *_ = utils.setup_model_dataset(args)

    imodel_path = args.dataset + '_' + args.arch + '_' + str(args.seed) + 'model.pth.tar'
    print('initial model : ', imodel_path)

    icheckpoint = torch.load(f'results/{args.dataset}/{imodel_path}', map_location=device, weights_only = False)
    imodel.load_state_dict(icheckpoint["state_dict"], strict=True)
    imodel.to(device)

    mlflow.start_run()
    mlflow.log_param("seed", args.seed)
    mlflow.log_param("save_dir", args.save_dir)
    mlflow.log_param("model", imodel_path)
    mlflow.log_param("unlearn", 'initial')
    mlflow.log_param("num_indexes_to_replace", args.num_indexes_to_replace)
    mlflow.log_param("unlearn_epochs", 0)
    mlflow.log_param("unlearn_lr", args.unlearn_lr) 
    mlflow.log_param("beta", args.beta)
    mlflow.log_param("quantile", args.quantile)
    mlflow.log_param("num_indexes_to_replace", args.num_indexes_to_replace)
    mlflow.log_param("class_to_replace", args.class_to_replace)
    mlflow.log_param("arch", args.arch) 
    mlflow.log_param("dataset", args.dataset)
    mlflow.log_metric("RTE", 0.)

    for name, loader in unlearn_data_loaders.items():
        utils.dataset_convert_to_test(loader.dataset, args)
        val_acc = validate(loader, imodel, criterion, args)
        mlflow.log_metric(name, val_acc)
        print(name, val_acc)
    
    MIA_trainer_loader = torch.utils.data.DataLoader(
                torch.utils.data.Subset(unlearn_data_loaders["retain"].dataset, list(range(len(unlearn_data_loaders["test"].dataset)))), batch_size=args.batch_size, shuffle=False
            )
    MIA_classifiers = evaluation.SVC_classifiers(MIA_trainer_loader, unlearn_data_loaders["test"], imodel)

    eval = evaluation.SVC_predict(MIA_classifiers, forget_loader, imodel)
    
    for key, val in eval.items():
        mlflow.log_metric("MIA_" + key, val)
        print(key, val)

    result = evaluation.relativeUA(imodel, forget_loader, args, device)
    print(result)

    mlflow.log_metric("relativeUA", result["rUA"])
    mlflow.log_metric("Fid", result["Fid"])

    mlflow.end_run()

if __name__ == '__main__':
    evaluate('ideal_50_-1_svhn_resnet18_8model.pth.tar')
    
    base_dir = 'results/'
    eval_file_path = os.path.join(base_dir, 'eval.txt')

    if os.path.exists(eval_file_path):
        with open(eval_file_path, 'r') as f:
            already_evaluated = set(line.strip() for line in f)
    else:
        already_evaluated = set()

    with open(eval_file_path, 'a') as eval_file:
        
        for subfolder in os.listdir(base_dir):
            subfolder_path = os.path.join(base_dir, subfolder)

            if os.path.isdir(subfolder_path):
                for item in os.listdir(subfolder_path):
                    if item.startswith("ideal") and item.endswith("model.pth.tar"):
                        if item not in already_evaluated:
                            try:
                                evaluate(item)
                                eval_file.write(item + '\n')
                                eval_file.flush()  # écrit immédiatement
                            except:  # noqa: E722
                                pass
                        else:
                            print(f"Déjà évalué : {item}")