import sys
import time
import torch
import torch.nn as nn
import utils
from .impl import iterative_unlearn
from .FT import FT

sys.path.append(".")
from imagenet import get_x_y_from_data_dict
from trainer import validate
import evaluation
import mlflow # type: ignore

def transpose_conv_kernels(model):
    """
    Transpose les noyaux (patchs) HxW de toutes les couches de convolution du modèle
    sans changer les dimensions globales des poids.
    """
    for module in model.modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            with torch.no_grad():
                weight = module.weight.data
                if weight.dim() == 4:  # Conv2d
                    print(f"Avant transpose: {weight.shape}")
                    transposed = weight.transpose(2, 3).contiguous()
                    module.weight.data.copy_(transposed)
                    print(f"Après transpose : {module.weight.shape}")
                elif weight.dim() == 3:  # Conv1d
                    # (out_channels, in_channels, kernel_size)
                    print(f"Conv1d non transposé (1D kernel)")
                elif weight.dim() == 5:  # Conv3d
                    # Transposer profondeur/hauteur ou autre combinaison possible
                    print(f"Conv3d non transposé (à définir si besoin)")
    return model

@iterative_unlearn
def CT(data_loaders, model, criterion, optimizer, epoch, args):

    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu))
        device = torch.device(f"cuda:{int(args.gpu)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")


    if epoch == 0:
        model = transpose_conv_kernels(model)

    else:
        model.train()

        forget_loader = data_loaders["forget"]
        retain_loader = torch.utils.data.DataLoader(data_loaders["retain"].dataset, batch_size = args.batch_size, shuffle=True)
        retain_loader_iter = enumerate(retain_loader)
        
        mlflow.start_run()
        mlflow.log_param("seed", args.seed)
        mlflow.log_param("save_dir", args.save_dir)
        mlflow.log_param("model", args.mask)
        mlflow.log_param("unlearn", args.unlearn)
        mlflow.log_param("num_indexes_to_replace", args.num_indexes_to_replace)
        mlflow.log_param("unlearn_epochs", epoch + 1)
        mlflow.log_param("unlearn_lr", args.unlearn_lr) 
        mlflow.log_param("beta", args.beta)
        mlflow.log_param("quantile", args.quantile)
        mlflow.log_param("num_indexes_to_replace", args.num_indexes_to_replace)
        mlflow.log_param("class_to_replace", args.class_to_replace)
        mlflow.log_param("arch", args.arch) 
        mlflow.log_param("dataset", args.dataset)

        losses = utils.AverageMeter()
        top1 = utils.AverageMeter()

        # switch to train mode
        model.train()

        if torch.cuda.is_available():
            torch.cuda.set_device(int(args.gpu))
            device = torch.device(f"cuda:{int(args.gpu)}")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        start = time.time()

        for i, (image, target) in enumerate(forget_loader):
            if epoch < args.warmup:
                utils.warmup_lr(
                    epoch, i + 1, optimizer, one_epoch_step=len(forget_loader), args=args
                )

            _, data = next(retain_loader_iter)
            image, target = data[0].to(device), data[1].to(device)
            
            # compute output
            output_clean = model(image)
            loss = criterion(output_clean, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            output = output_clean.float()
            loss = loss.float()
            # measure accuracy and record loss
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

        mlflow.log_metric("RTE", time.time() - start)

        for name, loader in data_loaders.items():
            utils.dataset_convert_to_test(loader.dataset, args)
            val_acc = validate(loader, model, criterion, args)
            mlflow.log_metric(name, val_acc)
        
        MIA_trainer_loader = torch.utils.data.DataLoader(
                    torch.utils.data.Subset(retain_loader.dataset, list(range(len(data_loaders["test"].dataset)))), batch_size=args.batch_size, shuffle=False
                )
        MIA_classifiers = evaluation.SVC_classifiers(MIA_trainer_loader, data_loaders["test"], model)
        # print(evaluation.SVC_predict(MIA_classifiers, forget_loader, model))
        eval = evaluation.SVC_predict(MIA_classifiers, forget_loader, model)
        print(eval)
        for key, val in eval.items():
            mlflow.log_metric("MIA_" + key, val)

        result = evaluation.relativeUA(model, data_loaders["forget"], args, device)
        mlflow.log_metric("relativeUA", result["rUA"])
        mlflow.log_metric("Fid", result["Fid"])

        mlflow.end_run()
        print("retain_accuracy {top1.avg:.3f}".format(top1=top1))

        return top1.avg
