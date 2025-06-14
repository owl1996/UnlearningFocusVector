import sys
import time
import torch # type: ignore
import utils
from .impl import iterative_unlearn

sys.path.append(".")
from imagenet import get_x_y_from_data_dict

import mlflow # type: ignore
from trainer import validate
import evaluation

@iterative_unlearn
def SalUn(data_loaders, model, criterion, optimizer, epoch, args):
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

    beta = args.beta

    forget_loader = data_loaders["forget"]
    retain_loader = torch.utils.data.DataLoader(data_loaders["retain"].dataset, batch_size = args.batch_size, shuffle=True)
    retain_loader_iter = enumerate(retain_loader)

    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()

    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu))
        device = torch.device(f"cuda:{int(args.gpu)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # switch to train mode
    num_classes = list(model.children())[-1].out_features
    
    model.train()

    start = time.time()    
    if args.imagenet_arch:
        for i, data in enumerate(forget_loader):
            image, target = get_x_y_from_data_dict(data, device)
            if epoch < args.warmup:
                utils.warmup_lr(
                    epoch, i + 1, optimizer, one_epoch_step=len(forget_loader), args=args
                )
            
             # compute output for masking salUn
            output_clean = model(image)

            loss = -criterion(output_clean, target)
            optimizer.zero_grad()
            loss.backward()
            
            # mask_grads = [torch.ones_like(param) for param in model.parameters()]
            mask_grads = []
            for idx_param, param in enumerate(model.parameters()):
                # salUn
                mask = (torch.abs(param.grad) >= torch.quantile(torch.abs(param.grad), args.quantile, interpolation='midpoint'))
                # imbriqué
                # mask_grad = mask * mask_grads[idx_param]
                # mask_grads[idx_param] = mask_grad
                mask_grads.append(mask)

            # compute output
            model.zero_grad()
            output_clean = model(image)

            # random target, target size
            target = (target + torch.randint(1, num_classes, target.shape, device=device)) % num_classes

            loss = criterion(output_clean, target)
            optimizer.zero_grad()
            loss.backward()

            output = output_clean.float()
            loss = loss.float()

            # copy the grads
            grads = []
            for param in model.parameters():
                grads.append(param.grad)
            model.zero_grad()

            # compute loss and grad on the retain
            _, data = next(retain_loader_iter)
            image, target = get_x_y_from_data_dict(data, device)

            output_clean = model(image)

            loss = criterion(output_clean, target)
            optimizer.zero_grad()
            loss.backward()

            for idx_param, param in enumerate(model.parameters()):
                
                param.grad = mask_grads[idx_param] * (beta * param.grad + (1 - beta) * grads[idx_param])
            optimizer.step()
            
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

            image = image.to(device)
            target = target.to(device)

            # compute output for masking salUn
            output_clean = model(image)

            loss = -criterion(output_clean, target)
            optimizer.zero_grad()
            loss.backward()

            mask_grads = []
            for idx_param, param in enumerate(model.parameters()):
                # salUn
                mask = (torch.abs(param.grad) >= torch.quantile(torch.abs(param.grad), args.quantile, interpolation='midpoint'))
                # imbriqué
                # mask_grad = mask * mask_grads[idx_param]
                # mask_grads[idx_param] = mask_grad
                mask_grads.append(mask)

            # compute output
            model.zero_grad()

            # random target, target size
            target = (target + torch.randint(1, num_classes, target.shape, device=device)) % num_classes

            # compute output
            output_clean = model(image)

            loss = criterion(output_clean, target)
            optimizer.zero_grad()
            loss.backward()

            output = output_clean.float()
            loss = loss.float()
            
            # copy the grads
            grads = []
            for param in model.parameters():
                grads.append(param.grad)
            model.zero_grad()

            # compute loss and grad on the retain
            _, data = next(retain_loader_iter)
            image, target = data[0].to(device), data[1].to(device)

            output_clean = model(image)

            loss = criterion(output_clean, target)
            optimizer.zero_grad()
            loss.backward()

            for idx_param, param in enumerate(model.parameters()):
                param.grad = mask_grads[idx_param] * (beta * param.grad + (1 - beta) * grads[idx_param])
            optimizer.step()

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