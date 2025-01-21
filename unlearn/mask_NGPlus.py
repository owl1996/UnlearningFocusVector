import sys
import time
import torch
import utils
from .impl import iterative_unlearn

sys.path.append(".")
from imagenet import get_x_y_from_data_dict

@iterative_unlearn
def mask_NGPlus(data_loaders, model, criterion, optimizer, epoch, args):
    forget_loader = data_loaders["forget"]
    retain_loader = data_loaders["retain"]
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
    mask_grads = [torch.ones_like(param) for param in model.parameters()]
    model.train()

    start = time.time()    
    if args.imagenet_arch:
        for i, data in enumerate(forget_loader):
            image, target = get_x_y_from_data_dict(data, device)
            if epoch < args.warmup:
                utils.warmup_lr(
                    epoch, i + 1, optimizer, one_epoch_step=len(forget_loader), args=args
                )

            # compute output
            output_clean = model(image)

            loss = -criterion(output_clean, target)
            optimizer.zero_grad()
            loss.backward()

            output = output_clean.float()
            loss = loss.float()
            # measure accuracy and record loss
            prec1 = utils.accuracy(output.data, target)[0]

            losses.update(loss.item(), image.size(0))
            top1.update(prec1.item(), image.size(0))

            # copy the grads
            grads = []
            for param in model.parameters():
                grads.append(param.grad)

            # compute loss and grad on the retain
            _, data = next(retain_loader_iter)
            image, target = get_x_y_from_data_dict(data, device)

            output_clean = model(image)

            loss = criterion(output_clean, target)
            optimizer.zero_grad()
            loss.backward()

            beta = 0.95
            for idx_param, param in enumerate(model.parameters()):
                mask = (param.grad * grads[idx_param] > 0)
                # imbriqué
                mask_grad = mask * mask_grads[idx_param]
                mask_grads[idx_param] = mask_grad
                # step_grad = _mask * special_aggregation(param.grad, grads[idx_param])
                # param.grad = mask_grad * utils.special_aggregation(param.grad, grads[idx_param])
                param.grad = mask_grad * beta * param.grad + (1 - beta) * grads[idx_param]

                # verbose
                # layer_ratio = 100 * torch.sum(param.grad > 0)/torch.tensor(param.size()).prod()
                layer_ratio = torch.sum(param.grad > 0)
                Layer_ratio.append(layer_ratio)
            print('Max Masking Grad :', max(Layer_ratio))

            optimizer.step()

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

            # compute output
            output_clean = model(image)

            loss = -criterion(output_clean, target)
            optimizer.zero_grad()
            loss.backward()

            output = output_clean.float()
            loss = loss.float()
            # measure accuracy and record loss
            prec1 = utils.accuracy(output.data, target)[0]
            losses.update(loss.item(), image.size(0))
            top1.update(prec1.item(), image.size(0))

            # copy the grads
            grads = []
            for param in model.parameters():
                grads.append(param.grad)

            # compute loss and grad on the retain
            _, data = next(retain_loader_iter)
            image, target = data[0].to(device), data[1].to(device)

            output_clean = model(image)

            loss = criterion(output_clean, target)
            optimizer.zero_grad()
            loss.backward()

            Layer_ratio = []
            for idx_param, param in enumerate(model.parameters()):
                mask = (param.grad * grads[idx_param] > 0)
                # imbrique
                mask_grad = mask * mask_grads[idx_param]
                mask_grads[idx_param] = mask_grad
                # step_grad = _mask * special_aggregation(param.grad, grads[idx_param])
                param.grad = mask_grad * utils.special_aggregation(param.grad, grads[idx_param])

                # verbose
                # layer_ratio = 100 * torch.sum(param.grad > 0)/torch.tensor(param.size()).prod()
                layer_ratio = torch.sum(param.grad > 0)
                Layer_ratio.append(layer_ratio)
            print('Max Masking Grad :', max(Layer_ratio))

            optimizer.step()

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