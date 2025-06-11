import sys
import time
import torch
import utils
import vutils
from .impl import iterative_unlearn

import mlflow

from trainer import validate

sys.path.append(".")
from imagenet import get_x_y_from_data_dict
import evaluation

normal_dist = torch.distributions.Normal(loc=0.0, scale=1.0)

@iterative_unlearn
def ProbGrad(data_loaders, model, criterion, optimizers, epoch, args):
    """
    ProbGrad unlearning method.

    We compute the different gradients (according to NegGradPlus) of the losses with respect to the model parameters.
    We estimate the variance of the gradients of the parameters.
    We put a probabilistic threshold so that we mask the parameters that are unsure to be updated.

    """
    
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

    # mask_grads = [torch.ones_like(param) for param in model.parameters()]
    
    optimizer, optimizer_forget, optimizer_retain = optimizers

    start = time.time()
    model.train()
    for i, (image, target) in enumerate(forget_loader):
        # NegGrad : Montée de Gradient sur le forget
        image = image.to(device)
        target = target.to(device)

        if epoch < args.warmup:
                utils.warmup_lr(
                    epoch, i + 1, optimizer, one_epoch_step=len(forget_loader), args=args
                )

        model.zero_grad()
        original_params = [param.detach().clone() for param in model.parameters()]

        loss = - criterion(model(image), target)
        loss.backward()

        grad_forget = [param.grad for param in model.parameters()]

        with torch.no_grad():
            optimizer_forget.step()
            for param, orig in zip(model.parameters(), original_params):
                param.copy_(orig)

        # Plus : Descente sur le retain
        _, data = next(retain_loader_iter)
        image, target = data[0].to(device), data[1].to(device)

        model.zero_grad()
        output_clean = model(image)
        loss = criterion(output_clean, target)
        loss.backward()

        with torch.no_grad():
            optimizer_retain.step()
            for param, orig in zip(model.parameters(), original_params):
                param.copy_(orig)
        
        del original_params

        # Compute the mask
        for idx_param, param in enumerate(model.parameters()):

            # saved momentums in optimizers
            m_forget, v_forget = optimizer_forget.state[param]["exp_avg"], optimizer_forget.state[param]["exp_avg_sq"]
            _, v_retain = optimizer_retain.state[param]["exp_avg"], optimizer_retain.state[param]["exp_avg_sq"]

            signal_noise_forget = m_forget / (v_forget + 1e-8).sqrt()
            signal_noise_retain = param.grad / (v_retain + 1e-8).sqrt()

            cdf_forget = normal_dist.cdf(signal_noise_forget)
            cdf_retain = normal_dist.cdf(signal_noise_retain)

            # quantiles mask
            vmask = cdf_forget * cdf_retain + (1. - cdf_forget) * (1. - cdf_retain)

            
            # Generate Bernouilli mask with probabilities vmask
            mask = torch.bernoulli(vmask)

            # mask = vmask >= args.quantile

            # # imbriqué
            # mask_grad = mask * mask_grads[idx_param]
            # mask_grads[idx_param] = mask_grad

            # update grad
            param.grad = mask * (args.beta * param.grad + (1 - args.beta) * grad_forget[idx_param])

            # print("Ratio of masked parameters : ", mask_grad.sum() / mask.numel())

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

    result = evaluation.relativeUA(model, data_loaders["test"], args, device)
    mlflow.log_metric("relativeUA", result["rUA"])
    mlflow.log_metric("Fid", result["Fid"])

    mlflow.end_run()
    print("retain_accuracy {top1.avg:.3f}".format(top1=top1))

    return top1.avg