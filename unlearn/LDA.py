import time
import torch
import torchvision.models as models
import torch.nn as nn
import utils
import mlflow # type: ignore
from trainer import validate
import evaluation

def get_features_with_hook(model, batch):
    try:
        last_layer = model.fc
    except AttributeError:
        # For models like VGG or AlexNet
        last_layer = model.classifier[-1]
    
    activation = {}

    def hook_fn(module, input, output):
        activation['features'] = input[0].detach() 

    handle = last_layer.register_forward_hook(hook_fn)

    model.eval()
    with torch.no_grad():
        _ = model(batch)

    handle.remove()

    return activation['features']

@torch.no_grad()
def LDA_unlearn(data_loaders, model, criterion, args):
    # INIT METHOD AND LOGGING
    mlflow.start_run()
    mlflow.log_param("seed", args.seed)
    mlflow.log_param("save_dir", args.save_dir)
    mlflow.log_param("model", args.mask)
    mlflow.log_param("unlearn", args.unlearn)
    mlflow.log_param("num_indexes_to_replace", args.num_indexes_to_replace)
    mlflow.log_param("unlearn_epochs", 100*args.unlearn_lr)
    mlflow.log_param("unlearn_lr", args.unlearn_lr) 
    mlflow.log_param("beta", args.beta)
    mlflow.log_param("quantile", args.quantile)
    mlflow.log_param("num_indexes_to_replace", args.num_indexes_to_replace)
    mlflow.log_param("class_to_replace", args.class_to_replace)
    mlflow.log_param("arch", args.arch) 
    mlflow.log_param("dataset", args.dataset)

    train_loader = data_loaders["retain"]
    retain_loader = torch.utils.data.DataLoader(train_loader.dataset, batch_size=args.batch_size, shuffle=False)
    forget_loader = data_loaders["forget"]

    top1 = utils.AverageMeter()

    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu))
        device = torch.device(f"cuda:{int(args.gpu)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    start = time.time()
    # --- UNLEARNING PROCESS ---
    try:
        last_layer = model.fc
    except AttributeError:
        last_layer = model.classifier[-1]

    num_classes = last_layer.out_features
    
    # Determine feature dimension from the first batch
    for batch in retain_loader:
        inputs_r, _ = batch
        inputs_r = inputs_r.to(device)
        features_r = get_features_with_hook(model, inputs_r)
        feat_dim = features_r.shape[1]
        break

    # Pass 1: compute class means
    class_sums = torch.zeros((num_classes, feat_dim), device=device)
    class_counts = torch.zeros(num_classes, device=device)

    for inputs_r, labels_r in retain_loader:
        inputs_r = inputs_r.to(device)
        labels_r = labels_r.to(device)
        features_r = get_features_with_hook(model, inputs_r)

        class_sums.scatter_add_(0, labels_r.unsqueeze(1).expand(-1, feat_dim), features_r)
        class_counts.scatter_add_(0, labels_r, torch.ones_like(labels_r, dtype=torch.float32))

    # Add a small epsilon to avoid division by zero
    class_means = class_sums / (class_counts.unsqueeze(1) + 1e-8)

    # Pass 2: compute within-class variance (diagonal covariance)
    variance_sum = torch.zeros(feat_dim, device=device)
    total_count = 0

    for inputs_r, labels_r in retain_loader:
        inputs_r = inputs_r.to(device)
        labels_r = labels_r.to(device)
        features_r = get_features_with_hook(model, inputs_r)

        means_for_batch = class_means[labels_r]
        diffs = features_r - means_for_batch
        variance_sum += (diffs ** 2).sum(dim=0)
        total_count += features_r.shape[0]

    # Diagonal approximation of pooled within-class covariance matrix
    # adding a small regularization term to avoid numerical instability
    diag_cov = variance_sum / max(total_count - num_classes, 1) + 1e-5
    inv_diag_cov = 1.0 / diag_cov

    # Compute new weights and biases
    W_new = class_means * inv_diag_cov.unsqueeze(0)

    prior_probs = class_counts / total_count
    bias_new = -0.5 * (class_means * W_new).sum(dim=1) + torch.log(prior_probs + 1e-8)

    # Replace weights
    try:
        model.fc.weight.data = W_new
        model.fc.bias.data = bias_new
    except AttributeError:
        model.classifier[-1].weight.data = W_new
        model.classifier[-1].bias.data = bias_new

    # EVALUATION AFTER UNLEARNING
    mlflow.log_metric("RTE", time.time() - start)
    print("Unlearning time: {:.3f}s".format(time.time() - start))

    for name, loader in data_loaders.items():
        utils.dataset_convert_to_test(loader.dataset, args)
        val_acc = validate(loader, model, criterion, args)
        mlflow.log_metric(name, val_acc)
        if name == "retain":
            top1.update(val_acc)
    
    MIA_trainer_loader = torch.utils.data.DataLoader(
                torch.utils.data.Subset(train_loader.dataset, list(range(len(data_loaders["test"].dataset)))), batch_size=args.batch_size, shuffle=False
            )
    MIA_classifiers = evaluation.SVC_classifiers(MIA_trainer_loader, data_loaders["test"], model)
    eval_res = evaluation.SVC_predict(MIA_classifiers, forget_loader, model)
    print(eval_res)
    for key, val in eval_res.items():
        mlflow.log_metric("MIA_" + key, val)

    result = evaluation.relativeUA(model, data_loaders["forget"], args, device)
    mlflow.log_metric("relativeUA", result["rUA"])
    mlflow.log_metric("Fid", result["Fid"])

    mlflow.end_run()
    print("retain_accuracy {:.3f}".format(top1.avg))

    return top1.avg
