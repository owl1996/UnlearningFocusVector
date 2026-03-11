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


@torch.no_grad()
def LDA_CG_unlearn(data_loaders, model, criterion, args):
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
    mlflow.log_param("cg_iterations", args.cg_iterations)

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

    class_means = class_sums / (class_counts.unsqueeze(1) + 1e-8)

    # Pass 2: compute full within-class covariance matrix
    cov_sum = torch.zeros((feat_dim, feat_dim), device=device)
    total_count = 0

    for inputs_r, labels_r in retain_loader:
        inputs_r = inputs_r.to(device)
        labels_r = labels_r.to(device)
        features_r = get_features_with_hook(model, inputs_r)

        means_for_batch = class_means[labels_r]
        diffs = features_r - means_for_batch
        
        # Batch covariance accumulation
        cov_sum += torch.mm(diffs.t(), diffs)
        total_count += features_r.shape[0]

    # Full covariance matrix with regularization
    Sigma = cov_sum / max(total_count - num_classes, 1) 
    Sigma += torch.eye(feat_dim, device=device) * 1e-5
    
    # Batched Conjugate Gradient solver
    # Solving Sigma * W^T = class_means^T  <=>  A * X = B
    def batched_cg(A, B, max_iter, tol=1e-6):
        X = torch.zeros_like(B)
        R = B - torch.mm(A, X)
        P = R.clone()
        for _ in range(max_iter):
            AP = torch.mm(A, P)
            alpha = torch.sum(R * R, dim=0) / (torch.sum(P * AP, dim=0) + 1e-8)
            X_new = X + alpha.unsqueeze(0) * P
            R_new = R - alpha.unsqueeze(0) * AP
            
            if torch.max(torch.sqrt(torch.sum(R_new * R_new, dim=0))) < tol:
                break
                
            beta = torch.sum(R_new * R_new, dim=0) / (torch.sum(R * R, dim=0) + 1e-8)
            P = R_new + beta.unsqueeze(0) * P
            X = X_new
            R = R_new
        return X

    W_new_t = batched_cg(Sigma, class_means.t(), max_iter=args.cg_iterations)
    W_new = W_new_t.t() # (num_classes, feat_dim)

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


@torch.no_grad()
def LDA_update_unlearn(data_loaders, model, criterion, args):
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
    mlflow.log_param("cg_iterations", args.cg_iterations)

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
    W_net = last_layer.weight.data.clone()
    
    # Determine feature dimension from the first batch
    for batch in retain_loader:
        inputs_r, _ = batch
        inputs_r = inputs_r.to(device)
        features_r = get_features_with_hook(model, inputs_r)
        feat_dim = features_r.shape[1]
        break

    # Pass 1: compute class means for retain and forget records
    class_sums_r = torch.zeros((num_classes, feat_dim), device=device)
    class_counts_r = torch.zeros(num_classes, device=device)

    for inputs_r, labels_r in retain_loader:
        inputs_r = inputs_r.to(device)
        labels_r = labels_r.to(device)
        features_r = get_features_with_hook(model, inputs_r)

        class_sums_r.scatter_add_(0, labels_r.unsqueeze(1).expand(-1, feat_dim), features_r)
        class_counts_r.scatter_add_(0, labels_r, torch.ones_like(labels_r, dtype=torch.float32))

    class_sums_u = torch.zeros((num_classes, feat_dim), device=device)
    class_counts_u = torch.zeros(num_classes, device=device)

    for inputs_u, labels_u in forget_loader:
        inputs_u = inputs_u.to(device)
        labels_u = labels_u.to(device)
        features_u = get_features_with_hook(model, inputs_u)

        class_sums_u.scatter_add_(0, labels_u.unsqueeze(1).expand(-1, feat_dim), features_u)
        class_counts_u.scatter_add_(0, labels_u, torch.ones_like(labels_u, dtype=torch.float32))

    # Calculate empirical class means
    M_r = class_sums_r / (class_counts_r.unsqueeze(1) + 1e-8)
    M_u = class_sums_u / (class_counts_u.unsqueeze(1) + 1e-8)
    
    n_r = class_counts_r.sum()
    n_u = class_counts_u.sum()
    n = n_r + n_u
    
    # Global mean M_0 matching Theorem calculations formulation
    M_0 = (class_counts_r.unsqueeze(1) * M_r + class_counts_u.unsqueeze(1) * M_u) / (n + 1e-8)
    
    # Delta_M (C, d) derivation
    Delta_M = M_0 - M_r

    # Pass 2: compute full within-class covariance Sigma_r and unlearn internal scatter Sigma_u0
    Sigma_r = torch.zeros((feat_dim, feat_dim), device=device)
    Sigma_u0 = torch.zeros((feat_dim, feat_dim), device=device)

    # retain set scatter: sum (X_r - M_r[y_r])(X_r - M_r[y_r])^T
    for inputs_r, labels_r in retain_loader:
        inputs_r = inputs_r.to(device)
        labels_r = labels_r.to(device)
        features_r = get_features_with_hook(model, inputs_r)

        means_for_batch_r = M_r[labels_r]
        diffs_r = features_r - means_for_batch_r
        Sigma_r += torch.mm(diffs_r.t(), diffs_r)

    # unlearn set scatter wrt M_0: sum (X_u - M_0[y_u])(X_u - M_0[y_u])^T
    for inputs_u, labels_u in forget_loader:
        inputs_u = inputs_u.to(device)
        labels_u = labels_u.to(device)
        features_u = get_features_with_hook(model, inputs_u)

        means_for_batch_u = M_0[labels_u]
        diffs_u = features_u - means_for_batch_u
        Sigma_u0 += torch.mm(diffs_u.t(), diffs_u)

    # Regularization tracking to preclude structural NaNs across identical elements
    Sigma_r += torch.eye(feat_dim, device=device) * 1e-5
    
    # Structural Whitening: Delta_S = Sigma_{u,0} + Delta_M^T D_{n_r} Delta_M
    scaled_Delta_M = class_counts_r.unsqueeze(1) * Delta_M
    term2 = torch.mm(Delta_M.t(), scaled_Delta_M)
    Delta_S = Sigma_u0 + term2

    # Formulate structural linear mapping constraints vector B matching: (Delta_M - W_{net} * Delta_S)
    B = Delta_M - torch.mm(W_net, Delta_S)
    
    # Batched Conjugate Gradient solver identifying inverse linear relationships
    # Solving Sigma_r * \Delta_{LDA}^T = B^T efficiently
    def batched_cg(A, B_val, max_iter, tol=1e-6):
        X = torch.zeros_like(B_val)
        R = B_val - torch.mm(A, X)
        P = R.clone()
        for _ in range(max_iter):
            AP = torch.mm(A, P)
            alpha = torch.sum(R * R, dim=0) / (torch.sum(P * AP, dim=0) + 1e-8)
            X_new = X + alpha.unsqueeze(0) * P
            R_new = R - alpha.unsqueeze(0) * AP
            
            if torch.max(torch.sqrt(torch.sum(R_new * R_new, dim=0))) < tol:
                break
                
            beta = torch.sum(R_new * R_new, dim=0) / (torch.sum(R * R, dim=0) + 1e-8)
            P = R_new + beta.unsqueeze(0) * P
            X = X_new
            R = R_new
        return X

    Delta_LDA_t = batched_cg(Sigma_r, B.t(), max_iter=args.cg_iterations)
    Delta_LDA = Delta_LDA_t.t() # (num_classes, feat_dim) layout returned

    # Subtract optimal isolated derivation difference vector exactly from tracking network
    W_new = W_net - Delta_LDA

    # Recalculate ultimate structural layer specific prior probabilities and exact structural biases
    prior_probs = class_counts_r / (n_r + 1e-8)
    bias_new = torch.log(prior_probs + 1e-8) - 0.5 * (W_new * M_r).sum(dim=1)

    # Replace generated final theoretical weights tracking
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