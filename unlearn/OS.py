import sys
import time
import copy
import torch
import torchvision.models as models
import torch.nn as nn

import utils
from .impl import iterative_unlearn

sys.path.append(".")
from imagenet import get_x_y_from_data_dict

import mlflow # type: ignore
from trainer import validate
import evaluation

def get_features_with_hook(model, batch):
        try:
            last_layer = model.fc
        except AttributeError:
            # For models like VGG or AlexNet
            last_layer = model.classifier[-1]
        # Dictionnaire ou liste pour stocker l'activation capturée
        activation = {}

        # 1. Définir la fonction du hook
        # Signature obligatoire : (module, input, output)
        def hook_fn(module, input, output):
            # 'input' est un tuple contenant les arguments passés à la couche.
            # Pour une couche Linear, input[0] est le tenseur de features aplati.
            activation['features'] = input[0].detach() 

        # 2. Enregistrer le hook sur la couche 'fc'
        # On utilise register_forward_hook pour intervenir pendant la passe avant
        handle = last_layer.register_forward_hook(hook_fn)

        # 3. Faire la passe avant (Inference)
        # Le hook va se déclencher automatiquement quand les données arrivent à 'fc'
        model.eval()
        with torch.no_grad():
            _ = model(batch) # On ignore la sortie finale (logits)

        # 4. Nettoyage : Toujours supprimer le hook après usage !
        # Sinon, ils s'accumulent et ralentissent le modèle ou créent des fuites de mémoire.
        handle.remove()

        return activation['features']
    
def append_bias_term(features):
    batch_size = features.shape[0]
    
    # Création d'une colonne de 1 (sur le même device que features)
    ones = torch.ones((batch_size, 1), device=features.device, dtype=features.dtype)
    
    # Concaténation sur la dimension des colonnes (dim=1)
    features_aug = torch.cat([features, ones], dim=1)
    
    return features_aug

@torch.no_grad()
def OS_unlearn(data_loaders, model, criterion, args):
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
    retain_loader = torch.utils.data.DataLoader(data_loaders["retain"].dataset, batch_size = args.batch_size, shuffle=True)
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
        # For models like VGG or AlexNet
        last_layer = model.classifier[-1]

    gram_retain = 0
    for batch_retain in retain_loader:
        inputs_r, _ = batch_retain
        inputs_r = inputs_r.to(device)
        features_r = get_features_with_hook(model, inputs_r)
        features_r = append_bias_term(features_r)
        gram_retain += features_r.T @ features_r

    inputs_u = torch.from_numpy(forget_loader.dataset.data).float().permute(0, 3, 1, 2).to(device)
    labels_u = torch.from_numpy(forget_loader.dataset.targets).long().to(device)
    one_hot_labels_u = torch.nn.functional.one_hot(labels_u, num_classes=last_layer.out_features).float()
    batch_size = inputs_u.shape[0]
    dtype = gram_retain.dtype

    inv_gram = torch.linalg.pinv(gram_retain)
    identity_batch = torch.eye(batch_size, device=device, dtype=dtype)
    identity_features = torch.eye(gram_retain.shape[0], device=device, dtype=dtype)

    features_forget = get_features_with_hook(model, inputs_u)
    features_forget = append_bias_term(features_forget)

    conjug_features = features_forget @ inv_gram @ features_forget.T # (batch_size, batch_size)
    K = inv_gram @ (features_forget.T @ torch.linalg.pinv(identity_batch + conjug_features))
    K = K.to(device)

    W_model = torch.cat([last_layer.weight, last_layer.bias.unsqueeze(1)], dim=1)
    
    uniform_labels = torch.ones_like(one_hot_labels_u) / one_hot_labels_u.shape[1]
    smooth_labels = args.beta * one_hot_labels_u + (1 - args.beta) * uniform_labels

    prob_labels = torch.softmax(model(inputs_u), dim=1)

    # update_forget = - (K @ prob_labels).T @ torch.linalg.pinv(identity_features - K @ features_forget)
    update_forget = - (K @ one_hot_labels_u).T @ torch.linalg.pinv(identity_features - K @ features_forget)
    W_forget = W_model + args.unlearn_lr * torch.norm(W_model) / torch.norm(update_forget) * update_forget

    try:
        model.fc.weight.data = W_forget[:, :-1]
        model.fc.bias.data = W_forget[:, -1]
    except AttributeError:
        model.classifier[-1].weight.data = W_forget[:, :-1]
        model.classifier[-1].bias.data = W_forget[:, -1]

    # EVALUATION AFTER UNLEARNING
    mlflow.log_metric("RTE", time.time() - start)
    print("Unlearning time: {:.3f}s".format(time.time() - start))

    for name, loader in data_loaders.items():
        utils.dataset_convert_to_test(loader.dataset, args)
        val_acc = validate(loader, model, criterion, args)
        mlflow.log_metric(name, val_acc)
    
    MIA_trainer_loader = torch.utils.data.DataLoader(
                torch.utils.data.Subset(train_loader.dataset, list(range(len(data_loaders["test"].dataset)))), batch_size=args.batch_size, shuffle=False
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

if __name__ == "__main__":
    pass