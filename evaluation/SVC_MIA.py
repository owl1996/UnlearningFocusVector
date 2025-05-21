import numpy as np
import torch
import torch.nn.functional as F
from sklearn.svm import SVC

from imagenet import get_x_y_from_data_dict


def entropy(p, dim=-1, keepdim=False):
    return -torch.where(p > 0, p * p.log(), p.new([0.0])).sum(dim=dim, keepdim=keepdim)



def m_entropy(p, labels, dim=-1, keepdim=False):
    # Calcul des log-probabilités en évitant log(0)
    log_prob = torch.where(p > 0, p.log(), torch.tensor(1e-30).to(p.device).log())
    reverse_prob = 1 - p
    log_reverse_prob = torch.where(
        reverse_prob > 0, reverse_prob.log(), torch.tensor(1e-30).to(p.device).log()
    )

    # Clone des probabilités pour éviter les modifications en place
    modified_probs = p.clone()
    modified_log_probs = log_prob.clone()

    # Indexation correcte par lot
    batch_indices = torch.arange(labels.size(0))
    modified_probs[batch_indices, labels] = reverse_prob[batch_indices, labels]
    modified_log_probs[batch_indices, labels] = log_reverse_prob[batch_indices, labels]

    return -torch.sum(modified_probs * modified_log_probs, dim=dim, keepdim=keepdim)


def collect_prob(data_loader, model):
    if data_loader is None:
        return torch.zeros([0, 10]), torch.zeros([0])

    prob = []
    targets = []

    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            try:
                batch = [tensor.to(next(model.parameters()).device) for tensor in batch]
                data, target = batch
            except:
                if torch.cuda.is_available():
                    torch.cuda.set_device(int(args.gpu))
                    device = torch.device(f"cuda:{int(args.gpu)}")
                elif torch.backends.mps.is_available():
                    device = torch.device("mps")
                else:
                    device = torch.device("cpu")
                data, target = get_x_y_from_data_dict(batch, device)
            with torch.no_grad():
                output = model(data)
                prob.append(F.softmax(output, dim=-1).data)
                targets.append(target)

    return torch.cat(prob), torch.cat(targets)




def SVC_fit_predict(shadow_train, shadow_test, target_train, target_test):
    n_shadow_train = shadow_train.shape[0]
    n_shadow_test = shadow_test.shape[0]
    n_target_train = target_train.shape[0]
    n_target_test = target_test.shape[0]

    X_shadow = (
        torch.cat([shadow_train, shadow_test])
        .cpu()
        .numpy()
        .reshape(n_shadow_train + n_shadow_test, -1)
    )
    Y_shadow = np.concatenate([np.ones(n_shadow_train), np.zeros(n_shadow_test)])

    clf = SVC(C=3, gamma="auto", kernel="rbf")
    clf.fit(X_shadow, Y_shadow)

    accs = []

    if n_target_train > 0:
        X_target_train = target_train.cpu().numpy().reshape(n_target_train, -1)
        acc_train = clf.predict(X_target_train).mean()
        accs.append(acc_train)

    if n_target_test > 0:
        X_target_test = target_test.cpu().numpy().reshape(n_target_test, -1)
        acc_test = 1 - clf.predict(X_target_test).mean()
        accs.append(acc_test)

    return np.mean(accs)



def SVC_MIA(shadow_train, target_train, target_test, shadow_test, model):
    shadow_train_prob, shadow_train_labels = collect_prob(shadow_train, model)
    shadow_test_prob, shadow_test_labels = collect_prob(shadow_test, model)

    target_train_prob, target_train_labels = collect_prob(target_train, model)
    target_test_prob, target_test_labels = collect_prob(target_test, model)

    shadow_train_corr = (
        torch.argmax(shadow_train_prob, axis=1) == shadow_train_labels
    ).int()
    shadow_test_corr = (
        torch.argmax(shadow_test_prob, axis=1) == shadow_test_labels
    ).int()
    target_train_corr = (
        torch.argmax(target_train_prob, axis=1) == target_train_labels
    ).int()
    target_test_corr = (
        torch.argmax(target_test_prob, axis=1) == target_test_labels
    ).int()

    shadow_train_conf = torch.gather(shadow_train_prob, 1, shadow_train_labels[:, None])
    shadow_test_conf = torch.gather(shadow_test_prob, 1, shadow_test_labels[:, None])
    target_train_conf = torch.gather(target_train_prob, 1, target_train_labels[:, None])
    target_test_conf = torch.gather(target_test_prob, 1, target_test_labels[:, None])

    shadow_train_entr = entropy(shadow_train_prob)
    shadow_test_entr = entropy(shadow_test_prob)
    target_train_entr = entropy(target_train_prob)
    target_test_entr = entropy(target_test_prob)

    shadow_train_m_entr = m_entropy(shadow_train_prob, shadow_train_labels)
    shadow_test_m_entr = m_entropy(shadow_test_prob, shadow_test_labels)
    if target_train is not None:
        target_train_m_entr = m_entropy(target_train_prob, target_train_labels)
    else:
        target_train_m_entr = target_train_entr
    if target_test is not None:
        target_test_m_entr = m_entropy(target_test_prob, target_test_labels)
    else:
        target_test_m_entr = target_test_entr

    acc_corr = SVC_fit_predict(
        shadow_train_corr, shadow_test_corr, target_train_corr, target_test_corr
    )
    acc_conf = SVC_fit_predict(
        shadow_train_conf, shadow_test_conf, target_train_conf, target_test_conf
    )
    acc_entr = SVC_fit_predict(
        shadow_train_entr, shadow_test_entr, target_train_entr, target_test_entr
    )
    acc_m_entr = SVC_fit_predict(
        shadow_train_m_entr, shadow_test_m_entr, target_train_m_entr, target_test_m_entr
    )
    acc_prob = SVC_fit_predict(
        shadow_train_prob, shadow_test_prob, target_train_prob, target_test_prob
    )
    m = {
        "correctness": acc_corr,
        "confidence": acc_conf,
        "entropy": acc_entr,
        "m_entropy": acc_m_entr,
        "prob": acc_prob,
    }
    print(m)
    return m

def SVC_classifier(shadow_train, shadow_test):
    "Return a classifier model"
    n_shadow_train = shadow_train.shape[0]
    n_shadow_test = shadow_test.shape[0]

    X_shadow = (
        torch.cat([shadow_train, shadow_test])
        .cpu()
        .numpy()
        .reshape(n_shadow_train + n_shadow_test, -1)
    )
    Y_shadow = np.concatenate([np.ones(n_shadow_train), np.zeros(n_shadow_test)])
    
    clf = SVC(C=3, gamma="auto", kernel="rbf")
    clf.fit(X_shadow, Y_shadow)
    return clf

def SVC_classifiers(shadow_train, shadow_test, model):
    """Return a dict of classifiers model"""
    shadow_train_prob, shadow_train_labels = collect_prob(shadow_train, model)
    shadow_test_prob, shadow_test_labels = collect_prob(shadow_test, model)

    shadow_train_corr = (
        torch.argmax(shadow_train_prob, axis=1) == shadow_train_labels
    ).int()
    shadow_test_corr = (
        torch.argmax(shadow_test_prob, axis=1) == shadow_test_labels
    ).int()

    shadow_train_conf = torch.gather(shadow_train_prob, 1, shadow_train_labels[:, None])
    shadow_test_conf = torch.gather(shadow_test_prob, 1, shadow_test_labels[:, None])

    shadow_train_entr = entropy(shadow_train_prob)
    shadow_test_entr = entropy(shadow_test_prob)

    shadow_train_m_entr = m_entropy(shadow_train_prob, shadow_train_labels)
    shadow_test_m_entr = m_entropy(shadow_test_prob, shadow_test_labels)

    clfs = {
        "correctness": SVC_classifier(
            shadow_train_corr, shadow_test_corr
        ),
        "confidence": SVC_classifier(
            shadow_train_conf, shadow_test_conf
        ),
        "entropy": SVC_classifier(
            shadow_train_entr, shadow_test_entr
        ),
        "m_entropy": SVC_classifier(
            shadow_train_m_entr, shadow_test_m_entr
        ),
        "prob": SVC_classifier(
            shadow_train_prob, shadow_test_prob
        )
    }

    return clfs

def SVC_predict(clfs, target_loader, model):
    target_prob, target_labels = collect_prob(target_loader, model)
    target_corr = (
        torch.argmax(target_prob, axis=1) == target_labels
    ).int()
    target_conf = torch.gather(target_prob, 1, target_labels[:, None])
    target_entr = entropy(target_prob)
    target_m_entr = m_entropy(target_prob, target_labels)

    results = {}
    for metric, clf in clfs.items():
        if metric == "prob":
            X_target = target_prob.cpu().numpy().reshape(target_prob.shape[0], -1)
            pred = clf.predict(X_target)
            results[metric] = pred.mean()
        elif metric == "correctness":
            X_target = target_corr.cpu().numpy().reshape(target_corr.shape[0], -1)
            pred = clf.predict(X_target)
            results[metric] = pred.mean()
        elif metric == "confidence":
            X_target = target_conf.cpu().numpy().reshape(target_conf.shape[0], -1)
            pred = clf.predict(X_target)
            results[metric] = pred.mean()
        elif metric == "entropy":
            X_target = target_entr.cpu().numpy().reshape(target_entr.shape[0], -1)
            pred = clf.predict(X_target)
            results[metric] = pred.mean()
        elif metric == "m_entropy":
            X_target = target_m_entr.cpu().numpy().reshape(target_m_entr.shape[0], -1)
            pred = clf.predict(X_target)
            results[metric] = pred.mean()
        else:
            raise ValueError(f"Unknown metric: {metric}")
    return results