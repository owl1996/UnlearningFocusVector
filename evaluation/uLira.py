import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import norm
from tqdm import tqdm
import random

def sample_dataset(include, full_dataset, size=5000):
    """
    Args:
        include: tuple (x, y) to forcibly include in the sampled dataset
        full_dataset: list or dataset-like [(x1, y1), (x2, y2), ...]
        size: total number of samples in returned dataset

    Returns:
        sampled dataset (list of (x, y)), including the specified sample
    """
    assert size >= 1, "Dataset size must be at least 1"
    
    # Filter out all points identical to include
    rest = [(x_, y_) for x_, y_ in full_dataset
            if not (torch.equal(x_, include[0]) and y_ == include[1])]

    assert len(rest) >= size - 1, "Not enough examples to sample from"

    sampled = random.sample(rest, size - 1)
    sampled.append(include)
    random.shuffle(sampled)
    return sampled

def logit(p):
    eps = 1e-12
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))

def fit_gaussian(values):
    mu = np.mean(values)
    sigma = np.std(values) + 1e-8  # to avoid division by zero
    return mu, sigma

def u_lira_attack(
    x, y, model_under_test, train_algo, unlearn_algo,
    n_shadow=20, device='cpu'
):
    """
    U-LiRA attack on a single example (x, y)

    Args:
        x, y: torch tensors, single input and label
        model_under_test: model θ* to evaluate (PyTorch model)
        train_algo: function(D) → trained_model
        unlearn_algo: function(model, (x, y)) → unlearned_model
        n_shadow: number of shadow model pairs
        device: 'cuda' or 'cpu'

    Returns:
        p_member: estimated membership probability
    """

    phi_f = []
    phi_r = []

    x = x.unsqueeze(0).to(device)
    y = y.unsqueeze(0).to(device)

    for _ in tqdm(range(n_shadow), desc="Generating shadow models"):
        # Step 1: Sample dataset including (x, y)
        D_in = sample_dataset(include=(x, y))
        D_out = [ex for ex in D_in if not torch.equal(ex[0], x)]

        # Step 2: Train model with (x, y)
        theta_o = train_algo(D_in)

        # Step 3: Unlearn (x, y)
        theta_f = unlearn_algo(theta_o, (x, y))

        # Step 4: Retrain from scratch without (x, y)
        theta_r = train_algo(D_out)

        # Step 5: Evaluate φ(f(x;θ)_y) for each model
        for model, lst in [(theta_f, phi_f), (theta_r, phi_r)]:
            model.eval()
            with torch.no_grad():
                out = model(x)
                prob = F.softmax(out, dim=1)[0, y.item()].item()
                lst.append(logit(prob))

    # Step 6: Fit Gaussians
    mu_f, sigma_f = fit_gaussian(phi_f)
    mu_r, sigma_r = fit_gaussian(phi_r)

    # Step 7: Evaluate model_under_test on (x, y)
    model_under_test.eval()
    with torch.no_grad():
        out = model_under_test(x)
        p = F.softmax(out, dim=1)[0, y.item()].item()
        phi_star = logit(p)

    # Step 8: Likelihood ratio and membership prob
    pdf_f = norm.pdf(phi_star, mu_f, sigma_f)
    pdf_r = norm.pdf(phi_star, mu_r, sigma_r)
    p_member = pdf_f / (pdf_f + pdf_r)

    return p_member
