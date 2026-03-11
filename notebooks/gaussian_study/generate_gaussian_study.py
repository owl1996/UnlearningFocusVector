import nbformat as nbf

# Create a new Jupyter notebook
nb = nbf.v4.new_notebook()

# Define the cells
cells = []

# Title and Imports
cells.append(nbf.v4.new_markdown_cell("# Feature Gaussianity Study (NeurIPS Format)\nInvestigating the per-class Gaussian hypothesis of penultimate features of a trained ResNet on CIFAR-10."))

code_imports = """
import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA

# NeurIPS formatting for matplotlib (without requiring external tex installation)
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "serif"],
    "mathtext.fontset": "stix",  # STIX fonts are math fonts that look similar to Times
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.titlesize": 12
})
"""
cells.append(nbf.v4.new_code_cell(code_imports))

# Model and Data Loading
cells.append(nbf.v4.new_markdown_cell("## 1. Loader functions and Setup\nLoad the CIFAR-10 train set and the trained `results/cifar10/cifar10_resnet18_1model.pth.tar` model."))

code_setup = """
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Import model architecture
import sys
sys.path.append('.')
from models import model_dict

# Load dataset (CIFAR-10 train set without augmentation for feature evaluation)
transform = list(model_dict['resnet18'](num_classes=10).values())[0] if isinstance(model_dict['resnet18'](num_classes=10), dict) else transforms.Compose([transforms.ToTensor()])
# A little hack to get the transform used in main_baseline/utils.py 
# We'll just define basic ToTensor since ResNet normalizes internally in this project.
transform = transforms.Compose([transforms.ToTensor()])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=256, shuffle=False, num_workers=2)

# Load model
model = model_dict['resnet18'](num_classes=10)
checkpoint_path = './results/cifar10/cifar10_resnet18_1model.pth.tar'
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    print("Model loaded successfully.")
else:
    print(f"Error: Model not found at {checkpoint_path}")

model = model.to(device)
model.eval()
"""
cells.append(nbf.v4.new_code_cell(code_setup))

# Feature Extraction
cells.append(nbf.v4.new_markdown_cell("## 2. Feature Extraction\nUse a forward hook to extract penultimate features."))

code_extract = """
def get_penultimate_features(model, dataloader):
    features_list = []
    labels_list = []
    
    try:
        last_layer = model.fc
    except AttributeError:
        last_layer = model.classifier[-1]

    def hook_fn(module, input, output):
        features_list.append(input[0].detach().cpu())

    handle = last_layer.register_forward_hook(hook_fn)

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            _ = model(inputs)
            labels_list.append(targets.cpu())

    handle.remove()
    
    return torch.cat(features_list, dim=0), torch.cat(labels_list, dim=0)

print("Extracting features...")
features, labels = get_penultimate_features(model, trainloader)
print(f"Extracted features shape: {features.shape}")
"""
cells.append(nbf.v4.new_code_cell(code_extract))

# Experiment 1
cells.append(nbf.v4.new_markdown_cell("## 3. Multivariate Gaussianity Analysis\nAssess Gaussianity using squared Mahalanobis distances against a $\\chi^2$ distribution."))

code_exp1 = """
def analyze_class_mahalanobis(features, labels, class_idx=0, n_components=50):
    # Get features for specific class
    class_features = features[labels == class_idx].numpy()
    
    # --- Original Features ---
    # Compute mean and covariance
    mean = np.mean(class_features, axis=0)
    # Add small epsilon to diagonal for numerical stability (regularization)
    cov = np.cov(class_features, rowvar=False) + np.eye(class_features.shape[1]) * 1e-4 
    
    try:
        inv_cov = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        inv_cov = np.linalg.pinv(cov)
        
    # Compute squared Mahalanobis distances
    diff = class_features - mean
    mahalanobis_sq = np.sum(np.dot(diff, inv_cov) * diff, axis=1)
    
    # Theoretical quantiles of Chi-square distribution
    d_freedom = class_features.shape[1]
    theoretical_quantiles = stats.chi2.ppf((np.arange(1, len(mahalanobis_sq) + 1) - 0.5) / len(mahalanobis_sq), df=d_freedom)
    
    # --- PCA Reduced Features ---
    pca = PCA(n_components=n_components)
    class_features_pca = pca.fit_transform(class_features)
    
    mean_pca = np.mean(class_features_pca, axis=0)
    cov_pca = np.cov(class_features_pca, rowvar=False) + np.eye(class_features_pca.shape[1]) * 1e-4
    
    try:
        inv_cov_pca = np.linalg.inv(cov_pca)
    except np.linalg.LinAlgError:
        inv_cov_pca = np.linalg.pinv(cov_pca)
        
    diff_pca = class_features_pca - mean_pca
    mahalanobis_sq_pca = np.sum(np.dot(diff_pca, inv_cov_pca) * diff_pca, axis=1)
    
    d_freedom_pca = class_features_pca.shape[1]
    theoretical_quantiles_pca = stats.chi2.ppf((np.arange(1, len(mahalanobis_sq_pca) + 1) - 0.5) / len(mahalanobis_sq_pca), df=d_freedom_pca)
    

    # Plot Q-Q plot
    fig, axes = plt.subplots(1, 2, figsize=(7, 3)) # Wider figure for two subplots
    
    # Subplot 1: Original Features
    ax = axes[0]
    ax.scatter(theoretical_quantiles, np.sort(mahalanobis_sq), s=5, alpha=0.6, color='#1f77b4', edgecolors='none')
    max_val = max(np.max(theoretical_quantiles), np.max(mahalanobis_sq))
    ax.plot([0, max_val], [0, max_val], 'r--', linewidth=1.5)
    ax.set_xlabel('Theoretical $\chi^2$ Quantiles')
    ax.set_ylabel('Empirical $D_M^2$')
    ax.set_title(f'Original Features ($d={class_features.shape[1]}$)')
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # Subplot 2: PCA Reduced Features
    ax = axes[1]
    ax.scatter(theoretical_quantiles_pca, np.sort(mahalanobis_sq_pca), s=5, alpha=0.6, color='#ff7f0e', edgecolors='none')
    max_val_pca = max(np.max(theoretical_quantiles_pca), np.max(mahalanobis_sq_pca))
    ax.plot([0, max_val_pca], [0, max_val_pca], 'r--', linewidth=1.5)
    ax.set_xlabel('Theoretical $\chi^2$ Quantiles')
    ax.set_title(f'PCA Reduced Features ($d={n_components}$)')
    ax.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(f'fig_mahalanobis_qq_class_{class_idx}_compared.pdf', bbox_inches='tight')
    plt.show()

# Analyze class 0 (airplane)
analyze_class_mahalanobis(features, labels, class_idx=0)
"""
cells.append(nbf.v4.new_code_cell(code_exp1))


# Experiment 2
cells.append(nbf.v4.new_markdown_cell("## 4. 1D Projections Analysis\nAnalyze Gaussianity along principal components since multivariate Gaussian projections must be univariate Gaussian."))

code_exp2 = """
def analyze_1d_projections(features, labels, class_idx=0):
    class_features = features[labels == class_idx].numpy()
    
    # Projection onto top 2 Principal Components
    pca = PCA(n_components=2)
    proj_features = pca.fit_transform(class_features)
    
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.5))
    
    for i in range(2):
        data = proj_features[:, i]
        
        # Histograms
        ax = axes[i]
        n, bins, patches = ax.hist(data, bins=30, density=True, alpha=0.6, color='steelblue', edgecolor='black', linewidth=0.5)
        
        # Overlay Gaussian
        mu, std = stats.norm.fit(data)
        xmin, xmax = ax.get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = stats.norm.pdf(x, mu, std)
        ax.plot(x, p, 'r', linewidth=1.5, label='Gaussian Fit')
        
        ax.set_title(f'Projection on PC {i+1}')
        ax.set_xlabel('Feature Value')
        if i == 0:
            ax.set_ylabel('Density')
        ax.legend(loc='best')
        ax.grid(True, linestyle=':', alpha=0.6)
        
    plt.tight_layout()
    plt.savefig(f'fig_1d_projections_class_{class_idx}.pdf', bbox_inches='tight')
    plt.show()

analyze_1d_projections(features, labels, class_idx=0)
"""
cells.append(nbf.v4.new_code_cell(code_exp2))

# Assign cells to notebook and save
nb.cells = cells
with open('gaussian_study.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Jupyter notebook 'gaussian_study.ipynb' created successfully.")
