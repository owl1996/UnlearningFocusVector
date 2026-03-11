<div align="center">

# Exact Unlearning Approximations

[![Status](https://img.shields.io/badge/status-active-success.svg)](https://github.com/anonymous/unlearning-repo) [![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)

*Anonymized Source Code for NeurIPS Submission*

</div>

---

## 📖 Abstract

As machine learning models become increasingly integrated into real-world applications, concerns over data confidentiality, user privacy, and compliance with the European GDPR "right to be forgotten" have intensified. Because models can contain accessible artifacts of their potentially sensitive training data, the ability to remove the detectability of specific training features from an already-trained model has emerged as a key mechanism for model confidentiality. 
---

## 🏗️ Repository Structure

The architecture of this repository provides an extensible framework mimicking robust Machine Learning lifecycles:

- **`models/`**: Deep Neural Network definitions (e.g., ResNet, VGG).
- **`evaluation/`**: Specialized MU metric code calculating relative Unlearning Accuracy (`rUA`), Retained Accuracy (Fidelity/`FID`), and Membership Inference Attacks (`MIA`).
- **`unlearn/`**: The extensible library implementing the core MU algorithms (e.g., `OS_unlearn`, `LDA`, `fisher`, `VarGrad`).
- **`notebooks/`**: Analytical Jupyter notebooks streamlining tracking and data analysis.
  - `gaussian_study.ipynb`: Analyzes the multivariate normality hypothesis of the extracted penultimate features using Chi-Square Q-Q plots.
  - `benchmark.ipynb`: An automated benchmarking loop to programmatically aggregate tracking data directly off `mlflow_forget.py`.
  - `results/mlflow_results_study.ipynb`: Consolidates and visualizes MLflow performance outcomes (`rUA` vs `RTE` vs `FID`).
- **`results/`**: Execution directory where `checkpoint` and `.pth.tar` weights are automatically stored.
- **`mlruns/`**: Integrated MLFlow tracking environment.

---

## 🛠️ Installation & Requirements
> Ensure your system is running Python 3.8+ and contains accessible local dataset targets.

1. **Clone the anonymized repository**
2. **Install the unified requirements**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Usage Guide

### 1. Training the Initial & Exact Ideal Baselines
The `main_baseline.py` and `main_ideal.py` scripts act as initialization protocols for subsequent model unlearning comparisons.

**Cifar10 Baseline Example:**
```bash
python -u main_baseline.py --data ./data --dataset cifar10 --arch resnet18 --save_dir ./results/cifar10 --epochs 10 --num_indexes_to_replace 4500
```
**Cifar100 Baseline Example:**
```bash
python -u main_baseline.py --data ./data --dataset cifar100 --arch resnet18 --save_dir ./results/cifar100 --epochs 50 --num_indexes_to_replace 450 --num_classes 100
```

### 2. Performing Unlearning
To scrub the target dataset, utilize the unified execution file `mlflow_forget.py`.
- `--class_to_replace`: Provide `-1` to randomly unlearn from any class, or `0` for specific target sampling.
- `--num_indexes_to_replace`: Volume of indexes to forcibly prune.

**Example Command (VarGrad):**
```bash
python -u mlflow_forget.py --save_dir ./results/cifar10 --mask ./results/cifar10/1resnet18_ep100model_SA_best.pth.tar --unlearn VarGrad --unlearn_epochs 4 --unlearn_lr 0.0001 --data ./data --dataset cifar10 --seed 1 --arch resnet18 --epochs 100 --num_indexes_to_replace 450 --beta 0.9 --quantile 0.55 --class_to_replace -1
```

---

## 🧠 Core Methodology: Exact LDA Update (`LDA_update_unlearn`)

The theoretical centerpiece of the unlearning framework is the Exact LDA Update routine (`unlearn/LDA.py:LDA_update_unlearn`). It avoids full retraining by algebraically isolating the theoretical contribution of the forgotten dataset to the readout layer using Linear Discriminant Analysis principles.

This involves computing the optimal linear weight deviation $\Delta_W$ analytically without backpropagation.

### Update Workflow:
1. **Mean Tracking**: Extracts the average penultimate activations generated per-class across the *retain* set ($M_r$) and *forget* set ($M_u$), alongside aggregating the theoretical dataset global mean ($M_0$). This exposes the shift in activation means: $\Delta_M = M_0 - M_r$.
2. **Scatter Matrices Gathering**: Computes the within-class scatter matrix for the retained data ($\Sigma_r$) and the internal scatter of the forget set features specifically surrounding the theoretical global centroid $M_0$ bounds ($\Sigma_{u0}$).
3. **Structural Whitening Approximation ($\Delta_S$)**: Approximates the difference in covariance distributions caused by the targeted removal: $\Delta_S = \Sigma_{u0} + n_r \Delta_M^T \Delta_M$.
4. **Isolating the Delta**: An analytical constraint $B$ is composed specifying the gap in the pre-activations: $B = \Delta_M - W_{net} \Delta_S$.
5. **Conjugate Gradient Solver**: Solves the linear derivation bounds formulated to generate the exact network update $\Delta_{LDA} = (\Delta_M - W_{net} \Delta_S) \Sigma_r^{-1}$. An optimized batched CG iterative solver calculates this inverse matrix multiplication efficiently on-device natively.
6. **Direct Weight Scrubbing**: Instead of descending on proxy gradients, the final network classifier weights logically deduct this precise theoretical impact directly: $W_{new} = W_{net} - \Delta_{LDA}$.

---

## 📊 Benchmarking & MLFlow Evaluation

This repository natively interfaces automated testing frameworks alongside MLflow integration.

1. **Setup Tracking Target**: Ensure the `results/` folder initializes a valid subdirectory reflecting your target dataset.
2. **Parameters Mapping**: Configure your array parameters inside `array_param_file.py` and invoke it to establish constraints.
3. **Execution Arrays**: Sequentially execute the arguments populated in `base_params.txt` followed by the method constraints parsed in `params.txt`.
4. **Notebook Benchmarking**: Alternately, streamline the execution lifecycle dynamically utilizing the provided script generated at `notebooks/benchmark.ipynb` via Python subprocess piping.

### MLFlow Parsers
Logged MLruns are accessible natively under `/mlruns/`. To extract and utilize visualization pipelines natively:
```bash
python mlruns_parser.py
```
This formulates a localized readable `mlruns_parsed.csv` schema that feeds directly into `notebooks/results/mlflow_results_study.ipynb` for empirical visualizations linking runtimes (`RTE`) and performance (`rUA` vs `FID`).

---

## 🔗 Extensibility
Each MU algorithm and Add-On logic resides under the modular `unlearn/` ecosystem. To introduce novel MU mechanisms:
1. Generate the implementation subset file under `unlearn/MethodName.py`.
2. Map the execution constraints mapping via the dispatch API `unlearn/impl.py` and `unlearn/__init__.py`.

---

## 📝 Acknowledgements
> *This structure has been forked from parallel MU research efforts; specific original team crediting will occur unconditionally following the conclusions of the double-blind review process.*