<div align="center">

# Masking Gradients Can Improve Machine Unlearning

## Abstract
In response to recent data regulation requirements, machine unlearning (MU) has emerged as a critical process to remove the influence of specific examples from a given model. Although exact unlearning can be achieved through complete model retraining using the remaining dataset, the associated computational costs have driven the development of efficient, approximate unlearning techniques. 
## Requirements
```
conda env create -f environment.yml
```

## Code Structure
The source code is organized as follows:

```evaluation```: contains MIA evaluation code.

```models```: contains the model definitions.

```utils.py```: contains the utility functions. 

```main_baseline.py```: contains the code for training a base model to unlearn. 

```main_forget.py```: contains the main executable code for unlearning. 

```mlflow_forget.py```: contains the main executable code for unlearning with mlflow tracking.

```main_backdoor.py```: contains the main executable code for backdoor cleanse.


## Commands

### Baseline

#### Cifar10

```python -u main_baseline.py --data ./data --dataset cifar10 --arch resnet18 --save_dir ./results/cifar10 --epochs 10 --num_indexes_to_replace 4500```

#### Cifar 100

```python -u main_baseline.py --data ./data --dataset cifar100 --arch resnet18 --save_dir ./results/cifar100 --epochs 10 -num_indexes_to_replace 450 --num_classes 100```

#### Benchmarking

```python benchmark.py```

```
