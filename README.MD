<div align="center">

# Masking Gradients Can Improve Machine Unlearning

## Abstract
In response to recent data regulation requirements, machine unlearning (MU) has emerged as a critical process to remove the influence of specific examples from a given model. Although exact unlearning can be achieved through complete model retraining using the remaining dataset, the associated computational costs have driven the development of efficient, approximate unlearning techniques. 


## Code Structure
The source code is organized as follows:

```evaluation```: contains MIA evaluation code.

```models```: contains the model definitions.

```utils.py```: contains the utility functions. 

```main_baseline.py```: contains the code for training a base model to unlearn. 

```main_forget.py```: contains the main executable code for unlearning. 

```mlflow_forget.py```: contains the main executable code for unlearning with mlflow tracking.

```main_backdoor.py```: contains the main executable code for backdoor cleanse.

## Requirements
```
pip install requirements.txt
```
## Commands

### Baseline

#### How to use (Example)
class_to_replace :  
    0 by default (the class in which take samples to unlearn)
    put -1 to unlearn from any class
num_indexes_to_replace : number of samples to unlearn

```python u_ mlflow_forget.py --save_dir ./results/cifar10 --mask ./results/cifar10/1resnet18_ep100model_SA_best.pth.tar --unlearn VarGrad --unlearn_epochs 4 --unlearn_lr 0.0001 --data ./data --dataset cifar10 --seed 1 --arch resnet18 --epochs 100 --num_indexes_to_replace 450 --beta 0.9 --quantile 0.55 --class_to_replace -1```

#### Cifar10

```python -u main_baseline.py --data ./data --dataset cifar10 --arch resnet18 --save_dir ./results/cifar10 --epochs 10 --num_indexes_to_replace 4500```

#### Cifar 100

```python -u main_baseline.py --data ./data --dataset cifar100 --arch resnet18 --save_dir ./results/cifar100 --epochs 50 --num_indexes_to_replace 450 --num_classes 100```

#### Benchmarking

```python benchmark.py```