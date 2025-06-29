<div align="center">

# Improving Unlearning with Model Updates Probably Aligned with Gradients

## Abstract
Machine learning models become increasingly integrated into real world applications. Concerns over data confidentiality, user privacy, and how to apply the European GDPR "right to be forgotten" \cite{Hoofnagle2019TheEU} have intensified.
Since models contain artifacts of potentially sensitive training data, the ability to remove the detectability of specific training data from a model has emerged as a key mechanism of model confidentiality. 
Bi-objective first-order approximate Machine Unlearning (MU) methods rely on computing gradients of a new utility loss $L_U$ over samples $\sD_U$ to update the parameters of a model that was already trained to minimize some cost loss $L_C$ over initial samples $\sD_C$.

This work aims to unify and improve bi-objectives first-order methods for approximate deep supervised MU through a single problem formulation on which we can plug efficient add-on techniques.


## Code Structure
The source code is organized as follows:

```evaluation```: contains rUA, FID and MIA evaluation code.

```models```: contains the model definitions.

```main_baseline.py```: contains the code for training a initial model to unlearn. 

```main_ideal.py```: contains the code for training a exact ideal model to unlearn w.r.t. the seed.

```mlflow_forget.py```: main executable to test the methods

## Requirements
```
pip install requirements.txt
```

## How to use (Example)
class_to_replace :  
    0 by default (the class in which take samples to unlearn)
    put -1 to unlearn from any class
num_indexes_to_replace : number of samples to unlearn

```python u_ mlflow_forget.py --save_dir ./results/cifar10 --mask ./results/cifar10/1resnet18_ep100model_SA_best.pth.tar --unlearn VarGrad --unlearn_epochs 4 --unlearn_lr 0.0001 --data ./data --dataset cifar10 --seed 1 --arch resnet18 --epochs 100 --num_indexes_to_replace 450 --beta 0.9 --quantile 0.55 --class_to_replace -1```

## Example Cifar10

```python -u main_baseline.py --data ./data --dataset cifar10 --arch resnet18 --save_dir ./results/cifar10 --epochs 10 --num_indexes_to_replace 4500```

## Example Cifar 100

```python -u main_baseline.py --data ./data --dataset cifar100 --arch resnet18 --save_dir ./results/cifar100 --epochs 50 --num_indexes_to_replace 450 --num_classes 100```

## Benchmarking
(0) ensure the folder results/ contains a folder named as your dataset (if not, create it).
(1) put your methods, parameters and hyper-parameters in array_param_file.py and run it.
(2) run every line of base_params.txt to initialize the models initial and exact ideal
(3) run every line of params.txt

Your results will be stored in the mlruns/ folder.
Your models will be stored in results/$your_dataset$/

## Parsing Results
To parse your results run the command :
```python mlruns_parser.py```
It creates a .csv file you can work on named 'mlruns_parsed.csv'.

## Methods and Add-On technique
Each method with or without add-on is coded following a certain scheme in the folder unlearn/
If you want to add a method, please make sure it goes well with unlearn/impl.py and unlearn/__init__.py