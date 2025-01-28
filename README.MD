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

```main_imp.py```: contains the code for training and pruning. 

```main_forget.py```: contains the main executable code for unlearning. 

```main_backdoor.py```: contains the main executable code for backdoor cleanse.


## Commands

### Baseline

#### INIT

```python -u main_baseline.py --data ./data --dataset $data --arch $arch  --save_dir ${save_dir} --epochs ${epochs} --num_workers 8```

#### Benchmarking

```python benchmark.py```

### Unlearning

#### Retrain

```python -u main_forget.py --save_dir ${save_dir} --mask ${mask_path} --unlearn retrain --num_indexes_to_replace 4500 --unlearn_epochs 160 --unlearn_lr 0.1```

#### FT

```python -u main_forget.py --save_dir ${save_dir} --mask ${mask_path} --unlearn FT --num_indexes_to_replace 4500 --unlearn_lr 0.01 --unlearn_epochs 10```

#### GA

```python -u main_forget.py --save_dir ${save_dir} --mask ${mask_path} --unlearn GA --num_indexes_to_replace 4500 --unlearn_lr 0.0001 --unlearn_epochs 5```

#### FF

```python -u main_forget.py --save_dir ${save_dir} --mask ${mask_path} --unlearn fisher_new --num_indexes_to_replace 4500 --alpha ${alpha}```

#### IU

```python -u main_forget.py --save_dir ${save_dir} --mask ${mask_path} --unlearn wfisher --num_indexes_to_replace 4500 --alpha ${alpha}```

#### $\ell_1$-sparse

```python -u main_forget.py --save_dir ${save_dir} --mask ${mask_path} --unlearn FT_prune --num_indexes_to_replace 4500 --alpha ${alpha} --unlearn_lr 0.01 --unlearn_epochs 10```

### Trojan model cleanse

```python -u main_backdoor.py --save_dir ${save_dir} --mask ${mask_path} --unlearn FT --num_indexes_to_replace 4500```

## BibTeX
If you find this repository or the ideas presented in our paper useful, please consider citing.
```
@inproceedings{jia2023model,
  title={Model Sparsity Can Simplify Machine Unlearning},
  author={Jia, Jinghan and Liu, Jiancheng and Ram, Parikshit and Yao, Yuguang and Liu, Gaowen and Liu, Yang and Sharma, Pranay and Liu, Sijia},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
  year={2023}
}
```
