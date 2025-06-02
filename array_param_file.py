import itertools
import os

# Liste de tes commandes sous forme de strings
baseline_train_epochs = {
    "cifar10": 100,
    "cifar100": 150,
    "tiny-imagenet": 1,
    "svhn": 200,
    "imagenet": 90,
    "imagenet100": 90
}

nums_index_to_replace = {
    "cifar10": {-1 : [2250, 4500, 22500],
                0 : [450, 2250, 3500]
                },
    "cifar100": {-1 : [2250, 4500, 22500],
                 0 : [45, 225, 350]
                }
}

base_script = "-u mlflow_forget.py"

dataset = ["cifar10"]
# unlearn = ["NGPlus", "mix_NGPlus", "SRL", "mix_SRL", "SalUn", "FT", "pSalUn"]
unlearn = ["NGPlus", "VarGrad", "SalGrad"]
unlearn_epochs = ["10"]
archs = ["resnet18"]
seeds = ["1", "2", "3", "4", "5"]
quantiles = ["0.3", "0,4", "0.5", "0,6", "0.7"]
class_to_replace = [-1, 0]

commands = [base_script
            + " --save_dir ./results/" + _dataset
            + " --mask ./results/" + _dataset + "/" + str(_class_to_replace) + "_" + _dataset + "_"  + _arch + "_" + _seed + "model.pth.tar" 
            + " --unlearn " + _unlearn
            + " --unlearn_epochs " + _unlearn_epochs
            + " --unlearn_lr 0.0001"
            + " --data ./data"
            + " --dataset " + _dataset
            + " --seed " + _seed
            + " --arch " + _arch
            + " --epochs " + str(baseline_train_epochs[_dataset])
            + " --num_indexes_to_replace " + str(_nums_index_to_replace)
            + " --class_to_replace " + str(_class_to_replace)
            + " --beta " + "0.95"
            for (_dataset, _unlearn, _unlearn_epochs, _seed, _arch, _class_to_replace) in itertools.product(dataset, unlearn, unlearn_epochs, seeds, archs, class_to_replace)
            for _nums_index_to_replace in nums_index_to_replace[_dataset][_class_to_replace]
]

need_quantile = ["VarGrad", "SalGrad"]
for command in commands:
    for _quantile in quantiles:
        if any(x in command for x in need_quantile):
            command = command + " --quantile " + _quantile

print(len(commands))

file_name = "params.txt"
with open(file_name, "w") as f:
    for commande in commands:
        f.write(commande + "\n")
