import itertools
import os

# Liste de tes commandes sous forme de strings
baseline_train_epochs = {
    "cifar10": 100,
    "cifar100": 120,
    "tiny-imagenet": 1,
    "svhn": 200,
    "imagenet": 90,
    "imagenet100": 90
}

base_script = "-u mlflow_forget.py"

dataset = ["cifar100"]
unlearn = ["NGPlus", "mix_NGPlus", "SRL", "mix_SRL", "SalUn", "FT"]
unlearn_epochs = ["1", "5", "10"]
beta = ["0.9", "0.95"]
quantile = ["0.3", "0.5"]
archs = ["resnet18", "vgg16_bn"]
seeds = ["0", "1", "2"]

commands = [base_script
            + " --save_dir ./results/" + _dataset
            + " --mask ./results/" + _dataset + "/" + _seed + _arch + "_ep" + str(baseline_train_epochs[_dataset]) + "model_SA_best.pth.tar"
            + " --unlearn " + _unlearn
            + " --unlearn_epochs " + _unlearn_epochs
            + " --unlearn_lr 0.1"
            + " --data ./data"
            + " --dataset " + _dataset
            + " --seed " + _seed
            + " --arch " + _arch
            + " --epochs " + str(baseline_train_epochs[_dataset])
            for (_dataset, _unlearn, _unlearn_epochs, _seed, _arch) in itertools.product(dataset, unlearn, unlearn_epochs, seeds, archs) 
]

new_commands = []
for command in commands:
    if ("SRL" in command) or ("SalUn" in command) or ("NGPlus" in command):
        commands.remove(command)
        for _beta in beta:
            new_command = command + " --beta " + _beta
            if ("SalUn" in command) or ("mix" in command):
                for _quantile in quantile:
                    new_command_ = new_command + " --quantile " + _quantile
                    new_commands.append(new_command_)
            else:
                new_commands.append(new_command)
    else:
        new_commands.append(command)
    
# print(new_commands)

base_commands = ["-u main_baseline.py"
            + " --save_dir ./results/" + _dataset
            + " --arch " + _arch
            + " --data ./data"
            + " --dataset " + _dataset
            + " --seed " + _seed
            + " --epochs " + str(baseline_train_epochs[_dataset])
            for (_arch, _dataset, _seed) in itertools.product(archs, dataset, seeds)]

ideal_commands = ["-u mlflow_forget.py"
            + " --save_dir ./results/" + _dataset
            + " --mask ./results/" + _dataset + "/" + _seed + _arch + "_ep" + str(baseline_train_epochs[_dataset]) + "model_SA_best.pth.tar"
            + " --unlearn ideal"
            + " --unlearn_epochs " + str(baseline_train_epochs[_dataset])
            + " --unlearn_lr 0.1"
            + " --data ./data"
            + " --dataset " + _dataset
            + " --seed " + _seed
            + " --arch " + _arch
            + " --epochs " + str(baseline_train_epochs[_dataset])
            for (_arch, _dataset, _seed) in itertools.product(archs, dataset, seeds)
            if "ideal" + "_uep" + str(baseline_train_epochs[_dataset]) + "_s" + _seed + _arch + "_ep" + str(baseline_train_epochs[_dataset]) + "checkpoint.pth.tar" not in os.listdir("./results/" + _dataset)]

file_name = "base_params.txt"

with open(file_name, "w") as f:
    for commande in base_commands:
        f.write(commande + "\n")

file_name = "ideal_params.txt"

with open(file_name, "w") as f:
    for commande in ideal_commands:
        f.write(commande + "\n")

nothing_commands = [base_script
            + " --save_dir ./results/" + _dataset
            + " --mask ./results/" + _dataset + "/" + _seed + _arch + "_ep" + str(baseline_train_epochs[_dataset]) + "model_SA_best.pth.tar"
            + " --unlearn nothing"
            + " --unlearn_epochs 1"
            + " --unlearn_lr 0.1"
            + " --data ./data"
            + " --dataset " + _dataset
            + " --seed " + _seed
            + " --arch " + _arch
            + " --epochs " + str(baseline_train_epochs[_dataset])
            for (_dataset, _seed, _arch) in itertools.product(dataset, seeds, archs)
]

ideal_nothing_commands =  [base_script
            + " --save_dir ./results/" + _dataset
            + " --mask ./results/" + _dataset + "/ideal" + "_uep" + str(baseline_train_epochs[_dataset]) + "_s" + _seed + _arch + "_ep" + str(baseline_train_epochs[_dataset])
            + " --unlearn nothing"
            + " --unlearn_epochs 1"
            + " --unlearn_lr 0.1"
            + " --data ./data"
            + " --dataset " + _dataset
            + " --seed " + _seed
            + " --arch " + _arch
            + " --epochs " + str(baseline_train_epochs[_dataset])
            for (_dataset, _seed, _arch) in itertools.product(dataset, seeds, archs) 
]

print(nothing_commands + ideal_nothing_commands)

file_name = "params.txt"
with open(file_name, "w") as f:
    for commande in new_commands + nothing_commands + ideal_nothing_commands:
        f.write(commande + "\n")
