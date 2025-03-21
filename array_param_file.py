import itertools

# Liste de tes commandes sous forme de strings
baseline_train_epochs = {
    "cifar10": 60,
    "cifar100": 120,
    "tiny-imagenet": 200,
    "svhn": 200,
    "imagenet": 90,
    "imagenet100": 90
}

base_script = "-u mlflow_forget.py"

dataset = ["cifar10", "cifar100"]
mask = ["model_SA_best.pth.tar"]
unlearn = ["NGPlus", "mask_NGPlus", "mix_NGPlus", "SRL", "mask_SRL", "mix_SRL", "SalUn", "FT"]
unlearn_epochs = ["5"]
beta = ["0.95"]
quantile = ["0.4", "0.5", "0.6"]
archs = ["resnet18"]
seeds = ["0", "1", "2"]

commands = [base_script
            + " --save_dir ./results/" + _dataset
            + " --mask ./results/" + _dataset + "/" + str(_seed) + _mask
            + " --unlearn " + _unlearn
            + " --unlearn_epochs " + _unlearn_epochs
            + " --unlearn_lr 0.1"
            + " --data ./data"
            + " --dataset " + _dataset
            + " --seed " + _seed
            for (_dataset, _mask, _unlearn, _unlearn_epochs, _seed) in itertools.product(dataset, mask, unlearn, unlearn_epochs, seeds) 
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
    


base_commands = ["-u main_baseline.py"
            + " --save_dir ./results/" + _dataset
            + " --arch " + _arch
            + " --data ./data"
            + " --dataset " + _dataset
            + " --seed " + _seed
            + " --epochs " + str(baseline_train_epochs[_dataset])
            for (_arch, _dataset, _seed) in itertools.product(archs, dataset, seeds)]



commands = base_commands
print(commands)

file_name = "params.txt"
with open(file_name, "w") as f:
    for commande in commands:
        f.write(commande + "\n")