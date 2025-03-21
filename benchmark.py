import subprocess
from concurrent.futures import ThreadPoolExecutor
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

base_script = "python -u mlflow_forget.py"

dataset = ["cifar10", "cifar100"]
mask = ["model_SA_best.pth.tar"]
unlearn = ["NGPlus", "mask_NGPlus", "mix_NGPlus", "SRL", "mask_SRL", "mix_SRL", "SalUn", "FT"]
unlearn_epochs = ["1", "2", "5"]
beta = ["0.95"]
quantile = ["0.4", "0.5"]
archs = ["resnet18"]
seeds = ["0", "1"]

commands = [base_script
            + " --save_dir ./results/" + _dataset
            + " --mask ./results/" + _dataset + "/" + _seed + _mask
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
    
# print(new_commands)

base_commands = ["python -u main_baseline.py"
            + " --save_dir ./results/" + _dataset
            + " --arch " + _arch
            + " --data ./data"
            + " --dataset " + _dataset
            + " --seed " + _seed
            + " --epochs " + str(baseline_train_epochs[_dataset])
            for (_arch, _dataset, _seed) in itertools.product(archs, dataset, seeds)]

def run_command(cmd):
    """Exécute une commande et gère les erreurs"""
    try:
        print(f"Démarrage: {cmd}")
        result = subprocess.run(
            cmd.split(), 
            check=True,
            capture_output=True,
            text=True
        )
        print(f"Succès: {cmd}\nSortie:\n{result.stdout[:200]}...")  # Affiche les 200 premiers caractères
        return True
    except subprocess.CalledProcessError as e:
        print(f"Erreur dans {cmd}\nCode: {e.returncode}\nErreur: {e.stderr[:200]}...")
        return False
    except Exception as e:
        print(f"Exception inattendue: {str(e)}")
        return False

# Exécution parallèle (ajuster max_workers selon ton CPU)
with ThreadPoolExecutor(max_workers=1) as executor:
    results = executor.map(run_command, base_commands)

# Vérification finale
if all(results):
    print(f"\nToutes les {len(new_commands)} commandes ont réussi !")
else:
    print("\nCertaines commandes ont échoué, vérifie les logs ci-dessus.")
