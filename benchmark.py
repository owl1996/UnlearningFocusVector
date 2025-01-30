import subprocess
from concurrent.futures import ThreadPoolExecutor
import itertools

# Liste de tes commandes sous forme de strings

base_script = "python -u mlflow_forget.py"

dataset = ["cifar10" , "cifar100"]
mask = ["/0model_SA_best.pth.tar"]
unlearn = ["NGPlus", "mask_NGPlus", "SRL", "mask_SRL", "SalUn", "FT"]
unlearn_epochs = ["1", "2", "5", "10"]

commands = [base_script
            + " --save_dir ./results/" + _dataset
            + " --mask ./results/" + _dataset + _mask
            + " --unlearn " + _unlearn
            + " --unlearn_epochs " + _unlearn_epochs
            + " --unlearn_lr 0.1"
            + " --data ./data"
            + " --dataset " + _dataset
            for (_dataset, _mask, _unlearn, _unlearn_epochs) in itertools.product(dataset, mask, unlearn, unlearn_epochs)
]

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
    results = executor.map(run_command, commands)

# Vérification finale
if all(results):
    print("\nToutes les commandes ont réussi !")
else:
    print("\nCertaines commandes ont échoué, vérifie les logs ci-dessus.")
