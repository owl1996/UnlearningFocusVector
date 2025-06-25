import os

def evaluate(filepath):
    print(f"Évaluation du modèle : {filepath}")
    # Place ici ta vraie logique d'évaluation

# Dossier principal
base_dir = 'results/'
eval_file_path = os.path.join(base_dir, 'eval.txt')

# Lecture ou création du fichier eval.txt
if os.path.exists(eval_file_path):
    with open(eval_file_path, 'r') as f:
        already_evaluated = set(line.strip() for line in f)
else:
    already_evaluated = set()

# Ouverture en mode ajout
with open(eval_file_path, 'a') as eval_file:
    for subfolder in os.listdir(base_dir):
        subfolder_path = os.path.join(base_dir, subfolder)

        if os.path.isdir(subfolder_path):
            for item in os.listdir(subfolder_path):
                if item.startswith("ideal") and item.endswith("model.pth.tar"):
                    if item not in already_evaluated:
                        evaluate(item)
                        eval_file.write(item + '\n')
                        eval_file.flush()  # écrit immédiatement
                    else:
                        print(f"Déjà évalué : {item}")
