import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl
from yaml import safe_load

# Dossier contenant les runs MLflow (expérience 0)
mlruns_path = "mlruns/0"


# Liste pour stocker les données combinées
combined_data = []

# Parcours des sous-dossiers de runs
for run_id in os.listdir(mlruns_path):
    run_dir = os.path.join(mlruns_path, run_id)
    if not os.path.isdir(run_dir):
        continue

    # Dictionnaire pour stocker les données de ce run
    run_data = {"run_id": run_id}

    # Récupération des params
    params_path = os.path.join(run_dir, "params")
    if os.path.exists(params_path):
        for param_file in os.listdir(params_path):
            with open(os.path.join(params_path, param_file), "r") as f:
                value = f.read().strip()
                run_data[param_file] = value  # clé = nom du paramètre

    # Récupération des metrics (dernière valeur de chaque métrique)
    metrics_path = os.path.join(run_dir, "metrics")
    if os.path.exists(metrics_path):
        for metric_file in os.listdir(metrics_path):
            with open(os.path.join(metrics_path, metric_file), "r") as f:
                lines = [line.strip() for line in f if line.strip()]
                if lines:
                    # Format: step timestamp value (on prend la dernière ligne)
                    last_value = float(lines[-1].split(" ")[1])
                    run_data[metric_file] = last_value  # clé = nom de la métrique

    combined_data.append(run_data)

# Création du DataFrame combiné
df = pd.DataFrame(combined_data)

df.loc[
    df["unlearn"].str.contains("GradMask") & (df["quantile"].astype(float) == 0.5),
    "unlearn"
] = df["unlearn"].str.replace("GradMask", "ANDMask")


# Activer LaTeX
mpl.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "axes.labelsize": 12,
    "font.size": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

# Dictionnaire des noms en LaTeX
method_rename = {
    "SRL": "SRL",
    "SalUn": "SalUn",
    "SRGradMask": "SRL - GradMask",
    "SRGradFocus": "SRL - GradFocus",
    "SRANDMask": "SRL - ANDMask",
    "NANDMask": "NGPlus - ANDMask",
    "NGPlus": "NGPlus",
    "NGradFocus": "NGPlus - GradFocus",
    "NGradMask": "NGPlus - GradMask",
    "NGSalUn": "NGPlus - SalUn",
    "SRGradFocusEnsure" : "SRL - GradFocusSecure",
    "NGGradFocusEnsure" : "NGPlus - GradFocusSecure",
    "SRGradFocusOPT" : "SRL - FocusOPT",
    "NGGradFocusOPT" : "NGPlus - FocusOPT",
    "SRL_OPT" : "SRL - OPT",
    "NGPlus_OPT" : "NGPlus - OPT"
}


# Ajouter une colonne avec les noms latexifiés
df.insert(0, "Methods", df["unlearn"].map(method_rename).fillna(df["unlearn"]))

df.drop(columns=["unlearn", "run_id", "beta", "save_dir", "model", "unlearn_lr", "val"], inplace=True)

metric_rename = {
    "RTE": "RTE",
    "Fid": "FID",
    "relativeUA": "rUA (%)",
    "forget": "UA (%)",
    "retain": "RA (%)",
    "test": "TA (%)",
    "MIA_prob": "MIA prob",
    "MIA_confidence": "MIA confidence",
    "MIA_correctness": "MIA correctness",
    "MIA_entropy": "MIA entropy",
    "MIA_m_entropy": "MIA mix entropy",
    "unlearn_epochs": "Unlearn epochs"
}

# Modifier les noms de colonnes pour les métriques
df.rename(columns=metric_rename, inplace=True)

# convert all metrics to float
for col in df.columns:
    if col not in ["Methods", "num_indexes_to_replace", "dataset", "arch"]:
        # print(col)
        df[col] = df[col].astype(float)

df = df.dropna()

for col in df.columns:
    if col not in ["Methods", "num_indexes_to_replace", "dataset", "arch"]:
        # print(col)
        df[col] = df[col].astype(float)

df["Unlearn epochs"] = df["Unlearn epochs"].astype(int)
df["seed"] = df["seed"].astype(int)
df["class_to_replace"] = df["class_to_replace"].astype(int)

df.sort_values(by=["num_indexes_to_replace", "Methods", "Unlearn epochs"], inplace=True)

# Change orders of columns
df = df[["Methods", "num_indexes_to_replace", "class_to_replace", "Unlearn epochs", "RTE", "FID", "rUA (%)", "UA (%)", "RA (%)", "TA (%)",
         "MIA correctness","MIA confidence","MIA prob", "MIA entropy",
         "MIA mix entropy", "arch", "dataset", "seed"]]

df.to_csv("mlruns_parsed.csv", index=False)