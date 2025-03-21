#!/bin/bash
#OAR -n my_array_jobs
#OAR -l /nodes=1/core=1/gpu=1,walltime=01:00:00
#OAR -O output_%jobid%_%arrayid%.log
#OAR -E error_%jobid%_%arrayid%.err

# Vérifier que le fichier de paramètres est bien passé en argument
if [ -z "$1" ]; then
    echo "Usage: $0 param_file"
    exit 1
fi

PARAM_FILE="$1"

# Vérifier si le fichier requirements.txt existe et installer les dépendances
if [ -f "requirements.txt" ]; then
    echo "Installation des dépendances Python depuis requirements.txt"
    pip install -r requirements.txt
else
    echo "requirements.txt non trouvé, aucune installation des dépendances"
fi

# Lire la ligne correspondant à l'index du job (OAR_ARRAY_INDEX commence à 1)
CMD=$(sed -n "${OAR_ARRAY_INDEX}p" "$PARAM_FILE")

# Vérifier si la commande est vide
if [ -z "$CMD" ]; then
    echo "Erreur: aucune commande trouvée à l'index $OAR_ARRAY_INDEX"
    exit 1
fi

# Exécuter la commande
echo "Exécution : $CMD"
eval $CMD
