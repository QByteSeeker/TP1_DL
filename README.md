# TP2 : Amélioration des Réseaux de Neurones Profonds

Ce projet est la suite du TP1 et se concentre sur les techniques avancées pour améliorer la performance et la robustesse des modèles de Deep Learning.

## Objectifs et Techniques Explorées

*   **Diagnostic de Performance** : Analyse du biais et de la variance en utilisant des ensembles d'entraînement et de validation distincts.
*   **Régularisation** : Mise en œuvre de la régularisation L2 et du Dropout pour combattre le surapprentissage.
*   **Optimisation Avancée** : Comparaison des performances des optimiseurs Adam, RMSprop et SGD avec momentum.
*   **Normalisation** : Utilisation de la Batch Normalization pour accélérer et stabiliser l'entraînement.

Toutes les expériences sont suivies et comparées à l'aide de **MLflow**.

## Structure du Projet

```
.
├── run_experiments.py  # Script principal pour lancer toutes les expériences
├── requirements.txt    # Dépendances Python
└── report_tp2.pdf      # Rapport résumant les concepts et les résultats
```

## Comment l'utiliser ?

### Prérequis

*   Python 3.8+
*   Avoir installé les dépendances listées dans `requirements.txt`.

### 1. Installation des dépendances

Clonez le dépôt et installez les bibliothèques nécessaires :

```bash
git clone https://github.com/QByteSeeker/TP_DL.git
cd TP_DL
checkout tp2

python3 -m venv venv # Si aucun environnement virtuel n'est défini
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Lancer les Expériences

Exécutez le script principal. Cela va entraîner séquentiellement les différents modèles et enregistrer les résultats dans MLflow.
```bash
python run_experiments.py
```

### 3. Visualiser les Résultats

Pour comparer les performances des différents modèles, lancez l'interface utilisateur de MLflow dans votre terminal :
```bash
mlflow ui
```
Ouvrez votre navigateur à l'adresse `http://127.0.0.1:5000` pour analyser les courbes d'apprentissage, les métriques et les paramètres de chaque exécution.