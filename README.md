# Projet : Cycle de Vie d'un Modèle Deep Learning

Ce projet illustre le cycle de vie complet d'un modèle de classification d'images (MNIST), de l'entraînement au déploiement via une API conteneurisée avec Docker.

## Technologies utilisées

*   **Python**
*   **TensorFlow / Keras** : Pour la construction et l'entraînement du modèle.
*   **MLflow** : Pour le suivi des expérimentations.
*   **Flask** : Pour la création de l'API web.
*   **Docker** : Pour la conteneurisation de l'application.

## Structure du Projet

```
.
├── train_model.py      # Script pour entraîner le modèle et le sauvegarder
├── app.py              # API Flask pour servir le modèle
├── requirements.txt    # Dépendances Python du projet
├── Dockerfile          # Instructions pour construire l'image Docker
└── mnist_model.h5      # Modèle entraîné (généré par train_model.py)
```

## Comment l'utiliser ?

### Prérequis

*   Python 3.8+
*   Docker

### 1. Installation des dépendances

Clonez le dépôt et installez les bibliothèques nécessaires :
```bash
git clone https://github.com/QByteSeeker/TP1_DL.git
cd TP1_DL
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Entraînement du modèle

Exécutez le script d'entraînement. Cela va créer le fichier `mnist_model.h5` et enregistrer les métriques avec MLflow.
```bash
python train_model.py
```
Pour visualiser les résultats de l'expérience, lancez l'interface de MLflow :
```bash
mlflow ui
```

### 3. Lancer l'API avec Docker

1.  **Construire l'image Docker :**
    ```bash
    docker build -t mnist-api .
    ```

2.  **Lancer le conteneur :**
    ```bash
    docker run -p 5000:5000 mnist-api
    ```

L'API est maintenant accessible à l'adresse `http://localhost:5000/predict`. Vous pouvez envoyer une requête POST avec une image 28x28 (aplatie en un vecteur de 784 pixels) pour obtenir une prédiction.