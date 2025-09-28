# Projet : Cycle de Vie des Modèles de Deep Learning

Ce dépôt GitHub contient les travaux pratiques réalisés dans le cadre du cours sur l'ingénierie des modèles de Deep Learning. Il couvre le cycle de vie complet d'un modèle, depuis sa conception initiale jusqu'à son amélioration et son déploiement.

Le projet est divisé en deux parties, chacune résidant sur sa propre branche Git.

## Structure du Dépôt

Pour isoler le travail de chaque TP, ce dépôt utilise des branches Git :

* **`main`** : Cette branche, qui contient cette présentation générale du projet.
* **`tp1`** : Contient tout le code et les rapports pour le premier TP, axé sur la création et le déploiement d'un modèle de base.
* **`tp2`** : Contient le code et les rapports pour le second TP, qui se concentre sur l'amélioration et l'optimisation de ce modèle.

---

## Contenu des Travaux Pratiques

### Branche `tp1` : De la Conception au Déploiement d'un Modèle

Cette première partie aborde les étapes fondamentales pour mettre en production un modèle de Deep Learning.

*   **Construction** d'un réseau de neurones avec Keras pour la classification sur MNIST.
*   **Versionnement** du code avec Git et GitHub.
*   **Suivi des expérimentations** avec MLflow.
*   **Création d'une API web** avec Flask pour servir les prédictions du modèle.
*   **Conteneurisation** de l'application avec Docker pour un déploiement reproductible.

### Branche `tp2` : Amélioration des Réseaux de Neurones Profonds

Cette seconde partie explore les techniques avancées pour améliorer la performance et la robustesse du modèle initial.

*   **Analyse de la performance** du modèle (biais et variance).
*   **Application de techniques de régularisation** (L2, Dropout) pour lutter contre le surapprentissage.
*   **Comparaison d'optimiseurs avancés** (Adam, RMSprop, SGD avec momentum).
*   **Utilisation de la Batch Normalization** pour accélérer et stabiliser l'entraînement.

---

## Comment Accéder au Code

Pour consulter le code de chaque TP, utilisez les commandes Git suivantes depuis votre terminal après avoir cloné le dépôt :

```bash
# Pour voir le code du TP1
git checkout tp1
```

```bash
# Pour voir le code du TP2
git checkout tp2
```

```bash
# Pour revenir à cette page d'accueil
git checkout main
```

## Technologies Principales

*   TensorFlow / Keras
*   MLflow
*   Flask
*   Docker
*   Git / GitHub