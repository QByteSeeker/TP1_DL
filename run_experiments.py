import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
import numpy as np
import mlflow
import mlflow.tensorflow

# Préparation des données
def load_and_prepare_data():
    """Charge et prépare les données MNIST en ensembles d'entraînement, validation et test."""
    (x_train_full, y_train_full), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Normalisation
    x_train_full = x_train_full.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Redimensionnement
    x_train_full = x_train_full.reshape(-1, 784)
    x_test = x_test.reshape(-1, 784)

    # Création des ensembles de validation (dev) et d'entraînement
    x_val = x_train_full[54000:]
    y_val = y_train_full[54000:]
    x_train = x_train_full[:54000]
    y_train = y_train_full[:54000]
    
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

# Fonction pour construire le modèle
def build_model(use_l2=False, use_dropout=False, use_batch_norm=False):
    """Construit un modèle Keras avec des options pour la régularisation et la batch norm."""
    
    layers = [keras.layers.Input(shape=(784,))]
    
    # Couche dense avec régularisation L2 optionnelle
    l2_reg = regularizers.l2(0.001) if use_l2 else None
    layers.append(keras.layers.Dense(512, activation='relu', kernel_regularizer=l2_reg))

    # Batch Normalization optionnelle (Exercice 2.4)
    if use_batch_norm:
        layers.append(keras.layers.BatchNormalization())
    
    # Dropout optionnel
    if use_dropout:
        layers.append(keras.layers.Dropout(0.2)) # Taux de dropout du TP1
        
    layers.append(keras.layers.Dense(10, activation='softmax'))
    
    model = keras.Sequential(layers)
    return model

# Script principal pour lancer les expériences
if __name__ == "__main__":
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_and_prepare_data()

    # Expérience 1: Analyse Biais/Variance (Modèle de base)
    print("\n--- Démarrage de l'expérience 1: Modèle de base ---")
    with mlflow.start_run(run_name="Base_Model_Bias_Variance"):
        model_base = build_model()
        model_base.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        history = model_base.fit(
            x_train, y_train,
            epochs=5,
            batch_size=128,
            validation_data=(x_val, y_val)
        )
        
        test_loss, test_acc = model_base.evaluate(x_test, y_test)
        mlflow.log_params({"regularization": "none", "batch_norm": False, "optimizer": "adam"})
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("final_val_accuracy", history.history['val_accuracy'][-1])

    # Expérience 2: Application de la Régularisation (L2 + Dropout)
    print("\n--- Démarrage de l'expérience 2: Modèle avec Régularisation ---")
    with mlflow.start_run(run_name="Regularized_Model"):
        model_reg = build_model(use_l2=True, use_dropout=True)
        model_reg.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        history = model_reg.fit(
            x_train, y_train,
            epochs=5,
            batch_size=128,
            validation_data=(x_val, y_val)
        )
        
        test_loss, test_acc = model_reg.evaluate(x_test, y_test)
        mlflow.log_params({"regularization": "L2+Dropout", "batch_norm": False, "optimizer": "adam"})
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("final_val_accuracy", history.history['val_accuracy'][-1])
        
    # Expérience 3: Comparaison des Optimiseurs
    print("\n--- Démarrage de l'expérience 3: Comparaison des Optimiseurs ---")
    optimizers = {
        'SGD_with_momentum': keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
        'RMSprop': 'rmsprop',
        'Adam': 'adam'
    }

    for opt_name, optimizer_instance in optimizers.items():
        with mlflow.start_run(run_name=f"Optimizer_Comparison_{opt_name}"):
            print(f"  Entraînement avec l'optimiseur : {opt_name}")
            # On utilise le modèle régularisé comme base de comparaison
            model_opt = build_model(use_l2=True, use_dropout=True)
            model_opt.compile(optimizer=optimizer_instance, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            
            history = model_opt.fit(
                x_train, y_train,
                epochs=5,
                batch_size=128,
                validation_data=(x_val, y_val),
                verbose=0 # Moins de logs pour la boucle
            )
            
            test_loss, test_acc = model_opt.evaluate(x_test, y_test, verbose=0)
            print(f"  Accuracy finale pour {opt_name}: {test_acc:.4f}")
            mlflow.log_params({"regularization": "L2+Dropout", "batch_norm": False, "optimizer": opt_name})
            mlflow.log_metric("test_accuracy", test_acc)
            mlflow.log_metric("final_val_accuracy", history.history['val_accuracy'][-1])
    
    # Expérience 4: Ajout de la Batch Normalization
    print("\n--- Démarrage de l'expérience 4: Modèle avec Batch Normalization ---")
    with mlflow.start_run(run_name="Model_With_Batch_Norm"):
        model_bn = build_model(use_l2=True, use_dropout=True, use_batch_norm=True)
        model_bn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        history = model_bn.fit(
            x_train, y_train,
            epochs=5,
            batch_size=128,
            validation_data=(x_val, y_val)
        )
        
        test_loss, test_acc = model_bn.evaluate(x_test, y_test)
        mlflow.log_params({"regularization": "L2+Dropout", "batch_norm": True, "optimizer": "adam"})
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("final_val_accuracy", history.history['val_accuracy'][-1])

    print("\nToutes les expériences sont terminées. Lancez 'mlflow ui' pour voir les résultats.")