import tensorflow as tf
from tensorflow import keras
import numpy as np
import mlflow
import mlflow.tensorflow


# 1. Paramètres d'entraînement (pour MLflow)

EPOCHS = 5
BATCH_SIZE = 128
DROPOUT_RATE = 0.2


# 2. Chargement du dataset MNIST

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalisation
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Flatten 28x28 → vecteur 784
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)


# 3. Lancement de la session MLflow

mlflow.tensorflow.autolog()  # Active le log automatique (optionnel mais pratique)

with mlflow.start_run():

    # Enregistrer les hyperparamètres
    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("dropout_rate", DROPOUT_RATE)


    # 4. Construction du modèle

    model = keras.Sequential([
        keras.layers.Dense(512, activation='relu', input_shape=(784,)),
        keras.layers.Dropout(DROPOUT_RATE),
        keras.layers.Dense(10, activation='softmax')
    ])

    # Compilation
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )


    # 5. Entraînement du modèle

    history = model.fit(
        x_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1
    )


    # 6. Évaluation

    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"Précision sur test : {test_acc:.4f}")

    # Enregistrer la métrique dans MLflow
    mlflow.log_metric("test_accuracy", test_acc)

  
    # 7. Sauvegarde du modèle
   
    model.save("mnist_model.h5")
    print("Modèle sauvegardé sous mnist_model.h5")

    # Log du modèle dans MLflow
    mlflow.keras.log_model(model, artifact_path="mnist-model")
