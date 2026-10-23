import os
import sys
from pathlib import Path

ROOT_DIR = str(Path(__file__).resolve().parent.parent)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import keras
from keras import layers, models, applications, callbacks, optimizers


def build_googlenetv3(num_classes=4, input_shape=(224, 224, 3), weights=None, lr=1e-4):
    """Construit et compile le modèle GoogLeNet v3 (InceptionV3) pour la classification."""
    base_model = applications.InceptionV3(
        weights=weights,
        include_top=False,
        input_shape=input_shape
    )
    base_model.trainable = False

    inputs = keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="GoogLeNetV3")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model, base_model


def export_model_summary(model, filepath="models/googlenetv3_summary.txt"):
    """Exporte le résumé du modèle dans un fichier texte."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        model.summary(print_fn=lambda line: f.write(line + "\n"))
    print(f"Résumé du modèle exporté dans : {filepath}")


def setup_fine_tuning(model, base_model, unfreeze_from=-50, lr=1e-5):
    """Dégèle les dernières couches du backbone et recompile avec un learning rate réduit."""
    base_model.trainable = True

    for layer in base_model.layers[:unfreeze_from]:
        layer.trainable = False

    for layer in base_model.layers[unfreeze_from:]:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def get_callbacks(checkpoint_path="models/best_googlenetv3.keras"):
    """Retourne les callbacks standards (EarlyStopping, Checkpoint, ReduceLROnPlateau)."""
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    return [
        callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor="val_loss", save_best_only=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, min_lr=1e-7),
    ]


def train_phase1_transfer(model, train_ds, val_ds, epochs=10, callbacks_list=None):
    """Entraîne la tête de classification avec la base gelée."""
    return model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks_list)


def train_phase2_finetune(
    model,
    base_model,
    train_ds,
    val_ds,
    initial_epoch=10,
    total_epochs=25,
    lr=1e-5,
    callbacks_list=None
):
    """Fine-tune les dernières couches du modèle."""
    setup_fine_tuning(model, base_model, lr=lr)
    return model.fit(
        train_ds,
        validation_data=val_ds,
        initial_epoch=initial_epoch,
        epochs=total_epochs,
        callbacks=callbacks_list
    )


def evaluate_classifier(model, test_ds):
    """Évalue le modèle sur le jeu de test."""
    return model.evaluate(test_ds)


if __name__ == "__main__":
    model, base_model = build_googlenetv3()
    export_model_summary(model)
