import os
import sys
import argparse
from pathlib import Path

ROOT_DIR = str(Path(__file__).resolve().parent.parent)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.classification.data_loader import load_data, prepare_data_pipeline
from src.classification.classifier import (
    build_googlenetv3,
    get_callbacks,
    train_phase1_transfer,
    train_phase2_finetune,
    evaluate_classifier,
    export_model_summary,
)


def run_training(
    classification_dir="data/classification",
    checkpoint_path="models/best_googlenetv3.keras",
    epochs_p1=10,
    epochs_p2=25,
    batch_size=32,
    img_size=(224, 224),
    seed=42,
    prepare_first=False
):
    """Exécute l'entraînement complet (Phase 1 Transfer Learning + Phase 2 Fine-Tuning)."""
    if prepare_first or not os.path.exists(os.path.join(classification_dir, "train")):
        print("Préparation du pipeline de données...")
        train_ds, val_ds, test_ds, class_names = prepare_data_pipeline(
            output_dir=classification_dir,
            img_size=img_size,
            batch_size=batch_size,
            seed=seed
        )
    else:
        print(f"Chargement des données depuis : {classification_dir}")
        train_ds, val_ds, test_ds, class_names = load_data(
            output_dir=classification_dir,
            img_size=img_size,
            batch_size=batch_size,
            seed=seed
        )

    print(f"Classes détectées ({len(class_names)}) : {class_names}")

    # 1. Construction du modèle
    model, base_model = build_googlenetv3(
        num_classes=len(class_names),
        input_shape=(img_size[0], img_size[1], 3),
        weights="imagenet",
        lr=1e-4
    )
    export_model_summary(model)

    callbacks_list = get_callbacks(checkpoint_path=checkpoint_path)

    # 2. Phase 1 : Transfer learning (base gelée)
    print(f"\n--- Démarrage Phase 1 (Transfer Learning : {epochs_p1} époques) ---")
    train_phase1_transfer(
        model=model,
        train_ds=train_ds,
        val_ds=val_ds,
        epochs=epochs_p1,
        callbacks_list=callbacks_list
    )

    # 3. Phase 2 : Fine-tuning
    print(f"\n--- Démarrage Phase 2 (Fine-Tuning : jusqu'à {epochs_p2} époques) ---")
    train_phase2_finetune(
        model=model,
        base_model=base_model,
        train_ds=train_ds,
        val_ds=val_ds,
        initial_epoch=epochs_p1,
        total_epochs=epochs_p2,
        lr=1e-5,
        callbacks_list=callbacks_list
    )

    # 4. Évaluation sur le jeu de test
    print("\n--- Évaluation finale sur l'ensemble de Test ---")
    loss, accuracy = evaluate_classifier(model, test_ds)
    print(f"Résultats finaux Test -> Loss: {loss:.4f} | Accuracy: {accuracy * 100:.2f}%")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CanScan Classifier Training")
    parser.add_argument("--data_dir", type=str, default="data/classification", help="Dossier des splits de données")
    parser.add_argument("--checkpoint", type=str, default="models/best_googlenetv3.keras", help="Chemin du checkpoint")
    parser.add_argument("--epochs_p1", type=int, default=10, help="Époques Phase 1")
    parser.add_argument("--epochs_p2", type=int, default=25, help="Époques totales Phase 2")
    parser.add_argument("--batch_size", type=int, default=32, help="Taille des batches")
    parser.add_argument("--prepare", action="store_true", help="Recalculer les splits et l'augmentation")

    args = parser.parse_args()
    run_training(
        classification_dir=args.data_dir,
        checkpoint_path=args.checkpoint,
        epochs_p1=args.epochs_p1,
        epochs_p2=args.epochs_p2,
        batch_size=args.batch_size,
        prepare_first=args.prepare
    )
