import os
import sys
from pathlib import Path

ROOT_DIR = str(Path(__file__).resolve().parent.parent)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.exploration import run_exploration
from src.processing import split_data, create_classification_splits, create_tf_datasets
from src.charts import draw_split_distribution


def prepare_data_pipeline(
    source_dir="data/Data",
    output_dir="data/classification",
    img_size=(224, 224),
    batch_size=32,
    target_count=1500,
    seed=42,
    generate_charts=True
):
    """Pipeline complet de préparation, split et chargement des données."""
    print("Démarrage de la préparation des données...")

    # 1. Exploration et vérification
    data = run_exploration(source_dir, output_dir, generate_charts=generate_charts)

    # 2. Partitionnement stratifié (70% Train, 15% Val, 15% Test)
    splits = split_data(data["valid_paths"], data["labels"], data["classes"], seed=seed)

    # 3. Sauvegarde des dossiers 'train', 'val', 'test' avec data augmentation
    split_counts = create_classification_splits(splits, data["classes"], output_dir, target_count=target_count)
    if generate_charts:
        draw_split_distribution(split_counts, os.path.join(output_dir, "split_distribution.png"))

    # 4. Chargement des datasets TensorFlow
    train_ds, val_ds, test_ds, class_names = create_tf_datasets(
        output_dir,
        img_size=img_size,
        batch_size=batch_size,
        seed=seed
    )

    print("Pipeline de données prêt !")
    return train_ds, val_ds, test_ds, class_names


def load_data(output_dir="data/classification", img_size=(224, 224), batch_size=32, seed=42):
    """Charge les datasets TensorFlow existants sans recalculer les splits."""
    return create_tf_datasets(output_dir, img_size=img_size, batch_size=batch_size, seed=seed)


if __name__ == "__main__":
    prepare_data_pipeline()