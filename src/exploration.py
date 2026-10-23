import os
import sys
from pathlib import Path

ROOT_DIR = str(Path(__file__).resolve().parent.parent)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from PIL import Image
import pandas as pd
from src.charts import draw_count_plot, show_examples

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def load_and_verify_data(data_dir="data/Data"):
    """Vérifie l'intégrité des images dans le dossier source et retourne les chemins valides."""
    valid_paths, labels, invalid_paths = [], [], []

    for class_name in sorted(os.listdir(data_dir)):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir) or class_name.startswith("."):
            continue

        for filename in sorted(os.listdir(class_dir)):
            filepath = os.path.join(class_dir, filename)
            if os.path.splitext(filename)[1].lower() not in VALID_EXTENSIONS:
                invalid_paths.append((filepath, "Extension non supportée"))
                continue

            try:
                with Image.open(filepath) as img:
                    img.verify()
                valid_paths.append(filepath)
                labels.append(class_name)
            except Exception as e:
                invalid_paths.append((filepath, str(e)))

    print(f"Chargement terminé : {len(valid_paths)} images valides, {len(invalid_paths)} écartées.")
    return valid_paths, labels, invalid_paths


def explore_classes(labels):
    """Calcule la distribution des classes."""
    classes = sorted(set(labels))
    class_counts = pd.Series(labels).value_counts().sort_index()
    return classes, class_counts


def run_exploration(data_dir="data/Data", output_charts_dir="data/classification", generate_charts=True):
    """Exécute l'exploration, vérification et génération des graphiques."""
    valid_paths, labels, invalid_paths = load_and_verify_data(data_dir)
    classes, class_counts = explore_classes(labels)

    if generate_charts and valid_paths:
        draw_count_plot(class_counts, os.path.join(output_charts_dir, "class_distribution.png"))
        show_examples(classes, valid_paths, labels, os.path.join(output_charts_dir, "class_samples.png"))

    return {
        "valid_paths": valid_paths,
        "labels": labels,
        "classes": classes,
        "class_counts": class_counts,
        "invalid_paths": invalid_paths,
    }
