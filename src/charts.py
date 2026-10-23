import os
import random
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def draw_count_plot(class_counts, output_path="data/classification/class_distribution.png"):
    """Trace et sauvegarde la distribution des classes."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(8, 4))
    df = pd.DataFrame({"Classe": class_counts.index, "Nombre": class_counts.values})
    sns.barplot(data=df, x="Classe", y="Nombre", hue="Classe", palette="viridis", legend=False)
    plt.title("Distribution des images par classe")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def show_examples(classes, valid_paths, labels, output_path="data/classification/class_samples.png"):
    """Sauvegarde un échantillon d'image par classe."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, axes = plt.subplots(1, len(classes), figsize=(4 * len(classes), 4))
    for ax, cls in zip(axes, classes):
        samples = [p for p, l in zip(valid_paths, labels) if l == cls]
        if samples:
            img = cv2.cvtColor(cv2.imread(random.choice(samples)), cv2.COLOR_BGR2RGB)
            ax.imshow(img)
        ax.set_title(cls)
        ax.axis("off")
    plt.suptitle("Échantillon par classe")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def draw_split_distribution(split_counts, output_path="data/classification/split_distribution.png"):
    """Trace la répartition des images par split et par classe."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    records = [
        {"Split": s, "Classe": c, "Nombre": n}
        for s, classes in split_counts.items()
        for c, n in classes.items()
    ]
    if not records:
        return
    plt.figure(figsize=(10, 5))
    sns.barplot(data=pd.DataFrame(records), x="Classe", y="Nombre", hue="Split", palette="Set2")
    plt.title("Répartition des images par split et classe")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()