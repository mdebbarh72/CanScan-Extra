import os
import shutil
import random
import numpy as np
import cv2
import keras
import tensorflow as tf


def split_data(valid_paths, labels, classes, train_ratio=0.70, val_ratio=0.15, seed=42):
    """Effectue une séparation stratifiée par classe en Train / Val / Test."""
    by_class = {cls: [] for cls in classes}
    for path, label in zip(valid_paths, labels):
        by_class[label].append(path)

    splits = {s: {"files": [], "labels": []} for s in ["train", "val", "test"]}
    rng = random.Random(seed)

    for cls, paths in by_class.items():
        rng.shuffle(paths)
        n = len(paths)
        n_train, n_val = int(train_ratio * n), int(val_ratio * n)
        partitions = {
            "train": paths[:n_train],
            "val": paths[n_train:n_train + n_val],
            "test": paths[n_train + n_val:]
        }
        for split_name, file_list in partitions.items():
            splits[split_name]["files"].extend(file_list)
            splits[split_name]["labels"].extend([cls] * len(file_list))

    return splits


def augment_image(img: np.ndarray) -> np.ndarray:
    """Applique une transformation aléatoire : flip, flou ou bruit gaussien."""
    choice = random.randint(0, 2)
    if choice == 0:
        return cv2.flip(img, 1)
    if choice == 1:
        return cv2.GaussianBlur(img, (5, 5), 0)
    noise = np.random.normal(0, 15, img.shape).astype(np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def save_split_files(files, labels, split_dir, classes, augment=False, target_count=1500):
    """Copie les images dans le dossier cible avec data augmentation optionnelle."""
    for cls in classes:
        os.makedirs(os.path.join(split_dir, cls), exist_ok=True)

    for src, label in zip(files, labels):
        shutil.copy2(src, os.path.join(split_dir, label, os.path.basename(src)))

    if not augment:
        return

    by_class = {cls: [f for f, l in zip(files, labels) if l == cls] for cls in classes}
    for cls, paths in by_class.items():
        needed = max(0, target_count - len(paths))
        if not paths or needed == 0:
            continue
        pool = (paths * ((needed // len(paths)) + 2))[:needed]
        for idx, src in enumerate(pool):
            img = cv2.imread(src)
            if img is not None:
                base, ext = os.path.splitext(os.path.basename(src))
                cv2.imwrite(os.path.join(split_dir, cls, f"{base}_aug{idx}{ext}"), augment_image(img))


def create_classification_splits(splits, classes, output_dir="data/classification", target_count=1500):
    """Crée les dossiers de split ('train', 'val', 'test') directement sous output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    counts = {}

    for name in ["train", "val", "test"]:
        split_dir = os.path.join(output_dir, name)
        if os.path.exists(split_dir):
            shutil.rmtree(split_dir)

        is_train = name == "train"
        save_split_files(
            splits[name]["files"],
            splits[name]["labels"],
            split_dir,
            classes,
            augment=is_train,
            target_count=target_count
        )
        counts[name] = {cls: len(os.listdir(os.path.join(split_dir, cls))) for cls in classes}
        print(f"Split '{name}' : {sum(counts[name].values())} images créées.")

    return counts


def create_tf_datasets(classification_dir="data/classification", img_size=(224, 224), batch_size=32, seed=42):
    """Charge les datasets TensorFlow avec normalisation [0, 1] et prefetch."""
    rescaling = keras.layers.Rescaling(1.0 / 255)
    raw_datasets = {}

    for name in ["train", "val", "test"]:
        raw_datasets[name] = keras.utils.image_dataset_from_directory(
            os.path.join(classification_dir, name),
            image_size=img_size,
            batch_size=batch_size,
            shuffle=(name == "train"),
            seed=seed if name == "train" else None
        )

    class_names = raw_datasets["train"].class_names
    datasets = [
        raw_datasets[name].map(lambda x, y: (rescaling(x), y)).prefetch(tf.data.AUTOTUNE)
        for name in ["train", "val", "test"]
    ]

    return datasets[0], datasets[1], datasets[2], class_names
