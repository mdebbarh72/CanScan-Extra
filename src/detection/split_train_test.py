

from pathlib import Path
import random
import shutil
import math

ROOT = Path("data/detection")

TEST_RATIO = 0.15
SEED = 42

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

train_images = ROOT / "images" / "train"
train_labels = ROOT / "labels" / "train"
test_images = ROOT / "images" / "test"
test_labels = ROOT / "labels" / "test"

# Create test directories if they do not exist.
test_images.mkdir(parents=True, exist_ok=True)
test_labels.mkdir(parents=True, exist_ok=True)

# Index annotations by base filename.
labels_by_stem = {
    p.stem: p for p in train_labels.glob("*.txt")
}

# Only select images with matching annotation files.
pairs = []

for image in train_images.iterdir():
    if image.is_file() and image.suffix.lower() in IMAGE_EXTENSIONS:
        label = labels_by_stem.get(image.stem)

        if label is not None:
            pairs.append((image, label))
        else:
            print(f"Skipping image without label: {image.name}")

# Ensure the destination does not already contain conflicting filenames.
for image, label in pairs:
    if (test_images / image.name).exists():
        raise FileExistsError(f"Test image already exists: {image.name}")

    if (test_labels / label.name).exists():
        raise FileExistsError(f"Test label already exists: {label.name}")

if len(pairs) < 2:
    raise ValueError("Not enough paired images to create a test set.")

# Random selection with reproducible results.
random.Random(SEED).shuffle(pairs)

number_to_move = max(1, math.floor(len(pairs) * TEST_RATIO))
selected = pairs[:number_to_move]

# Move each image together with its corresponding annotation.
for image, label in selected:
    shutil.move(str(image), str(test_images / image.name))
    shutil.move(str(label), str(test_labels / label.name))

print(f"Paired images originally available: {len(pairs)}")
print(f"Images moved to test: {number_to_move}")
print(f"Remaining paired training images: {len(pairs) - number_to_move}")
print(f"Test images directory: {test_images}")
print(f"Test labels directory: {test_labels}")