from collections import Counter
from pathlib import Path
import random
import shutil

import cv2
import matplotlib.pyplot as plt

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IGNORED_LABEL_FILES = {"classes.txt"}  # metadata, not a per-image label


# --------------------------------------------------------------------------
# Pairing helpers
# --------------------------------------------------------------------------
def _list_files(images_dir, labels_dir):
    images = [
        p for p in sorted(images_dir.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    labels = [
        p for p in sorted(labels_dir.iterdir())
        if p.is_file()
        and p.suffix.lower() == ".txt"
        and p.name not in IGNORED_LABEL_FILES
    ]
    return images, labels


def find_unpaired(images_dir, labels_dir):
    """Return (images without labels, labels without images) as Path lists."""
    images, labels = _list_files(images_dir, labels_dir)
    image_stems = {p.stem for p in images}
    label_stems = {p.stem for p in labels}

    images_without_labels = [p for p in images if p.stem not in label_stems]
    labels_without_images = [p for p in labels if p.stem not in image_stems]
    return images_without_labels, labels_without_images


# --------------------------------------------------------------------------
# Step 1: log unpaired files to artifacts/, then drop them from the dataset
# --------------------------------------------------------------------------
def remove_unpaired(
    dataset_path="data/detection",
    splits=("train", "val", "test"),
    action="move",
    dry_run=True,
    artifacts_dir="artifacts",
):
    """
    Log unpaired images/labels to artifacts/unpaired_files.txt, then remove them.

    action:
        "move"   -> move files to artifacts/removed_unpaired/{images,labels}/{split}/
                    (recoverable, recommended)
        "delete" -> permanently delete files
    dry_run:
        True  -> only write the log, do not touch any dataset files
        False -> write the log and then remove the files

    The log is always written BEFORE any file is removed.
    """
    if action not in {"move", "delete"}:
        raise ValueError("action must be 'move' or 'delete'.")

    dataset_path = Path(dataset_path)
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    quarantine_dir = artifacts_dir / "removed_unpaired"

    # Collect unpaired files for every split that exists.
    found = {}
    for split in splits:
        images_dir = dataset_path / "images" / split
        labels_dir = dataset_path / "labels" / split

        if not images_dir.is_dir() or not labels_dir.is_dir():
            print(f"[{split}] skipped (images or labels directory missing)")
            continue

        found[split] = find_unpaired(images_dir, labels_dir)

    # Write the log first.
    log_path = artifacts_dir / "unpaired_files.txt"
    mode = "DRY RUN (nothing removed)" if dry_run else action.upper()

    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"Mode: {mode}\n")

        for split, (imgs, lbls) in found.items():
            f.write(f"\n=== {split} ===\n")
            f.write(f"Images without labels ({len(imgs)}):\n")
            f.writelines(f"  {p.name}\n" for p in imgs)
            f.write(f"Labels without images ({len(lbls)}):\n")
            f.writelines(f"  {p.name}\n" for p in lbls)

    print(f"Unpaired log saved to: {log_path}")

    # Then remove.
    total = 0
    for split, (imgs, lbls) in found.items():
        for kind, files in (("images", imgs), ("labels", lbls)):
            for path in files:
                total += 1
                if dry_run:
                    continue

                if action == "move":
                    dest_dir = quarantine_dir / kind / split
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(path), str(dest_dir / path.name))
                else:
                    path.unlink()

    if dry_run:
        print(f"Dry run: {total} unpaired file(s) would be removed (action='{action}').")
    else:
        print(f"{action.capitalize()}d {total} unpaired file(s).")

    return found


# --------------------------------------------------------------------------
# Step 2: verify the dataset is suitable for YOLO
# --------------------------------------------------------------------------
def validate_yolo_structure(
    dataset_path="data/detection",
    required_splits=("train", "val"),
    optional_splits=("test",),
    num_classes=None,
    artifacts_dir="artifacts",
    max_printed=30,
):
    """
    Check the dataset follows the YOLO detection layout:

        dataset/
        ├── images/{train,val,test}/
        └── labels/{train,val,test}/

    Checks:
      - required split directories exist (images and labels)
      - each split has images
      - every image has a label and every label has an image
      - no two images in a split share a stem (e.g. a.jpg and a.png)
      - label lines are: class_id x_center y_center width height
        (integer class >= 0 and < num_classes if given, coordinates
         normalized to [0, 1], width and height > 0)
      - no filename overlap between splits (possible train/val leakage)

    Empty label files are valid (background images).
    Full report is saved to artifacts/yolo_validation.txt.
    Returns True if no problems were found.
    """
    dataset_path = Path(dataset_path)
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    problems = []
    summary = []
    stems_by_split = {}

    for split in tuple(required_splits) + tuple(optional_splits):
        required = split in required_splits
        images_dir = dataset_path / "images" / split
        labels_dir = dataset_path / "labels" / split

        has_images, has_labels = images_dir.is_dir(), labels_dir.is_dir()

        if not has_images and not has_labels and not required:
            summary.append(f"[{split}] not present (optional)")
            continue

        if not has_images:
            problems.append(f"[{split}] missing images directory: {images_dir}")
        if not has_labels:
            problems.append(f"[{split}] missing labels directory: {labels_dir}")
        if not (has_images and has_labels):
            continue

        images, labels = _list_files(images_dir, labels_dir)
        stems_by_split[split] = {p.stem for p in images}

        if not images:
            problems.append(f"[{split}] no images found in {images_dir}")

        # Pairing.
        imgs_no_lbl, lbls_no_img = find_unpaired(images_dir, labels_dir)
        if imgs_no_lbl:
            problems.append(
                f"[{split}] {len(imgs_no_lbl)} image(s) without labels, "
                f"e.g. {imgs_no_lbl[0].name}"
            )
        if lbls_no_img:
            problems.append(
                f"[{split}] {len(lbls_no_img)} label(s) without images, "
                f"e.g. {lbls_no_img[0].name}"
            )

        # Duplicate stems with different extensions.
        stem_counts = Counter(p.stem for p in images)
        for stem, count in stem_counts.items():
            if count > 1:
                problems.append(
                    f"[{split}] {count} images share the stem '{stem}' "
                    "(they would map to the same label file)"
                )

        # Label content.
        n_boxes = 0
        n_empty = 0
        class_counts = Counter()

        for label_path in labels:
            lines = label_path.read_text(encoding="utf-8").splitlines()
            lines = [ln for ln in lines if ln.strip()]

            if not lines:
                n_empty += 1
                continue

            for line_no, line in enumerate(lines, start=1):
                where = f"[{split}] {label_path.name}:{line_no}"
                parts = line.split()

                if len(parts) != 5:
                    problems.append(
                        f"{where} expected 5 values, got {len(parts)}"
                    )
                    continue

                try:
                    class_id = int(parts[0])
                except ValueError:
                    problems.append(f"{where} class id '{parts[0]}' is not an integer")
                    continue

                try:
                    x, y, w, h = map(float, parts[1:])
                except ValueError:
                    problems.append(f"{where} non-numeric coordinates")
                    continue

                if class_id < 0 or (num_classes is not None and class_id >= num_classes):
                    problems.append(f"{where} class id {class_id} out of range")
                if not all(0.0 <= v <= 1.0 for v in (x, y, w, h)):
                    problems.append(f"{where} coordinates not normalized to [0, 1]")
                if w <= 0 or h <= 0:
                    problems.append(f"{where} non-positive width/height")

                n_boxes += 1
                class_counts[class_id] += 1

        summary.append(
            f"[{split}] {len(images)} images, {len(labels)} labels, "
            f"{n_boxes} boxes, {n_empty} empty label files, "
            f"classes seen: {sorted(class_counts)}"
        )

    # Leakage between splits.
    split_names = list(stems_by_split)
    for i, a in enumerate(split_names):
        for b in split_names[i + 1:]:
            overlap = stems_by_split[a] & stems_by_split[b]
            if overlap:
                problems.append(
                    f"{len(overlap)} filename(s) appear in both '{a}' and '{b}', "
                    f"e.g. {sorted(overlap)[0]}"
                )

    # Report.
    report_path = artifacts_dir / "yolo_validation.txt"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("SUMMARY\n")
        f.writelines(f"  {line}\n" for line in summary)
        f.write(f"\nPROBLEMS ({len(problems)})\n")
        f.writelines(f"  {line}\n" for line in problems)

    print("\n".join(summary))
    if problems:
        print(f"\n{len(problems)} problem(s) found:")
        for line in problems[:max_printed]:
            print(f"  - {line}")
        if len(problems) > max_printed:
            print(f"  ... and {len(problems) - max_printed} more")
    else:
        print("\nDataset structure looks valid for YOLO.")
    print(f"Validation report saved to: {report_path}")

    return not problems


# --------------------------------------------------------------------------
# Step 3: display samples (unchanged apart from the shared helpers)
# --------------------------------------------------------------------------
def display_samples(
    dataset_path="data/detection",
    split="train",
    class_names=None,
    num_samples=8,
    seed=42,
):
    """Display and save sample images with YOLO bounding boxes."""
    dataset_path = Path(dataset_path)
    images_dir = dataset_path / "images" / split
    labels_dir = dataset_path / "labels" / split

    if not images_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {images_dir}")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Label directory not found: {labels_dir}")
    if num_samples < 1:
        raise ValueError("num_samples must be at least 1.")

    images, _ = _list_files(images_dir, labels_dir)
    image_paths = [p for p in images if (labels_dir / f"{p.stem}.txt").is_file()]

    if not image_paths:
        raise ValueError(f"No labeled images found in {images_dir}")

    rng = random.Random(seed)
    selected_paths = rng.sample(image_paths, min(num_samples, len(image_paths)))

    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    n_samples = len(selected_paths)
    n_cols = min(4, n_samples)
    n_rows = (n_samples + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), squeeze=False
    )

    for ax, image_path in zip(axes.flat, selected_paths):
        label_path = labels_dir / f"{image_path.stem}.txt"

        image = cv2.imread(str(image_path))
        if image is None:
            ax.set_title(f"Failed to load: {image_path.name}")
            ax.axis("off")
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_height, image_width = image.shape[:2]

        with label_path.open("r", encoding="utf-8") as file:
            annotations = file.readlines()

        for annotation in annotations:
            values = annotation.strip().split()
            if len(values) != 5:
                continue

            try:
                class_id = int(values[0])
                x_center, y_center, box_width, box_height = map(float, values[1:])
            except ValueError:
                continue

            x_center *= image_width
            y_center *= image_height
            box_width *= image_width
            box_height *= image_height

            x1 = max(0, int(x_center - box_width / 2))
            y1 = max(0, int(y_center - box_height / 2))
            x2 = min(image_width - 1, int(x_center + box_width / 2))
            y2 = min(image_height - 1, int(y_center + box_height / 2))

            if isinstance(class_names, dict):
                class_name = class_names.get(class_id, str(class_id))
            elif isinstance(class_names, (list, tuple)):
                class_name = (
                    class_names[class_id]
                    if 0 <= class_id < len(class_names)
                    else str(class_id)
                )
            else:
                class_name = str(class_id)

            color = (255, 0, 0)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                image, class_name, (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA,
            )

        ax.imshow(image)
        ax.set_title(image_path.name)
        ax.axis("off")

    for ax in axes.flat[n_samples:]:
        ax.axis("off")

    fig.suptitle(f"YOLO Bounding Box Samples — {split}", fontsize=16)
    fig.tight_layout()

    output_path = artifacts_dir / f"{split}_samples.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.show()
    plt.close(fig)

    print(f"Figure saved to: {output_path}")
    return output_path


# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------
if __name__ == "__main__":
    DATASET = "data/detection"

    # 1. Log unpaired files and remove them.
    #    Start with dry_run=True to review artifacts/unpaired_files.txt,
    #    then set dry_run=False to actually move the files.
    remove_unpaired(DATASET, action="move", dry_run=False)

    # 2. Verify the dataset is valid for YOLO (set num_classes if you know it).
    is_valid = validate_yolo_structure(DATASET, num_classes=None)

    # 3. Display samples only if the structure checks out.
    if is_valid:
        display_samples(
            dataset_path=DATASET,
            split="train",
            class_names=None,
            num_samples=8,
            seed=42,
        )