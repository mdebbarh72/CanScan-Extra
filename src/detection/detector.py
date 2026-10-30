"""
YOLOv8 pipeline for a 4-class object detection dataset.

Requirement -> function
    1. Data augmentation parameters .......... get_augmentation_config()
    2. Training with hyperparameters adapted
       to the dataset and the hardware ....... get_resource_info(), suggest_training_plan(),
                                               build_model(), train_model(), tune_model()
    3. Detection metrics (precision, recall,
       mAP@0.5, mAP@0.5:0.95) ................ evaluate_model()
    4. Results analysis and generalization
       on the test set ....................... test_model(), analyze_generalization(),
                                               plot_training_curves()
    5. Saving the model and the artifacts
       needed to use it ...................... save_artifacts(), predict_images()

Install:
    pip install ultralytics

Dataset layout:
    data/detection/
    ├── images/{train,val,test}/
    └── labels/{train,val,test}/

data.yaml structure (write this by hand):

    path: /absolute/path/to/data/detection   # dataset root
    train: images/train                      # relative to `path`
    val: images/val                          # relative to `path`
    test: images/test                        # optional

    names:
      0: class_a
      1: class_b
      2: class_c
      3: class_d

Everything is saved under artifacts/.
"""

from datetime import datetime
import json
import os
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import pandas as pd
import ultralytics
import yaml
from ultralytics import YOLO

NUM_CLASSES = 4
DATA_YAML = Path("data.yaml")
ARTIFACTS_DIR = Path("artifacts").resolve()
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ==========================================================================
# Config check
# ==========================================================================
def check_data_yaml(data_yaml=DATA_YAML, num_classes=NUM_CLASSES):
    """Load data.yaml and verify it matches the expected structure."""
    data_yaml = Path(data_yaml)
    if not data_yaml.is_file():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

    with data_yaml.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    for key in ("path", "train", "val", "names"):
        if key not in config:
            raise KeyError(f"data.yaml is missing the '{key}' key.")

    if len(config["names"]) != num_classes:
        raise ValueError(
            f"data.yaml lists {len(config['names'])} class names, "
            f"expected {num_classes}."
        )

    root = Path(config["path"])
    for key in ("train", "val", "test"):
        if key in config and not (root / config[key]).is_dir():
            raise FileNotFoundError(
                f"'{key}' folder not found: {root / config[key]} "
                "(use an absolute `path` in data.yaml)"
            )

    return config


def _class_names(config):
    """Return class names as a list ordered by class id."""
    names = config["names"]
    if isinstance(names, dict):
        return [names[k] for k in sorted(names)]
    return list(names)


def count_images(config, split="train"):
    """Count the images of a split declared in data.yaml."""
    folder = Path(config["path"]) / config[split]
    return sum(
        1 for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


# ==========================================================================
# 1. Data augmentation
# ==========================================================================
AUGMENTATION_PRESETS = {
    # Large datasets: the data already carries a lot of variety.
    "light": {
        "hsv_h": 0.015, "hsv_s": 0.5, "hsv_v": 0.3,   # color jitter (hue/sat/value)
        "degrees": 0.0, "translate": 0.1, "scale": 0.3,
        "shear": 0.0, "perspective": 0.0,
        "flipud": 0.0, "fliplr": 0.5,
        "mosaic": 0.5, "mixup": 0.0, "close_mosaic": 10,
    },
    # Reasonable default for medium datasets.
    "default": {
        "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4,
        "degrees": 0.0, "translate": 0.1, "scale": 0.5,
        "shear": 0.0, "perspective": 0.0,
        "flipud": 0.0, "fliplr": 0.5,
        "mosaic": 1.0, "mixup": 0.0, "close_mosaic": 10,
    },
    # Small datasets: stronger augmentation to limit overfitting.
    "strong": {
        "hsv_h": 0.02, "hsv_s": 0.8, "hsv_v": 0.5,
        "degrees": 10.0, "translate": 0.15, "scale": 0.6,
        "shear": 2.0, "perspective": 0.0005,
        "flipud": 0.0, "fliplr": 0.5,
        "mosaic": 1.0, "mixup": 0.15, "close_mosaic": 15,
    },
}


def get_augmentation_config(preset="default", horizontal_flip=True, **overrides):
    """
    Return the augmentation parameters passed to YOLOv8's train().

    Parameters:
        hsv_h / hsv_s / hsv_v   color jitter (hue, saturation, brightness)
        degrees, shear          rotation and shear in degrees
        translate, scale        shift and zoom (fractions)
        perspective             perspective warp (keep tiny, e.g. 0.0005)
        flipud / fliplr         probability of vertical / horizontal flip
        mosaic                  probability of mixing 4 images into one
        mixup                   probability of blending 2 images
        close_mosaic            disable mosaic for the last N epochs

    preset: "light", "default" or "strong".
    horizontal_flip: set False if left/right orientation matters for your
        classes (e.g. text, directional signs).
    overrides: any parameter above, e.g. get_augmentation_config(scale=0.7).
    """
    if preset not in AUGMENTATION_PRESETS:
        raise ValueError(f"preset must be one of {list(AUGMENTATION_PRESETS)}")

    config = dict(AUGMENTATION_PRESETS[preset])
    if not horizontal_flip:
        config["fliplr"] = 0.0
    config.update(overrides)
    return config


# ==========================================================================
# 2. Hyperparameters adapted to the dataset and the hardware
# ==========================================================================
def get_resource_info():
    """Detect the available accelerator (CUDA GPU, Apple MPS or CPU)."""
    import torch

    cpu_count = os.cpu_count() or 2

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        return {
            "device": 0,
            "name": props.name,
            "vram_gb": props.total_memory / 1024 ** 3,
            "cpu_count": cpu_count,
        }

    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return {"device": "mps", "name": "Apple MPS", "vram_gb": None, "cpu_count": cpu_count}

    return {"device": "cpu", "name": "CPU", "vram_gb": None, "cpu_count": cpu_count}


def suggest_training_plan(data_yaml=DATA_YAML, num_classes=NUM_CLASSES):
    """
    Choose model size, batch size, epochs, optimizer and augmentation from
    the number of training images and the available hardware.

    Returns {"weights", "hyperparameters", "augmentation"}.
    These are sensible starting points, not guarantees: use tune_model()
    to refine them.
    """
    config = check_data_yaml(data_yaml, num_classes)
    n_train = count_images(config, "train")
    if n_train == 0:
        raise ValueError("No training images found.")

    resources = get_resource_info()
    vram = resources["vram_gb"]
    reasons = []

    # Model size, image size and batch from the hardware.
    if resources["device"] == "cpu":
        weights, imgsz, batch = "yolov8n.pt", 416, 8
        reasons.append("CPU only: smallest model, imgsz 416, batch 8")
    elif vram is None or vram < 6:
        weights, imgsz, batch = "yolov8n.pt", 640, 8
        reasons.append("low memory accelerator: yolov8n, batch 8")
    elif vram < 12:
        weights, imgsz, batch = "yolov8s.pt", 640, 16
        reasons.append(f"{vram:.0f} GB VRAM: yolov8s, batch 16")
    elif vram < 24:
        weights, imgsz, batch = "yolov8m.pt", 640, 16
        reasons.append(f"{vram:.0f} GB VRAM: yolov8m, batch 16")
    else:
        weights, imgsz, batch = "yolov8m.pt", 640, 32
        reasons.append(f"{vram:.0f} GB VRAM: yolov8m, batch 32")

    # Epochs, optimizer and augmentation from the dataset size.
    if n_train < 1000 and weights in ("yolov8m.pt", "yolov8l.pt", "yolov8x.pt"):
        weights = "yolov8s.pt"
        reasons.append("small dataset: capped model size at yolov8s to limit overfitting")

    if n_train < 500:
        epochs, patience = 150, 30
    elif n_train < 3000:
        epochs, patience = 100, 20
    else:
        epochs, patience = 60, 15

    if resources["device"] == "cpu":
        epochs = min(epochs, 50)
        reasons.append("CPU only: epochs capped at 50")

    small = n_train < 3000
    hyperparameters = {
        "imgsz": imgsz,
        "batch": batch,
        "epochs": epochs,
        "patience": patience,
        "workers": min(8, max(2, resources["cpu_count"] // 2)),
        "device": resources["device"],
        "optimizer": "AdamW" if small else "SGD",
        "lr0": 0.001 if small else 0.01,
        "lrf": 0.01,
        "weight_decay": 0.0005,
        "warmup_epochs": 3,
        "cos_lr": small,
    }
    if n_train < 300:
        hyperparameters["freeze"] = 10  # freeze the backbone, train the rest
        reasons.append("very small dataset: backbone frozen (freeze=10)")

    if n_train < 1000:
        preset = "strong"
    elif n_train < 5000:
        preset = "default"
    else:
        preset = "light"
    reasons.append(f"{n_train} training images: '{preset}' augmentation preset")

    print(f"Hardware: {resources['name']}")
    for reason in reasons:
        print(f"  - {reason}")

    return {
        "weights": weights,
        "hyperparameters": hyperparameters,
        "augmentation": get_augmentation_config(preset),
    }


def build_model(weights="yolov8n.pt", data_yaml=DATA_YAML, num_classes=NUM_CLASSES):
    """
    Load a YOLOv8 model after checking data.yaml.

    weights:
        "yolov8n.pt" / "yolov8s.pt" / "yolov8m.pt" ...  pretrained (recommended)
        "yolov8n.yaml"                                   random init, no pretraining
        path to your own best.pt                         continue from a checkpoint

    The detection head is rebuilt automatically for the number of classes
    declared in data.yaml when training starts.
    """
    check_data_yaml(data_yaml, num_classes)
    return YOLO(weights)


def plot_training_curves(results_csv, out_dir=ARTIFACTS_DIR, prefix="yolov8"):
    """
    Save loss and accuracy charts from an Ultralytics results.csv.

    Detection has no single "accuracy" number, so the accuracy chart shows
    precision, recall, mAP@0.50 and mAP@0.50:0.95 per epoch.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()
    epochs = df["epoch"]

    loss_path = out_dir / f"{prefix}_loss_curves.png"
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (key, title) in zip(
        axes,
        [("box_loss", "Box loss"), ("cls_loss", "Class loss"), ("dfl_loss", "DFL loss")],
    ):
        ax.plot(epochs, df[f"train/{key}"], label="train")
        ax.plot(epochs, df[f"val/{key}"], label="val")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("Training and validation loss", fontsize=16)
    fig.tight_layout()
    fig.savefig(loss_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    acc_path = out_dir / f"{prefix}_accuracy_curves.png"
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    for ax, (key, title) in zip(
        axes,
        [
            ("metrics/precision(B)", "Precision"),
            ("metrics/recall(B)", "Recall"),
            ("metrics/mAP50(B)", "mAP@0.50"),
            ("metrics/mAP50-95(B)", "mAP@0.50:0.95"),
        ],
    ):
        ax.plot(epochs, df[key], color="tab:green")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
    fig.suptitle("Validation accuracy metrics", fontsize=16)
    fig.tight_layout()
    fig.savefig(acc_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Loss chart saved to: {loss_path}")
    print(f"Accuracy chart saved to: {acc_path}")
    return loss_path, acc_path


def train_model(
    model,
    data_yaml=DATA_YAML,
    hyperparameters=None,
    augmentation=None,
    tuned_hyperparameters=None,
    seed=42,
    run_name="yolov8_train",
):
    """
    Train the model, save the config and the loss/accuracy charts, and
    return the path to best.pt.

    hyperparameters:       dict from suggest_training_plan()["hyperparameters"]
                           (computed automatically if None)
    augmentation:          dict from get_augmentation_config()
                           (the "default" preset if None)
    tuned_hyperparameters: optional dict or path to the best_hyperparameters.yaml
                           returned by tune_model(); applied last
    """
    if hyperparameters is None:
        hyperparameters = suggest_training_plan(data_yaml)["hyperparameters"]
    if augmentation is None:
        augmentation = get_augmentation_config("default")

    tuned = {}
    if tuned_hyperparameters is not None:
        if isinstance(tuned_hyperparameters, (str, Path)):
            with open(tuned_hyperparameters, "r", encoding="utf-8") as f:
                tuned_hyperparameters = yaml.safe_load(f)
        tuned = dict(tuned_hyperparameters)

    train_args = {**hyperparameters, **augmentation, **tuned}

    # Keep a record of exactly what was used.
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    config_path = ARTIFACTS_DIR / f"{run_name}_config.yaml"
    with config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "data": str(Path(data_yaml).resolve()),
                "seed": seed,
                "hyperparameters": hyperparameters,
                "augmentation": augmentation,
                "tuned_overrides": tuned,
            },
            f,
            sort_keys=False,
        )
    print(f"Training config saved to: {config_path}")

    model.train(
        data=str(data_yaml),
        seed=seed,
        project=str(ARTIFACTS_DIR),
        name=run_name,
        exist_ok=False,
        plots=True,
        **train_args,
    )

    run_dir = Path(model.trainer.save_dir)
    best_weights = run_dir / "weights" / "best.pt"

    plot_training_curves(run_dir / "results.csv", ARTIFACTS_DIR, prefix=run_name)

    print(f"Best weights: {best_weights}")
    return best_weights


def tune_model(
    model,
    data_yaml=DATA_YAML,
    iterations=30,
    epochs=30,
    imgsz=640,
    batch=16,
    device=None,
    run_name="yolov8_tune",
):
    """
    Search for better hyperparameters with the Ultralytics genetic tuner.

    Cost is roughly iterations x epochs of training, so keep both modest.
    Returns the path to best_hyperparameters.yaml, which can be passed to
    train_model(tuned_hyperparameters=...).
    """
    model.tune(
        data=str(data_yaml),
        iterations=iterations,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        optimizer="AdamW",
        project=str(ARTIFACTS_DIR),
        name=run_name,
        plots=False,
        save=False,
        val=False,
    )

    candidates = sorted(
        ARTIFACTS_DIR.glob(f"{run_name}*/best_hyperparameters.yaml"),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(
            f"best_hyperparameters.yaml not found under {ARTIFACTS_DIR}"
        )

    best_path = candidates[-1]
    print(f"Best hyperparameters saved to: {best_path}")
    return best_path


# ==========================================================================
# 3. Evaluation metrics
# ==========================================================================
def evaluate_model(
    weights_path,
    data_yaml=DATA_YAML,
    split="val",
    imgsz=640,
    batch=16,
    device=None,
    save_outputs=True,
):
    """
    Evaluate trained weights on a split.

    Metrics (overall and per class): precision, recall, mAP@0.50, mAP@0.50:0.95.

    With save_outputs=True, saves to artifacts/:
        {split}_confusion_matrix.png
        {split}_confusion_matrix_normalized.png
        {split}_metrics.csv

    Falls back to "val" if the requested split is not in data.yaml.
    Returns (ultralytics metrics object, DataFrame).
    """
    config = check_data_yaml(data_yaml)

    if split not in config:
        print(f"'{split}' not found in data.yaml, using 'val' instead.")
        split = "val"

    run_name = f"yolov8_eval_{split}" if save_outputs else f"yolov8_check_{split}"

    model = YOLO(str(weights_path))
    metrics = model.val(
        data=str(data_yaml),
        split=split,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(ARTIFACTS_DIR),
        name=run_name,
        exist_ok=True,
        plots=save_outputs,   # generates the confusion matrices
    )

    names = metrics.names
    rows = [{
        "class": "all",
        "precision": metrics.box.mp,
        "recall": metrics.box.mr,
        "mAP50": metrics.box.map50,
        "mAP50-95": metrics.box.map,
    }]
    for position, class_id in enumerate(metrics.box.ap_class_index):
        precision, recall, ap50, ap = metrics.box.class_result(position)
        rows.append({
            "class": names[int(class_id)],
            "precision": precision,
            "recall": recall,
            "mAP50": ap50,
            "mAP50-95": ap,
        })
    table = pd.DataFrame(rows)

    print(f"\nResults on '{split}':")
    print(table.round(3).to_string(index=False))

    if save_outputs:
        eval_dir = Path(metrics.save_dir)
        for source_name in ("confusion_matrix.png", "confusion_matrix_normalized.png"):
            source = eval_dir / source_name
            if source.is_file():
                destination = ARTIFACTS_DIR / f"{split}_{source_name}"
                shutil.copy2(source, destination)
                print(f"Saved: {destination}")

        table_path = ARTIFACTS_DIR / f"{split}_metrics.csv"
        table.to_csv(table_path, index=False)
        print(f"Metrics saved to: {table_path}")

    return metrics, table


# ==========================================================================
# 4. Test set and generalization analysis
# ==========================================================================
def test_model(weights_path, data_yaml=DATA_YAML, imgsz=640, batch=16, device=None):
    """Evaluate on the test split and save the confusion matrices and metrics."""
    return evaluate_model(
        weights_path, data_yaml, split="test",
        imgsz=imgsz, batch=batch, device=device, save_outputs=True,
    )


def analyze_generalization(
    weights_path,
    data_yaml=DATA_YAML,
    imgsz=640,
    batch=16,
    device=None,
    include_train=True,
    overfit_gap=0.15,
    val_test_gap=0.05,
):
    """
    Compare train / val / test performance to check generalization.

    Saves to artifacts/:
        generalization_summary.csv
        generalization_comparison.png
        generalization_report.txt

    Thresholds (absolute differences in a metric on a 0-1 scale) are
    rules of thumb:
        overfit_gap   train mAP@0.5:0.95 minus test mAP@0.5:0.95
        val_test_gap  val mAP@0.5:0.95 minus test mAP@0.5:0.95

    Returns (summary DataFrame, list of findings).
    """
    config = check_data_yaml(data_yaml)

    splits = [s for s in ("train", "val", "test") if s in config]
    if not include_train and "train" in splits:
        splits.remove("train")

    tables = {}
    for split in splits:
        _, table = evaluate_model(
            weights_path, data_yaml, split=split,
            imgsz=imgsz, batch=batch, device=device, save_outputs=False,
        )
        tables[split] = table

    columns = ["precision", "recall", "mAP50", "mAP50-95"]
    summary = pd.DataFrame(
        {split: t[t["class"] == "all"].iloc[0][columns] for split, t in tables.items()}
    ).T.astype(float)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(ARTIFACTS_DIR / "generalization_summary.csv")

    # Grouped bar chart: metrics on the x axis, one bar per split.
    chart_path = ARTIFACTS_DIR / "generalization_comparison.png"
    fig, ax = plt.subplots(figsize=(10, 5))
    summary.T.plot(kind="bar", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Train vs validation vs test")
    ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=0)
    fig.tight_layout()
    fig.savefig(chart_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Findings.
    findings = []
    final_split = "test" if "test" in summary.index else "val"
    final = summary.loc[final_split]

    if "train" in summary.index:
        gap = summary.loc["train", "mAP50-95"] - final["mAP50-95"]
        if gap > overfit_gap:
            findings.append(
                f"Likely overfitting: train mAP@0.5:0.95 is {gap:.2f} higher than "
                f"{final_split}. Try stronger augmentation, a smaller model, "
                "more data, or fewer epochs."
            )
        else:
            findings.append(
                f"Train vs {final_split} gap is {gap:.2f} (below {overfit_gap}): "
                "no strong sign of overfitting."
            )

    if "val" in summary.index and "test" in summary.index:
        drop = summary.loc["val", "mAP50-95"] - summary.loc["test", "mAP50-95"]
        if drop > val_test_gap:
            findings.append(
                f"Test mAP@0.5:0.95 is {drop:.2f} lower than validation. The "
                "validation set may be optimistic (it drives early stopping and "
                "tuning) or the test images may differ from the rest of the data."
            )
        elif drop < -val_test_gap:
            findings.append(
                f"Test mAP@0.5:0.95 is {-drop:.2f} higher than validation, which "
                "can happen with small or easier test sets. Interpret with care."
            )
        else:
            findings.append(
                f"Validation and test mAP@0.5:0.95 differ by {drop:.2f}: results "
                "are consistent, which supports good generalization."
            )

    if abs(final["precision"] - final["recall"]) > 0.15:
        if final["precision"] < final["recall"]:
            findings.append(
                f"On {final_split}, precision ({final['precision']:.2f}) is well below "
                f"recall ({final['recall']:.2f}): many false detections."
            )
        else:
            findings.append(
                f"On {final_split}, recall ({final['recall']:.2f}) is well below "
                f"precision ({final['precision']:.2f}): many missed objects."
            )

    per_class = tables[final_split]
    per_class = per_class[per_class["class"] != "all"]
    weak = per_class[per_class["mAP50-95"] < final["mAP50-95"] - 0.10]
    for _, row in weak.iterrows():
        findings.append(
            f"Weak class on {final_split}: '{row['class']}' "
            f"(mAP@0.5:0.95 {row['mAP50-95']:.2f}, recall {row['recall']:.2f}). "
            "Check label quality and the number of examples for this class."
        )

    report_path = ARTIFACTS_DIR / "generalization_report.txt"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("Overall metrics per split\n")
        f.write(summary.round(3).to_string())
        f.write("\n\nFindings\n")
        f.writelines(f"- {line}\n" for line in findings)

    print("\n" + summary.round(3).to_string())
    print("\nFindings:")
    for line in findings:
        print(f"  - {line}")
    print(f"Comparison chart saved to: {chart_path}")
    print(f"Report saved to: {report_path}")

    return summary, findings


# ==========================================================================
# 5. Saving the model and what is needed to use it
# ==========================================================================
def save_artifacts(
    weights_path,
    data_yaml=DATA_YAML,
    imgsz=640,
    summary=None,
    export_formats=("onnx",),
    out_dir=None,
):
    """
    Collect everything needed to use the model in artifacts/model/:
        best.pt          trained weights
        data.yaml        dataset config (class names and ids)
        model_info.json  classes, input size, recommended thresholds,
                         library version, date and metrics
        best.onnx        optional exported formats

    summary: the DataFrame returned by analyze_generalization(), if available.
    export_formats: e.g. ("onnx",), ("onnx", "torchscript"), or () for none.
    """
    out_dir = Path(out_dir) if out_dir else ARTIFACTS_DIR / "model"
    out_dir.mkdir(parents=True, exist_ok=True)

    config = check_data_yaml(data_yaml)

    best_copy = out_dir / "best.pt"
    shutil.copy2(weights_path, best_copy)
    shutil.copy2(data_yaml, out_dir / "data.yaml")

    info = {
        "classes": _class_names(config),
        "imgsz": imgsz,
        "recommended_conf": 0.25,
        "recommended_iou": 0.7,
        "ultralytics_version": ultralytics.__version__,
        "created": datetime.now().isoformat(timespec="seconds"),
        "metrics": summary.round(4).to_dict(orient="index") if summary is not None else None,
    }
    with (out_dir / "model_info.json").open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    for fmt in export_formats:
        try:
            exported = YOLO(str(best_copy)).export(format=fmt, imgsz=imgsz)
            print(f"Exported {fmt}: {exported}")
        except Exception as error:  # export needs extra packages (e.g. onnx)
            print(f"Export to '{fmt}' failed: {error}")

    print(f"Model artifacts saved to: {out_dir}")
    return out_dir


def predict_images(
    weights_path,
    source,
    conf=0.25,
    iou=0.7,
    imgsz=640,
    device=None,
    run_name="yolov8_predictions",
):
    """
    Run the trained model on an image, a folder, or a list of images and
    save the annotated results to artifacts/{run_name}/.
    """
    model = YOLO(str(weights_path))
    results = model.predict(
        source=str(source) if isinstance(source, (str, Path)) else [str(s) for s in source],
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device,
        save=True,
        project=str(ARTIFACTS_DIR),
        name=run_name,
        exist_ok=True,
    )
    print(f"Predictions saved to: {ARTIFACTS_DIR / run_name}")
    return results


# ==========================================================================
# Example usage
# ==========================================================================
if __name__ == "__main__":
    # Hardware- and dataset-aware settings (see printed reasons).
    plan = suggest_training_plan(DATA_YAML, NUM_CLASSES)
    imgsz = plan["hyperparameters"]["imgsz"]

    # To change augmentation yourself instead of using the suggested preset:
    # plan["augmentation"] = get_augmentation_config("strong", horizontal_flip=False, scale=0.7)

    model = build_model(plan["weights"], DATA_YAML, NUM_CLASSES)

    # Optional and slow: tune hyperparameters, then pass the result to train_model().
    # tuned = tune_model(model, DATA_YAML, iterations=30, epochs=30, imgsz=imgsz)
    # model = build_model(plan["weights"], DATA_YAML, NUM_CLASSES)
    tuned = None

    best_weights = train_model(
        model,
        DATA_YAML,
        hyperparameters=plan["hyperparameters"],
        augmentation=plan["augmentation"],
        tuned_hyperparameters=tuned,
    )

    evaluate_model(best_weights, DATA_YAML, split="val", imgsz=imgsz)
    test_model(best_weights, DATA_YAML, imgsz=imgsz)
    summary, findings = analyze_generalization(best_weights, DATA_YAML, imgsz=imgsz)

    save_artifacts(best_weights, DATA_YAML, imgsz=imgsz, summary=summary)