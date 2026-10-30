"""
Airflow 3 DAG for the CanScan Object Detection Pipeline (YOLOv8).

1. Verify images and matching YOLO label files
2. Split into train / val / test lists
3. Train YOLOv8
4. Evaluate on the test split (precision, recall, mAP@0.5, mAP@0.5:0.95)
5. Export weights and metrics as deployment artifacts

Requirements:
- `ultralytics` installed in the Airflow image (airflow/Dockerfile)
- Volumes mounted in docker-compose (x-airflow-volumes):
    - ./data:/opt/airflow/data:ro
    - ./artifacts:/opt/airflow/artifacts
"""

import json
import logging
import random
import shutil
from datetime import datetime, timedelta
from pathlib import Path

from airflow.sdk import DAG, task

logger = logging.getLogger(__name__)

DATASET_ROOT = Path("/opt/airflow/data/detection")
ARTIFACTS_DIR = Path("/opt/airflow/artifacts/detection")
WORK_DIR = ARTIFACTS_DIR / "work"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
SEED = 42

TRAIN_PARAMS = {
    "model": "yolov8m.pt",
    "epochs": 50,
    "imgsz": 640,
    "batch": 16,
}

default_args = {
    "owner": "canscan",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="canscan_detection_pipeline",
    description="Automated pipeline for CanScan YOLOv8 object detection",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule=None,  # manual trigger; use e.g. "@weekly" to automate
    catchup=False,
    max_active_runs=1,
    tags=["canscan", "detection", "yolo", "mlops"],
) as dag:

    @task
    def verify_dataset() -> dict:
        """Fail if images and YOLO label files don't match."""
        images_dir = DATASET_ROOT / "images"
        labels_dir = DATASET_ROOT / "labels"

        for d in (DATASET_ROOT, images_dir, labels_dir):
            if not d.exists():
                raise FileNotFoundError(f"{d} not found. Check the ./data volume mount.")

        images = {p.stem: p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS}
        labels = {p.stem for p in labels_dir.glob("*.txt")}

        if not images:
            raise ValueError(f"No images found in {images_dir}")

        missing_labels = sorted(set(images) - labels)
        orphan_labels = sorted(labels - set(images))
        if missing_labels:
            logger.warning("%d images without labels, e.g. %s", len(missing_labels), missing_labels[:5])
        if orphan_labels:
            logger.warning("%d labels without images, e.g. %s", len(orphan_labels), orphan_labels[:5])
        if missing_labels or orphan_labels:
            raise ValueError(
                f"Dataset mismatch: {len(missing_labels)} images lack labels, "
                f"{len(orphan_labels)} labels lack images."
            )

        logger.info("Dataset verified: %d image/label pairs", len(images))
        return {"dataset_root": str(DATASET_ROOT), "num_pairs": len(images)}

    @task
    def split_dataset(verify_info: dict) -> dict:
        """Create train/val/test image lists and a data.yaml for Ultralytics."""
        images = sorted(
            p for p in (DATASET_ROOT / "images").iterdir() if p.suffix.lower() in IMAGE_EXTS
        )
        random.Random(SEED).shuffle(images)

        n = len(images)
        n_train = int(n * SPLIT_RATIOS["train"])
        n_val = int(n * SPLIT_RATIOS["val"])
        splits = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:],
        }

        WORK_DIR.mkdir(parents=True, exist_ok=True)
        for name, files in splits.items():
            (WORK_DIR / f"{name}.txt").write_text("\n".join(str(f) for f in files) + "\n")
            logger.info("%s: %d images", name, len(files))

        # Ultralytics finds labels by swapping /images/ for /labels/ in each path.
        # Update names/nc to match your classes (or point to your existing data.yaml).
        existing_yaml = DATASET_ROOT / "data.yaml"
        data_yaml = WORK_DIR / "data.yaml"
        if existing_yaml.exists():
            import yaml

            cfg = yaml.safe_load(existing_yaml.read_text())
        else:
            raise FileNotFoundError(
                f"{existing_yaml} not found. It must define `names` for your classes."
            )
        cfg.update({
            "path": str(DATASET_ROOT),
            "train": str(WORK_DIR / "train.txt"),
            "val": str(WORK_DIR / "val.txt"),
            "test": str(WORK_DIR / "test.txt"),
        })
        data_yaml.write_text(yaml.safe_dump(cfg))

        return {"data_yaml": str(data_yaml), "counts": {k: len(v) for k, v in splits.items()}}

    @task(execution_timeout=timedelta(hours=6))
    def train_detector(split_info: dict) -> dict:
        """Train YOLOv8 and return the path to the best weights."""
        from ultralytics import YOLO

        model = YOLO(TRAIN_PARAMS["model"])
        results = model.train(
            data=split_info["data_yaml"],
            epochs=TRAIN_PARAMS["epochs"],
            imgsz=TRAIN_PARAMS["imgsz"],
            batch=TRAIN_PARAMS["batch"],
            seed=SEED,
            project=str(ARTIFACTS_DIR / "runs"),
            name="train",
            exist_ok=True,
        )
        best = Path(results.save_dir) / "weights" / "best.pt"
        if not best.exists():
            raise FileNotFoundError(f"Training finished but {best} was not created")

        return {"weights": str(best), "data_yaml": split_info["data_yaml"], **TRAIN_PARAMS}

    @task(execution_timeout=timedelta(hours=1))
    def evaluate_detector(train_info: dict) -> dict:
        """Evaluate the trained weights on the test split."""
        from ultralytics import YOLO

        if not train_info or "weights" not in train_info:
            raise ValueError("No training output received from train_detector")

        logger.info("Evaluating weights: %s", train_info["weights"])
        model = YOLO(train_info["weights"])
        res = model.val(data=train_info["data_yaml"], split="test")

        metrics = {
            "precision": float(res.box.mp),
            "recall": float(res.box.mr),
            "mAP50": float(res.box.map50),
            "mAP50-95": float(res.box.map),
        }
        logger.info("Detection evaluation results: %s", metrics)
        return {"weights": train_info["weights"], "metrics": metrics}

    @task
    def export_artifacts(eval_info: dict) -> dict:
        """Copy weights and metrics into the artifacts folder."""
        weights_dir = ARTIFACTS_DIR / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)

        dest = weights_dir / "best.pt"
        shutil.copy2(eval_info["weights"], dest)
        (ARTIFACTS_DIR / "metrics.json").write_text(json.dumps(eval_info["metrics"], indent=2))

        logger.info("Artifacts stored in %s", ARTIFACTS_DIR)
        return {"artifacts_dir": str(ARTIFACTS_DIR), "weights": str(dest), "status": "exported"}

    # Dependencies are inferred from the data passed between tasks
    verified = verify_dataset()
    split = split_dataset(verified)
    trained = train_detector(split)
    evaluated = evaluate_detector(trained)
    export_artifacts(evaluated)