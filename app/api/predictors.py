
from pathlib import Path

import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from ultralytics import YOLO

from app.api.schemas import (
    BoundingBox,
    ClassificationResponse,
    DetectionResponse,
    DetectionResult,
    PredictionResponse,
)


BASE_DIR = Path(__file__).resolve().parent.parent

YOLO_MODEL_PATH = BASE_DIR / "models" / "best.pt"
GOOGLENET_MODEL_PATH = BASE_DIR / "models" / "googlenet.keras"

# Must match your classifier's training label order.
CLASS_NAMES = [
    "glioma",
    "meningioma",
    "no_tumor",
    "pituitary",
]

yolo_model = None
googlenet_model = None


def load_models():
    global yolo_model, googlenet_model

    yolo_model = YOLO(str(YOLO_MODEL_PATH))
    googlenet_model = load_model(str(GOOGLENET_MODEL_PATH))


def predict_image(image: Image.Image) -> PredictionResponse:
    if yolo_model is None or googlenet_model is None:
        raise RuntimeError("Models have not been loaded.")

    # YOLOv8 detection
    result = yolo_model.predict(
        source=np.array(image),
        conf=0.25,
        imgsz=640,
        verbose=False,
    )[0]

    detections = []

    for box in result.boxes:
        class_id = int(box.cls.item())
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        detections.append(
            DetectionResult(
                class_id=class_id,
                class_name=result.names[class_id],
                confidence=float(box.conf.item()),
                bbox=BoundingBox(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                ),
            )
        )

    # GoogLeNet classification
    resized = image.resize((224, 224))
    array = np.asarray(resized, dtype=np.float32) / 255.0
    batch = np.expand_dims(array, axis=0)

    probabilities = googlenet_model.predict(batch, verbose=0)[0]
    predicted_class_id = int(np.argmax(probabilities))

    class_name = (
        CLASS_NAMES[predicted_class_id]
        if predicted_class_id < len(CLASS_NAMES)
        else str(predicted_class_id)
    )

    classification = ClassificationResponse(
        class_id=predicted_class_id,
        class_name=class_name,
        confidence=float(probabilities[predicted_class_id]),
        probabilities={
            (
                CLASS_NAMES[i]
                if i < len(CLASS_NAMES)
                else str(i)
            ): float(probability)
            for i, probability in enumerate(probabilities)
        },
    )

    return PredictionResponse(
        classification=classification,
        detection=DetectionResponse(
            count=len(detections),
            detections=detections,
        ),
    )
