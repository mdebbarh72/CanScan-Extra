
from pydantic import BaseModel


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class DetectionResult(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox: BoundingBox


class DetectionResponse(BaseModel):
    count: int
    detections: list[DetectionResult]


class ClassificationResponse(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    probabilities: dict[str, float]


class PredictionResponse(BaseModel):
    classification: ClassificationResponse
    detection: DetectionResponse
