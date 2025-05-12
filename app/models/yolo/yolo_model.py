from ultralytics import YOLO
from PIL import Image
import io

import torch
from ultralytics.nn.tasks import DetectionModel

torch.serialization.add_safe_globals([DetectionModel])

class YOLOModel:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    def predict(self, image_bytes: bytes):
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        results = self.model(image)
        return results[0].tojson()
