"""
model_loader.py
===============
Layer 1 — MobileNetV2 model loader and inference runner.
Loads the .pth model once at startup and exposes a predict() method.
"""

import os
import io
import logging
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

logger = logging.getLogger(__name__)

# ── 38 PlantVillage class names (must match training order) ─────
CLASS_NAMES: List[str] = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

# ── Image preprocessing transform ─────────────────────────────────
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


class PlantDiseaseModel:
    """
    Singleton model loader.
    Loads MobileNetV2 from .pth checkpoint and runs inference on image bytes.
    """

    def __init__(self, model_path: str = "mobilenetv2_plant.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"🔧 Loading model on device: {self.device}")
        self.model = self._load_model(model_path)
        logger.info(f"✅ Model loaded: {model_path}")

    def _load_model(self, path: str) -> nn.Module:
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(model.last_channel, len(CLASS_NAMES)),
        )
        state = torch.load(path, map_location=self.device)
        model.load_state_dict(state)
        model.to(self.device)
        model.eval()
        return model

    def predict(
        self,
        image_bytes: bytes,
        top_k: int = 3,
    ) -> Tuple[str, float, List[dict]]:
        """
        Run inference on raw image bytes.

        Returns
        -------
        disease_key   : str   — top predicted class key
        confidence    : float — softmax probability (0–1)
        top_k_results : list  — [{"rank", "disease_key", "confidence"}, ...]
        """
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = TRANSFORM(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(tensor)
            probs  = torch.softmax(output, dim=1)[0]

        k = min(top_k, len(CLASS_NAMES))
        topk_probs, topk_indices = torch.topk(probs, k)

        disease_key = CLASS_NAMES[topk_indices[0].item()]
        confidence  = float(topk_probs[0].item())

        top_k_results = [
            {
                "rank":        i + 1,
                "disease_key": CLASS_NAMES[topk_indices[i].item()],
                "confidence":  round(float(topk_probs[i].item()) * 100, 2),
            }
            for i in range(k)
        ]

        return disease_key, confidence, top_k_results


# ── Singleton ─────────────────────────────────────────────────────
_model_instance: Optional[PlantDiseaseModel] = None

def get_model(model_path: Optional[str] = None) -> PlantDiseaseModel:
    global _model_instance
    if _model_instance is None:
        path = model_path or os.getenv("MODEL_PATH", "mobilenetv2_plant.pth")
        _model_instance = PlantDiseaseModel(path)
    return _model_instance
