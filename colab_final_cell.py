# ╔══════════════════════════════════════════════════════════════════╗
# ║  CELL 5 — INTELLIGENCE ENGINE (PASTE THIS IN YOUR LAST CELL)   ║
# ║  Replaces the old: print("Predicted class index:", pred_idx)    ║
# ╚══════════════════════════════════════════════════════════════════╝
#
# HOW TO USE IN GOOGLE COLAB:
# 1. Upload 'disease_protocols.json' to Colab (use the file upload cell)
# 2. Upload 'intelligence_engine.py' to Colab (use the file upload cell)  
# 3. Run this cell AFTER all previous cells have run
# ────────────────────────────────────────────────────────────────────

import json
import datetime
import os
import sys

# ── Step 1: Install requests if not available ─────────────────────
try:
    import requests
except ImportError:
    os.system("pip install requests -q")
    import requests

# ── Step 2: Load the intelligence engine ──────────────────────────
# Make sure intelligence_engine.py is uploaded to Colab
from intelligence_engine import IntelligenceEngine, print_intelligence_report

# ── Step 3: Define class names (38 PlantVillage classes) ───────────
CLASS_NAMES = [
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

# ── Step 4: Load intelligence engine ─────────────────────────────
engine = IntelligenceEngine("disease_protocols.json")

# ── Step 5: Get top-k predictions ─────────────────────────────────
TOP_K = 3
topk_probs, topk_indices = torch.topk(probs, TOP_K)

pred_idx        = topk_indices[0].item()
pred_confidence = topk_probs[0].item()
disease_key     = CLASS_NAMES[pred_idx]

# ── Step 6: Optional — set your location & farm details ───────────
# Get from Priyanshu's GPS module. For now, set manually:
LOCATION = None          # Replace with (lat, lon) e.g. (21.1458, 79.0882)
CROP_AREA_ACRES = 2.0   # How many acres the farmer has
MARKET_PRICE = 1500     # Rs per quintal (current mandi price)

# ── Step 7: Run intelligence engine ────────────────────────────────
result = engine.analyze(
    disease_key       = disease_key,
    confidence        = pred_confidence,
    location          = LOCATION,
    crop_area_acres   = CROP_AREA_ACRES,
    market_price_rs_per_quintal = MARKET_PRICE,
)

# ── Step 8: Add top-k alternatives ────────────────────────────────
result["top_k_predictions"] = [
    {
        "rank":        i + 1,
        "disease_key": CLASS_NAMES[topk_indices[i].item()],
        "confidence":  round(topk_probs[i].item() * 100, 2),
    }
    for i in range(TOP_K)
]

# ── Step 9: Print the full intelligence report ────────────────────
print_intelligence_report(result)

# ── Step 10: Print raw JSON (for DB / backend) ────────────────────
print("\n📦 RAW JSON PAYLOAD (Send to Backend/DB):")
print("─" * 65)
print(json.dumps(result, indent=2, ensure_ascii=False))
