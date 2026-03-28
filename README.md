# 🌿 Crop Disease Detection — Multi-Layer AI System
### Phase 1: Layer 1 (DL Model) + Layer 2 (Intelligence Engine)

> **Built for Google Colab. Part of a larger AgriTech platform.**  
> Handles: Disease Detection → Severity → Remedy → Weather × Economic Intelligence → DB JSON

---

## 📁 Folder Structure

```
crop disease/
│
├── doom.ipynb                  ← Google Colab notebook (upload here)
├── mobilenetv2_plant.pth       ← Trained MobileNetV2 model (38 classes)
├── leaf.jpg                    ← Sample test image
│
├── intelligence_engine.py      ← 🧠 Layer 2: Intelligence Engine
├── disease_protocols.json      ← 📋 Rule database: 38 diseases × all fields
├── colab_final_cell.py         ← 📌 Copy-paste this into last cell of doom.ipynb
│
└── README.md                   ← This file
```

---

## 🏗️ Architecture Overview

```
                     ┌─────────────────┐
                     │   Leaf Image    │
                     └────────┬────────┘
                              │
                     ┌────────▼────────┐
                     │  LAYER 1 (DL)   │
                     │  MobileNetV2    │
                     │  38 Classes     │
                     └────────┬────────┘
                              │
              disease_key + confidence (0–1)
                              │
                     ┌────────▼────────────────────────┐
                     │    LAYER 2: Intelligence Engine  │
                     │                                  │
                     │  ✅ Severity Flag                │
                     │  ✅ First-Aid Remedy             │
                     │  ✅ Granular Action Plan         │
                     │  ✅ Weather-Contextual Advice    │
                     │  ✅ Yield & Economic Impact      │
                     │  ✅ Marketplace Routing          │
                     └────────┬────────────────────────┘
                              │
                 ┌────────────┴────────────┐
                 │                         │
        ┌────────▼──────┐       ┌──────────▼────────┐
        │  Human-Readable│       │  JSON → DB Payload │
        │  Printed Report│       │  (sent to backend) │
        └───────────────┘       └───────────────────┘
```

---

## ⚡ Quick Start (Google Colab)

### Step 1: Upload these 4 files to Colab
```
1. mobilenetv2_plant.pth
2. leaf.jpg  (or your own image)
3. disease_protocols.json
4. intelligence_engine.py
```

### Step 2: Run existing cells 1–4 in `doom.ipynb`
(They load the model + transform the image)

### Step 3: Replace the LAST cell with `colab_final_cell.py`
Open `colab_final_cell.py`, copy everything, paste into the last cell of your notebook.

### Step 4: Run! You'll get output like:
```
═════════════════════════════════════════════════════════════
  🌿  CROP DISEASE INTELLIGENCE REPORT
═════════════════════════════════════════════════════════════
  🦠  Disease Name    : Tomato Late Blight (Phytophthora infestans) — EMERGENCY
  🌾  Crop            : Tomato
  📊  Confidence      : 91.20%
  🚨  Severity        : 🔴 HIGH — Immediate action required

  💊  FIRST-AID REMEDY:
      EMERGENCY: Can destroy field in 3 days. Apply Metalaxyl + Mancozeb IMMEDIATELY.

  📋  GRANULAR ACTION PLAN:
      ✅ Step 1: STOP ALL IRRIGATION
      ✅ Step 2: Apply Metalaxyl + Mancozeb (Ridomil Gold) 2.5g/L IMMEDIATELY
      ...

  🌦️  WEATHER-CONTEXTUAL ADVICE:
      🌧️ RAIN EXPECTED — delay spraying. Wait for dry window.

  📉  YIELD & ECONOMIC IMPACT:
      Expected Yield Loss : 80%
      Estimated Loss      : ₹56,000

  🛒  MARKETPLACE ROUTING:
      🔹 Metalaxyl + Mancozeb (Ridomil Gold)
      🔹 Cymoxanil 8% + Mancozeb 64%
      📌 Contact your nearest Krishi Seva Kendra or AgriShop.
═════════════════════════════════════════════════════════════
```

---

## 🗂️ JSON Payload Sent to Backend

```json
{
  "disease": "Tomato Late Blight — EMERGENCY",
  "disease_key": "Tomato___Late_blight",
  "crop": "Tomato",
  "confidence": 91.2,
  "severity": "🔴 HIGH — Immediate action required",
  "first_aid": "EMERGENCY: Apply Metalaxyl + Mancozeb IMMEDIATELY...",
  "action_plan": ["Step 1: STOP IRRIGATION", "Step 2: Apply Ridomil Gold..."],
  "weather_advice": "Rain expected — delay spraying",
  "yield_loss_pct": 80,
  "economic_loss_rs": 56000,
  "marketplace": {
    "recommended_products": ["Metalaxyl + Mancozeb", "Cymoxanil + Mancozeb"],
    "product_type": "Systemic Fungicide — EMERGENCY"
  },
  "location": { "lat": 21.1458, "lon": 79.0882 },
  "timestamp": "2026-03-28T12:30:00Z",
  "top_k_predictions": [
    {"rank": 1, "disease_key": "Tomato___Late_blight", "confidence": 91.2},
    {"rank": 2, "disease_key": "Tomato___Early_blight", "confidence": 5.1},
    {"rank": 3, "disease_key": "Potato___Late_blight",  "confidence": 2.3}
  ]
}
```

---

## 🧩 Key Files Explained

### `intelligence_engine.py`

| Class/Function | Purpose |
|---|---|
| `IntelligenceEngine.__init__()` | Loads `disease_protocols.json` |
| `.analyze(disease_key, confidence, location, crop_area_acres)` | Main method — returns full result dict |
| `._get_severity(confidence)` | HIGH / MEDIUM / LOW based on confidence |
| `._get_weather_advice(location, disease_key)` | Calls OpenWeatherMap API |
| `._compute_economic_impact()` | Yield loss × area × market price |
| `print_intelligence_report(result)` | Human-readable terminal output |

### `disease_protocols.json`

Contains all **38 PlantVillage disease classes** with:
- `display_name` — clean readable name
- `crop` — which crop it affects
- `first_aid` — immediate action in plain language
- `action_plan` — step-by-step treatment list
- `yield_loss_pct` — expected yield damage
- `economic_loss_per_acre_rs` — financial impact
- `marketplace.recommended_products` — what to buy

---

## 🔧 Configuration

### Add your location (for weather advice)
In `colab_final_cell.py`, line:
```python
LOCATION = None  # Replace with (lat, lon) e.g. (21.1458, 79.0882)
```
→ Get this from Priyanshu's GPS integration

### Add OpenWeatherMap API key (FREE)
In `intelligence_engine.py`, line:
```python
WEATHER_API_KEY = "YOUR_OPENWEATHERMAP_API_KEY"
```
→ Get free key at: https://openweathermap.org/api (Free tier: 60 calls/min)

---

## 🚀 Next Steps (Phase 2+)

| Phase | Task | Owner |
|---|---|---|
| Phase 2 | FastAPI backend server | You |
| Phase 2 | MongoDB/PostgreSQL DB integration | Priyanshu |
| Phase 2 | GPS/location module | Priyanshu |
| Phase 3 | Sarvam STT online API | You |
| Phase 3 | Sarvam Edge (offline, 30MB) | You |
| Phase 3 | Sarvam Bulbul TTS (Hindi/Marathi) | Priyanshu |
| Phase 4 | Retrain model on 2 lakh+ dataset | You |
| Phase 4 | EfficientNet upgrade (better accuracy) | You |

---

## 🌐 Model Details

| Attribute | Value |
|---|---|
| Architecture | MobileNetV2 |
| Output classes | 38 (PlantVillage dataset) |
| Input size | 224 × 224 RGB |
| Normalization | [0.485, 0.456, 0.406] / [0.229, 0.224, 0.225] |
| File | `mobilenetv2_plant.pth` |

---

## 📞 Contacts

| Task | Person |
|---|---|
| GPS / Location integration | Priyanshu |
| Sarvam TTS voice output | Priyanshu |
| Backend DB / server | Team discussion |
| Model training (2L+ dataset) | You |

---

## 🏛️ Disease Classes Supported (38)

<details>
<summary>Click to expand all 38 classes</summary>

1. Apple — Apple Scab
2. Apple — Black Rot
3. Apple — Cedar Apple Rust
4. Apple — Healthy
5. Blueberry — Healthy
6. Cherry — Powdery Mildew
7. Cherry — Healthy
8. Corn/Maize — Gray Leaf Spot
9. Corn/Maize — Common Rust
10. Corn/Maize — Northern Leaf Blight
11. Corn/Maize — Healthy
12. Grape — Black Rot
13. Grape — Esca (Black Measles)
14. Grape — Leaf Blight
15. Grape — Healthy
16. Orange — Citrus Greening (HLB)
17. Peach — Bacterial Spot
18. Peach — Healthy
19. Bell Pepper — Bacterial Spot
20. Bell Pepper — Healthy
21. Potato — Early Blight
22. Potato — Late Blight ⚠️
23. Potato — Healthy
24. Raspberry — Healthy
25. Soybean — Healthy
26. Squash — Powdery Mildew
27. Strawberry — Leaf Scorch
28. Strawberry — Healthy
29. Tomato — Bacterial Spot
30. Tomato — Early Blight
31. Tomato — Late Blight ⚠️
32. Tomato — Leaf Mold
33. Tomato — Septoria Leaf Spot
34. Tomato — Spider Mites
35. Tomato — Target Spot
36. Tomato — Yellow Leaf Curl Virus ⚠️
37. Tomato — Mosaic Virus
38. Tomato — Healthy

</details>
