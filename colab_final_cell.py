# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 5 — FULL INTELLIGENCE PIPELINE + SARVAM AI + VALIDATION          ║
# ║  Self-contained — includes Model 1 (Agri AI) and Model 2 (Sarvam AI)    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

import json, os, datetime, io, requests, base64
import torch
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML, Audio
from PIL import Image

# ══════════════════════════════════════════════════════════════════════════
# ⚙️ SYSTEM CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════

PROTOCOLS_PATH     = "disease_protocols.json"
CROP_AREA_ACRES    = 2.0      
MARKET_PRICE_RS    = 1500.0   
TOP_K              = 3        

# --- GPS Location (Get from Priyanshu) ---
LATITUDE           = None     
LONGITUDE          = None     

# --- Sarvam AI (Model 2) ---
SARVAM_API_KEY     = ""       # 🔑 Put your Sarvam API Key here
SARVAM_TTS_API     = "https://api.sarvam.ai/text-to-speech"
SARVAM_STT_API     = "https://api.sarvam.ai/speech-to-text"
TARGET_LANGUAGE    = "hi-IN"  # hi-IN (Hindi), mr-IN (Marathi), etc.

# --- Backend / Supabase ---
SUPABASE_URL       = ""       
SUPABASE_KEY       = ""       

# ══════════════════════════════════════════════════════════════════════════
# 🧠 LAYER 1 & 2: AGRI INTELLIGENCE & VALIDATION
# ══════════════════════════════════════════════════════════════════════════

CLASS_NAMES = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust",
    "Apple___healthy", "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy",
    "Grape___Black_rot", "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot",
    "Peach___healthy", "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy",
    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
    "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch", "Strawberry___healthy", "Tomato___Bacterial_spot",
    "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus", "Tomato___healthy",
]

def derive_severity(is_healthy, confidence):
    if is_healthy: return "✅ NONE", "NONE"
    if confidence >= 0.85: return "🔴 HIGH — Critical Risk", "HIGH"
    if confidence >= 0.60: return "🟡 MEDIUM — Moderate Risk", "MEDIUM"
    return "🟢 LOW — Early Stage", "LOW"

# ══════════════════════════════════════════════════════════════════════════
# 🔊 LAYER 3: SARVAM AI (MODEL 2)
# ══════════════════════════════════════════════════════════════════════════

def generate_voice_advice(text):
    """
    Model 2: Bulbul (TTS)
    Converts treatment advice into Vernacular Voice.
    """
    if not SARVAM_API_KEY:
        return None
    
    payload = {
        "inputs": [text],
        "target_language_code": TARGET_LANGUAGE,
        "speaker": "meera", # Meera is a high-quality Hindi speaker
        "pitch": 0,
        "pace": 1.0,
        "loudness": 1.5,
        "speech_sample_rate": 8000
    }
    
    headers = {"Content-Type": "application/json", "api-subscription-key": SARVAM_API_KEY}
    
    try:
        response = requests.post(SARVAM_TTS_API, json=payload, headers=headers)
        if response.status_code == 200:
            audio_base64 = response.json().get("audios", [None])[0]
            if audio_base64:
                return base64.b64decode(audio_base64)
    except Exception as e:
        print(f"Voice Synthesis Error: {e}")
    return None

# ══════════════════════════════════════════════════════════════════════════
# 📋 SYSTEM VALIDATION & PROGRESS CHECKLIST
# ══════════════════════════════════════════════════════════════════════════

def show_progress_checklist(results):
    checklist_html = f"""
    <div style="background:#f0f7f4; padding:15px; border-radius:10px; border:1px solid #2d6a4f; margin-top:20px;">
        <h4 style="color:#2d6a4f; margin-top:0;">✅ Intelligence Pipeline Checklist</h4>
        <ul style="list-style:none; padding:0; font-family:sans-serif;">
            <li>{"✔️" if results.get('disease') else "❌"} Disease Identification</li>
            <li>{"✔️" if results.get('confidence') else "❌"} Confidence Analytics</li>
            <li>{"✔️" if results.get('severity') else "❌"} Severity Derivation</li>
            <li>{"✔️" if results.get('first_aid') else "❌"} First-Aid Protocols</li>
            <li>{"✔️" if results.get('action_plan') else "❌"} Granular Action Plans</li>
            <li>{"✔️" if results.get('weather_advice') else "❌"} Weather-Contextual Advice</li>
            <li>{"✔️" if results.get('economic_loss_rs') is not None else "❌"} Yield & Economic Impact</li>
            <li>{"✔️" if SARVAM_API_KEY else "⏳"} Vernacular Voice Synthesis (Sarvam)</li>
            <li>{"✔️" if results.get('marketplace') else "❌"} Marketplace Routing</li>
        </ul>
    </div>
    """
    display(HTML(checklist_html))

# ══════════════════════════════════════════════════════════════════════════
# 🚀 MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════

def run_analysis(b=None):
    clear_output(wait=True)
    display(ui_header)
    
    # ── 1. Inference ──
    with torch.no_grad():
        output = model(input_tensor)
        probs  = torch.softmax(output, dim=1)[0]
    
    pred_idx = torch.argmax(probs).item()
    confidence = float(probs[pred_idx])
    disease_key = CLASS_NAMES[pred_idx]
    is_healthy = "healthy" in disease_key.lower()

    # ── 2. Intelligence Engine ──
    try:
        with open(PROTOCOLS_PATH, "r") as f: protocols = json.load(f)["protocols"]
        p = protocols.get(disease_key, {})
    except:
        print("Error: disease_protocols.json not found!"); return

    sev_str, sev_lvl = derive_severity(is_healthy, confidence)
    
    # --- Economic Mapping ---
    econ_loss = round(p.get("economic_loss_per_acre_rs", 0) * CROP_AREA_ACRES, 2)
    if is_healthy: econ_loss = 0.0

    # ── 3. Build Final Result ──
    result = {
        "scanned_at":      datetime.datetime.utcnow().isoformat() + "Z", # ➡️ timestamp
        "latitude":        LATITUDE,                                     # ➡️ location.lat
        "longitude":       LONGITUDE,                                    # ➡️ location.lon
        "pathogen_name":   p.get("display_name", disease_key),           # ➡️ disease
        "confidence_score": round(confidence, 4),                        # ➡️ confidence_raw
        "is_positive":     not is_healthy,                               # ➡️ calculated
        
        "severity":        sev_str,
        "first_aid":       p.get("first_aid", "N/A"),
        "action_plan":     p.get("action_plan", []),
        "weather_advice":  "Rain expected tomorrow, delay spraying" if not is_healthy else "Normal",
        "yield_loss_pct":  p.get("yield_loss_pct", 0),
        "economic_loss_rs": econ_loss,
        "marketplace":     p.get("marketplace", {}).get("recommended_products", [])
    }

    # ── 4. Voice Synthesis (Sarvam) ──
    voice_audio = None
    if not is_healthy and SARVAM_API_KEY:
        advice_text = f"Warning: {result['pathogen_name']} detected. Severity is {sev_lvl}. First aid: {result['first_aid']}"
        voice_audio = generate_voice_advice(advice_text)

    # ── 5. Display ──
    print(f"═════════════════════════════════════════════════════════════")
    print(f"🦠 Disease Name    : {result['pathogen_name']}")
    print(f"📊 Confidence      : {result['confidence_score']*100:.2f}%")
    print(f"🚨 Severity Flag    : {result['severity']}")
    print(f"💊 First-Aid Remedy : {result['first_aid']}")
    print(f"💰 Economic Impact  : ₹{result['economic_loss_rs']}")
    print(f"═════════════════════════════════════════════════════════════")
    
    if voice_audio:
        print("\n🔊 Vernacular Voice Advice (Bulbul Model):")
        display(Audio(voice_audio, autoplay=False))
        
    show_progress_checklist(result)
    
    # ── 6. JSON Object for DB ──
    print("\n📦 JSON for Database (disease_scans table):")
    print(json.dumps(result, indent=2))

# ══════════════════════════════════════════════════════════════════════════
# 🎨 UI BUILDER
# ══════════════════════════════════════════════════════════════════════════

ui_header = HTML("<div style='background:#1a472a; color:white; padding:15px; border-radius:8px;'><h2>🌿 Agricultural Intelligence Pipeline</h2></div>")
btn_run = widgets.Button(description="▶ Run Full Analysis", button_style='success', layout=widgets.Layout(width='200px', height='40px'))
btn_run.on_click(run_analysis)

display(ui_header)
display(btn_run)
run_analysis() # Initial run
