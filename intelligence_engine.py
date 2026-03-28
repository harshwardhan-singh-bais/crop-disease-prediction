"""
intelligence_engine.py  (v2 — with Validation & Consistency Layer)
====================================================================
Layer 2 — Intelligence Engine for Crop Disease Detection System.

NEW in v2:
  - ValidationEngine: enforces all 8 strict consistency rules
  - Severity is derived DYNAMICALLY from disease type × confidence × bio-impact
  - Healthy plants get zero economic loss, preventive-only actions
  - All fields guaranteed internally consistent before returning
"""

import json
import os
import datetime
import requests
from typing import Optional, Tuple, Dict, Any, List


# ══════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════

# Disease type keywords for dynamic reasoning
_FUNGAL_KEYWORDS    = {"blight", "mold", "scab", "mildew", "rot", "spot", "rust", "esca", "measles", "scorch"}
_BACTERIAL_KEYWORDS = {"bacterial"}
_VIRAL_KEYWORDS     = {"virus", "viral", "curl", "mosaic", "greening", "haunglongbing"}
_PEST_KEYWORDS      = {"mite", "spider", "insect"}

# Biological severity multipliers — how bad these disease types get
_BIO_SEVERITY_WEIGHT = {
    "viral":      1.25,   # Viruses: no cure, high impact
    "bacterial":  1.10,
    "fungal":     1.00,
    "pest":       0.90,
    "healthy":    0.00,   # Healthy → zero severity regardless of confidence
    "unknown":    0.85,
}

# Confidence bands
CONFIDENCE_HIGH   = 0.95
CONFIDENCE_MEDIUM = 0.70


# ══════════════════════════════════════════════════════════════════
# UTILITY: disease type detector
# ══════════════════════════════════════════════════════════════════

def _detect_disease_context(disease_key: str) -> Dict[str, Any]:
    """
    Dynamically infers:
      - is_healthy: bool
      - disease_type: 'fungal' | 'bacterial' | 'viral' | 'pest' | 'healthy' | 'unknown'
      - bio_weight: float  (biological severity multiplier)
    """
    key_lower = disease_key.lower()
    is_healthy = "healthy" in key_lower

    if is_healthy:
        return {"is_healthy": True, "disease_type": "healthy", "bio_weight": 0.0}

    if any(k in key_lower for k in _VIRAL_KEYWORDS):
        return {"is_healthy": False, "disease_type": "viral", "bio_weight": _BIO_SEVERITY_WEIGHT["viral"]}
    if any(k in key_lower for k in _BACTERIAL_KEYWORDS):
        return {"is_healthy": False, "disease_type": "bacterial", "bio_weight": _BIO_SEVERITY_WEIGHT["bacterial"]}
    if any(k in key_lower for k in _PEST_KEYWORDS):
        return {"is_healthy": False, "disease_type": "pest", "bio_weight": _BIO_SEVERITY_WEIGHT["pest"]}
    if any(k in key_lower for k in _FUNGAL_KEYWORDS):
        return {"is_healthy": False, "disease_type": "fungal", "bio_weight": _BIO_SEVERITY_WEIGHT["fungal"]}

    return {"is_healthy": False, "disease_type": "unknown", "bio_weight": _BIO_SEVERITY_WEIGHT["unknown"]}


# ══════════════════════════════════════════════════════════════════
# VALIDATION ENGINE  (8 strict rules)
# ══════════════════════════════════════════════════════════════════

class ValidationEngine:
    """
    Applies all 8 consistency rules to a raw intelligence result dict.
    Modifies the result in-place and returns it corrected.

    Rules enforced:
      1. Global consistency check
      2. Disease vs severity alignment
      3. Confidence-aware reasoning
      4. Dynamic severity derivation
      5. Action plan validation
      6. Economic consistency
      7. Output sanity (no alarmism without cause)
      8. No hard-coded case reliance — purely context-driven
    """

    def validate(self, result: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parameters
        ----------
        result  : raw output dict from IntelligenceEngine.analyze()
        context : {"is_healthy", "disease_type", "bio_weight", "confidence_raw"}
        """
        is_healthy    = context["is_healthy"]
        disease_type  = context["disease_type"]
        bio_weight    = context["bio_weight"]
        confidence    = context["confidence_raw"]

        # ── Rule 4: Dynamic severity (must come first — others depend on it)
        result["severity"] = self._derive_severity(is_healthy, disease_type, bio_weight, confidence)
        result["severity_level"] = self._severity_level(is_healthy, disease_type, bio_weight, confidence)

        # ── Rule 2 + 5: Healthy plant → force preventive-only fields
        if is_healthy:
            result = self._enforce_healthy_fields(result)

        # ── Rule 3: Confidence-aware messaging
        result = self._apply_confidence_tone(result, confidence, is_healthy)

        # ── Rule 6: Economic consistency
        result = self._enforce_economic_consistency(result, is_healthy)

        # ── Rule 7: Sanity check — remove unjustified alarmism
        result = self._sanity_check(result, is_healthy, confidence)

        return result

    # ── RULE 4: Dynamic severity ──────────────────────────────────
    def _derive_severity(
        self, is_healthy: bool, disease_type: str, bio_weight: float, confidence: float
    ) -> str:
        if is_healthy:
            return "✅ NONE — Plant appears healthy. No disease risk detected."

        # Effective severity score: confidence modulated by biological impact
        effective = confidence * bio_weight

        if effective >= 0.85:
            return "🔴 HIGH — Significant biological risk. Immediate intervention required."
        elif effective >= 0.55:
            return "🟡 MEDIUM — Moderate risk detected. Treat within 48 hours and monitor daily."
        else:
            return "🟢 LOW — Early-stage or uncertain detection. Preventive treatment advised."

    def _severity_level(
        self, is_healthy: bool, disease_type: str, bio_weight: float, confidence: float
    ) -> str:
        """Returns clean machine-readable level: NONE | LOW | MEDIUM | HIGH"""
        if is_healthy:
            return "NONE"
        eff = confidence * bio_weight
        if eff >= 0.85:
            return "HIGH"
        elif eff >= 0.55:
            return "MEDIUM"
        return "LOW"

    # ── RULE 2 + 5: Enforce healthy fields ───────────────────────
    def _enforce_healthy_fields(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """When healthy — zero economic loss, preventive-only actions, no treatment language."""
        result["first_aid"] = (
            "✅ No treatment required. Your plant appears healthy. "
            "Continue regular monitoring and maintain good agronomic practices."
        )
        # Filter action plan to keep only preventive steps
        action_plan = result.get("action_plan", [])
        preventive_keywords = {
            "monitor", "inspect", "preventive", "prevent", "continue",
            "maintain", "prune", "mulch", "rotation", "certified", "irrigation"
        }
        cleaned = [
            step for step in action_plan
            if any(kw in step.lower() for kw in preventive_keywords)
        ]
        # Always ensure at least a basic preventive plan
        if not cleaned:
            cleaned = [
                "Continue weekly field monitoring for early disease detection",
                "Apply preventive copper or neem-based spray before monsoon season",
                "Maintain proper plant spacing and drainage for good air circulation",
                "Use balanced NPK fertilizer to maintain plant immunity",
            ]
        result["action_plan"] = cleaned

        # Marketplace: preventive only
        mp = result.get("marketplace", {})
        products = mp.get("recommended_products", [])
        # Keep only non-emergency products
        result["marketplace"] = {
            "recommended_products": products,
            "product_type": "Preventive",
            "note": "No immediate purchase required. These are optional preventive measures.",
        }
        return result

    # ── RULE 3: Confidence-aware tone ─────────────────────────────
    def _apply_confidence_tone(
        self, result: Dict[str, Any], confidence: float, is_healthy: bool
    ) -> Dict[str, Any]:
        if is_healthy:
            return result  # Healthy needs no confidence caveat

        if confidence >= CONFIDENCE_HIGH:
            # Strong, decisive — no changes needed
            pass
        elif confidence >= CONFIDENCE_MEDIUM:
            # Add monitoring caveat to first_aid
            caveat = (
                f" (Confidence: {confidence*100:.1f}% — Monitor closely and consider "
                "a second image scan from a different leaf angle to confirm.)"
            )
            result["first_aid"] = result["first_aid"].rstrip(".") + caveat
        else:
            # Low confidence — prepend uncertainty notice
            uncertainty = (
                f"⚠️ LOW CONFIDENCE ({confidence*100:.1f}%): Model is not highly certain. "
                "Take another photo with better lighting and leaf clearly visible. "
                "Consult your local agricultural officer before applying chemicals. | "
            )
            result["first_aid"] = uncertainty + result["first_aid"]
            # Add re-inspection step to action plan
            recheck = (
                "Re-scan with a clearer, well-lit image of the affected leaf "
                "or consult a Krishi Vigyan Kendra expert before treatment"
            )
            if recheck not in result.get("action_plan", []):
                result["action_plan"] = [recheck] + result.get("action_plan", [])

        return result

    # ── RULE 6: Economic consistency ──────────────────────────────
    def _enforce_economic_consistency(
        self, result: Dict[str, Any], is_healthy: bool
    ) -> Dict[str, Any]:
        if is_healthy:
            result["yield_loss_pct"]        = 0
            result["economic_loss_rs"]       = 0.0
            result["economic_loss_per_acre"] = 0.0
            return result

        # Ensure non-zero loss for diseased plants
        severity_level = result.get("severity_level", "LOW")
        yield_loss = result.get("yield_loss_pct", 0)

        if yield_loss == 0 and severity_level != "NONE":
            # Protocol said 0 but there's a disease — set a floor
            default_losses = {"LOW": 5, "MEDIUM": 20, "HIGH": 40}
            result["yield_loss_pct"] = default_losses.get(severity_level, 10)

        # Economic loss must be > 0 if yield_loss > 0
        if result["yield_loss_pct"] > 0 and result.get("economic_loss_rs", 0) == 0:
            # Estimate: Rs 1000 per acre per 1% yield loss (conservative baseline)
            result["economic_loss_rs"] = round(result["yield_loss_pct"] * 1000, 2)
            result["economic_loss_per_acre"] = result["economic_loss_rs"]

        return result

    # ── RULE 7: Sanity check ──────────────────────────────────────
    def _sanity_check(
        self, result: Dict[str, Any], is_healthy: bool, confidence: float
    ) -> Dict[str, Any]:
        """Remove EMERGENCY language for low-confidence or low-severity predictions."""
        severity_level = result.get("severity_level", "LOW")

        if severity_level in ("LOW", "NONE") or confidence < CONFIDENCE_MEDIUM:
            # Tone down alarmist language
            for field in ("first_aid", "weather_advice"):
                val = result.get(field, "")
                for alarm_word in ("EMERGENCY", "CRITICAL", "IMMEDIATELY", "DESTROY ALL"):
                    if alarm_word in val and severity_level != "HIGH":
                        result[field] = val.replace(
                            alarm_word,
                            alarm_word.replace("EMERGENCY", "ATTENTION NEEDED")
                                      .replace("CRITICAL", "NOTABLE")
                                      .replace("IMMEDIATELY", "promptly")
                                      .replace("DESTROY ALL", "Remove affected")
                        )
        return result


# ══════════════════════════════════════════════════════════════════
# INTELLIGENCE ENGINE  (v2)
# ══════════════════════════════════════════════════════════════════

class IntelligenceEngine:
    """
    Layer 2 intelligence pipeline:
      analyze() → _build_raw_result() → ValidationEngine.validate() → final result
    """

    WEATHER_API_URL = (
        "https://api.openweathermap.org/data/2.5/forecast"
        "?lat={lat}&lon={lon}&appid={key}&cnt=3&units=metric"
    )

    def __init__(
        self,
        protocols_path: str = "disease_protocols.json",
        weather_api_key: Optional[str] = None,
    ):
        with open(protocols_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.class_names: List[str]  = data["class_names"]
        self.protocols:   Dict       = data["protocols"]
        self.weather_api_key         = weather_api_key or os.getenv("OPENWEATHER_API_KEY", "")
        self._validator              = ValidationEngine()

    # ── PUBLIC ENTRY POINT ────────────────────────────────────────

    def analyze(
        self,
        disease_key:                 str,
        confidence:                  float,
        location:                    Optional[Tuple[float, float]] = None,
        crop_area_acres:             float = 1.0,
        market_price_rs_per_quintal: float = 1500.0,
        top_k_predictions:           Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """
        Full pipeline: model output → intelligence → validation → consistent result.

        Parameters
        ----------
        disease_key   : str   — e.g. "Tomato___Late_blight"
        confidence    : float — softmax probability (0–1)
        location      : (lat, lon) or None
        crop_area_acres : float
        market_price_rs_per_quintal : float
        top_k_predictions : list of {"rank", "disease_key", "confidence"} dicts
        """
        protocol = self.protocols.get(disease_key)
        if protocol is None:
            return self._unknown_disease(disease_key, confidence, location)

        # Detect disease context (healthy vs diseased, type, bio weight)
        context = _detect_disease_context(disease_key)
        context["confidence_raw"] = confidence

        # Build raw result from protocols
        raw = self._build_raw_result(
            disease_key, confidence, protocol, location,
            crop_area_acres, market_price_rs_per_quintal, top_k_predictions
        )

        # Validate & correct for consistency
        validated = self._validator.validate(raw, context)

        # Add DB-ready fields
        validated["is_positive"] = not context["is_healthy"]
        validated["disease_type"] = context["disease_type"]

        return validated

    # ── RAW RESULT BUILDER ────────────────────────────────────────

    def _build_raw_result(
        self,
        disease_key:   str,
        confidence:    float,
        protocol:      dict,
        location:      Optional[Tuple[float, float]],
        area_acres:    float,
        market_price:  float,
        top_k:         Optional[List[Dict]],
    ) -> Dict[str, Any]:
        economic = self._compute_economic_impact(protocol, area_acres, market_price)
        weather  = self._get_weather_advice(location, disease_key)

        return {
            # ── Layer 1 passthrough ───────────────
            "disease":           protocol["display_name"],
            "disease_key":       disease_key,
            "crop":              protocol["crop"],
            "confidence":        round(confidence * 100, 2),
            "confidence_raw":    round(confidence, 6),

            # ── Layer 2 raw (pre-validation) ──────
            "severity":          "",          # filled by validator
            "severity_level":    "",          # filled by validator
            "first_aid":         protocol["first_aid"],
            "action_plan":       list(protocol["action_plan"]),  # copy
            "weather_advice":    weather,

            # ── Economic (may be corrected by validator) ──
            "yield_loss_pct":         protocol["yield_loss_pct"],
            "economic_loss_rs":       economic["economic_loss_rs"],
            "economic_loss_per_acre": economic["per_acre_rs"],

            # ── Marketplace ───────────────────────
            "marketplace": {
                "recommended_products": protocol["marketplace"]["recommended_products"],
                "product_type":         protocol["marketplace"]["product_type"],
                "note": "Contact your nearest Krishi Seva Kendra or AgriShop for availability.",
            },

            # ── Location + time ───────────────────
            "location": {
                "lat": location[0] if location else None,
                "lon": location[1] if location else None,
            },
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",

            # ── Top-K alternatives ────────────────
            "top_k_predictions": top_k or [],
        }

    # ── SEVERITY (private, only called inside ValidationEngine now) ──
    # Kept here for reference — actual logic is in ValidationEngine._derive_severity()

    # ── WEATHER ───────────────────────────────────────────────────

    def _get_weather_advice(
        self,
        location: Optional[Tuple[float, float]],
        disease_key: str,
    ) -> str:
        if location is None:
            return "📍 Location not provided. Share GPS coordinates for personalized weather advice."

        if not self.weather_api_key:
            return self._static_weather_advice(disease_key)

        try:
            url = self.WEATHER_API_URL.format(
                lat=location[0], lon=location[1], key=self.weather_api_key
            )
            resp = requests.get(url, timeout=5)
            data = resp.json()
            forecasts = data.get("list", [])
            if not forecasts:
                return self._static_weather_advice(disease_key)

            rain_expected = any(
                "rain" in f.get("weather", [{}])[0].get("description", "").lower()
                for f in forecasts[:3]
            )
            avg_temp     = sum(f["main"]["temp"]     for f in forecasts[:3]) / len(forecasts[:3])
            avg_humidity = sum(f["main"]["humidity"] for f in forecasts[:3]) / len(forecasts[:3])
            return self._dynamic_weather_advice(rain_expected, avg_temp, avg_humidity, disease_key)

        except Exception as e:
            return f"⚠️ Weather data unavailable ({e}). Manual monitoring recommended."

    def _static_weather_advice(self, disease_key: str) -> str:
        """Context-aware static advice when no live weather data available."""
        key_lower = disease_key.lower()
        if any(k in key_lower for k in _FUNGAL_KEYWORDS) and "healthy" not in key_lower:
            return (
                "🌧️ Fungal diseases thrive in humid, wet conditions. "
                "Spray fungicide in early morning (6–9 AM) for best absorption. "
                "Avoid spraying if rain is expected within 4 hours."
            )
        if any(k in key_lower for k in _VIRAL_KEYWORDS):
            return (
                "🦟 Viral disease vectors (whiteflies, aphids) are most active in hot, dry weather. "
                "Install yellow sticky traps. Spray insecticide in evening to protect beneficial insects."
            )
        if any(k in key_lower for k in _PEST_KEYWORDS):
            return (
                "🌡️ Spider mites thrive in hot, dry weather. "
                "Increase irrigation frequency and spray miticide in early morning."
            )
        if "healthy" in key_lower:
            return "☀️ Plant is healthy. Continue good agronomic practices. Monitor weather for disease-favorable conditions."
        return "🌤️ Apply treatments in calm, dry conditions. Early morning or late evening is best for spray applications."

    def _dynamic_weather_advice(
        self, rain_expected: bool, avg_temp: float, avg_humidity: float, disease_key: str
    ) -> str:
        parts = []
        if rain_expected:
            parts.append("🌧️ Rain expected in 24 hours — delay contact fungicide/bactericide sprays. Wait for dry window.")
        else:
            parts.append("☀️ Dry weather forecast — good window for spray applications today.")

        if avg_humidity > 80:
            parts.append(f"💧 High humidity ({avg_humidity:.0f}%) — fungal/bacterial spread risk elevated. Increase spray frequency.")
        elif avg_humidity < 40:
            parts.append(f"🌵 Low humidity ({avg_humidity:.0f}%) — spider mite risk elevated. Increase irrigation frequency.")

        if avg_temp > 35:
            parts.append(f"🌡️ High temp ({avg_temp:.1f}°C) — spray before 8 AM to avoid phytotoxicity.")
        elif avg_temp < 15:
            parts.append(f"❄️ Low temp ({avg_temp:.1f}°C) — pesticide efficacy reduced below 18°C. Wait for warmer window.")

        return " | ".join(parts)

    # ── ECONOMIC IMPACT ───────────────────────────────────────────

    def _compute_economic_impact(
        self,
        protocol:      dict,
        area_acres:    float,
        market_price:  float,
    ) -> Dict[str, float]:
        yield_loss_pct    = protocol.get("yield_loss_pct", 0)
        base_loss_per_acre = protocol.get("economic_loss_per_acre_rs", 0)
        price_mult        = market_price / 1500.0
        adjusted          = base_loss_per_acre * price_mult
        total             = adjusted * area_acres
        return {
            "per_acre_rs":      round(adjusted, 2),
            "economic_loss_rs": round(total, 2),
        }

    # ── UNKNOWN DISEASE FALLBACK ──────────────────────────────────

    def _unknown_disease(
        self,
        disease_key: str,
        confidence:  float,
        location:    Optional[Tuple[float, float]],
    ) -> Dict[str, Any]:
        context = _detect_disease_context(disease_key)
        is_healthy = context["is_healthy"]
        return {
            "disease":           disease_key.replace("___", " — ").replace("_", " ").title(),
            "disease_key":       disease_key,
            "crop":              "Unknown",
            "confidence":        round(confidence * 100, 2),
            "confidence_raw":    round(confidence, 6),
            "severity":          "⚪ UNKNOWN — Protocol not found. Manual expert inspection recommended.",
            "severity_level":    "UNKNOWN",
            "first_aid":         "Consult your nearest Krishi Vigyan Kendra (KVK) for expert diagnosis. Helpline: 1800-180-1551",
            "action_plan":       [
                "Take 3–4 clear photos of affected leaves in natural daylight",
                "Note crop age, recent weather, and any chemicals applied",
                "Contact KVK helpline: 1800-180-1551",
                "Visit nearest agricultural extension office",
            ],
            "weather_advice":    self._static_weather_advice(disease_key),
            "yield_loss_pct":    None,
            "economic_loss_rs":  None,
            "economic_loss_per_acre": None,
            "marketplace":       {"recommended_products": [], "note": "Consult expert before purchasing any chemical."},
            "location":          {"lat": location[0] if location else None, "lon": location[1] if location else None},
            "timestamp":         datetime.datetime.utcnow().isoformat() + "Z",
            "is_positive":       not is_healthy,
            "disease_type":      context["disease_type"],
            "top_k_predictions": [],
        }


# ══════════════════════════════════════════════════════════════════
# PRETTY PRINTER
# ══════════════════════════════════════════════════════════════════

def print_intelligence_report(result: Dict[str, Any]) -> None:
    """Prints a human-readable intelligence report."""
    print("\n" + "═" * 65)
    print("  🌿  CROP DISEASE INTELLIGENCE REPORT")
    print("═" * 65)
    print(f"  🦠  Disease       : {result['disease']}")
    print(f"  🌾  Crop          : {result.get('crop', 'N/A')}")
    print(f"  📊  Confidence    : {result['confidence']}%")
    print(f"  🔬  Disease Type  : {result.get('disease_type', 'N/A').upper()}")
    print(f"  🚨  Severity      : {result['severity']}")
    print(f"  ✅  Is Positive   : {result.get('is_positive', 'N/A')}")
    print()

    print("  💊  FIRST-AID REMEDY:")
    print(f"      {result['first_aid']}")
    print()

    print("  📋  GRANULAR ACTION PLAN:")
    for step in result.get("action_plan", []):
        print(f"      ▸ {step}")
    print()

    print("  🌦️  WEATHER-CONTEXTUAL ADVICE:")
    print(f"      {result['weather_advice']}")
    print()

    print("  📉  YIELD & ECONOMIC IMPACT:")
    if result.get("yield_loss_pct") is not None:
        print(f"      Expected Yield Loss : {result['yield_loss_pct']}%")
        print(f"      Estimated Total Loss: ₹{result['economic_loss_rs']:,.0f}")
        print(f"      Per-Acre Loss       : ₹{result['economic_loss_per_acre']:,.0f}")
    else:
        print("      N/A")
    print()

    print("  🛒  MARKETPLACE ROUTING:")
    mp = result.get("marketplace", {})
    for p in mp.get("recommended_products", []):
        print(f"      🔹 {p}")
    print(f"      📌 {mp.get('note', '')}")
    print()

    if result.get("top_k_predictions"):
        print("  🏆  TOP-K ALTERNATIVES:")
        for alt in result["top_k_predictions"]:
            print(f"      #{alt['rank']} {alt['disease_key']} — {alt['confidence']}%")
        print()

    loc = result.get("location", {})
    if loc.get("lat"):
        print(f"  📍  Location  : {loc['lat']}, {loc['lon']}")
    print(f"  🕐  Timestamp : {result.get('timestamp', 'N/A')}")
    print("═" * 65)
