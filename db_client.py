"""
db_client.py
============
Supabase database client — inserts disease scan results
into the `disease_scans` table.

Table schema expected in Supabase (SQL):
-----------------------------------------
CREATE TABLE disease_scans (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    scanned_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    latitude        DOUBLE PRECISION,
    longitude       DOUBLE PRECISION,
    pathogen_name   TEXT NOT NULL,
    confidence_score DOUBLE PRECISION NOT NULL,
    is_positive     BOOLEAN NOT NULL,
    severity_level  TEXT,
    disease_type    TEXT,
    yield_loss_pct  DOUBLE PRECISION,
    economic_loss_rs DOUBLE PRECISION,
    raw_payload     JSONB,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
"""

import os
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# DB CLIENT
# ══════════════════════════════════════════════════════════════════

class DBClient:
    """
    Thin wrapper around Supabase client.
    Handles the field mapping from intelligence result → disease_scans table.
    """

    def __init__(self):
        url = os.getenv("SUPABASE_URL", "")
        key = os.getenv("SUPABASE_KEY", "")

        if not url or not key:
            logger.warning(
                "SUPABASE_URL or SUPABASE_KEY not set. "
                "DB writes will be skipped (dry-run mode)."
            )
            self._client = None
            return

        try:
            from supabase import create_client
            self._client = create_client(url, key)
            logger.info("✅ Supabase client initialized.")
        except ImportError:
            logger.error("supabase package not installed. Run: uv add supabase")
            self._client = None
        except Exception as e:
            logger.error(f"Supabase connection error: {e}")
            self._client = None

    @property
    def is_connected(self) -> bool:
        return self._client is not None

    # ── FIELD MAPPING ─────────────────────────────────────────────
    @staticmethod
    def _map_to_db_record(result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Maps intelligence engine output fields → disease_scans table columns.

        Mapping:
          result["timestamp"]        → scanned_at
          result["location"]["lat"]  → latitude
          result["location"]["lon"]  → longitude
          result["disease"]          → pathogen_name
          result["confidence_raw"]   → confidence_score
          result["is_positive"]      → is_positive  (True = diseased, False = healthy)
          result["severity_level"]   → severity_level
          result["disease_type"]     → disease_type
          result["yield_loss_pct"]   → yield_loss_pct
          result["economic_loss_rs"] → economic_loss_rs
          full result dict           → raw_payload (JSONB)
        """
        loc = result.get("location", {})

        db_record = {
            "scanned_at":       result.get("timestamp"),
            "latitude":         loc.get("lat"),
            "longitude":        loc.get("lon"),
            "pathogen_name":    result.get("disease") or result.get("disease_key", "unknown"),
            "confidence_score": result.get("confidence_raw", 0.0),
            "is_positive":      result.get("is_positive", True),
            "severity_level":   result.get("severity_level", "UNKNOWN"),
            "disease_type":     result.get("disease_type", "unknown"),
            "yield_loss_pct":   result.get("yield_loss_pct"),
            "economic_loss_rs": result.get("economic_loss_rs"),
            "raw_payload":      result,   # full JSONB blob
        }

        return db_record

    # ── INSERT ────────────────────────────────────────────────────
    def insert_scan(self, result: Dict[str, Any]) -> Optional[Dict]:
        """
        Inserts a scan record into disease_scans table.

        Returns inserted row dict on success, None on failure.
        """
        record = self._map_to_db_record(result)

        if not self.is_connected:
            logger.warning(f"[DRY-RUN] Would insert: {json.dumps(record, indent=2, default=str)}")
            return {"dry_run": True, "record": record}

        try:
            response = (
                self._client.table("disease_scans")
                .insert(record)
                .execute()
            )
            inserted = response.data[0] if response.data else None
            logger.info(f"✅ Scan inserted to DB: id={inserted.get('id') if inserted else 'N/A'}")
            return inserted
        except Exception as e:
            logger.error(f"❌ DB insert failed: {e}")
            return None


# ── Singleton ─────────────────────────────────────────────────────
_db_instance: Optional[DBClient] = None

def get_db() -> DBClient:
    global _db_instance
    if _db_instance is None:
        _db_instance = DBClient()
    return _db_instance
