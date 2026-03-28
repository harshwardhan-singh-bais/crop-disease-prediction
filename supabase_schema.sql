-- ═══════════════════════════════════════════════════════════════
-- Supabase SQL — Run this in your Supabase SQL Editor
-- Table: disease_scans
-- ═══════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS disease_scans (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- When the scan happened (maps from result["timestamp"])
    scanned_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- GPS location (maps from result["location"]["lat"/"lon"])
    latitude         DOUBLE PRECISION,
    longitude        DOUBLE PRECISION,

    -- Disease identification (maps from result["disease"])
    pathogen_name    TEXT NOT NULL,

    -- Raw model confidence 0.0–1.0 (maps from result["confidence_raw"])
    confidence_score DOUBLE PRECISION NOT NULL CHECK (confidence_score BETWEEN 0 AND 1),

    -- True = diseased, False = healthy (derived: "healthy" NOT in disease_key)
    is_positive      BOOLEAN NOT NULL,

    -- Severity level: NONE | LOW | MEDIUM | HIGH | UNKNOWN
    severity_level   TEXT,

    -- Disease type: fungal | bacterial | viral | pest | healthy | unknown
    disease_type     TEXT,

    -- Economic data
    yield_loss_pct   DOUBLE PRECISION,
    economic_loss_rs DOUBLE PRECISION,

    -- Full intelligence report as JSONB (entire result dict)
    raw_payload      JSONB,

    -- Auto-timestamp
    created_at       TIMESTAMPTZ DEFAULT NOW()
);

-- Index for location-based queries (useful for Priyanshu's geo features)
CREATE INDEX IF NOT EXISTS idx_disease_scans_location
    ON disease_scans (latitude, longitude);

-- Index for time-series queries
CREATE INDEX IF NOT EXISTS idx_disease_scans_scanned_at
    ON disease_scans (scanned_at DESC);

-- Index for disease filtering
CREATE INDEX IF NOT EXISTS idx_disease_scans_pathogen
    ON disease_scans (pathogen_name);

-- Index for positive/negative filtering
CREATE INDEX IF NOT EXISTS idx_disease_scans_is_positive
    ON disease_scans (is_positive);

-- Enable Row Level Security (recommended for production)
ALTER TABLE disease_scans ENABLE ROW LEVEL SECURITY;

-- Allow all inserts from service role (backend uses service key)
CREATE POLICY "Allow service role insert"
    ON disease_scans FOR INSERT
    TO service_role
    WITH CHECK (true);

-- Allow all reads for authenticated users
CREATE POLICY "Allow authenticated read"
    ON disease_scans FOR SELECT
    TO authenticated
    USING (true);
