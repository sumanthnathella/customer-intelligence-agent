"""Shared configuration — all tunable knobs live here."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
GBRAIN_DIR = ROOT / "gbrain" / "store"
REPORTS_DIR = ROOT / "reports"
TAXONOMY_PATH = DATA_DIR / "taxonomy.json"

for _d in (DATA_DIR, GBRAIN_DIR, GBRAIN_DIR / "snapshots", REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
TAGGER_MODEL: str = os.getenv("TAGGER_MODEL", "qwen2.5:7b")
EMBED_MODEL: str = os.getenv("EMBED_MODEL", "nomic-embed-text")
AGENT_MODEL: str = os.getenv("AGENT_MODEL", "openrouter/nvidia/nemotron-3-ultra:free")
AGENT_MODEL_FALLBACK: str = os.getenv("AGENT_MODEL_FALLBACK", "ollama/gemma4:31b")

# ---------------------------------------------------------------------------
# Tagging
# ---------------------------------------------------------------------------
TAGGER_BATCH_SIZE: int = int(os.getenv("TAGGER_BATCH_SIZE", "10"))
API_TAGGER_BATCH_SIZE: int = int(os.getenv("API_TAGGER_BATCH_SIZE", "25"))
TAGGER_CACHE_DIR: Path = DATA_DIR / "tagger_cache"
TAGGER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
PACK_VERSION: str = os.getenv("PACK_VERSION", "v1")
PROMPT_VERSION: str = os.getenv("PROMPT_VERSION", "v1")

# ---------------------------------------------------------------------------
# Analytics — severity rubric
# ---------------------------------------------------------------------------
SEVERITY_BASE: dict[str, int] = {
    "very_neg": 3,
    "neg": 2,
    "neutral": 1,
    "pos": 1,
}
SEVERITY_SIGNAL_WEIGHTS: dict[str, int] = {
    "churn_intent": 1,
    "financial_harm": 1,
    "safety_legal": 2,
    "repeat_contact": 1,  # either repeat_contact OR unresolved adds +1
    "unresolved": 1,
}
SEVERITY_MIN: int = 1
SEVERITY_MAX: int = 5

# ---------------------------------------------------------------------------
# Analytics — z-score baseline
# ---------------------------------------------------------------------------
ZSCORE_BASELINE_WEEKS: int = int(os.getenv("ZSCORE_BASELINE_WEEKS", "8"))
ZSCORE_MIN_PERIODS: int = int(os.getenv("ZSCORE_MIN_PERIODS", "3"))
ZSCORE_STD_FLOOR: float = float(os.getenv("ZSCORE_STD_FLOOR", "0.5"))
ZSCORE_SPIKE_THRESHOLD: float = float(os.getenv("ZSCORE_SPIKE_THRESHOLD", "2.0"))

# ---------------------------------------------------------------------------
# Analytics — egregiousness weights (must sum to 1.0)
# ---------------------------------------------------------------------------
EGREG_WEIGHT_VOLUME: float = float(os.getenv("EGREG_WEIGHT_VOLUME", "0.35"))
EGREG_WEIGHT_SEVERITY: float = float(os.getenv("EGREG_WEIGHT_SEVERITY", "0.25"))
EGREG_WEIGHT_SPIKE: float = float(os.getenv("EGREG_WEIGHT_SPIKE", "0.25"))
EGREG_WEIGHT_VALUE: float = float(os.getenv("EGREG_WEIGHT_VALUE", "0.15"))

# ---------------------------------------------------------------------------
# Analytics — driver analysis
# ---------------------------------------------------------------------------
DRIVER_MIN_SUPPORT: int = int(os.getenv("DRIVER_MIN_SUPPORT", "30"))
DRIVER_FDR_ALPHA: float = float(os.getenv("DRIVER_FDR_ALPHA", "0.05"))
DRIVER_TOP_K_PER_DIM: int = int(os.getenv("DRIVER_TOP_K_PER_DIM", "10"))
DRIVER_MAX_2WAY: int = int(os.getenv("DRIVER_MAX_2WAY", "1"))

# ---------------------------------------------------------------------------
# Analytics — build window
# ---------------------------------------------------------------------------
DEFAULT_PERIOD: str = os.getenv("DEFAULT_PERIOD", "latest")  # ISO week or 'latest'

# ---------------------------------------------------------------------------
# Analytics — systemic ("bridge") drivers (Harness-1 evidence-graph inspired)
# ---------------------------------------------------------------------------
# A dimension value that is a *significant* driver across >= BRIDGE_MIN_L5
# distinct L5s is flagged as a systemic operational problem ("bridge").
BRIDGE_MIN_L5: int = int(os.getenv("BRIDGE_MIN_L5", "3"))

# ---------------------------------------------------------------------------
# Curated set — importance tiers + cap (warm-start → refine, with memory)
# ---------------------------------------------------------------------------
CURATED_CAP: int = int(os.getenv("CURATED_CAP", "30"))
IMPORTANCE_VERY_HIGH: float = float(os.getenv("IMPORTANCE_VERY_HIGH", "0.85"))
IMPORTANCE_HIGH: float = float(os.getenv("IMPORTANCE_HIGH", "0.60"))
IMPORTANCE_FAIR: float = float(os.getenv("IMPORTANCE_FAIR", "0.35"))

# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
TOP_L5_N: int = int(os.getenv("TOP_L5_N", "10"))
EXEMPLAR_K: int = int(os.getenv("EXEMPLAR_K", "3"))
EXEMPLAR_COMPRESS_SENTENCES: int = int(os.getenv("EXEMPLAR_COMPRESS_SENTENCES", "2"))
