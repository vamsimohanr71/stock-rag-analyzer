"""
Centralized configuration. Reads from Streamlit's secrets manager when
deployed on Streamlit Community Cloud (st.secrets), and falls back to a
local .env file for local development - so the exact same code runs in
both places without changes.
"""
import os

from dotenv import load_dotenv

load_dotenv()

try:
    import streamlit as st
    _has_streamlit_secrets = hasattr(st, "secrets")
except Exception:
    _has_streamlit_secrets = False


def _get(key: str, default: str = "") -> str:
    """Check Streamlit secrets first (for cloud deploys), then env vars."""
    if _has_streamlit_secrets:
        try:
            if key in st.secrets:
                return str(st.secrets[key])
        except Exception:
            pass  # secrets.toml not present (e.g. local run) - fall through
    return os.getenv(key, default)


class Settings:
    NVIDIA_API_KEY = _get("NVIDIA_API_KEY", "")
    NVIDIA_BASE_URL = _get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")

    FINNHUB_API_KEY = _get("FINNHUB_API_KEY", "")
    SEC_USER_AGENT = _get("SEC_USER_AGENT", "StockRAGApp contact@example.com")

    # Streamlit Community Cloud's filesystem is ephemeral - it resets on
    # every redeploy/restart/sleep-wake cycle. Chroma data will NOT persist
    # long-term on the free tier; each cold start re-ingests as needed.
    CHROMA_DB_PATH = _get("CHROMA_DB_PATH", "./chroma_store")

    EMBEDDING_MODEL = _get("EMBEDDING_MODEL", "nvidia/nv-embedqa-e5-v5")
    LLM_MODEL = _get("LLM_MODEL", "meta/llama-3.1-70b-instruct")

    NEWS_LOOKBACK_DAYS = int(_get("NEWS_LOOKBACK_DAYS", "14"))
    MAX_ARTICLES_PER_INGEST = int(_get("MAX_ARTICLES_PER_INGEST", "15"))
    WATCHLIST = [t.strip().upper() for t in _get("WATCHLIST", "AAPL,MSFT,GOOGL").split(",") if t.strip()]
    INGEST_INTERVAL_HOURS = int(_get("INGEST_INTERVAL_HOURS", "6"))


settings = Settings()

REQUIRED_KEYS = ["NVIDIA_API_KEY", "FINNHUB_API_KEY"]


def validate_settings():
    missing = [k for k in REQUIRED_KEYS if not getattr(settings, k)]
    if missing:
        raise EnvironmentError(
            f"Missing required secrets/env vars: {missing}. "
            f"Locally: copy .env.example to .env and fill these in. "
            f"On Streamlit Cloud: add them under app Settings -> Secrets."
        )
