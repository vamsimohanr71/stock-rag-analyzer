"""
Direct HTTP client for NVIDIA's NIM embeddings endpoint, using `requests`
instead of the `openai` SDK.

Why: the openai Python package (Stainless-generated) attaches extra
diagnostic headers (X-Stainless-*) that NVIDIA's API gateway appears to
mis-route, returning a plain-text 404 even with a valid key/model/body.
A raw curl call with just Authorization + Content-Type headers works
correctly - this module reproduces that exact request shape.

Chat completions (llm_analysis.py) are unaffected and continue to use the
openai SDK - only embeddings hit this issue in testing.
"""
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from config import settings

EMBEDDINGS_URL = f"{settings.NVIDIA_BASE_URL.rstrip('/')}/embeddings"


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def create_embeddings(texts: list[str], input_type: str) -> list[list[float]]:
    """
    texts: list of strings to embed (pass a single-item list for one query)
    input_type: 'query' for search queries, 'passage' for stored documents
                (NIM's QA embedding models embed these two asymmetrically -
                getting it backwards won't error, it just hurts retrieval
                quality, so keep call sites consistent)
    """
    if not texts:
        return []

    print(f"[nvidia_embeddings] Starting request: {len(texts)} text(s), model={settings.EMBEDDING_MODEL}")

    headers = {
        "Authorization": f"Bearer {settings.NVIDIA_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        # NVIDIA's gateway appears to 404 requests carrying the default
        # python-requests User-Agent (likely basic bot filtering) while an
        # identical request via curl succeeds - spoofing a curl-style UA
        # fixed it in testing.
        "User-Agent": "curl/8.4.0",
    }
    payload = {
        "input": texts,
        "model": settings.EMBEDDING_MODEL,
        "input_type": input_type,
    }

    resp = requests.post(EMBEDDINGS_URL, headers=headers, json=payload, timeout=30)
    print(f"[nvidia_embeddings] Got response: status={resp.status_code}")
    if resp.status_code != 200:
        # Temporary debug logging - shows exactly what NVIDIA sent back,
        # since a 404 here could mean several different things.
        print("=== NVIDIA EMBEDDINGS DEBUG ===")
        print("URL:", resp.url)
        print("STATUS:", resp.status_code)
        print("RESPONSE HEADERS:", dict(resp.headers))
        print("RESPONSE BODY:", resp.text[:2000])
        print("REQUEST HEADERS SENT:", headers)
        print("REQUEST BODY SENT:", payload)
        print("================================")
    resp.raise_for_status()
    data = resp.json()
    # Results aren't guaranteed to come back in input order - sort by index to be safe.
    ordered = sorted(data["data"], key=lambda d: d["index"])
    return [d["embedding"] for d in ordered]
