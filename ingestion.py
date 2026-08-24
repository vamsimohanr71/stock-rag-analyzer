"""
Data ingestion pipeline: pulls price history, news, and SEC filings for a
ticker, chunks the text, embeds it, and upserts into the vector DB.

Designed to be safe to re-run: uses deterministic IDs (hash of ticker+source+
date+chunk index) so re-ingestion overwrites rather than duplicates.
"""
import hashlib
import logging
import socket
from datetime import datetime, timedelta, timezone

import requests
import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_exponential

from config import settings
from indian_news import get_company_name, get_google_news
from nvidia_embeddings import create_embeddings
from vector_store import get_collection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ingestion")

# Hard safety net: yfinance doesn't expose a clean per-call timeout for
# fast_info/history, and cloud-hosted IPs (e.g. Streamlit Community Cloud)
# are sometimes rate-limited or silently stalled by Yahoo Finance. Without
# this, one slow/stuck network call can hang page loads indefinitely.
socket.setdefaulttimeout(10)


# ---------- Data pulls ----------

def get_price_data(ticker: str, period: str = "6mo"):
    """Historical OHLCV data. yfinance needs no API key."""
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period)
    if hist.empty:
        raise ValueError(f"No price data returned for {ticker}. Check the ticker symbol.")
    return hist


def get_quick_price(ticker: str) -> dict:
    """
    Fast quote lookup for a ticker banner - uses yfinance's fast_info
    instead of downloading full price history, so it's cheap to call
    frequently (e.g. every 15s) for several tickers at once.
    Data carries the same delay as the rest of the app (~15-20min for
    most exchanges via Yahoo Finance's free feed) - not true real-time.
    A module-level socket timeout (see top of file) keeps a stuck network
    call from hanging the whole page indefinitely.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.fast_info
        last_price = info.get("lastPrice") if hasattr(info, "get") else info["lastPrice"]
        prev_close = info.get("previousClose") if hasattr(info, "get") else info["previousClose"]
        if last_price is None or prev_close in (None, 0):
            raise ValueError(f"No quote data available for {ticker}.")
        change_pct = (last_price - prev_close) / prev_close * 100
        return {
            "ticker": ticker,
            "price": round(float(last_price), 2),
            "change_pct": round(float(change_pct), 2),
            "currency": getattr(info, "currency", None) or (info.get("currency") if hasattr(info, "get") else None),
        }
    except (socket.timeout, TimeoutError) as e:
        raise ValueError(f"Timed out fetching quote for {ticker}: {e}")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def get_news(ticker: str, lookback_days: int = None):
    """Company news via Finnhub. Free tier: 60 calls/min."""
    lookback_days = lookback_days or settings.NEWS_LOOKBACK_DAYS
    today = datetime.now(timezone.utc).date()
    frm = today - timedelta(days=lookback_days)
    url = (
        f"https://finnhub.io/api/v1/company-news"
        f"?symbol={ticker}&from={frm.isoformat()}&to={today.isoformat()}"
        f"&token={settings.FINNHUB_API_KEY}"
    )
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    articles = resp.json()
    if not isinstance(articles, list):
        logger.warning(f"Unexpected Finnhub response for {ticker}: {articles}")
        return []
    return articles


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def get_sec_filings(ticker: str, limit: int = 5):
    """
    Recent SEC filings via EDGAR full text search. No API key required,
    but SEC requires a descriptive User-Agent header with contact info.
    """
    url = "https://efts.sec.gov/LATEST/search-index"
    params = {"q": ticker, "forms": "10-K,10-Q,8-K", "dateRange": "custom"}
    headers = {"User-Agent": settings.SEC_USER_AGENT}
    resp = requests.get(url, params=params, headers=headers, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    hits = data.get("hits", {}).get("hits", [])[:limit]
    return hits


# ---------- Chunking + embedding ----------

def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50):
    words = text.split()
    if not words:
        return []
    chunks = []
    step = max(chunk_size - overlap, 1)
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Batch embedding call for documents being stored (not queries).
    input_type='passage' here vs 'query' in retrieval.py - NIM's QA
    embedding models embed these two asymmetrically; getting it backwards
    doesn't error, it just hurts retrieval quality later.
    """
    return create_embeddings(texts, input_type="passage")


def make_id(ticker: str, source: str, doc_key: str, chunk_idx: int) -> str:
    raw = f"{ticker}:{source}:{doc_key}:{chunk_idx}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def embed_and_store_batch(ticker: str, source: str, items: list[dict]) -> int:
    """
    Batches ALL chunks from ALL given items (articles/filings) into a
    single embedding API call, instead of one call per item. This is the
    single biggest speed win for ingestion - 15 articles previously meant
    15 separate network round-trips to NVIDIA; now it's 1 (or a couple, if
    an item produces many chunks).
    Each item needs: doc_key, text, date, and optional extra_meta.
    """
    collection = get_collection()

    all_chunks, chunk_owners = [], []  # chunk_owners[i] = (doc_key, date, extra_meta)
    for item in items:
        chunks = chunk_text(item["text"])
        for chunk in chunks:
            all_chunks.append(chunk)
            chunk_owners.append((item["doc_key"], item["date"], item.get("extra_meta")))

    if not all_chunks:
        return 0

    embeddings = embed_texts(all_chunks)

    # Track a running chunk index per doc_key so IDs stay unique and stable
    per_doc_idx = {}
    ids, metadatas, docs = [], [], []
    for chunk, emb, (doc_key, date, extra_meta) in zip(all_chunks, embeddings, chunk_owners):
        idx = per_doc_idx.get(doc_key, 0)
        per_doc_idx[doc_key] = idx + 1
        ids.append(make_id(ticker, source, doc_key, idx))
        meta = {"ticker": ticker, "source": source, "date": date, "doc_key": doc_key}
        if extra_meta:
            meta.update(extra_meta)
        metadatas.append(meta)
        docs.append(chunk)

    collection.upsert(ids=ids, embeddings=embeddings, documents=docs, metadatas=metadatas)
    return len(ids)


# ---------- Orchestration ----------

def ingest_ticker(ticker: str) -> dict:
    """Full ingestion run for one ticker. Returns a summary of what was stored."""
    ticker = ticker.upper().strip()
    summary = {"ticker": ticker, "news_chunks": 0, "filing_chunks": 0, "errors": []}
    is_indian = ticker.endswith(".NS") or ticker.endswith(".BO")

    try:
        if is_indian:
            # Finnhub/SEC barely cover Indian equities - use Google News
            # RSS by company name instead, which works for any company.
            company_name = get_company_name(ticker)
            logger.info(f"[{ticker}] Indian ticker detected - fetching Google News for '{company_name}'...")
            raw_articles = get_google_news(company_name, max_results=settings.MAX_ARTICLES_PER_INGEST)
            articles = [
                {
                    "headline": a["headline"],
                    "summary": a["summary"],
                    "url": a["url"],
                    "date": a["date"],
                    "id": a["url"],
                }
                for a in raw_articles
            ]
        else:
            logger.info(f"[{ticker}] Fetching news from Finnhub...")
            raw = get_news(ticker)
            # Cap how many articles get embedded per ingestion run - a
            # heavily covered ticker (AAPL, TSLA, etc.) can return hundreds
            # of articles over a 14-day window, and each one is a separate
            # embedding API call, which is slow on a free-tier rate limit.
            raw = sorted(raw, key=lambda a: a.get("datetime", 0), reverse=True)
            raw = raw[: settings.MAX_ARTICLES_PER_INGEST]
            articles = [
                {
                    "headline": a.get("headline", ""),
                    "summary": a.get("summary", ""),
                    "url": a.get("url", ""),
                    "date": datetime.fromtimestamp(a.get("datetime", 0), tz=timezone.utc).date().isoformat(),
                    "id": str(a.get("id", a.get("url", ""))),
                }
                for a in raw
            ]

        logger.info(f"[{ticker}] Using {len(articles)} articles. Batch-embedding all at once...")
        news_items = [
            {
                "doc_key": a["id"],
                "text": f"{a['headline']}. {a['summary']}",
                "date": a["date"],
                "extra_meta": {"url": a["url"], "headline": a["headline"]},
            }
            for a in articles
            if len(f"{a['headline']}. {a['summary']}".strip()) >= 20
        ]
        summary["news_chunks"] = embed_and_store_batch(ticker, "news", news_items)
        logger.info(f"[{ticker}] News ingestion done: {summary['news_chunks']} chunks stored.")
    except Exception as e:
        logger.exception(f"News ingestion failed for {ticker}")
        summary["errors"].append(f"news: {e}")

    if is_indian:
        # SEC EDGAR is US-only and will never have data for NSE/BSE tickers -
        # skip it entirely rather than waste a call that always returns nothing.
        logger.info(f"[{ticker}] Indian ticker - skipping SEC filings (US-only source).")
    else:
        try:
            logger.info(f"[{ticker}] Fetching SEC filings...")
            filings = get_sec_filings(ticker)
            logger.info(f"[{ticker}] Got {len(filings)} filings. Batch-embedding all at once...")
            filing_items = []
            for f in filings:
                src = f.get("_source", {})
                text = " ".join(str(v) for v in src.values() if isinstance(v, str))
                if len(text.strip()) < 20:
                    continue
                filing_items.append({
                    "doc_key": f.get("_id", ""),
                    "text": text,
                    "date": src.get("file_date", datetime.now(timezone.utc).date().isoformat()),
                    "extra_meta": {"form": src.get("root_form", "")},
                })
            summary["filing_chunks"] = embed_and_store_batch(ticker, "filing", filing_items)
            logger.info(f"[{ticker}] Filing ingestion done: {summary['filing_chunks']} chunks stored.")
        except Exception as e:
            logger.exception(f"Filing ingestion failed for {ticker}")
            summary["errors"].append(f"filings: {e}")

    return summary


def ingest_watchlist():
    results = []
    for ticker in settings.WATCHLIST:
        logger.info(f"Ingesting {ticker}...")
        results.append(ingest_ticker(ticker))
    return results
