"""
Data ingestion pipeline: pulls price history, news, and SEC filings for a
ticker, chunks the text, embeds it, and upserts into the vector DB.

Designed to be safe to re-run: uses deterministic IDs (hash of ticker+source+
date+chunk index) so re-ingestion overwrites rather than duplicates.
"""
import hashlib
import logging
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
    """
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


def embed_and_store(ticker: str, source: str, doc_key: str, text: str, date: str, extra_meta: dict = None):
    collection = get_collection()
    chunks = chunk_text(text)
    if not chunks:
        return 0
    embeddings = embed_texts(chunks)
    ids, metadatas, docs = [], [], []
    for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
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

        logger.info(f"[{ticker}] Using {len(articles)} articles. Embedding each...")
        for idx, a in enumerate(articles):
            text = f"{a['headline']}. {a['summary']}"
            if len(text.strip()) < 20:
                continue
            logger.info(f"[{ticker}] Embedding article {idx + 1}/{len(articles)}...")
            n = embed_and_store(
                ticker, "news", a["id"], text, a["date"],
                extra_meta={"url": a["url"], "headline": a["headline"]}
            )
            summary["news_chunks"] += n
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
            logger.info(f"[{ticker}] Got {len(filings)} filings. Embedding each...")
            for idx, f in enumerate(filings):
                src = f.get("_source", {})
                text = " ".join(str(v) for v in src.values() if isinstance(v, str))
                if len(text.strip()) < 20:
                    continue
                logger.info(f"[{ticker}] Embedding filing {idx + 1}/{len(filings)}...")
                n = embed_and_store(
                    ticker, "filing", f.get("_id", ""), text,
                    date=src.get("file_date", datetime.now(timezone.utc).date().isoformat()),
                    extra_meta={"form": src.get("root_form", "")}
                )
                summary["filing_chunks"] += n
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
