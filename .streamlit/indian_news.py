"""
Free, keyless news source using Google News RSS - primarily added to cover
Indian equities (NSE/BSE), since Finnhub and SEC EDGAR barely cover them.

Honesty note: real per-company RSS feeds from Moneycontrol/Economic Times
don't reliably exist as a public, stable API - so rather than build
something fragile against undocumented endpoints, this uses Google News
RSS, which is genuinely per-company (searches by company name) and works
for any company worldwide, not just India.
"""
import logging
import xml.etree.ElementTree as ET
from datetime import datetime
from urllib.parse import quote

import requests
import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger("indian_news")

# yf.Ticker(ticker).info is notoriously slow (fetches a large payload of
# fields we only need one of). Caching by ticker avoids repeating that
# slow call every time the same ticker is analyzed again in a session.
_company_name_cache: dict[str, str] = {}


def get_company_name(ticker: str) -> str:
    """
    Google News searches by company name, not ticker symbol, for decent
    results. Tries yfinance's company name lookup; falls back to the raw
    ticker (stripped of exchange suffix) if that lookup fails or is slow.
    """
    if ticker in _company_name_cache:
        return _company_name_cache[ticker]

    name = None
    try:
        info = yf.Ticker(ticker).info
        name = info.get("longName") or info.get("shortName")
    except Exception as e:
        logger.warning(f"Could not fetch company name for {ticker}: {e}")

    resolved = name or ticker.split(".")[0]  # fallback: strip exchange suffix
    _company_name_cache[ticker] = resolved
    return resolved


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def get_google_news(query: str, max_results: int = 10, region: str = "IN") -> list[dict]:
    """
    Fetches recent news for `query` via Google News RSS. No API key needed.
    region: 'IN' biases results toward Indian sources/English-India edition;
    use 'US' or omit region bias for other markets if reusing this function.
    """
    encoded_query = quote(query)
    url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-{region}&gl={region}&ceid={region}:en"

    resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()

    root = ET.fromstring(resp.content)
    items = root.findall(".//item")[:max_results]

    articles = []
    for item in items:
        title = item.findtext("title", default="")
        link = item.findtext("link", default="")
        pub_date_raw = item.findtext("pubDate", default="")
        description = item.findtext("description", default="")

        date_iso = datetime.now().date().isoformat()
        if pub_date_raw:
            try:
                date_iso = datetime.strptime(pub_date_raw, "%a, %d %b %Y %H:%M:%S %Z").date().isoformat()
            except ValueError:
                pass  # keep today's date as a safe fallback

        articles.append({
            "headline": title,
            "summary": description,
            "url": link,
            "date": date_iso,
        })
    return articles
