"""
Stock RAG Analyzer - single-file deployable version for Streamlit Community
Cloud (free hosting). Calls the ingestion/retrieval/LLM pipeline directly
as Python functions instead of going through a separate FastAPI backend -
this is what makes a single free Streamlit Cloud app sufficient to host
the whole thing, no second service needed.

Local dev:   streamlit run app.py   (reads secrets from .env via config.py)
Cloud deploy: push this folder to GitHub, deploy on share.streamlit.io,
              add secrets under app Settings -> Secrets (see .streamlit/secrets.toml.example)
"""
import json

import streamlit as st
import streamlit.components.v1 as components

from config import validate_settings
from ingestion import ingest_ticker, get_price_data, get_quick_price
from retrieval import retrieve_context
from technical import get_technical_signals
from llm_analysis import generate_analysis

st.set_page_config(page_title="Stock RAG Analyzer", layout="centered", page_icon="📊")

# Tickers shown in the scrolling banner. Mix US and Indian (.NS/.BO) freely.
TICKER_BANNER_SYMBOLS = ["AAPL", "MSFT", "NVDA", "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS"]


def render_ticker_banner(symbols: list[str], refresh_seconds: int = 30):
    """
    Self-updating scrolling price banner. Fetches prices via
    get_quick_price() at page-load time (Streamlit reruns the whole script
    on interaction, so this refreshes naturally on any user action; the
    CSS animation just gives it a live-feeling scroll in between).
    Prices carry the same delay as the rest of the app (~15-20min via
    Yahoo Finance's free feed) - not a true real-time feed.
    """
    items = []
    for sym in symbols:
        try:
            p = get_quick_price(sym)
            items.append(p)
        except Exception:
            items.append({"ticker": sym, "price": None, "change_pct": None, "currency": None})

    spans = []
    for p in items:
        if p["price"] is None:
            spans.append(f'<span style="color:#888; margin-right:40px;">{p["ticker"]}: n/a</span>')
            continue
        up = p["change_pct"] >= 0
        color = "#2ecc71" if up else "#e74c3c"
        arrow = "▲" if up else "▼"
        currency = "₹" if p.get("currency") == "INR" else "$"
        spans.append(
            f'<span style="margin-right:40px;"><b>{p["ticker"]}</b> '
            f'{currency}{p["price"]} '
            f'<span style="color:{color}">{arrow} {abs(p["change_pct"])}%</span></span>'
        )
    track_html = "".join(spans) * 2  # duplicate for a seamless scroll loop

    html = f"""
    <div style="width:100%; overflow:hidden; background:#0e1117; border-radius:6px;
                padding:10px 0; white-space:nowrap; font-family:monospace; font-size:15px;">
      <div style="display:inline-block; padding-left:100%; animation: scroll-left 30s linear infinite;">
        {track_html}
      </div>
    </div>
    <style>
      @keyframes scroll-left {{ from {{ transform: translateX(0); }} to {{ transform: translateX(-100%); }} }}
    </style>
    """
    components.html(html, height=50)


try:
    validate_settings()
except EnvironmentError as e:
    st.error(str(e))
    st.stop()

render_ticker_banner(TICKER_BANNER_SYMBOLS)

st.title("📊 Stock RAG Analyzer")
st.caption("RAG-grounded news/filings analysis + technical indicators. Not financial advice.")

ticker = st.text_input("Ticker symbol (use .NS or .BO suffix for Indian stocks, e.g. RELIANCE.NS)", "AAPL")
ticker = ticker.upper().strip()

col1, col2 = st.columns(2)
run_btn = col1.button("Analyze", type="primary")
refresh_btn = col2.button("Force refresh data")

if refresh_btn and ticker:
    with st.spinner(f"Re-ingesting latest news for {ticker}..."):
        try:
            summary = ingest_ticker(ticker)
            st.success(f"Ingestion complete: {summary}")
        except Exception as e:
            st.error(f"Ingestion failed: {e}")

if run_btn and ticker:
    with st.spinner(f"Analyzing {ticker}... (first run may take longer while data is ingested)"):
        try:
            hist = get_price_data(ticker)
            technicals = get_technical_signals(hist)

            docs = retrieve_context(ticker, f"{ticker} recent news, earnings, and outlook")
            if not docs:
                st.info("No cached data found for this ticker - ingesting now...")
                ingest_ticker(ticker)
                docs = retrieve_context(ticker, f"{ticker} recent news, earnings, and outlook")

            analysis = generate_analysis(ticker, docs, technicals)

            st.subheader("Technical Indicators")
            t = technicals
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Price", f"{t['current_price']}", f"{t['day_change_pct']}%")
            m2.metric("RSI (14)", t.get("rsi_14", "—"))
            m3.metric("Trend", t.get("trend", "—"))
            m4.metric("Volume", f"{t.get('volume', 0):,}")
            st.json(t)

            st.subheader("AI Analysis (RAG-grounded)")
            st.write(analysis)

            if docs:
                with st.expander(f"Sources used ({len(docs)})"):
                    for d in docs:
                        line = f"- **{d['source']}** ({d['date']})"
                        if d.get("url"):
                            line += f" — [link]({d['url']})"
                        st.markdown(line)
            else:
                st.info("No news/filing sources were retrieved for this ticker.")

            st.caption("This is AI-generated research context, not financial advice.")
        except ValueError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Something went wrong: {e}")
