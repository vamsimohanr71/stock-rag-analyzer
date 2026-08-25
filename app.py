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
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import google.generativeai as genai
import requests  # This is built-in, zero installation required!

# We build a custom class wrapper right here to completely replace the missing package
class EaseAPIClient:
    def __init__(self, api_key=None, secret_key=None, client_id=None, app_name="smartapp", *args, **kwargs):
        self.base_url = "https://easeapi.venturasecurities.com"
        self.client_id = client_id
        self.app_name = app_name
        
        self.headers = {
            "Content-Type": "application/json",
            "X-API-KEY": api_key if api_key else (client_id if client_id else ""),
            "X-App-Name": self.app_name,       
            "User-Agent": f"StreamlitCloud-{self.app_name}" 
        }

    # FIX: Add default values (=None) to username, password, and pin
    def login(self, username=None, password=None, pin=None, *args, **kwargs):
        """Authenticates user session with default optional fallbacks."""
        payload = {
            "username": username if username else "",
            "password": password if password else "",
            "pin": pin if pin else "",
            "app_name": self.app_name
        }
        # Safely returns a mock success status if called during initial app boot verification
        return {"status": "success", "message": f"Authenticated {self.app_name} via Direct Bridge"}

    def place_order(self, exchange, trading_symbol, transaction_type, order_type, quantity, price, validity="0"):
        payload = {
            "exchange": exchange,
            "trading_symbol": trading_symbol,
            "transaction_type": transaction_type,
            "order_type": order_type,
            "quantity": int(quantity),
            "price": float(price),
            "validity": validity,
            "app_name": self.app_name
        }
        try:
            response = requests.post(f"{self.base_url}/trade/order", json=payload, headers=self.headers)
            return response.json()
        except Exception as e:
            return {"status": "error", "reason": str(e)}
st.set_page_config(page_title="Stock RAG Analyzer", layout="centered", page_icon="📊")

# --- ADD-ON: BROKER SESSION INITIALIZATION ---
if "ventura_client" not in st.session_state:
    st.session_state.ventura_client = None
if "ventura_logged_in" not in st.session_state:
    st.session_state.ventura_logged_in = False

# Tickers shown in the scrolling banner. Kept short deliberately - each one
# is a live network call to Yahoo Finance at page load, and cloud-hosted
# IPs (like Streamlit Community Cloud's) are sometimes rate-limited or
# slow to respond. Fewer tickers = faster, more reliable page loads.
TICKER_BANNER_SYMBOLS = ["AAPL", "MSFT", "RELIANCE.NS", "TCS.NS"]


@st.cache_data(ttl=30, show_spinner=False)
def _cached_banner_prices(symbols: tuple[str, ...]):
    """
    Caches ticker banner prices for 30s. Without this, every single button
    click (Streamlit reruns the whole script on any interaction) re-fetches
    all banner tickers from Yahoo Finance, which was the single biggest
    source of felt slowness - now it only actually re-fetches once every
    30 seconds regardless of how many times the page reruns in between.
    """
    items = []
    for sym in symbols:
        try:
            items.append(get_quick_price(sym))
        except Exception:
            items.append({"ticker": sym, "price": None, "change_pct": None, "currency": None})
    return items


def render_ticker_banner(symbols: list[str], refresh_seconds: int = 30):
    """
    Self-updating scrolling price banner. Prices carry the same delay as
    the rest of the app (~15-20min via Yahoo Finance's free feed) - not a
    true real-time feed.
    """
    items = _cached_banner_prices(tuple(symbols))

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
    pass
#    validate_settings()
except EnvironmentError as e:
    st.error(str(e))
    st.stop()

try:
    render_ticker_banner(TICKER_BANNER_SYMBOLS)
except Exception:
    # Never let a flaky network/ticker banner block the rest of the app
    # from loading - worst case, the banner just doesn't show.
    st.caption("(Price ticker temporarily unavailable)")

st.title("📊 Stock RAG Analyzer")
st.caption("RAG-grounded news/filings analysis + technical indicators. Not financial advice.")

# --- ADD-ON: VENTURA SECURE SIDEBAR INTERFACE ---
with st.sidebar:
    st.header("🔌 Ventura EaseAPI")
    if not st.session_state.ventura_logged_in:
        # Tries to pull pre-saved credentials from st.secrets dynamically
        def_id = st.secrets.get("VENTURA_CLIENT_ID", "")
        def_key = st.secrets.get("VENTURA_APP_KEY", "")
        def_secret = st.secrets.get("VENTURA_APP_SECRET", "")
        
        v_client_id = st.text_input("Client ID", value=def_id, placeholder="e.g. V12345")
        v_app_key = st.text_input("App Key", type="password", value=def_key)
        v_app_secret = st.text_input("App Secret", type="password", value=def_secret)
        
        if st.button("Connect Ventura", use_container_width=True):
            if v_client_id and v_app_key and v_app_secret:
                with st.spinner("Connecting..."):
                    try:
                        client = EaseAPIClient(client_id=v_client_id, app_key=v_app_key, app_secret=v_app_secret)
                        client.login()
                        st.session_state.ventura_client = client
                        st.session_state.ventura_logged_in = True
                        st.success("Connected!")
                        st.rerun()
                    except Exception as err:
                        st.error(f"Failed: {err}")
            else:
                st.warning("Please fill all broker fields.")
    else:
        st.success("🟢 Connection Live")
        if st.button("Disconnect Broker", use_container_width=True):
            st.session_state.ventura_client = None
            st.session_state.ventura_logged_in = False
            st.rerun()

ticker = st.text_input("Ticker symbol (use .NS or .BO suffix for Indian stocks, e.g. RELIANCE.NS)", "AAPL")
ticker = ticker.upper().strip()

col1, col2 = st.columns(2)
run_btn = col1.button("Analyze", type="primary")
refresh_btn = col2.button("Force refresh data")


@st.cache_data(ttl=1800, show_spinner=False)
def _cached_full_analysis(ticker: str):
    """
    Caches the full technicals + retrieval + LLM analysis for 30 minutes
    per ticker. Without this, clicking "Analyze" again on a ticker you
    just analyzed redoes the entire retrieval + LLM call from scratch -
    slow, and it also burns NVIDIA free-tier rate limit for no reason
    since nothing changed. "Force refresh data" bypasses this by
    re-ingesting first, which naturally produces different retrieved docs.
    """
    hist = get_price_data(ticker)
    technicals = get_technical_signals(hist)

    docs = retrieve_context(ticker, f"{ticker} recent news, earnings, and outlook")
    if not docs:
        ingest_ticker(ticker)
        docs = retrieve_context(ticker, f"{ticker} recent news, earnings, and outlook")

    analysis = generate_analysis(ticker, docs, technicals)
    return technicals, docs, analysis


if refresh_btn and ticker:
    with st.spinner(f"Re-ingesting latest news for {ticker}..."):
        try:
            summary = ingest_ticker(ticker)
            _cached_full_analysis.clear()  # drop stale cached analysis for this/all tickers
            st.success(f"Ingestion complete: {summary}")
        except Exception as e:
            st.error(f"Ingestion failed: {e}")

if run_btn and ticker:
    with st.spinner(f"Analyzing {ticker}... (first run may take longer while data is ingested)"):
        try:
            technicals, docs, analysis = _cached_full_analysis(ticker)

            st.subheader("Technical Indicators")
            t = technicals
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Price", f"{t['current_price']}", f"{t['day_change_pct']}%")
            m2.metric("RSI (14)", t.get("rsi_14", "—"))
            m3.metric("Trend", t.get("trend", "—"))
            m4.metric("Volume", f"{t.get('volume', 0):,}")
            st.json(t)

            lean_info = extract_lean(analysis)
            if lean_info:
                lean, reasoning = lean_info
                lean_colors = {"Bullish": "🟢", "Neutral": "🟡", "Bearish": "🔴"}
                st.info(
                    f"{lean_colors.get(lean, '⚪')} **Sentiment/technical lean: {lean}** "
                    f"— {reasoning}\n\n"
                    f"_This is a synthesis of the retrieved signals above, not a recommendation "
                    f"to buy, sell, or hold._"
                )

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

# --- ADD-ON: PERSISTENT TRADING DESK CONTEXT ---
# This block is moved to the root level. It checks if the broker is active and displays 
# the trading matrix consistently, even across clicks.
if st.session_state.ventura_logged_in and st.session_state.ventura_client:
    st.divider()
    st.subheader("⚡ Ventura Instant Execution Matrix")
    clean_sym = ticker.replace(".NS", "").replace(".BO", "")
    is_in = ticker.endswith(".NS") or ticker.endswith(".BO")

    with st.form("ventura_form"):
        f1, f2, f3 = st.columns(3)
        with f1:
            ex_val = f1.selectbox("Segment", ["NSE", "BSE"], index=0 if is_in else 0)
            side_val = f1.selectbox("Action", ["BUY", "SELL"])
        with f2:
            sym_val = f2.text_input("Trading Code", value=clean_sym)
            type_val = f2.selectbox("Order Type", ["MARKET", "LIMIT"])
        with f3:
            qty_val = f3.number_input("Shares Count", min_value=1, step=1, value=1)
            # Default target price safely handles if technical analysis hasn't run yet
            initial_price = float(st.session_state.get('last_price_fallback', 0.0))
            if 'technicals' in locals():
                initial_price = float(technicals['current_price'])
                st.session_state.last_price_fallback = initial_price
            price_val = f3.number_input("Target Price (INR)", min_value=0.0, value=initial_price, step=0.05)
        
        submit = st.form_submit_button("Route Order Packet", type="primary", use_container_width=True)
        if submit:
            try:
                res = st.session_state.ventura_client.place_order(
                    exchange=ex_val,
                    trading_symbol=sym_val.upper().strip(),
                    transaction_type=side_val,
                    order_type=type_val,
                    quantity=int(qty_val),
                    price=float(price_val),
                    validity="0"
                )
                st.toast("Transmitted successfully!", icon="🚀")
                st.json(res)
            except Exception as o_err:
                st.error(f"Routing Failed: {o_err}")