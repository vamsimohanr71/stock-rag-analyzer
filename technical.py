"""
Quantitative signal generation from price history. This is deliberately
kept separate from the RAG/LLM path - it's the numeric half of the hybrid
design (see README for rationale).
"""
import pandas as pd
import ta


def get_technical_signals(hist_df: pd.DataFrame) -> dict:
    df = hist_df.copy()
    df["RSI"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()
    macd = ta.trend.MACD(df["Close"])
    df["MACD"] = macd.macd()
    df["MACD_signal"] = macd.macd_signal()
    df["SMA_50"] = df["Close"].rolling(50).mean()
    df["SMA_200"] = df["Close"].rolling(200).mean()
    bb = ta.volatility.BollingerBands(df["Close"])
    df["BB_high"] = bb.bollinger_hband()
    df["BB_low"] = bb.bollinger_lband()

    latest = df.iloc[-1]
    prev_close = df["Close"].iloc[-2] if len(df) > 1 else latest["Close"]

    sma_trend = None
    if pd.notna(latest["SMA_50"]) and pd.notna(latest["SMA_200"]):
        sma_trend = "bullish (50D > 200D)" if latest["SMA_50"] > latest["SMA_200"] else "bearish (50D < 200D)"

    return {
        "current_price": round(float(latest["Close"]), 2),
        "day_change_pct": round(float((latest["Close"] - prev_close) / prev_close * 100), 2),
        "rsi_14": round(float(latest["RSI"]), 2) if pd.notna(latest["RSI"]) else None,
        "macd": round(float(latest["MACD"]), 4) if pd.notna(latest["MACD"]) else None,
        "macd_signal": round(float(latest["MACD_signal"]), 4) if pd.notna(latest["MACD_signal"]) else None,
        "sma_50": round(float(latest["SMA_50"]), 2) if pd.notna(latest["SMA_50"]) else None,
        "sma_200": round(float(latest["SMA_200"]), 2) if pd.notna(latest["SMA_200"]) else None,
        "trend": sma_trend,
        "bollinger_high": round(float(latest["BB_high"]), 2) if pd.notna(latest["BB_high"]) else None,
        "bollinger_low": round(float(latest["BB_low"]), 2) if pd.notna(latest["BB_low"]) else None,
        "volume": int(latest["Volume"]),
    }
