import yfinance as yf
import pandas as pd
import numpy as np

# Columns that are targets or raw OHLCV — never fed into the model as features
EXCLUDE_COLS = ["Return", "Open", "High", "Low", "Close", "Volume"]


def get_live_data(ticker: str, period: str = "5d") -> pd.DataFrame:
    """
    Fetch 1-minute OHLCV bars via yfinance.
    period="5d" ensures enough history for all rolling warm-up windows
    (SMA_20 needs 20 bars, RSI needs 14, ATR needs 14, etc.)
    """
    df = yf.download(
        ticker,
        period=period,
        interval="1m",
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if df.empty:
        raise ValueError(f"No data fetched for {ticker}")

    # yfinance >= 0.2 returns MultiIndex columns for single-ticker downloads
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.dropna(inplace=True)
    df.sort_index(inplace=True)
    return df


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range — measures market volatility."""
    high_low   = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift())
    low_close  = np.abs(df["Low"]  - df["Close"].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(period).mean()


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer all technical indicators used by EnsembleStockPredictor.
    Drops NaN rows produced by rolling warm-up windows before returning.
    """
    df = df.copy()

    # ── Returns ───────────────────────────────────────────────────────────
    df["Return"] = df["Close"].pct_change()

    # ── Trend / Moving Averages ───────────────────────────────────────────
    df["SMA_5"]  = df["Close"].rolling(5).mean()
    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()

    # ── MACD ──────────────────────────────────────────────────────────────
    df["MACD"]        = df["EMA_12"] - df["EMA_26"]
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"]   = df["MACD"] - df["MACD_Signal"]

    # ── RSI ───────────────────────────────────────────────────────────────
    delta = df["Close"].diff()
    gain  = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df["RSI"] = 100 - (100 / (1 + gain / loss))

    # ── Bollinger Bands ───────────────────────────────────────────────────
    df["BB_Middle"]   = df["Close"].rolling(20).mean()
    bb_std            = df["Close"].rolling(20).std()
    df["BB_Upper"]    = df["BB_Middle"] + bb_std * 2
    df["BB_Lower"]    = df["BB_Middle"] - bb_std * 2
    df["BB_Width"]    = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Middle"]
    df["BB_Position"] = (df["Close"] - df["BB_Lower"]) / (df["BB_Upper"] - df["BB_Lower"])

    # ── Volatility & Range ────────────────────────────────────────────────
    df["Volatility"]       = df["Return"].rolling(10).std()
    df["ATR"]              = calculate_atr(df, period=14)
    df["High_Low_Ratio"]   = (df["High"] - df["Low"]) / df["Close"]
    df["Close_Open_Ratio"] = (df["Close"] - df["Open"]) / df["Open"]

    # ── Volume ────────────────────────────────────────────────────────────
    df["Volume_SMA"]    = df["Volume"].rolling(20).mean()
    df["Volume_Ratio"]  = df["Volume"] / df["Volume_SMA"]
    df["Volume_Change"] = df["Volume"].pct_change()

    # ── Momentum ──────────────────────────────────────────────────────────
    df["ROC"]      = df["Close"].pct_change(10) * 100
    df["Momentum"] = df["Close"] - df["Close"].shift(4)

    # ── Lagged Returns ────────────────────────────────────────────────────
    df["Return_Lag1"] = df["Return"].shift(1)
    df["Return_Lag2"] = df["Return"].shift(2)
    df["Return_Lag3"] = df["Return"].shift(3)

    # ── Time Features ─────────────────────────────────────────────────────
    df["Hour"]      = df.index.hour
    df["Minute"]    = df.index.minute
    df["DayOfWeek"] = df.index.dayofweek

    # Replace any inf/-inf produced by division (e.g. Volume_Ratio when
    # Volume_SMA is zero, BB_Position when band width is zero) with NaN
    # so the subsequent dropna() removes those rows cleanly.
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df