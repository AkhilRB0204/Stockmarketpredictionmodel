import yfinance as yf
import pandas as pd
import numpy as np


def get_live_data(ticker):
    """
    Fetch most recent 1-day, 1-minute stock data from Yahoo Finance
    """
    df = yf.download(
        ticker,
        period="1d",
        interval="1m",
        auto_adjust=True,
        progress=False
    )

    # Raise error if no data returned
    if df.empty:
        raise ValueError(f"No data fetched for {ticker}")

    # Keep only relevant columns
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.dropna(inplace=True)
    df.sort_index(inplace=True)  # Ensure datetime order
    return df


def add_features(df):
    """
    Add technical indicators for the model:
      - Return: next-period percentage change
      - SMA_5 and SMA_20: simple moving averages
      - Volatility: 10-period rolling standard deviation of returns
    """
    df = df.copy()

    # Compute returns
    df["Return"] = df["Close"].pct_change()

    # Simple moving averages
    df["SMA_5"] = df["Close"].rolling(5).mean()
    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()

    # MACD
    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    # Relative Strength Index
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    #  Bollinger Bands
    df["BB_Middle"] = df["Close"].rolling(20).mean()
    bb_std = df["Close"].rolling(20).std()
    df["BB_Upper"] = df["BB_Middle"] + (bb_std * 2)
    df["BB_Lower"] = df["BB_Middle"] - (bb_std * 2)
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Middle"]
    df["BB_Position"] = (df["Close"] - df["BB_Lower"]) / (df["BB_Upper"] - df["BB_Lower"])

    # Volatility (std of returns over last 10 periods)
    df["Volatility"] = df["Return"].rolling(10).std()
    df["ATR"] = calculate_atr(df, period=14)

    #  Volume Features 
    df["Volume_SMA"] = df["Volume"].rolling(20).mean()
    df["Volume_Ratio"] = df["Volume"] / df["Volume_SMA"]
    df["Volume_Change"] = df["Volume"].pct_change()

    #  Price Action Features 
    df["High_Low_Ratio"] = (df["High"] - df["Low"]) / df["Close"]
    df["Close_Open_Ratio"] = (df["Close"] - df["Open"]) / df["Open"]

    #  Momentum Indicators 
    df["ROC"] = ((df["Close"] - df["Close"].shift(10)) / df["Close"].shift(10)) * 100  # Rate of Change
    df["Momentum"] = df["Close"] - df["Close"].shift(4)

    #  Lagged Returns (helps capture patterns) 
    df["Return_Lag1"] = df["Return"].shift(1)
    df["Return_Lag2"] = df["Return"].shift(2)
    df["Return_Lag3"] = df["Return"].shift(3)
    
    #  Time-based Features 
    df["Hour"] = df.index.hour
    df["Minute"] = df.index.minute
    df["DayOfWeek"] = df.index.dayofweek

    # Drop rows with NaN values (from rolling calculations)
    df.dropna(inplace=True)

    return df

def calculate_atr(df, period=14):

    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift())
    low_close = np.abs(df["Low"] - df["Close"].shift())

    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    atr = true_range.rolling(period).mean()

    return atr