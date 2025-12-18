import yfinance as yf
import pandas as pd

def get_live_data(ticker):
    """
    Fetch most recent 1-day, 1-minute stock data
    """
    df = yf.download(
        ticker,
        period="1d",
        interval="1m",
        auto_adjust=True,
        progress=False
    )

    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.dropna(inplace=True)

    return df


def add_features(df):
    """
    Add technical indicators for the model
    """
    df = df.copy()

    # Returns
    df["Return"] = df["Close"].pct_change()

    # Moving averages
    df["SMA_5"] = df["Close"].rolling(5).mean()
    df["SMA_20"] = df["Close"].rolling(20).mean()

    # Volatility
    df["Volatility"] = df["Return"].rolling(10).std()

    df.dropna(inplace=True)

    return df
