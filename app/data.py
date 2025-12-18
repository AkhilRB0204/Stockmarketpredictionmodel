import yfinance as yf
import pandas as pd

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

    # Volatility (std of returns over last 10 periods)
    df["Volatility"] = df["Return"].rolling(10).std()

    # Drop rows with NaN values (from rolling calculations)
    df.dropna(inplace=True)

    return df
