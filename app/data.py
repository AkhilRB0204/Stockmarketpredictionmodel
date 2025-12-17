import yfinance as yf
import pandas as pd

def get_live_data(ticker):
    # grab the most recent 1-day stock data at 1-minute intervals
    # auto_adjust=True to avoid future warning and adjust for splits/dividends
    df = yf.download(ticker, period="1d", interval="1m", auto_adjust=True)
    return df

def add_features(df):
    # percentage return from previous close
    df["Return"] = df["Close"].pct_change()

    # simple moving averages
    df["SMA_5"] = df["Close"].rolling(5).mean()
    df["SMA_20"] = df["Close"].rolling(20).mean()

    # rolling volatility (how unstable price has been)
    df["Volatility"] = df["Return"].rolling(10).std()

    # drop rows created by rolling calculations
    df.dropna(inplace=True)
    return df
