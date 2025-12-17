import yfinance as yf
import pandas as pd

def get_live_data(ticker):
    # grab the latest 1-day stock data at 1-min intervals
    df = yf.download(ticker, period="1d", interval="1m")
    return df

def add_features(df):
    # calculate percentage change from previous close
    df['Return'] = df['Close'].pct_change()

    # add simple moving averages (5-min and 20-min)
    df['SMA_5'] = df['Close'].rolling(5).mean()
    df['SMA_20'] = df['Close'].rolling(20).mean()

    # calculate rolling volatility over 10 periods
    df['Volatility'] = df['Return'].rolling(10).std()

    # drop the first few rows with NaNs from rolling calculations
    df.dropna(inplace=True)
    return df
