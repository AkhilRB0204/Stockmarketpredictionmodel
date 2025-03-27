import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Fetch stock data
def get_stock_data(ticker, start="2020-01-01", end="2025-01-01"):
    stock = yf.download(ticker, start=start, end=end)
    return stock[['Close']]

# Prepare data for training
def prepare_data(data, prediction_days=30):
    data['Prediction'] = data['Close'].shift(-prediction_days)
    X = np.array(data.drop(['Prediction'], axis=1))[:-prediction_days]
    y = np.array(data['Prediction'])[:-prediction_days]
    return X, y

# Train and predict
def train_and_predict(ticker):
    data = get_stock_data(ticker)
    X, y = prepare_data(data)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predict next 30 days
    future_data = np.array(data.drop(['Prediction'], axis=1))[-30:]
    future_predictions = model.predict(future_data)
    
    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(data.index[-100:], data['Close'].iloc[-100:], label="Actual Prices")
    plt.plot(data.index[-30:], future_predictions, label="Predicted Prices", linestyle="dashed")
    plt.legend()
    plt.title(f"Stock Price Prediction for {ticker}")
    plt.show()
    
    return future_predictions

# Run predictor
ticker = "AAPL"  # Example stock
predictions = train_and_predict(ticker)
print(predictions)
