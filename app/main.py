import time
import matplotlib.pyplot as plt
from data import get_live_data, add_features
from model import train_model
import os

# Ensure output folder exists
os.makedirs("/app/output", exist_ok=True)

TICKER = "AAPL"

print("Training the model with some historical data...")

# Get recent stock data and calculate features
historical = add_features(get_live_data(TICKER))

# Train the ML model once
model = train_model(historical)

# Lists to store live data for plotting
timestamps = []
actual_prices = []
predicted_prices = []

print("Starting real-time predictions...\n")

while True:
    try:
        # Fetch the latest data
        live_data = add_features(get_live_data(TICKER))

        # Latest features
        latest_features = live_data.iloc[-1][["SMA_5", "SMA_20", "Volatility"]].values.astype(float)

        # Last actual price
        last_price = float(live_data["Close"].iloc[-1].item())

        # Predict next return
        predicted_return = model.predict([latest_features])[0]

        # Convert predicted return into predicted price
        predicted_price = last_price * (1 + predicted_return)

        # Append to lists for plotting
        timestamps.append(live_data.index[-1])
        actual_prices.append(last_price)
        predicted_prices.append(predicted_price)

        # Print to console
        print(f"Last: ${last_price:.2f} | Predicted next: ${predicted_price:.2f}")

        # --- Plot live graph ---
        plt.clf()  # Clear previous plot
        plt.plot(timestamps, actual_prices, label="Actual", marker='o')
        plt.plot(timestamps, predicted_prices, label="Predicted", linestyle='--', marker='x')
        plt.title(f"{TICKER} Live Stock Price Prediction")
        plt.xlabel("Time")
        plt.ylabel("Price (USD)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("/app/output/live_stock_plot.png")  # Save graph
        plt.pause(0.1)  # Small pause to update

        # Wait 60 seconds
        time.sleep(60)

    except Exception as e:
        print(f"Error: {e}. Retrying in 60 seconds...")
        time.sleep(60)
