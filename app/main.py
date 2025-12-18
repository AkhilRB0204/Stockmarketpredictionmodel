import time
import matplotlib.pyplot as plt
from data import get_live_data, add_features
from model import train_model
import os

# Ensure output folder exists
os.makedirs("/app/output", exist_ok=True)

TICKER = "AAPL"
RETRAIN_EVERY = 30  # retrain model every 30 iterations

print("Training the model with historical data...")

# Initial training
historical = add_features(get_live_data(TICKER))
model = train_model(historical)

# Storage for plotting
timestamps = []
actual_prices = []
predicted_prices = []

counter = 0

print("Starting real-time predictions...\n")

while True:
    try:
        # Fetch latest data
        live_data = add_features(get_live_data(TICKER))

        # Guard against duplicate candles
        if timestamps and live_data.index[-1] == timestamps[-1]:
            print("No new candle yet...")
            time.sleep(60)
            continue

        # Extract features safely
        feature_cols = ["SMA_5", "SMA_20", "Volatility"]
        latest_features = (
            live_data[feature_cols]
            .iloc[-1]
            .to_numpy(dtype=float)
        )

        # Last actual price (fixed float warning)
        last_price = float(live_data["Close"].iloc[-1].item())

        # Predict next return
        predicted_return = model.predict([latest_features])[0]

        # Convert return to price
        predicted_price = last_price * (1 + predicted_return)

        # Store values
        timestamps.append(live_data.index[-1])
        actual_prices.append(last_price)
        predicted_prices.append(predicted_price)

        # Console output
        print(f"Last: ${last_price:.2f} | Predicted next: ${predicted_price:.2f}")

        # ---- Plot ----
        plt.clf()
        plt.plot(timestamps, actual_prices, label="Actual", marker="o")
        plt.plot(timestamps, predicted_prices, label="Predicted", linestyle="--", marker="x")
        plt.title(f"{TICKER} Live Stock Price Prediction")
        plt.xlabel("Time")
        plt.ylabel("Price (USD)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("/app/output/live_stock_plot.png")
        plt.close()  # prevent memory leaks

        # Retrain model periodically
        counter += 1
        if counter % RETRAIN_EVERY == 0:
            print("Retraining model with latest data...")
            historical = add_features(get_live_data(TICKER))
            model = train_model(historical)

        # Wait before next poll
        time.sleep(60)

    except Exception as e:
        print(f"Error: {e}. Retrying in 60 seconds...")
        time.sleep(60)