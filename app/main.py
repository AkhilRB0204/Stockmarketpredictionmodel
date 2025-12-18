import time
import matplotlib
matplotlib.use("Agg")  # ensure plotting works in Docker/headless

import matplotlib.pyplot as plt
from data import get_live_data, add_features
from model import train_model
import os

# Make sure the output folder exists
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

        # skip if no new candle yet
        if timestamps and live_data.index[-1] == timestamps[-1]:
            print("No new candle yet...")
            time.sleep(60)
            continue

        # Extract features safely
        feature_cols = ["SMA_5", "SMA_20", "Volatility"]
        latest_features = live_data[feature_cols].iloc[-1].to_numpy(dtype=float)

        # Last actual price
        last_price = float(live_data["Close"].iloc[-1].item())

        # Predict next return
        predicted_return = model.predict([latest_features])[0]

        # Convert return to price
        predicted_price = last_price * (1 + predicted_return)

        # Store for plotting
        timestamps.append(live_data.index[-1])
        actual_prices.append(last_price)
        predicted_prices.append(predicted_price)

        # Print to console
        print(f"Last: ${last_price:.2f} | Predicted next: ${predicted_price:.2f}")

        # ----- Plotting section -----
        plt.figure(figsize=(14,6))  # bigger figure
        plt.plot(timestamps, actual_prices, label="Actual", color="#1f77b4", linewidth=2)  # solid line
        # dashed line for predictions; replace None with np.nan if needed
        preds = [p if p is not None else float('nan') for p in predicted_prices]
        plt.plot(timestamps, preds, label="Predicted", color="#ff7f0e", linestyle="--", linewidth=2)

        # Title, labels, grid
        plt.title(f"{TICKER} Live Stock Price Prediction", fontsize=16, fontweight='bold')
        plt.xlabel("Time")
        plt.ylabel("Price (USD)")
        plt.grid(True, linestyle='--', linewidth=0.5)
        plt.xticks(rotation=45)

        # Dynamic y-axis
        all_prices = [p for p in actual_prices + [p for p in predicted_prices if p is not None]]
        plt.ylim(min(all_prices)*0.995, max(all_prices)*1.005)

        plt.legend()
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