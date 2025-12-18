import time
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # ensure plotting works in Docker/headless
import matplotlib.pyplot as plt

from data import get_live_data, add_features
from model import train_model

# ---------------------------
# Configuration
# ---------------------------
TICKER = "AAPL"
RETRAIN_EVERY = 30  # retrain model every N iterations
OUTPUT_DIR = "/app/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)  # Ensure output folder exists

# ---------------------------
# Initial model training
# ---------------------------
print("Training the model with historical data...")
historical = add_features(get_live_data(TICKER))
model = train_model(historical)

# Storage for plotting
timestamps = []
actual_prices = []
predicted_prices = []

counter = 0
print("Starting real-time predictions...\n")

# ---------------------------
# Main real-time loop
# ---------------------------
while True:
    try:
        # Fetch latest 1-minute data
        live_data = add_features(get_live_data(TICKER))

        # Skip iteration if no new candle
        if timestamps and live_data.index[-1] == timestamps[-1]:
            print("No new candle yet...")
            time.sleep(60)
            continue

        # Extract latest features safely
        feature_cols = ["SMA_5", "SMA_20", "Volatility"]
        latest_features = live_data[feature_cols].iloc[-1].to_numpy(dtype=float)

        # Last actual closing price
        last_price = float(live_data["Close"].iloc[-1].item())

        # Predict next-period return
        predicted_return = model.predict([latest_features])[0]

        # Convert return to predicted price
        predicted_price = last_price * (1 + predicted_return)

        # Store values for plotting
        timestamps.append(live_data.index[-1])
        actual_prices.append(last_price)
        predicted_prices.append(predicted_price)

        # Print to console
        print(f"Last: ${last_price:.2f} | Predicted next: ${predicted_price:.2f}")

        # ---------------------------
        # Plotting section
        # ---------------------------
        if timestamps:  # ensure we have data to plot
            plt.figure(figsize=(14,6))

            # Create DataFrame for plotting
            df_plot = pd.DataFrame({
                "Actual": actual_prices,
                "Predicted": predicted_prices
            }, index=pd.to_datetime(timestamps))

            # Resample monthly: take last value in each month
            df_monthly = df_plot.resample('M').last()

            # Plot actual vs predicted
            df_resampled = df_plot.resample('5T').last()  # every 5 minutes
            plt.plot(df_monthly.index, df_monthly["Actual"], label="Actual", color="#1f77b4", linewidth=2)
            plt.plot(df_monthly.index, df_monthly["Predicted"], label="Predicted", color="#ff7f0e", linestyle="--", linewidth=2)

            # Title, labels, grid
            plt.title(f"{TICKER} Live Stock Price Prediction", fontsize=16, fontweight='bold')
            plt.xlabel("Time")
            plt.ylabel("Price (USD)")
            plt.grid(True, linestyle='--', linewidth=0.5)
            plt.xticks(rotation=45)

            # Dynamic y-axis
            all_prices = df_monthly["Actual"].tolist() + df_monthly["Predicted"].dropna().tolist()
            if all_prices:
                plt.ylim(min(all_prices)*0.995, max(all_prices)*1.005)

            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, "live_stock_plot.png"))
            plt.close()  # prevent memory leaks

        # ---------------------------
        # Retrain model periodically
        # ---------------------------
        counter += 1
        if counter % RETRAIN_EVERY == 0:
            print("Retraining model with latest data...")
            historical = add_features(get_live_data(TICKER))
            model = train_model(historical)

        # Wait before next iteration
        time.sleep(60)

    except Exception as e:
        print(f"Error: {e}. Retrying in 60 seconds...")
        time.sleep(60)
