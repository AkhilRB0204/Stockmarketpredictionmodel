import time
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # ensure plotting works in Docker/headless
import matplotlib.pyplot as plt
from datetime import datetime
from data import get_live_data, add_features
from model import train_model

# Configuration
TICKER = "AAPL"
RETRAIN_EVERY = 30  # retrain model every N iterations
OUTPUT_DIR = "/app/output"
MODEL_PATH = os.path.join(OUTPUT_DIR, "model.pkl")
os.makedirs(OUTPUT_DIR, exist_ok=True)  # Ensure output folder exists

# Initialize validator
validator = ModelValidator()

# Initial model training
print("Training the model with historical data...")
historical = add_features(get_live_data(TICKER))
model = train_model(historical)
save_model(model, MODEL_PATH)


# Get feature importance
if hasattr(model, 'get_feature_importance'):
    importance = model.get_feature_importance(model.feature_names)
    print("\nTop 10 Important Features:")
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
    for feat, imp in sorted_importance:
        print(f"  {feat}: {imp:.4f}")

# Storage for plotting
timestamps = []
actual_prices = []
predicted_prices = []
confidence_intervals = []


counter = 0
last_retrain = 0

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
        exclude_cols = ["Return", "Open", "High", "Low", "Close", "Volume"]
        feature_cols = [col for col in live_data.columns if col not in exclude_cols]
        latest_features = live_data[feature_cols].iloc[-1].to_numpy(dtype=float).reshape(1, -1)
        # Last actual closing price
        last_price = float(live_data["Close"].iloc[-1].item())

        # Predict next-period return
        predicted_return = model.predict([latest_features])[0]

        # Convert return to predicted price
        predicted_price = last_price * (1 + predicted_return)

        # Calculate confidence interval (simple approach using recent volatility)
        recent_volatility = live_data["Volatility"].iloc[-1]
        confidence = current_price * recent_volatility * 1.96  # 95% CI

        # Store values for plotting
        timestamp = live_data.index[-1]
        timestamps.append(timestamp)
        actual_prices.append(current_price)
        predicted_prices.append(predicted_price)
        confidence_intervals.append(confidence)

        # Add to validator
        if len(actual_prices) > 1:
            validator.add_prediction(
                timestamp=timestamps[-2],
                predicted=predicted_prices[-2],
                actual=actual_prices[-1]
            )

        # Print to console
        direction = "↑" if predicted_return > 0 else "↓"
        print(f"[{timestamp.strftime('%H:%M')}] Current: ${current_price:.2f} | "
              f"Predicted: ${predicted_price:.2f} {direction} | "
              f"±${confidence:.2f}")
        
        # Show performance metrics every 10 iterations
        if counter > 0 and counter % 10 == 0:
            metrics = validator.get_recent_performance(n=10)
            if metrics:
                print(f"\n📊 Recent Performance (last 10):")
                print(f"   MAE: ${metrics['MAE']:.4f}")
                print(f"   RMSE: ${metrics['RMSE']:.4f}")
                print(f"   Direction Accuracy: {metrics['Direction_Accuracy']:.1f}%\n")

        # Plotting section
        if len(timestamps) > 1:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
            
            # Create DataFrame for plotting
            df_plot = pd.DataFrame({
                "Actual": actual_prices,
                "Predicted": predicted_prices,
                "Upper_CI": [p + c for p, c in zip(predicted_prices, confidence_intervals)],
                "Lower_CI": [p - c for p, c in zip(predicted_prices, confidence_intervals)]
            }, index=pd.to_datetime(timestamps))
            
            # Resample for cleaner visualization
            df_resampled = df_plot.resample('5T').last()
            
            # Plot 1: Price predictions
            ax1.plot(df_resampled.index, df_resampled["Actual"], 
                    label="Actual", color="#1f77b4", linewidth=2)
            ax1.plot(df_resampled.index, df_resampled["Predicted"], 
                    label="Predicted", color="#ff7f0e", linestyle="--", linewidth=2)
            
            # Add confidence interval
            ax1.fill_between(df_resampled.index, 
                           df_resampled["Lower_CI"], 
                           df_resampled["Upper_CI"],
                           alpha=0.2, color="#ff7f0e", label="95% CI")
            
            ax1.set_title(f"{TICKER} Live Stock Price Prediction", 
                         fontsize=16, fontweight='bold')
            ax1.set_ylabel("Price (USD)")
            ax1.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
            ax1.legend(loc='upper left')
            
            # Plot 2: Prediction errors
            if len(actual_prices) > 2:
                errors = [actual_prices[i] - predicted_prices[i-1] 
                         for i in range(1, len(actual_prices))]
                error_times = timestamps[1:]
                
                ax2.bar(error_times, errors, width=0.003, 
                       color=['g' if e > 0 else 'r' for e in errors], alpha=0.6)
                ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
                ax2.set_title("Prediction Errors", fontsize=14)
                ax2.set_xlabel("Time")
                ax2.set_ylabel("Error (USD)")
                ax2.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
            
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, "live_stock_plot.png"), dpi=100)
            plt.close()
        
        # ---------------------------
        # Smart Retraining Logic
        # ---------------------------
        counter += 1
        should_retrain = False
        
        # Regular interval retraining
        if counter % RETRAIN_EVERY == 0:
            should_retrain = True
            reason = "regular interval"
        
        # Performance-based retraining
        elif (counter - last_retrain) >= MIN_RETRAIN_INTERVAL and validator.should_retrain():
            should_retrain = True
            reason = "poor performance"
        
        if should_retrain:
            print(f"\n🔄 Retraining model ({reason})...")
            historical = add_features(get_live_data(TICKER))
            model = train_model(historical)
            save_model(model, MODEL_PATH)
            last_retrain = counter
            print("Model retrained successfully\n")
        
        # Wait before next iteration
        time.sleep(60)
        
    except Exception as e:
        print(f" Error: {e}. Retrying in 60 seconds...")
        time.sleep(60)