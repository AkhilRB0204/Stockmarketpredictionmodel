import time
from data import get_live_data, add_features
from model import train_model

# Stock symbol to predict — change if you want
TICKER = "AAPL"

print("Training the model with some historical data...")

# Get recent stock data and calculate features
historical = add_features(get_live_data(TICKER))

# Train the ML model once
model = train_model(historical)

print("Starting real-time predictions...\n")

while True:
    try:
        # Fetch the latest 1-day, 1-minute interval data
        live_data = add_features(get_live_data(TICKER))

        # Ensure latest features are numeric array for prediction
        latest_features = live_data.iloc[-1][["SMA_5", "SMA_20", "Volatility"]].values.astype(float)

        # Last actual price as a float
        last_price = float(live_data["Close"].iloc[-1])

        # Predict the next return
        predicted_return = model.predict([latest_features])[0]

        # Convert predicted return into predicted price
        predicted_price = last_price * (1 + predicted_return)

        # Print nicely
        print(f"Last: ${last_price:.2f} | Predicted next: ${predicted_price:.2f}")

        # Wait 60 seconds before the next prediction
        time.sleep(60)

    except Exception as e:
        # In case something goes wrong, don't crash the loop
        print(f"Error: {e}. Retrying in 60 seconds...")
        time.sleep(60)
