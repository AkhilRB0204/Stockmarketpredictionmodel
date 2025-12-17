import time
from data import get_live_data, add_features
from model import train_model

# which stock we wanna predict
TICKER = "AAPL"

print("Training the model with some historical data...")

# get historical data and add our features
historical = add_features(get_live_data(TICKER))

# train the model
model = train_model(historical)

print("Starting real-time predictions...")

while True:
    # get latest data and add features
    live = add_features(get_live_data(TICKER))

    # take the most recent row of features
    latest = live.iloc[-1][['SMA_5', 'SMA_20', 'Volatility']]
    last_price = live['Close'].iloc[-1]

    # predict next return
    predicted_return = model.predict([latest])[0]

    # convert that return into a predicted price
    predicted_price = last_price * (1 + predicted_return)

    # print what’s happening
    print(f"Last: ${last_price:.2f} | Predicted next: ${predicted_price:.2f}")

    # wait a minute before running again
    time.sleep(60)
