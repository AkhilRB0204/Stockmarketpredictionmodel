# Real-Time Stock Predictor 🚀

Hey! This is my real-time stock predictor project.  
It’s a Python app that grabs live stock data, calculates some basic indicators, and tries to predict the next price.  
Also, it’s fully Dockerized so you can just run it anywhere.

---

## What it does

- Pulls live stock data from Yahoo Finance (1-minute intervals)  
- Adds a few features like moving averages, volatility, and returns  
- Trains a Random Forest model to predict the next price movement  
- Prints out the last price and predicted next price every minute  
- Runs in Docker so it’s super easy to set up  

---

## How to run it

### Build the Docker image
```bash
docker build -t stock-predictor .
