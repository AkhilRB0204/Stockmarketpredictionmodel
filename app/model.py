from sklearn.ensemble import RandomForestRegressor

def train_model(df):
    """
    Train a Random Forest model to predict next-period returns
    """

    feature_cols = ["SMA_5", "SMA_20", "Volatility"]

    # Features
    X = df[feature_cols].iloc[:-1].values

    # Target: next-period return
    y = df["Return"].shift(-1).dropna().values

    # Random Forest Regressor
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X, y)

    return model
