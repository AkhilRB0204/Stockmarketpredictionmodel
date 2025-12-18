from sklearn.ensemble import RandomForestRegressor

def train_model(df):
    """
    Train a Random Forest model to predict next-period returns.
    Uses SMA_5, SMA_20, Volatility as features.
    """
    feature_cols = ["SMA_5", "SMA_20", "Volatility"]

    # Features (exclude last row as its target is unknown)
    X = df[feature_cols].iloc[:-1].values

    # Target: next-period return (aligned with features)
    y = df["Return"].shift(-1).iloc[:-1].values

    # Initialize Random Forest Regressor
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    # Fit model
    model.fit(X, y)

    return model

