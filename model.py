from sklearn.ensemble import RandomForestRegressor

def train_model(df):
    # pick the columns we want to use for predicting returns
    X = df[['SMA_5', 'SMA_20', 'Volatility']]

    # target is next period return
    y = df['Return'].shift(-1).dropna()

    # align X with y by dropping last row
    X = X.iloc[:-1]

    # create a Random Forest model
    model = RandomForestRegressor(
        n_estimators=200,  # 200 trees should be enough
        random_state=42    # keep results consistent
    )

    # train it on our features
    model.fit(X, y)
    return model
