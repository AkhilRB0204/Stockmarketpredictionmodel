import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from data import EXCLUDE_COLS

MAE_RETRAIN_THRESH = 0.50   # dollars — trigger early retrain if exceeded


class ModelValidator:
    """
    Tracks rolling predictions vs actuals and computes performance metrics.

    Call pattern (in the worker loop):
        validator.add_prediction(ts, predicted=pred_price, actual=actual_price)
        metrics = validator.get_recent_performance(n=20)
        if validator.should_retrain(): ...
    """

    def __init__(self):
        self.predictions: list[float] = []
        self.actuals:     list[float] = []
        self.timestamps:  list        = []

    def add_prediction(self, timestamp, predicted: float, actual: float = None):
        self.timestamps.append(timestamp)
        self.predictions.append(predicted)
        if actual is not None:
            self.actuals.append(actual)

    # ── Metrics ───────────────────────────────────────────────────────────

    def calculate_metrics(self) -> dict | None:
        """Full-history metrics."""
        if len(self.actuals) < 2:
            return None
        a = np.array(self.actuals)
        p = np.array(self.predictions[: len(self.actuals)])
        return {
            "RMSE":               float(np.sqrt(mean_squared_error(a, p))),
            "MAE":                float(mean_absolute_error(a, p)),
            "R2":                 float(r2_score(a, p)),
            "MAPE":               float(np.mean(np.abs((a - p) / a)) * 100),
            "Direction_Accuracy": self._direction_accuracy(a, p),
        }

    def get_recent_performance(self, n: int = 30) -> dict | None:
        """Rolling metrics over the last `n` actuals."""
        if len(self.actuals) < 2:
            return None
        n   = min(n, len(self.actuals))
        a   = np.array(self.actuals[-n:])
        # predictions list leads actuals by 1 — align carefully
        offset = len(self.predictions) - len(self.actuals)
        p   = np.array(self.predictions[offset:][-n:])
        min_len = min(len(a), len(p))
        a, p = a[-min_len:], p[-min_len:]
        if min_len < 2:
            return None
        return {
            "RMSE":               float(np.sqrt(mean_squared_error(a, p))),
            "MAE":                float(mean_absolute_error(a, p)),
            "R2":                 float(r2_score(a, p)),
            "Direction_Accuracy": self._direction_accuracy(a, p),
        }

    def should_retrain(self, threshold_mae: float = MAE_RETRAIN_THRESH) -> bool:
        metrics = self.get_recent_performance(n=20)
        return metrics is not None and metrics["MAE"] > threshold_mae

    @staticmethod
    def _direction_accuracy(a: np.ndarray, p: np.ndarray) -> float:
        if len(a) < 2:
            return 0.0
        return float(np.mean(np.diff(a) > 0 == np.diff(p) > 0) * 100)


# ── Walk-forward validation ────────────────────────────────────────────────

def walk_forward_validation(df, model_class, train_size: float = 0.7,
                             step: int = 10) -> dict:
    """
    Sliding-window backtest on a historical feature DataFrame.
    model_class must be callable with no args (e.g. EnsembleStockPredictor).
    """
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    n         = len(df)
    train_end = int(n * train_size)
    preds, actuals = [], []

    for i in range(train_end, n - 1, step):
        train = df.iloc[:i]
        X_tr  = train[feature_cols].iloc[:-1].values
        y_tr  = train["Return"].shift(-1).iloc[:-1].values
        model = model_class()
        model.fit(X_tr, y_tr)
        X_te = df[feature_cols].iloc[i: i + 1].values
        preds.append(float(model.predict(X_te)[0]))
        actuals.append(float(df["Return"].iloc[i + 1]))

    if len(actuals) < 2:
        raise ValueError("Not enough data for walk-forward validation.")

    a, p = np.array(actuals), np.array(preds)
    return {
        "RMSE":               float(np.sqrt(mean_squared_error(a, p))),
        "MAE":                float(mean_absolute_error(a, p)),
        "R2":                 float(r2_score(a, p)),
        "Direction_Accuracy": float(np.mean(np.sign(a) == np.sign(p)) * 100),
    }