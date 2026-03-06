import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression

from data import EXCLUDE_COLS


class EnsembleStockPredictor:
    """
    Weighted ensemble: RandomForest (50%) + GradientBoosting (30%) + Ridge (20%).
    Applies StandardScaler and SelectKBest before each sub-model.
    """

    def __init__(self, n_features: int = 15):
        self.scaler           = StandardScaler()
        self.feature_selector = SelectKBest(score_func=f_regression, k=n_features)
        self.models = {
            "rf": RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            ),
            "gb": GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
            ),
            "ridge": Ridge(alpha=1.0),
        }
        self.weights: dict[str, float]  = {"rf": 0.5, "gb": 0.3, "ridge": 0.2}
        self.feature_names: list | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "EnsembleStockPredictor":
        X_scaled   = self.scaler.fit_transform(X)
        X_selected = self.feature_selector.fit_transform(X_scaled, y)
        for model in self.models.values():
            model.fit(X_selected, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled   = self.scaler.transform(X)
        X_selected = self.feature_selector.transform(X_scaled)
        return sum(
            self.models[name].predict(X_selected) * weight
            for name, weight in self.weights.items()
        )

    def get_feature_importance(self, feature_names: list[str]) -> dict[str, float]:
        """Sorted feature importances from the Random Forest sub-model."""
        mask     = self.feature_selector.get_support()
        selected = [f for f, s in zip(feature_names, mask) if s]
        imps     = self.models["rf"].feature_importances_
        return dict(sorted(zip(selected, imps), key=lambda x: -x[1]))


def train_model(df) -> EnsembleStockPredictor:
    """Fit a fresh ensemble on a feature-engineered DataFrame."""
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    X = df[feature_cols].iloc[:-1].values
    y = df["Return"].shift(-1).iloc[:-1].values
    model = EnsembleStockPredictor(n_features=min(15, len(feature_cols)))
    model.fit(X, y)
    model.feature_names = feature_cols
    return model


def save_model(model: EnsembleStockPredictor, filepath: str) -> None:
    with open(filepath, "wb") as f:
        pickle.dump(model, f)


def load_model(filepath: str) -> EnsembleStockPredictor:
    with open(filepath, "rb") as f:
        return pickle.load(f)