from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor 
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
import numpy as np
import pickle


class EnsembleStockPredictor:
    """
    Ensemble model combining multiple algorithms with feature selection
    """
    def __init__(self, n_features=15):
        self.scaler = StandardScaler()
        self.feature_selector = SelectKBest(score_func=f_regression, k=n_features)
        
        # Multiple models for ensemble
        self.models = {
            'rf': RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'gb': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            ),
            'ridge': Ridge(alpha=1.0)
        }
        
        # Weights for each model (can be tuned based on validation performance)
        self.weights = {'rf': 0.5, 'gb': 0.3, 'ridge': 0.2}
        self.feature_names = None
        
    def fit(self, X, y):
        """Train the ensemble model"""
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Select best features
        X_selected = self.feature_selector.fit_transform(X_scaled, y)
        
        # Train each model
        for name, model in self.models.items():
            model.fit(X_selected, y)
        
        return self
    
    def predict(self, X):
        """Predict using weighted ensemble"""
        # Scale and select features
        X_scaled = self.scaler.transform(X)
        X_selected = self.feature_selector.transform(X_scaled)
        
        # Get predictions from each model
        predictions = {}
        for name, model in self.models.items():
            predictions[name] = model.predict(X_selected)
        
        # Weighted average
        ensemble_pred = sum(
            predictions[name] * self.weights[name] 
            for name in self.models.keys()
        )
        
        return ensemble_pred
    
    def get_feature_importance(self, feature_names):
        """Get feature importance from Random Forest"""
        if 'rf' in self.models:
            # Get selected feature indices
            selected_mask = self.feature_selector.get_support()
            selected_features = [f for f, s in zip(feature_names, selected_mask) if s]
            
            importances = self.models['rf'].feature_importances_
            return dict(zip(selected_features, importances))
        return {}



def train_model(df):
    """
    Train ensemble model to predict next-period returns
    """
    # Define all feature columns (exclude target and time features for now)
    exclude_cols = ["Return", "Open", "High", "Low", "Close", "Volume"]
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Features (exclude last row as its target is unknown)
    X = df[feature_cols].iloc[:-1].values
    
    # Target: next-period return
    y = df["Return"].shift(-1).iloc[:-1].values
    
    # Initialize and train ensemble model
    model = EnsembleStockPredictor(n_features=min(15, len(feature_cols)))
    model.fit(X, y)
    model.feature_names = feature_cols
    
    return model

def save_model(model, filepath):
    """Save trained model to disk"""
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)

def load_model(filepath):
    """Load trained model from disk"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)
