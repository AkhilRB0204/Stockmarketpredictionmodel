import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd

class ModelValidator:
    """
    Track and validate model performance over time
    """
    def __init__(self):
        self.predictions = []
        self.actuals = []
        self.timestamps = []
        
    def add_prediction(self, timestamp, predicted, actual=None):
        """Add a prediction for tracking"""
        self.timestamps.append(timestamp)
        self.predictions.append(predicted)
        if actual is not None:
            self.actuals.append(actual)
    
    def calculate_metrics(self):
        """Calculate performance metrics"""
        if len(self.actuals) < 2:
            return None
        
        actuals = np.array(self.actuals)
        predictions = np.array(self.predictions[:len(self.actuals)])
        
        metrics = {
            'RMSE': np.sqrt(mean_squared_error(actuals, predictions)),
            'MAE': mean_absolute_error(actuals, predictions),
            'R2': r2_score(actuals, predictions),
            'MAPE': np.mean(np.abs((actuals - predictions) / actuals)) * 100,
            'Direction_Accuracy': self._direction_accuracy(actuals, predictions)
        }
        
        return metrics
    
    def _direction_accuracy(self, actuals, predictions):
        """Calculate how often we predict the correct direction"""
        if len(actuals) < 2:
            return 0
        
        actual_directions = np.diff(actuals) > 0
        pred_directions = np.diff(predictions) > 0
        
        return np.mean(actual_directions == pred_directions) * 100
    
    def get_recent_performance(self, n=30):
        """Get performance over last n predictions"""
        if len(self.actuals) < n:
            return self.calculate_metrics()
        
        recent_actuals = np.array(self.actuals[-n:])
        recent_predictions = np.array(self.predictions[-n:])
        
        metrics = {
            'RMSE': np.sqrt(mean_squared_error(recent_actuals, recent_predictions)),
            'MAE': mean_absolute_error(recent_actuals, recent_predictions),
            'Direction_Accuracy': self._direction_accuracy(recent_actuals, recent_predictions)
        }
        
        return metrics
    
    def should_retrain(self, threshold_mae=0.02):
        """Decide if model should be retrained based on recent performance"""
        recent_metrics = self.get_recent_performance(n=20)
        
        if recent_metrics is None:
            return False
        
        # Retrain if recent MAE is too high
        return recent_metrics['MAE'] > threshold_mae

def walk_forward_validation(df, model_class, train_size=0.7, step=10):
    """
    Perform walk-forward validation on historical data
    """
    n = len(df)
    train_end = int(n * train_size)
    
    predictions = []
    actuals = []
    
    exclude_cols = ["Return", "Open", "High", "Low", "Close", "Volume"]
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    for i in range(train_end, n - 1, step):
        # Train on data up to current point
        train_df = df.iloc[:i]
        X_train = train_df[feature_cols].iloc[:-1].values
        y_train = train_df["Return"].shift(-1).iloc[:-1].values
        
        # Train model
        model = model_class()
        model.fit(X_train, y_train)
        
        # Predict next point
        X_test = df[feature_cols].iloc[i:i+1].values
        y_test = df["Return"].iloc[i+1]
        
        pred = model.predict(X_test)[0]
        
        predictions.append(pred)
        actuals.append(y_test)
    
    # Calculate metrics
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    metrics = {
        'RMSE': np.sqrt(mean_squared_error(actuals, predictions)),
        'MAE': mean_absolute_error(actuals, predictions),
        'R2': r2_score(actuals, predictions),
        'Direction_Accuracy': np.mean((np.sign(actuals) == np.sign(predictions))) * 100
    }
    
    return metrics