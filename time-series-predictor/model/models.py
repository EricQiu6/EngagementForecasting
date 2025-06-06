import numpy as np

class AveragePredictor:
    """
    Simple baseline predictor that predicts the historical average
    """
    def __init__(self):
        self.mean_value = None
    
    def fit(self, X, y):
        """Fit by computing the mean of training targets"""
        self.mean_value = np.mean(y)
        return self
    
    def predict(self, X):
        """Predict the training mean for all samples"""
        if self.mean_value is None:
            raise ValueError("Must fit before predicting")
        return np.full(len(X), self.mean_value)
    
    def get_params(self, deep=True):
        """For sklearn compatibility"""
        return {}
    
