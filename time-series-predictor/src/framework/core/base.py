from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader

class TimeSeriesModel(ABC):
    """
    Abstract base class for all time series models.
    Supports both sklearn-style and deep learning workflows.
    """
    
    @abstractmethod
    def fit(self, 
            train_data: Union[DataLoader, Tuple[np.ndarray, np.ndarray]], 
            val_data: Optional[Union[DataLoader, Tuple[np.ndarray, np.ndarray]]] = None,
            **kwargs) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            train_data: Training data (DataLoader for DL, arrays for sklearn)
            val_data: Validation data (optional)
            **kwargs: Model-specific training parameters
            
        Returns:
            Training history/metrics
        """
        pass
    
    @abstractmethod
    def predict(self, data: Union[DataLoader, np.ndarray]) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            data: Input data
            
        Returns:
            Predictions as numpy array
        """
        pass
    
    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """Get model hyperparameters for reproducibility."""
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Save model to disk."""
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """Load model from disk.""" 
        pass


class TimeSeriesDataset(ABC):
    """
    Abstract base class for time series datasets.
    Supports both in-memory and streaming data loading.
    """
    
    @abstractmethod
    def __len__(self) -> int:
        pass
    
    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        pass
    
    @abstractmethod
    def get_splits(self, n_splits: int = 5, test_size: int = 1) -> list:
        """
        Generate time series cross-validation splits.
        
        Returns:
            List of (train_indices, val_indices) tuples
        """
        pass


class MetricsCalculator:
    """
    Unified metrics calculation for all model types.
    """
    
    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate standard time series metrics."""
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        smape = 100 * np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))
        
        # Additional metrics
        mape = 100 * np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8)))
        r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
        
        # Convert numpy types to Python floats for JSON serialization
        return {
            'mae': float(mae),
            'rmse': float(rmse),
            'smape': float(smape),
            'mape': float(mape),
            'r2': float(r2),
            'n_samples': int(len(y_true))
        }


class CrossValidator:
    """
    Time series cross-validation engine that works with any model type.
    """
    
    def __init__(self, model: TimeSeriesModel, dataset: TimeSeriesDataset):
        self.model = model
        self.dataset = dataset
        
    def cross_validate(self, 
                      n_splits: int = 5, 
                      test_size: int = 1,
                      **fit_kwargs) -> Dict[str, Any]:
        """
        Perform time series cross-validation.
        
        Args:
            n_splits: Number of CV folds
            test_size: Size of test set per fold
            **fit_kwargs: Arguments passed to model.fit()
            
        Returns:
            Aggregated results across folds
        """
        fold_results = []
        splits = self.dataset.get_splits(n_splits, test_size)
        
        for fold_idx, (train_indices, val_indices) in enumerate(splits):
            print(f"Fold {fold_idx + 1}/{n_splits}")
            
            # Get fold data
            train_data = self._get_fold_data(train_indices)
            val_data = self._get_fold_data(val_indices)
            
            # Train model
            history = self.model.fit(train_data, val_data, **fit_kwargs)
            
            # Evaluate
            y_pred = self.model.predict(val_data)
            y_true = self._extract_targets(val_data)
            
            metrics = MetricsCalculator.calculate_metrics(y_true, y_pred)
            metrics['fold'] = fold_idx
            fold_results.append(metrics)
            
        return self._aggregate_results(fold_results)
    
    def _get_fold_data(self, indices):
        """Extract data for specific fold indices."""
        from ..core.data import DataLoaderFactory
        from ..adapters.sklearn_adapter import SKLearnAdapter
        
        # For sklearn models, use direct numpy arrays to avoid DataLoader overhead
        if isinstance(self.model, SKLearnAdapter):
            return self._get_fold_data_numpy(indices)
        else:
            # Use DataLoader for PyTorch models
            batch_size = 32
            dataloader = DataLoaderFactory.create_dataloader(
                self.dataset, 
                indices=indices, 
                batch_size=batch_size, 
                shuffle=False
            )
            return dataloader
    
    def _get_fold_data_numpy(self, indices):
        """Extract data as numpy arrays directly for sklearn models (OPTIMIZED PATH)."""
        # Use the dataset's preprocessing by getting individual samples
        X_list = []
        y_list = []
        
        for idx in indices:
            features, target = self.dataset[idx]
            
            # Convert to numpy and flatten features for sklearn
            features_np = features.numpy()  # Shape: (seq_len, n_features)
            target_np = target.numpy().item()  # Single value
            
            # For sklearn, we need to flatten the sequence into a feature vector
            # Extract features similar to sklearn adapter approach
            seq_len, n_features = features_np.shape
            lag_window = getattr(self.model, 'lag_window', 5)
            
            # Extract week (first column of last timestep)
            week = features_np[-1, 0]
            
            # Extract lag features (target column from all timesteps)
            if n_features >= 5:  # We have 5 features: week, minutes, problems, opportunities, prev_proficiency
                lags_all = features_np[:, 4]  # Previous proficiency values (last column)
            else:
                lags_all = features_np[:, -1]  # Last column as fallback
            
            # Handle lag window sizing
            if seq_len > lag_window:
                # Take last lag_window values
                lags_processed = lags_all[-lag_window:]
            elif seq_len < lag_window:
                # Pad with zeros at the beginning
                lags_processed = np.pad(lags_all, (lag_window - seq_len, 0), mode='constant', constant_values=0)
            else:
                lags_processed = lags_all
            
            # Combine features: [week, lag1, lag2, ..., lagN]
            row_features = np.concatenate([[week], lags_processed])
            
            X_list.append(row_features)
            y_list.append(target_np)
        
        return np.array(X_list), np.array(y_list)
    
    def _extract_targets(self, data):
        """Extract target values from data (DataLoader or numpy arrays)."""
        if isinstance(data, tuple) and len(data) == 2:
            # Direct numpy arrays (X, y)
            return data[1]
        else:
            # DataLoader
            targets = []
            for _, batch_y in data:
                targets.extend(batch_y.numpy().flatten())
            return np.array(targets)
    
    def _aggregate_results(self, fold_results):
        """Aggregate metrics across folds."""
        metric_names = ['mae', 'rmse', 'smape', 'mape', 'r2']
        aggregated = {}
        
        for metric in metric_names:
            values = [result[metric] for result in fold_results]
            aggregated[f'{metric}_mean'] = float(np.mean(values))
            aggregated[f'{metric}_std'] = float(np.std(values))
            
        aggregated['n_folds'] = int(len(fold_results))
        aggregated['total_samples'] = int(sum(result['n_samples'] for result in fold_results))
        aggregated['fold_results'] = fold_results  # fold_results now contains Python floats
        
        return aggregated 