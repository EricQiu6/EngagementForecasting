"""
Schema-driven PyTorch adapter that eliminates hardcoded feature indices.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Any, Union, Tuple, Optional
import pickle
from pathlib import Path

from ..core.base import TimeSeriesModel
from ..core.schema import DataSchema, FeatureExtractor


class SchemaBasedPyTorchAdapter(TimeSeriesModel):
    """
    Adapter to make PyTorch models work with the framework using schema-based configuration.
    Eliminates hardcoded indices and column assumptions.
    """
    
    def __init__(self, pytorch_model: nn.Module, schema: DataSchema, 
                 learning_rate: float = 0.001, epochs: int = 100, batch_size: int = 32):
        """
        Args:
            pytorch_model: PyTorch model (nn.Module)
            schema: DataSchema defining the data structure
            learning_rate: Learning rate for training
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        self.pytorch_model = pytorch_model
        self.schema = schema
        self.feature_extractor = FeatureExtractor(schema)
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.is_fitted = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move model to device
        self.pytorch_model.to(self.device)
        
        # Build feature mapping from schema
        self._build_feature_mapping()
        
    def _build_feature_mapping(self):
        """Build mapping of feature names to indices based on schema."""
        self.feature_map = {}
        
        # Map all feature names to indices
        for i, feature in enumerate(self.schema.feature_columns):
            self.feature_map[feature] = i
            
        # Special mappings for common features
        if self.schema.time_column in self.schema.feature_columns:
            self.feature_map['time'] = self.schema.feature_columns.index(self.schema.time_column)
        if self.schema.target_column in self.schema.feature_columns:
            self.feature_map['target'] = self.schema.feature_columns.index(self.schema.target_column)
        
    def fit(self, 
            train_data: Union[DataLoader, Tuple[np.ndarray, np.ndarray]], 
            val_data: Optional[Union[DataLoader, Tuple[np.ndarray, np.ndarray]]] = None,
            **kwargs) -> Dict[str, Any]:
        """
        Train the PyTorch model using schema-based data handling.
        """
        
        if isinstance(train_data, DataLoader):
            train_loader = train_data
        else:
            # Convert numpy arrays to DataLoader
            X_train, y_train = train_data
            X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        # Setup optimizer and loss
        optimizer = torch.optim.Adam(self.pytorch_model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        
        # Training loop
        self.pytorch_model.train()
        train_losses = []
        
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = self.pytorch_model(batch_X)
                
                # Handle different output shapes
                if len(outputs.shape) > len(batch_y.shape):
                    outputs = outputs.squeeze(-1)
                if len(batch_y.shape) == 1:
                    batch_y = batch_y.unsqueeze(-1)
                    
                loss = criterion(outputs, batch_y)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
            train_losses.append(avg_loss)
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}/{self.epochs}, Loss: {avg_loss:.4f}")
        
        self.is_fitted = True
        
        return {
            'train_losses': train_losses,
            'final_loss': train_losses[-1] if train_losses else 0,
            'epochs_trained': self.epochs,
            'status': 'completed'
        }
    
    def predict(self, data: Union[DataLoader, np.ndarray, Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        """
        Make predictions using the PyTorch model.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        self.pytorch_model.eval()
        predictions = []
        
        with torch.no_grad():
            if isinstance(data, DataLoader):
                for batch_X, _ in data:
                    batch_X = batch_X.to(self.device)
                    outputs = self.pytorch_model(batch_X)
                    predictions.append(outputs.cpu().numpy())
            elif isinstance(data, tuple) and len(data) == 2:
                # Direct (X, y) tuple
                X, _ = data
                X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
                outputs = self.pytorch_model(X_tensor)
                predictions.append(outputs.cpu().numpy())
            else:
                # Direct numpy array
                X_tensor = torch.tensor(data, dtype=torch.float32).to(self.device)
                outputs = self.pytorch_model(X_tensor)
                predictions.append(outputs.cpu().numpy())
        
        # Concatenate all predictions
        all_predictions = np.concatenate(predictions, axis=0)
        
        # Ensure proper shape
        if len(all_predictions.shape) > 1 and all_predictions.shape[1] == 1:
            all_predictions = all_predictions.flatten()
            
        return all_predictions
    
    def get_params(self) -> Dict[str, Any]:
        """Get model hyperparameters."""
        params = {
            'model_type': type(self.pytorch_model).__name__,
            'learning_rate': self.learning_rate,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'schema': self.schema.to_config()
        }
        
        # Add model-specific parameters if available
        if hasattr(self.pytorch_model, 'get_coefficients'):
            params['coefficients'] = self.pytorch_model.get_coefficients()
            
        return params
    
    def save(self, path: str) -> None:
        """Save model to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        save_dict = {
            'model_state_dict': self.pytorch_model.state_dict(),
            'model_class': type(self.pytorch_model).__name__,
            'schema': self.schema.to_config(),
            'learning_rate': self.learning_rate,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'is_fitted': self.is_fitted
        }
        
        torch.save(save_dict, path)
    
    def load(self, path: str) -> None:
        """Load model from disk."""
        save_dict = torch.load(path, map_location=self.device)
        
        self.pytorch_model.load_state_dict(save_dict['model_state_dict'])
        self.schema = DataSchema.from_config(save_dict['schema'])
        self.learning_rate = save_dict['learning_rate']
        self.epochs = save_dict['epochs']
        self.batch_size = save_dict['batch_size']
        self.is_fitted = save_dict['is_fitted']
        
        # Rebuild mappings
        self.feature_extractor = FeatureExtractor(self.schema)
        self._build_feature_mapping()


def create_student_ability_model(schema: DataSchema, model_type: str = 'linear', **kwargs):
    """
    Create a student ability model compatible with the schema.
    
    Args:
        schema: Data schema defining the structure
        model_type: 'linear' or 'neural'
        **kwargs: Additional model parameters
        
    Returns:
        SchemaBasedPyTorchAdapter with the student ability model
    """
    from ..models.student_ability_model import StudentAbilityLinearModel, StudentAbilityNeuralModel
    
    # Validate that schema has required features for student ability model
    required_features = ['student_ability', 'student_learning_rate', 'week_difficulty']
    missing_features = [f for f in required_features if f not in schema.feature_columns]
    
    if missing_features:
        print(f"Warning: Missing features for student ability model: {missing_features}")
        print("The model may not work as expected.")
    
    # Create the appropriate model
    history_window = kwargs.get('history_window', 5)
    
    if model_type == 'linear':
        pytorch_model = StudentAbilityLinearModel(
            history_window=history_window,
            schema=schema  # Pass schema for feature mapping
        )
    elif model_type == 'neural':
        hidden_size = kwargs.get('hidden_size', 32)
        pytorch_model = StudentAbilityNeuralModel(
            history_window=history_window,
            hidden_size=hidden_size,
            schema=schema  # Pass schema for feature mapping
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    # Create adapter
    adapter = SchemaBasedPyTorchAdapter(
        pytorch_model=pytorch_model,
        schema=schema,
        learning_rate=kwargs.get('learning_rate', 0.001),
        epochs=kwargs.get('epochs', 100),
        batch_size=kwargs.get('batch_size', 32)
    )
    
    return adapter 