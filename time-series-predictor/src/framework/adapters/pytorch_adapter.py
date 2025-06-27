"""
Unified PyTorch adapter with comprehensive training infrastructure and optional schema support.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, Any, Union, Tuple, Optional, List
from pathlib import Path
import json
import copy
from tqdm import tqdm

from ..core.base import TimeSeriesModel
from ..utils.device import get_device


class PyTorchAdapter(TimeSeriesModel):
    """
    Unified adapter for PyTorch models with comprehensive deep learning infrastructure.
    Supports GPU acceleration, training loops, validation, early stopping, and optional schema-based configuration.
    """
    
    def __init__(self, 
                 pytorch_model: nn.Module,
                 schema: Optional['DataSchema'] = None,  # Optional schema support
                 device: Optional[str] = None,
                 loss_fn: Optional[nn.Module] = None,
                 optimizer_class: type = optim.Adam,
                 optimizer_kwargs: Optional[Dict] = None):
        """
        Args:
            pytorch_model: PyTorch model (nn.Module)
            schema: Optional DataSchema for schema-based configuration
            device: Device to use ('cuda', 'cpu', or None for auto-detection)
            loss_fn: Loss function (default: MSELoss for regression)
            optimizer_class: Optimizer class (default: Adam)
            optimizer_kwargs: Optimizer parameters (default: {'lr': 1e-3})
        """
        self.device = get_device() if device is None else device
        self.model = pytorch_model.to(self.device)
        self.schema = schema
        self.loss_fn = loss_fn or nn.MSELoss()
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs or {'lr': 1e-3}
        
        self.optimizer = None
        self.scheduler = None
        self.is_fitted = False
        self.training_history = []
        
        # Build feature mapping if schema is provided
        if self.schema:
            self._build_feature_mapping()
            
    def _build_feature_mapping(self):
        """Build mapping of feature names to indices based on schema."""
        self.feature_map = {}
        
        if not self.schema:
            return
            
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
            epochs: int = 100,
            batch_size: int = 32,
            early_stopping_patience: int = 10,
            min_delta: float = 1e-4,
            scheduler_kwargs: Optional[Dict] = None,
            verbose: bool = True,
            **kwargs) -> Dict[str, Any]:
        """
        Train the PyTorch model with comprehensive training infrastructure.
        
        Args:
            train_data: Training DataLoader or (X, y) arrays
            val_data: Validation DataLoader or (X, y) arrays
            epochs: Number of training epochs
            batch_size: Batch size for training (used if converting arrays to DataLoader)
            early_stopping_patience: Patience for early stopping
            min_delta: Minimum change to qualify as improvement
            scheduler_kwargs: Learning rate scheduler parameters
            verbose: Whether to show progress bar
            **kwargs: Additional training parameters
            
        Returns:
            Training history dictionary
        """
        
        # Convert numpy arrays to DataLoaders if needed
        if not isinstance(train_data, DataLoader):
            train_data = self._arrays_to_dataloader(train_data, batch_size=batch_size, shuffle=True)
        if val_data is not None and not isinstance(val_data, DataLoader):
            val_data = self._arrays_to_dataloader(val_data, batch_size=batch_size, shuffle=False)
            
        # Initialize optimizer and scheduler
        self.optimizer = self.optimizer_class(self.model.parameters(), **self.optimizer_kwargs)
        
        if scheduler_kwargs:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, **scheduler_kwargs
            )
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        self.training_history = []
        
        epoch_iterator = range(epochs)
        if verbose:
            epoch_iterator = tqdm(epoch_iterator, desc="Training")
        
        for epoch in epoch_iterator:
            # Training phase
            train_loss = self._train_epoch(train_data)
            
            # Validation phase
            val_loss = None
            if val_data is not None:
                val_loss = self._validate_epoch(val_data)
                
                # Early stopping check
                if val_loss < best_val_loss - min_delta:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = copy.deepcopy(self.model.state_dict())
                else:
                    patience_counter += 1
                    
                # Learning rate scheduling
                if self.scheduler:
                    self.scheduler.step(val_loss)
            
            # Record history
            epoch_history = {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'lr': self.optimizer.param_groups[0]['lr']
            }
            self.training_history.append(epoch_history)
            
            # Update progress bar
            if verbose:
                desc = f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}"
                if val_loss is not None:
                    desc += f" - Val Loss: {val_loss:.4f}"
                epoch_iterator.set_description(desc)
            
            # Early stopping
            if val_data is not None and patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                break
        
        # Restore best model if we used validation
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        self.is_fitted = True
        
        return {
            'history': self.training_history,
            'best_val_loss': best_val_loss,
            'epochs_trained': len(self.training_history),
            'status': 'completed'
        }
    
    def predict(self, data: Union[DataLoader, np.ndarray, Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            data: Input data (DataLoader, numpy array, or (X, y) tuple)
            
        Returns:
            Predictions as numpy array
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        self.model.eval()
        predictions = []
        
        # Handle different input types
        if isinstance(data, tuple) and len(data) == 2:
            # (X, y) tuple - extract X only
            data = data[0]
        
        if not isinstance(data, DataLoader):
            # Convert to DataLoader for consistent handling
            if isinstance(data, np.ndarray):
                # Create dummy targets for DataLoader compatibility
                dummy_y = np.zeros(len(data))
                data = self._arrays_to_dataloader((data, dummy_y), batch_size=32, shuffle=False)
        
        with torch.no_grad():
            for batch in data:
                # Handle both (X, y) and X-only batches
                if isinstance(batch, (list, tuple)):
                    batch_X = batch[0]
                else:
                    batch_X = batch
                    
                batch_X = batch_X.to(self.device)
                batch_pred = self.model(batch_X)
                
                # Handle different output shapes
                if len(batch_pred.shape) > 1 and batch_pred.shape[1] == 1:
                    batch_pred = batch_pred.squeeze(-1)
                    
                predictions.extend(batch_pred.cpu().numpy())
        
        return np.array(predictions)
    
    def get_params(self) -> Dict[str, Any]:
        """Get model hyperparameters."""
        params = {
            'model_type': type(self.model).__name__,
            'device': str(self.device),
            'loss_fn': type(self.loss_fn).__name__,
            'optimizer_class': self.optimizer_class.__name__,
            'optimizer_kwargs': self.optimizer_kwargs,
            'model_parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }
        
        # Add schema info if available
        if self.schema:
            params['schema'] = self.schema.__class__.__name__
            params['n_features'] = len(self.schema.feature_columns)
            
        return params
    
    def save(self, path: str) -> None:
        """Save model and training state."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'model_class': type(self.model).__name__,
            'optimizer_kwargs': self.optimizer_kwargs,
            'is_fitted': self.is_fitted,
            'training_history': self.training_history,
            'device': str(self.device)
        }
        
        if self.optimizer:
            save_dict['optimizer_state_dict'] = self.optimizer.state_dict()
            
        if self.schema:
            save_dict['schema'] = self.schema.to_config() if hasattr(self.schema, 'to_config') else None
        
        torch.save(save_dict, path)
        
        # Also save hyperparameters as JSON for easy inspection
        json_path = Path(path).with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump(self.get_params(), f, indent=2, default=str)
    
    def load(self, path: str) -> None:
        """Load model and training state."""
        save_dict = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(save_dict['model_state_dict'])
        self.optimizer_kwargs = save_dict['optimizer_kwargs']
        self.is_fitted = save_dict['is_fitted']
        self.training_history = save_dict.get('training_history', [])
        
        # Reinitialize optimizer if needed
        if 'optimizer_state_dict' in save_dict:
            self.optimizer = self.optimizer_class(self.model.parameters(), **self.optimizer_kwargs)
            self.optimizer.load_state_dict(save_dict['optimizer_state_dict'])
            
        # Load schema if available
        if 'schema' in save_dict and save_dict['schema']:
            try:
                from ..core.schema import DataSchema
                self.schema = DataSchema.from_config(save_dict['schema'])
                self._build_feature_mapping()
            except:
                pass  # Schema loading failed, continue without it
    
    def _train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            # Handle both (X, y) and other batch formats
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                batch_X, batch_y = batch[0], batch[1]
            else:
                raise ValueError("Unexpected batch format in train_loader")
                
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(batch_X)
            
            # Handle shape mismatches
            if len(predictions.shape) > len(batch_y.shape):
                predictions = predictions.squeeze(-1)
            if len(batch_y.shape) == 1 and len(predictions.shape) > 1:
                batch_y = batch_y.unsqueeze(-1)
                
            loss = self.loss_fn(predictions, batch_y)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def _validate_epoch(self, val_loader: DataLoader) -> float:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Handle both (X, y) and other batch formats
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    batch_X, batch_y = batch[0], batch[1]
                else:
                    raise ValueError("Unexpected batch format in val_loader")
                    
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                predictions = self.model(batch_X)
                
                # Handle shape mismatches
                if len(predictions.shape) > len(batch_y.shape):
                    predictions = predictions.squeeze(-1)
                if len(batch_y.shape) == 1 and len(predictions.shape) > 1:
                    batch_y = batch_y.unsqueeze(-1)
                    
                loss = self.loss_fn(predictions, batch_y)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def _arrays_to_dataloader(self, data: Tuple[np.ndarray, np.ndarray], 
                             batch_size: int = 32, shuffle: bool = False) -> DataLoader:
        """Convert numpy arrays to DataLoader."""
        X, y = data
        
        # Convert to tensors
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        
        # Create TensorDataset
        dataset = TensorDataset(X_tensor, y_tensor)
        
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    def get_training_history(self) -> List[Dict]:
        """Get training history for analysis and plotting."""
        return self.training_history
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training history (requires matplotlib)."""
        try:
            import matplotlib.pyplot as plt
            
            if not self.training_history:
                print("No training history available")
                return
            
            epochs = [h['epoch'] for h in self.training_history]
            train_losses = [h['train_loss'] for h in self.training_history]
            val_losses = [h['val_loss'] for h in self.training_history if h['val_loss'] is not None]
            
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, train_losses, label='Training Loss', marker='.')
            if val_losses:
                val_epochs = [h['epoch'] for h in self.training_history if h['val_loss'] is not None]
                plt.plot(val_epochs, val_losses, label='Validation Loss', marker='.')
            
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training History')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"Plot saved to {save_path}")
            else:
                plt.show()
                
        except ImportError:
            print("matplotlib not available for plotting") 


# Alias for backward compatibility
SchemaBasedPyTorchAdapter = PyTorchAdapter 