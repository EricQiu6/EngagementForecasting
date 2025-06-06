import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
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
    Adapter for PyTorch models that provides proper deep learning training infrastructure.
    Supports GPU acceleration, training loops, validation, and early stopping.
    """
    
    def __init__(self, 
                 pytorch_model: nn.Module,
                 device: Optional[str] = None,
                 loss_fn: Optional[nn.Module] = None,
                 optimizer_class: type = optim.Adam,
                 optimizer_kwargs: Optional[Dict] = None):
        """
        Args:
            pytorch_model: PyTorch model (nn.Module)
            device: Device to use ('cuda', 'cpu', or None for auto-detection)
            loss_fn: Loss function (default: MSELoss for regression)
            optimizer_class: Optimizer class (default: Adam)
            optimizer_kwargs: Optimizer parameters (default: {'lr': 1e-3})
        """
        self.device = get_device() if device is None else device
        self.model = pytorch_model.to(self.device)
        self.loss_fn = loss_fn or nn.MSELoss()
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs or {'lr': 1e-3}
        
        self.optimizer = None
        self.scheduler = None
        self.is_fitted = False
        self.training_history = []
        
    def fit(self, 
            train_data: Union[DataLoader, Tuple[np.ndarray, np.ndarray]], 
            val_data: Optional[Union[DataLoader, Tuple[np.ndarray, np.ndarray]]] = None,
            epochs: int = 100,
            early_stopping_patience: int = 10,
            min_delta: float = 1e-4,
            scheduler_kwargs: Optional[Dict] = None,
            verbose: bool = True,
            **kwargs) -> Dict[str, Any]:
        """
        Train the PyTorch model with proper training loop.
        
        Args:
            train_data: Training DataLoader or (X, y) arrays
            val_data: Validation DataLoader or (X, y) arrays
            epochs: Number of training epochs
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
            train_data = self._arrays_to_dataloader(train_data, shuffle=True)
        if val_data is not None and not isinstance(val_data, DataLoader):
            val_data = self._arrays_to_dataloader(val_data, shuffle=False)
            
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
    
    def predict(self, data: Union[DataLoader, np.ndarray]) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            data: Input data (DataLoader or numpy array)
            
        Returns:
            Predictions as numpy array
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        self.model.eval()
        predictions = []
        
        if not isinstance(data, DataLoader):
            # Handle prediction-only case (no targets)
            if isinstance(data, np.ndarray):
                # Create dummy targets for DataLoader compatibility
                dummy_y = np.zeros(len(data))
                data = self._arrays_to_dataloader((data, dummy_y), shuffle=False)
            else:
                data = self._arrays_to_dataloader(data, shuffle=False)
        
        with torch.no_grad():
            for batch_X, _ in data:
                batch_X = batch_X.to(self.device)
                batch_pred = self.model(batch_X)
                predictions.extend(batch_pred.cpu().numpy())
        
        return np.array(predictions).flatten()
    
    def get_params(self) -> Dict[str, Any]:
        """Get model hyperparameters."""
        return {
            'model_type': type(self.model).__name__,
            'device': self.device,
            'loss_fn': type(self.loss_fn).__name__,
            'optimizer_class': self.optimizer_class.__name__,
            'optimizer_kwargs': self.optimizer_kwargs,
            'model_parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }
    
    def save(self, path: str) -> None:
        """Save model and training state."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'model_class': type(self.model).__name__,
            'optimizer_kwargs': self.optimizer_kwargs,
            'is_fitted': self.is_fitted,
            'training_history': self.training_history,
            'device': self.device
        }
        
        if self.optimizer:
            save_dict['optimizer_state_dict'] = self.optimizer.state_dict()
        
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
    
    def _train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(batch_X)
            loss = self.loss_fn(predictions, batch_y)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def _validate_epoch(self, val_loader: DataLoader) -> float:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                predictions = self.model(batch_X)
                loss = self.loss_fn(predictions, batch_y)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def _arrays_to_dataloader(self, data: Tuple[np.ndarray, np.ndarray], 
                             batch_size: int = 32, shuffle: bool = False) -> DataLoader:
        """Convert numpy arrays to DataLoader."""
        X, y = data
        
        class ArrayDataset:
            def __init__(self, X, y):
                self.X = torch.tensor(X, dtype=torch.float32)
                self.y = torch.tensor(y, dtype=torch.float32)
                if len(self.y.shape) == 1:
                    self.y = self.y.unsqueeze(1)  # Add dimension for loss calculation
            
            def __len__(self):
                return len(self.X)
            
            def __getitem__(self, idx):
                return self.X[idx], self.y[idx]
        
        dataset = ArrayDataset(X, y)
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
            plt.plot(epochs, train_losses, label='Training Loss')
            if val_losses:
                val_epochs = [h['epoch'] for h in self.training_history if h['val_loss'] is not None]
                plt.plot(val_epochs, val_losses, label='Validation Loss')
            
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training History')
            plt.legend()
            plt.grid(True)
            
            if save_path:
                plt.savefig(save_path)
            else:
                plt.show()
                
        except ImportError:
            print("matplotlib not available for plotting") 