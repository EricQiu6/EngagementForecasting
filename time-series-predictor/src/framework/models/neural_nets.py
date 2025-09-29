import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

# Import baseline models
from .baselines import DLinearWrapper
from .student_ability_model import StudentAbilityLinearModel, StudentAbilityNeuralModel
from .zero_inflated_model import ZeroInflatedPoissonAdapter, ImprovedStudentModel


class SimpleMLP(nn.Module):
    """
    Simple Multi-Layer Perceptron for time series prediction.
    Good baseline for traditional lag-based features.
    """
    
    def __init__(self, input_size: int, hidden_sizes: list = [64, 32], dropout: float = 0.2):
        super().__init__()
        self.input_size = input_size
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        # Handle both 2D and 3D inputs
        if len(x.shape) == 3:
            # (batch_size, seq_len, features) -> flatten last two dims
            batch_size = x.shape[0]
            x = x.view(batch_size, -1)
        
        return self.network(x)


class SimpleLSTM(nn.Module):
    """
    LSTM-based model for sequence prediction.
    Handles variable-length sequences and captures temporal dependencies.
    """
    
    def __init__(self, 
                 input_size: int = 2, 
                 hidden_size: int = 64, 
                 num_layers: int = 2,
                 dropout: float = 0.2,
                 bidirectional: bool = False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        self.fc = nn.Linear(lstm_output_size, 1)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use last time step output
        if self.bidirectional:
            # Concatenate forward and backward hidden states
            last_output = lstm_out[:, -1, :]
        else:
            last_output = lstm_out[:, -1, :]
        
        # Apply dropout and final linear layer
        output = self.dropout(last_output)
        output = self.fc(output)
        
        return output


class TimeSeriesCNN(nn.Module):
    """
    1D CNN for time series prediction.
    Good for capturing local patterns and trends.
    """
    
    def __init__(self,
                 input_size: int = 2,
                 num_filters: list = [32, 64, 128],
                 kernel_sizes: list = [3, 3, 3],
                 dropout: float = 0.2):
        super().__init__()
        self.input_size = input_size
        
        conv_layers = []
        in_channels = input_size
        
        for num_filter, kernel_size in zip(num_filters, kernel_sizes):
            conv_layers.extend([
                nn.Conv1d(in_channels, num_filter, kernel_size, padding=kernel_size//2),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(dropout)
            ])
            in_channels = num_filter
        
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Final classifier
        self.fc = nn.Linear(num_filters[-1], 1)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        # Conv1d expects: (batch_size, input_size, seq_len)
        x = x.transpose(1, 2)
        
        # Apply convolutions
        x = self.conv_layers(x)
        
        # Global pooling
        x = self.global_pool(x)  # (batch_size, num_filters, 1)
        x = x.squeeze(-1)        # (batch_size, num_filters)
        
        # Final prediction
        output = self.fc(x)
        
        return output


class AttentionLSTM(nn.Module):
    """
    LSTM with attention mechanism.
    Allows the model to focus on different parts of the sequence.
    """
    
    def __init__(self,
                 input_size: int = 2,
                 hidden_size: int = 64,
                 num_layers: int = 2,
                 dropout: float = 0.2):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Attention mechanism
        self.attention = nn.Linear(hidden_size, 1)
        
        # Final prediction layer
        self.fc = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_len, hidden_size)
        
        # Attention weights
        attention_weights = F.softmax(self.attention(lstm_out), dim=1)  # (batch_size, seq_len, 1)
        
        # Weighted sum
        context = torch.sum(attention_weights * lstm_out, dim=1)  # (batch_size, hidden_size)
        
        # Final prediction
        output = self.dropout(context)
        output = self.fc(output)
        
        return output


class SimpleTransformer(nn.Module):
    """
    Simple Transformer model for time series prediction.
    Uses positional encoding and self-attention.
    """
    
    def __init__(self,
                 input_size: int = 2,
                 d_model: int = 64,
                 nhead: int = 8,
                 num_layers: int = 3,
                 dropout: float = 0.1,
                 max_seq_len: int = 100):
        super().__init__()
        self.input_size = input_size
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layer
        self.fc = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        
        # Project to d_model dimensions
        x = self.input_projection(x)  # (batch_size, seq_len, d_model)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Transformer encoding
        x = self.transformer(x)  # (batch_size, seq_len, d_model)
        
        # Use last time step for prediction
        x = x[:, -1, :]  # (batch_size, d_model)
        
        # Final prediction
        output = self.dropout(x)
        output = self.fc(output)
        
        return output


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""
    
    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # (max_len, 1, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        x = x + self.pe[:seq_len, :, :].transpose(0, 1)
        return x





# Factory function for easy model creation
def create_model(model_type: str, **kwargs):
    """
    Factory function to create models.
    
    Args:
        model_type: Type of model ('mlp', 'lstm', 'cnn', 'attention_lstm', 'transformer', 'dlinear', 'dlinear_simple',
                   'student_ability_linear', 'student_ability_neural', 'zip', 'improved_student')
        **kwargs: Model-specific arguments
        
    DLinear Options:
        - 'dlinear': Original research implementation with full series decomposition (18 params)
        - 'dlinear_simple': Simplified custom implementation (12 params)
        
    Returns:
        PyTorch model instance
    """
    
    if model_type == 'mlp':
        return SimpleMLP(**kwargs)
    elif model_type == 'lstm':
        return SimpleLSTM(**kwargs)
    elif model_type == 'cnn':
        return TimeSeriesCNN(**kwargs)
    elif model_type == 'attention_lstm':
        return AttentionLSTM(**kwargs)
    elif model_type == 'transformer':
        return SimpleTransformer(**kwargs)
    elif model_type == 'dlinear':
        # Return the wrapper that works with SKLearnAdapter
        return DLinearWrapper(**kwargs)
    elif model_type == 'student_ability_linear':
        return StudentAbilityLinearModel(**kwargs)
    elif model_type == 'student_ability_neural':
        return StudentAbilityNeuralModel(**kwargs)
    elif model_type == 'zip':
        return ZeroInflatedPoissonAdapter(**kwargs)
    elif model_type == 'improved_student':
        return ImprovedStudentModel(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}") 