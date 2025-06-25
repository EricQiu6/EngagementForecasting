"""
Schema-driven time series dataset implementation.
Replaces hardcoded column names with configurable schemas.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional, Union
import os
import sys
from pathlib import Path

from .schema import DataSchema, DataValidator, FeatureExtractor, week_string_to_numeric

# Add data processing module to path
# Go up from src/framework/core to project root, then to data
project_root = os.path.join(os.path.dirname(__file__), '..', '..', '..')
data_path = os.path.join(project_root, 'data')
sys.path.append(data_path)

try:
    from data_processing import create_time_series_splits, get_fold_data
except ImportError:
    print("Warning: Could not import legacy data processing. Some features may not work.")


class SchemaBasedTimeSeriesDataset(Dataset):
    """
    Schema-driven dataset for student time series data.
    Eliminates hardcoded column names and provides configurable data handling.
    """
    
    def __init__(self, 
                 data_path: str,
                 schema: DataSchema,
                 sequence_length: int = 5,
                 load_in_memory: bool = True,
                 validate_data: bool = True):
        """
        Args:
            data_path: Path to the CSV data file
            schema: DataSchema defining columns and validation rules
            sequence_length: Number of historical steps to include
            load_in_memory: Whether to load all data in memory (False for large datasets)
            validate_data: Whether to validate data against schema
        """
        self.data_path = data_path
        self.schema = schema
        self.sequence_length = sequence_length
        self.load_in_memory = load_in_memory
        
        # Initialize components
        self.validator = DataValidator(schema)
        self.feature_extractor = FeatureExtractor(schema)
        
        # Load data first
        if self.load_in_memory:
            self.data = pd.read_csv(data_path)
            
            # Validate if requested
            if validate_data:
                is_valid, issues = self.validator.validate_dataset(self.data)
                if not is_valid:
                    print(f"Data validation issues found:")
                    for issue in issues:
                        print(f"  - {issue}")
                    raise ValueError(f"Data validation failed with {len(issues)} issues")
        else:
            self.data = None
            
        # Build index of valid sequences
        self.sequence_index = self._build_sequence_index()
            
    def _build_sequence_index(self) -> List[Dict]:
        """
        Build an index of all valid sequences using schema-defined columns.
        """
        # Load data to build index (even for streaming mode)
        if self.data is not None:
            data = self.data
        else:
            data = pd.read_csv(self.data_path)
        
        sequences = []
        
        # Use schema-defined columns
        student_col = self.schema.student_column
        time_col = self.schema.time_column
        
        for student in data[student_col].unique():
            student_data = data[data[student_col] == student].sort_values(time_col)
            
            # Create sequences for this student
            for i in range(self.sequence_length, len(student_data)):
                sequences.append({
                    'student': student,
                    'target_time': student_data.iloc[i][time_col],
                    'sequence_start_idx': i - self.sequence_length,
                    'sequence_end_idx': i,
                    'data_start_idx': student_data.index[i - self.sequence_length],
                    'data_end_idx': student_data.index[i]
                })
        
        return sequences
    
    def __len__(self) -> int:
        return len(self.sequence_index)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sequence using schema-based feature extraction.
        
        Returns:
            features: Shape (sequence_length, n_features)
            target: Shape (1,)
        """
        seq_info = self.sequence_index[idx]
        
        if self.load_in_memory:
            data = self.data
        else:
            # For streaming: load only the student's data
            data = pd.read_csv(self.data_path)
            data = data[data[self.schema.student_column] == seq_info['student']]
        
        # Get student data
        student_data = data[data[self.schema.student_column] == seq_info['student']].sort_values(
            self.schema.time_column
        )
        
        # Extract sequence
        start_idx = seq_info['sequence_start_idx']
        end_idx = seq_info['sequence_end_idx']
        
        # Features: use schema-based extraction
        sequence_data = student_data.iloc[start_idx:end_idx]
        
        features = []
        for _, row in sequence_data.iterrows():
            # Add time feature if it's in the feature columns
            if self.schema.time_column in self.schema.feature_columns:
                time_value = week_string_to_numeric(
                    row[self.schema.time_column], 
                    self.schema.time_format
                )
                # Find time column index in features
                time_idx = self.schema.feature_columns.index(self.schema.time_column)
                feature_vec = self.feature_extractor.extract_features(row)
                feature_vec[time_idx] = time_value  # Replace with numeric version
            else:
                # Time not in features, extract normally
                feature_vec = self.feature_extractor.extract_features(row)
            
            features.append(feature_vec)
        
        # Target: next value using schema-defined target column
        target_value = row[self.schema.target_column] if self.schema.target_column in row else 0.0
        if hasattr(student_data.iloc[end_idx], self.schema.target_column):
            target_value = student_data.iloc[end_idx][self.schema.target_column]
        
        # Safe conversion
        target = self.feature_extractor._safe_float_conversion(target_value)
        
        return torch.tensor(features, dtype=torch.float32), torch.tensor([target], dtype=torch.float32)
    
    def get_splits(self, n_splits: int = 5, test_size: int = 1) -> List[Tuple[List[int], List[int]]]:
        """
        Generate time series cross-validation splits.
        """
        try:
            # Use legacy splitting if available
            data_path = os.path.expanduser(self.data_path)
            
            # Create a temporary file with legacy column names for compatibility
            import tempfile
            
            # Load data and rename columns to match legacy expectations
            data = pd.read_csv(data_path)
            legacy_data = data.copy()
            
            # Map schema columns to legacy column names
            column_mapping = {
                self.schema.student_column: 'name',
                self.schema.time_column: 'week',
                self.schema.target_column: 'proficient'
            }
            
            # Rename columns
            legacy_data = legacy_data.rename(columns=column_mapping)
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
                legacy_data.to_csv(tmp_file.name, index=False)
                temp_path = tmp_file.name
            
            try:
                # Use legacy splitting with temporary file
                global_timeline, legacy_splits = create_time_series_splits(temp_path, n_splits, test_size)
                
                # Map legacy column names to schema columns
                timeline_time_col = 'week'  # Legacy uses 'week'
                
                # Create a mapping from time values to sequence indices
                time_to_seq_indices = {}
                for seq_idx, seq_info in enumerate(self.sequence_index):
                    target_time = seq_info['target_time']
                    if target_time not in time_to_seq_indices:
                        time_to_seq_indices[target_time] = []
                    time_to_seq_indices[target_time].append(seq_idx)
                
                # Convert legacy splits to sequence indices
                splits = []
                for train_week_indices, val_week_indices in legacy_splits:
                    # Get actual week values (not indices)
                    train_weeks = set(global_timeline.iloc[i][timeline_time_col] for i in train_week_indices)
                    val_weeks = set(global_timeline.iloc[i][timeline_time_col] for i in val_week_indices)
                    
                    # Map weeks to sequence indices
                    train_seq_indices = []
                    val_seq_indices = []
                    
                    for time_val, seq_indices in time_to_seq_indices.items():
                        if time_val in train_weeks:
                            train_seq_indices.extend(seq_indices)
                        elif time_val in val_weeks:
                            val_seq_indices.extend(seq_indices)
                    
                    splits.append((train_seq_indices, val_seq_indices))
                
                return splits
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            
        except Exception as e:
            print(f"Warning: Could not use legacy splitting ({e}). Using simple split.")
            return self._simple_time_split(n_splits, test_size)
    
    def _simple_time_split(self, n_splits: int, test_size: int) -> List[Tuple[List[int], List[int]]]:
        """Fallback simple time-based splitting using schema columns."""
        # Group sequences by time
        time_to_indices = {}
        for idx, seq_info in enumerate(self.sequence_index):
            time_val = seq_info['target_time']
            if time_val not in time_to_indices:
                time_to_indices[time_val] = []
            time_to_indices[time_val].append(idx)
        
        sorted_times = sorted(time_to_indices.keys())
        
        splits = []
        total_times = len(sorted_times)
        
        for fold in range(n_splits):
            # Calculate split points
            split_size = total_times // n_splits
            val_start = fold * split_size
            val_end = min(val_start + test_size, total_times)
            
            val_times = sorted_times[val_start:val_end]
            train_times = [t for t in sorted_times if t not in val_times]
            
            # Get indices
            train_indices = []
            val_indices = []
            
            for time_val in train_times:
                train_indices.extend(time_to_indices[time_val])
            for time_val in val_times:
                val_indices.extend(time_to_indices[time_val])
            
            splits.append((train_indices, val_indices))
        
        return splits


class DataLoaderFactory:
    """Factory for creating DataLoaders with schema-based configuration."""
    
    @staticmethod
    def create_dataloader(dataset: SchemaBasedTimeSeriesDataset,
                         indices: Optional[List[int]] = None,
                         batch_size: int = 32,
                         shuffle: bool = False,
                         num_workers: int = 0) -> DataLoader:
        """Create a DataLoader for the dataset."""
        if indices is not None:
            # Create subset
            subset = torch.utils.data.Subset(dataset, indices)
            return DataLoader(subset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        else:
            return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def convert_legacy_format(X: np.ndarray, y: np.ndarray, 
                         schema: DataSchema, 
                         batch_size: int = 32) -> DataLoader:
    """
    Convert legacy numpy arrays to DataLoader format with schema validation.
    """
    
    class ArrayDataset(Dataset):
        def __init__(self, X, y, schema):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.float32)
            self.schema = schema
            
            # Validate dimensions match schema
            expected_features = len(schema.feature_columns)
            if X.shape[-1] != expected_features:
                print(f"Warning: Feature dimension {X.shape[-1]} doesn't match "
                      f"schema features {expected_features}")
        
        def __len__(self):
            return len(self.X)
        
        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]
    
    dataset = ArrayDataset(X, y, schema)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False) 