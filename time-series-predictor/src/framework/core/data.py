"""
Schema-driven time series dataset implementation.
Replaces hardcoded column names with configurable schemas.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional, Union
from sklearn.model_selection import TimeSeriesSplit

from .schema import DataSchema, DataValidator, FeatureExtractor, week_string_to_numeric


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
        # Load data if not already loaded
        if self.data is not None:
            data = self.data
        else:
            data = pd.read_csv(self.data_path)
        
        # Create global timeline sorted by time
        global_timeline = data.sort_values([self.schema.time_column, self.schema.student_column]).reset_index(drop=True)
        
        # Get unique weeks for splitting
        unique_weeks = sorted(global_timeline[self.schema.time_column].unique())
        n_weeks = len(unique_weeks)
        
        # Create TimeSeriesSplit on week level
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
        
        # Generate splits based on week indices
        week_splits = list(tscv.split(unique_weeks))
        
        # Pre-compute mapping from week to sequence indices for efficiency
        week_to_seq_indices = {}
        for seq_idx, seq_info in enumerate(self.sequence_index):
            target_week = seq_info['target_time']
            if target_week not in week_to_seq_indices:
                week_to_seq_indices[target_week] = []
            week_to_seq_indices[target_week].append(seq_idx)
        
        # Convert week splits to sequence indices
        splits = []
        for train_week_idx, test_week_idx in week_splits:
            train_weeks = set(unique_weeks[i] for i in train_week_idx)
            test_weeks = set(unique_weeks[i] for i in test_week_idx)
            
            train_seq_indices = []
            val_seq_indices = []
            
            # Map weeks to sequence indices using pre-computed mapping
            for week, seq_indices in week_to_seq_indices.items():
                if week in train_weeks:
                    train_seq_indices.extend(seq_indices)
                elif week in test_weeks:
                    val_seq_indices.extend(seq_indices)
            
            splits.append((train_seq_indices, val_seq_indices))
        
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