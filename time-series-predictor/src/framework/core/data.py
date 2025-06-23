import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional, Union
import os
import sys
from pathlib import Path

# Add data processing module to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))

try:
    from data_processing import create_time_series_splits, get_fold_data
except ImportError:
    print("Warning: Could not import legacy data processing. Some features may not work.")


def week_string_to_numeric(week_str):
    """Convert week string like '2011-W36' to a numeric representation."""
    if pd.isna(week_str) or week_str == 'NA':
        return 0
    try:
        if isinstance(week_str, str) and '-W' in week_str:
            year, week = week_str.split('-W')
            return int(year) * 100 + int(week)
        else:
            return float(week_str) if week_str != 'NA' else 0
    except:
        return 0


def safe_float_conversion(value):
    """Safely convert value to float, handling NA strings."""
    if pd.isna(value) or value == 'NA' or value == '':
        return 0.0
    try:
        return float(value)
    except:
        return 0.0


class StudentTimeSeriesDataset(Dataset):
    """
    Scalable dataset for student time series data.
    Supports streaming and efficient memory usage.
    """
    
    def __init__(self, 
                 data_path: str,
                 sequence_length: int = 5,
                 target_column: str = 'avg_proficiency',
                 student_column: str = 'anon_student_id',
                 time_column: str = 'week_id',
                 load_in_memory: bool = True):
        """
        Args:
            data_path: Path to the CSV data file
            sequence_length: Number of historical steps to include
            target_column: Column name for target variable (default: avg_proficiency)
            student_column: Column name for student identifier (default: anon_student_id)
            time_column: Column name for time identifier (default: week_id)
            load_in_memory: Whether to load all data in memory (False for large datasets)
        """
        self.data_path = data_path
        self.sequence_length = sequence_length
        self.target_column = target_column
        self.student_column = student_column
        self.time_column = time_column
        self.load_in_memory = load_in_memory
        
        # Load data first
        if self.load_in_memory:
            self.data = pd.read_csv(data_path)
        else:
            self.data = None
            
        # Build index of valid sequences
        self.sequence_index = self._build_sequence_index()
            
    def _build_sequence_index(self) -> List[Dict]:
        """
        Build an index of all valid sequences.
        Each entry contains student, start_week, end_week for a sequence.
        """
        # Load data to build index (even for streaming mode)
        if self.data is not None:
            data = self.data
        else:
            data = pd.read_csv(self.data_path)
        
        sequences = []
        
        for student in data[self.student_column].unique():
            student_data = data[data[self.student_column] == student].sort_values(self.time_column)
            
            # Create sequences for this student
            for i in range(self.sequence_length, len(student_data)):
                sequences.append({
                    'student': student,
                    'target_week': student_data.iloc[i][self.time_column],
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
        Get a single sequence.
        
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
            data = data[data[self.student_column] == seq_info['student']]
        
        # Get student data
        student_data = data[data[self.student_column] == seq_info['student']].sort_values(self.time_column)
        
        # Extract sequence
        start_idx = seq_info['sequence_start_idx']
        end_idx = seq_info['sequence_end_idx']
        
        # Features: historical values + time information
        sequence_data = student_data.iloc[start_idx:end_idx]
        
        features = []
        for _, row in sequence_data.iterrows():
            # Include relevant features for predicting proficiency
            feature_vector = []
            
            # Time-based features - convert week string to numeric
            if self.time_column in row:
                feature_vector.append(week_string_to_numeric(row[self.time_column]))
            
            # Study behavior features - safely convert to float
            if 'minutes_per_week' in row:
                feature_vector.append(safe_float_conversion(row['minutes_per_week']))
            
            if 'problems_solved' in row:
                feature_vector.append(safe_float_conversion(row['problems_solved']))
                
            if 'total_opportunities' in row:
                feature_vector.append(safe_float_conversion(row['total_opportunities']))
            
            # Previous proficiency - safely convert to float
            if self.target_column in row:
                feature_vector.append(safe_float_conversion(row[self.target_column]))
            
            # Additional features for student ability model
            if 'n_skills_measured' in row:
                feature_vector.append(safe_float_conversion(row['n_skills_measured']))
            
            if 'week_difficulty' in row:
                feature_vector.append(safe_float_conversion(row['week_difficulty']))
                
            if 'student_ability' in row:
                feature_vector.append(safe_float_conversion(row['student_ability']))
                
            if 'student_learning_rate' in row:
                feature_vector.append(safe_float_conversion(row['student_learning_rate']))
            
            # If we have fewer features than expected, pad with basic features
            if len(feature_vector) < 2:
                # Fallback to basic features for compatibility
                feature_vector = [
                    week_string_to_numeric(row[self.time_column]) if self.time_column in row else 0,
                    safe_float_conversion(row[self.target_column]) if self.target_column in row else 0
                ]
            
            features.append(feature_vector)
        
        # Target: next proficiency value - safely convert to float
        target = safe_float_conversion(student_data.iloc[end_idx][self.target_column])
        
        return torch.tensor(features, dtype=torch.float32), torch.tensor([target], dtype=torch.float32)
    
    def get_splits(self, n_splits: int = 5, test_size: int = 1) -> List[Tuple[List[int], List[int]]]:
        """
        Generate time series cross-validation splits.
        OPTIMIZED VERSION: Pre-compute mappings to avoid nested loops.
        
        Returns:
            List of (train_indices, val_indices) where indices refer to self.sequence_index
        """
        try:
            # Use legacy splitting if available
            data_path = os.path.expanduser(self.data_path)
            global_timeline, legacy_splits = create_time_series_splits(data_path, n_splits, test_size)
            
            # PRE-COMPUTE: Create a mapping from week to sequence indices (FAST)
            week_to_seq_indices = {}
            for seq_idx, seq_info in enumerate(self.sequence_index):
                target_week = seq_info['target_week']
                if target_week not in week_to_seq_indices:
                    week_to_seq_indices[target_week] = []
                week_to_seq_indices[target_week].append(seq_idx)
            
            # PRE-COMPUTE: Create a set of weeks from global timeline for fast lookup
            timeline_weeks = set(global_timeline['week'].values)
            
            # Convert legacy splits to sequence indices (OPTIMIZED)
            splits = []
            for train_week_indices, val_week_indices in legacy_splits:
                # Get actual week values (not indices)
                train_weeks = set(global_timeline.iloc[i]['week'] for i in train_week_indices)
                val_weeks = set(global_timeline.iloc[i]['week'] for i in val_week_indices)
                
                # Map weeks to sequence indices using pre-computed mapping
                train_seq_indices = []
                val_seq_indices = []
                
                for week, seq_indices in week_to_seq_indices.items():
                    if week in train_weeks:
                        train_seq_indices.extend(seq_indices)
                    elif week in val_weeks:
                        val_seq_indices.extend(seq_indices)
                
                splits.append((train_seq_indices, val_seq_indices))
            
            return splits
            
        except Exception as e:
            print(f"Warning: Could not use legacy splitting ({e}). Using simple split.")
            return self._simple_time_split(n_splits, test_size)
    
    def _simple_time_split(self, n_splits: int, test_size: int) -> List[Tuple[List[int], List[int]]]:
        """Fallback simple time-based splitting."""
        # Group sequences by week
        week_to_indices = {}
        for idx, seq_info in enumerate(self.sequence_index):
            week = seq_info['target_week']
            if week not in week_to_indices:
                week_to_indices[week] = []
            week_to_indices[week].append(idx)
        
        sorted_weeks = sorted(week_to_indices.keys())
        
        splits = []
        total_weeks = len(sorted_weeks)
        
        for fold in range(n_splits):
            # Calculate split points
            split_size = total_weeks // n_splits
            val_start = fold * split_size
            val_end = min(val_start + test_size, total_weeks)
            
            val_weeks = sorted_weeks[val_start:val_end]
            train_weeks = [w for w in sorted_weeks if w not in val_weeks]
            
            # Get indices
            train_indices = []
            val_indices = []
            
            for week in train_weeks:
                train_indices.extend(week_to_indices[week])
            for week in val_weeks:
                val_indices.extend(week_to_indices[week])
            
            splits.append((train_indices, val_indices))
        
        return splits


class DataLoaderFactory:
    """
    Factory for creating DataLoaders with appropriate configurations.
    """
    
    @staticmethod
    def create_dataloader(dataset: StudentTimeSeriesDataset,
                         indices: Optional[List[int]] = None,
                         batch_size: int = 32,
                         shuffle: bool = False,
                         num_workers: int = 0) -> DataLoader:
        """
        Create a DataLoader for the dataset.
        
        Args:
            dataset: The dataset
            indices: Subset indices (for train/val splits)
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes
            
        Returns:
            DataLoader instance
        """
        if indices is not None:
            # Create subset
            subset = torch.utils.data.Subset(dataset, indices)
            return DataLoader(subset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        else:
            return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def convert_legacy_format(X: np.ndarray, y: np.ndarray, batch_size: int = 32) -> DataLoader:
    """
    Convert legacy numpy arrays to DataLoader format.
    Useful for backwards compatibility.
    """
    
    class ArrayDataset(Dataset):
        def __init__(self, X, y):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.float32)
        
        def __len__(self):
            return len(self.X)
        
        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]
    
    dataset = ArrayDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False) 