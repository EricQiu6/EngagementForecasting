#!/usr/bin/env python3
"""
Data processing pipeline for time-series prediction
Handles data tidying and chronological splitting for model-agnostic training

This implements step 3 of the baseline specification:
3. Split chronologically (per student)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from pathlib import Path

def create_global_timeline(data_path='data_tidied.csv'):
    """
    Convert per-student data to global chronological timeline
    
    Args:
        data_path: Path to tidied data file
        
    Returns:
        DataFrame: [week, student_id, proficient] sorted chronologically
    """
    # Load data
    df = pd.read_csv(data_path)
    
    # Sort by week first (chronological), then by student (for consistency)
    global_timeline = df.sort_values(['week', 'name']).reset_index(drop=True)
    
    return global_timeline

def create_time_series_splits(data_path='data_tidied.csv', n_splits=5, test_size=1):
    """
    Create TimeSeriesSplit cross-validation folds
    
    Args:
        data_path: Path to tidied data
        n_splits: Number of CV folds
        test_size: Number of weeks in test set per fold
        
    Returns:
        tuple: (global_timeline, list of (train_indices, test_indices))
    """
    # Create global timeline
    global_timeline = create_global_timeline(data_path)
    
    # Get unique weeks for splitting
    unique_weeks = sorted(global_timeline['week'].unique())
    n_weeks = len(unique_weeks)
    
    # Create TimeSeriesSplit on week level
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
    
    # Generate splits based on week indices
    week_splits = list(tscv.split(unique_weeks))
    
    # Convert week indices to row indices in global timeline
    row_splits = []
    for train_week_idx, test_week_idx in week_splits:
        train_weeks = [unique_weeks[i] for i in train_week_idx]
        test_weeks = [unique_weeks[i] for i in test_week_idx]
        
        train_row_indices = global_timeline[global_timeline['week'].isin(train_weeks)].index.tolist()
        test_row_indices = global_timeline[global_timeline['week'].isin(test_weeks)].index.tolist()
        
        row_splits.append((train_row_indices, test_row_indices))
    
    return global_timeline, row_splits

def get_fold_data(global_timeline, train_indices, test_indices):
    """
    Extract train/test data for a specific fold
    
    Args:
        global_timeline: DataFrame with global chronological data
        train_indices: Row indices for training data
        test_indices: Row indices for test data
        
    Returns:
        tuple: (train_data, test_data) DataFrames in original format
    """
    train_data = global_timeline.iloc[train_indices].copy()
    test_data = global_timeline.iloc[test_indices].copy()
    
    # Convert back to original format [name, week, proficient]
    train_data = train_data.rename(columns={'student_id': 'name'})
    test_data = test_data.rename(columns={'student_id': 'name'})
    
    return train_data, test_data

def analyze_data_for_split(df):
    """
    Analyze the data to determine appropriate K value for chronological split
    
    Args:
        df: DataFrame with columns [name, week, proficient]
    
    Returns:
        dict: Analysis results including recommended K value
    """
    print("Analyzing data structure for chronological split...")
    
    # Calculate weeks per student
    weeks_per_student = df.groupby('name')['week'].count()
    
    analysis = {
        'total_students': df['name'].nunique(),
        'week_range': (df['week'].min(), df['week'].max()),
        'total_weeks': df['week'].max() - df['week'].min() + 1,
        'weeks_per_student_stats': {
            'min': weeks_per_student.min(),
            'max': weeks_per_student.max(),
            'mean': weeks_per_student.mean(),
            'median': weeks_per_student.median(),
            'std': weeks_per_student.std()
        },
        'weeks_distribution': weeks_per_student.value_counts().sort_index().to_dict()
    }
    
    # Recommend K based on data characteristics
    # Rule: Use ~25-30% of data for testing, but ensure minimum training length
    median_weeks = weeks_per_student.median()
    min_training_weeks = 5  # Minimum for meaningful AR model
    max_test_weeks = max(1, int(median_weeks * 0.3))
    
    # Ensure we have enough training data
    recommended_k = min(max_test_weeks, int(median_weeks - min_training_weeks))
    recommended_k = max(1, recommended_k)  # At least 1 week for testing
    
    analysis['recommended_k'] = recommended_k
    analysis['reasoning'] = f"With median {median_weeks:.1f} weeks per student, K={recommended_k} provides ~{recommended_k/median_weeks*100:.1f}% test data while ensuring ≥{min_training_weeks} training weeks"
    
    return analysis

# Backward compatibility - keep original function
def split_chronologically(df=None, K=None, data_path='data_tidied.csv'):
    """
    DEPRECATED: Original per-student chronological split
    Use create_time_series_splits() for proper time series cross-validation
    """
    print("WARNING: split_chronologically() is deprecated.")
    print("Use create_time_series_splits() for proper time series cross-validation.")
    
    # Keep original implementation for backward compatibility
    print("=" * 60)
    print("STEP 3: Split chronologically")
    print("=" * 60)
    
    # Load data if not provided
    if df is None:
        print(f"Loading tidied data from: {data_path}")
        df = pd.read_csv(data_path)
        print(f"Loaded data shape: {df.shape}")
        print()
    
    # Analyze data to determine K if not provided
    if K is None:
        analysis = analyze_data_for_split(df)
        K = analysis['recommended_k']
        
        print("Data analysis for split determination:")
        print(f"  Total students: {analysis['total_students']}")
        print(f"  Week range: {analysis['week_range'][0]} to {analysis['week_range'][1]}")
        print(f"  Weeks per student - median: {analysis['weeks_per_student_stats']['median']:.1f}, range: {analysis['weeks_per_student_stats']['min']}-{analysis['weeks_per_student_stats']['max']}")
        print(f"  Recommended K: {K}")
        print(f"  Reasoning: {analysis['reasoning']}")
        print()
    else:
        print(f"Using provided K = {K}")
        print()
    
    # Perform chronological split per student
    train_data = []
    test_data = []
    split_info = []
    
    print(f"Splitting data with K = {K} test weeks per student...")
    
    students_processed = 0
    students_insufficient_data = 0
    
    for student in df['name'].unique():
        student_data = df[df['name'] == student].sort_values('week').reset_index(drop=True)
        
        # Check if student has enough data for meaningful split
        if len(student_data) <= K:
            # Not enough data for both train and test
            students_insufficient_data += 1
            continue
            
        # Split into train (everything except last K weeks) and test (last K weeks)
        split_point = len(student_data) - K
        
        student_train = student_data.iloc[:split_point].copy()
        student_test = student_data.iloc[split_point:].copy()
        
        # Add split metadata
        student_train['split'] = 'train'
        student_test['split'] = 'test'
        
        train_data.append(student_train)
        test_data.append(student_test)
        
        # Record split information
        split_info.append({
            'student': student,
            'total_weeks': len(student_data),
            'train_weeks': len(student_train),
            'test_weeks': len(student_test),
            'train_week_range': (student_train['week'].min(), student_train['week'].max()),
            'test_week_range': (student_test['week'].min(), student_test['week'].max())
        })
        
        students_processed += 1
    
    # Combine all student data
    train_df = pd.concat(train_data, ignore_index=True) if train_data else pd.DataFrame()
    test_df = pd.concat(test_data, ignore_index=True) if test_data else pd.DataFrame()
    
    # Create metadata
    metadata = {
        'K': K,
        'total_students': df['name'].nunique(),
        'students_processed': students_processed,
        'students_insufficient_data': students_insufficient_data,
        'train_samples': len(train_df),
        'test_samples': len(test_df),
        'original_samples': len(df)
    }
    
    # Print summary
    print("Chronological split complete!")
    print(f"  Students processed: {students_processed}")
    print(f"  Students with insufficient data (≤{K} weeks): {students_insufficient_data}")
    print(f"  Training samples: {len(train_df)}")
    print(f"  Test samples: {len(test_df)}")
    print(f"  Training weeks per student: {train_df.groupby('name').size().describe()}")
    print(f"  Test weeks per student: {test_df.groupby('name').size().describe()}")
    print()
    
    # Create model-agnostic output format
    result = {
        'train_data': train_df,
        'test_data': test_df,
        'split_info': pd.DataFrame(split_info),
        'metadata': metadata,
        'target_column': 'proficient',
        'student_column': 'name',
        'time_column': 'week'
    }
    
    print("=" * 60)
    print("CHRONOLOGICAL SPLIT COMPLETE")
    print("=" * 60)
    print("Output format (model-agnostic):")
    print("  result['train_data'] - Training data DataFrame")
    print("  result['test_data'] - Test data DataFrame") 
    print("  result['split_info'] - Per-student split statistics")
    print("  result['metadata'] - Overall split metadata")
    print("  result['target_column'] - Name of target variable")
    print("  result['student_column'] - Name of student identifier")
    print("  result['time_column'] - Name of time variable")
    
    return result

if __name__ == "__main__":
    # Demo of new time series cross-validation
    print("=== TIME SERIES CROSS-VALIDATION DEMO ===")
    
    global_timeline, splits = create_time_series_splits(n_splits=3, test_size=1)
    
    print(f"Global timeline shape: {global_timeline.shape}")
    print(f"Week range: {global_timeline['week'].min()} to {global_timeline['week'].max()}")
    print(f"Number of CV folds: {len(splits)}")
    print()
    
    for i, (train_idx, test_idx) in enumerate(splits):
        train_data, test_data = get_fold_data(global_timeline, train_idx, test_idx)
        
        print(f"Fold {i+1}:")
        print(f"  Train weeks: {train_data['week'].min()} to {train_data['week'].max()}")
        print(f"  Test weeks: {test_data['week'].min()} to {test_data['week'].max()}")
        print(f"  Train samples: {len(train_data)}")
        print(f"  Test samples: {len(test_data)}")
        print()