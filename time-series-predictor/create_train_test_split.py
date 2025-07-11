#!/usr/bin/env python3
"""
Create Train/Validation Split for Time Series Data
================================================

This script creates a student-level train/validation split to avoid
the computational overhead of cross-validation while still preventing
data leakage through student-specific features.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Any
import argparse
import json
from datetime import datetime


def analyze_dataset(df: pd.DataFrame, student_col: str, time_col: str) -> Dict[str, Any]:
    """Analyze the dataset to understand its structure."""
    analysis = {
        'total_students': df[student_col].nunique(),
        'total_records': len(df),
        'time_range': (df[time_col].min(), df[time_col].max()),
        'avg_records_per_student': len(df) / df[student_col].nunique(),
        'student_record_distribution': df[student_col].value_counts().describe()
    }
    
    print("Dataset Analysis:")
    print(f"  Total students: {analysis['total_students']}")
    print(f"  Total records: {analysis['total_records']}")
    print(f"  Time range: {analysis['time_range']}")
    print(f"  Avg records per student: {analysis['avg_records_per_student']:.1f}")
    print(f"  Student record distribution:")
    print(f"    Min: {analysis['student_record_distribution']['min']:.0f}")
    print(f"    Max: {analysis['student_record_distribution']['max']:.0f}")
    print(f"    Mean: {analysis['student_record_distribution']['mean']:.1f}")
    print(f"    Std: {analysis['student_record_distribution']['std']:.1f}")
    
    return analysis


def create_student_level_split(df: pd.DataFrame, 
                             student_col: str,
                             train_ratio: float = 0.7,
                             stratify_by: str = None,
                             min_records_per_student: int = 5,
                             random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create a student-level train/validation split.
    
    Args:
        df: Input dataframe
        student_col: Column name for student IDs
        train_ratio: Proportion of students for training
        stratify_by: Column to use for stratification (optional)
        min_records_per_student: Minimum records required per student
        random_state: Random seed for reproducibility
        
    Returns:
        (train_df, val_df): Training and validation dataframes
    """
    np.random.seed(random_state)
    
    # Filter students with minimum records
    student_counts = df[student_col].value_counts()
    valid_students = student_counts[student_counts >= min_records_per_student].index
    
    print(f"Students with >= {min_records_per_student} records: {len(valid_students)}/{len(student_counts)}")
    
    # Filter dataset to valid students
    df_filtered = df[df[student_col].isin(valid_students)].copy()
    
    if stratify_by and stratify_by in df_filtered.columns:
        # Stratified split
        train_students, val_students = stratified_student_split(
            df_filtered, student_col, stratify_by, train_ratio, random_state
        )
    else:
        # Random split
        students = df_filtered[student_col].unique()
        np.random.shuffle(students)
        n_train = int(len(students) * train_ratio)
        train_students = students[:n_train]
        val_students = students[n_train:]
    
    # Create splits
    train_df = df_filtered[df_filtered[student_col].isin(train_students)].copy()
    val_df = df_filtered[df_filtered[student_col].isin(val_students)].copy()
    
    print(f"Split Results:")
    print(f"  Training: {len(train_students)} students, {len(train_df)} records")
    print(f"  Validation: {len(val_students)} students, {len(val_df)} records")
    print(f"  Ratio: {len(train_df)/(len(train_df)+len(val_df)):.3f} train")
    
    return train_df, val_df


def stratified_student_split(df: pd.DataFrame, 
                           student_col: str,
                           stratify_col: str,
                           train_ratio: float,
                           random_state: int) -> Tuple[List[str], List[str]]:
    """Create stratified split based on a column (e.g., performance level)."""
    np.random.seed(random_state)
    
    # Calculate student-level stratification variable
    student_strat = df.groupby(student_col)[stratify_col].mean()
    
    # Create quartiles for stratification
    quartiles = student_strat.quantile([0.25, 0.5, 0.75]).values
    
    def get_quartile(value):
        if value <= quartiles[0]:
            return 0
        elif value <= quartiles[1]:
            return 1
        elif value <= quartiles[2]:
            return 2
        else:
            return 3
    
    student_quartiles = student_strat.apply(get_quartile)
    
    train_students = []
    val_students = []
    
    # Split within each quartile
    for quartile in range(4):
        quartile_students = student_quartiles[student_quartiles == quartile].index.tolist()
        np.random.shuffle(quartile_students)
        
        n_train = int(len(quartile_students) * train_ratio)
        train_students.extend(quartile_students[:n_train])
        val_students.extend(quartile_students[n_train:])
    
    print(f"Stratified split by {stratify_col} quartiles:")
    for q in range(4):
        q_students = student_quartiles[student_quartiles == q]
        q_train = len([s for s in train_students if s in q_students.index])
        q_val = len([s for s in val_students if s in q_students.index])
        print(f"  Quartile {q}: {q_train} train, {q_val} val")
    
    return train_students, val_students


def create_temporal_split(df: pd.DataFrame,
                        student_col: str,
                        time_col: str,
                        train_ratio: float = 0.7) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create a temporal split where early students go to training.
    This maintains some temporal structure while avoiding data leakage.
    """
    # Get first appearance time for each student
    student_first_time = df.groupby(student_col)[time_col].min().sort_values()
    
    # Split by temporal order of student entry
    n_train = int(len(student_first_time) * train_ratio)
    train_students = student_first_time.iloc[:n_train].index
    val_students = student_first_time.iloc[n_train:].index
    
    train_df = df[df[student_col].isin(train_students)].copy()
    val_df = df[df[student_col].isin(val_students)].copy()
    
    print(f"Temporal Split Results:")
    print(f"  Training: {len(train_students)} students (early), {len(train_df)} records")
    print(f"  Validation: {len(val_students)} students (late), {len(val_df)} records")
    print(f"  Train time range: {train_df[time_col].min()} to {train_df[time_col].max()}")
    print(f"  Val time range: {val_df[time_col].min()} to {val_df[time_col].max()}")
    
    return train_df, val_df


def save_split_info(train_df: pd.DataFrame, 
                   val_df: pd.DataFrame,
                   config: Dict[str, Any],
                   output_dir: Path):
    """Save information about the split for reproducibility."""
    split_info = {
        'timestamp': datetime.now().isoformat(),
        'config': config,
        'train_stats': {
            'n_students': train_df['anon_student_id'].nunique(),
            'n_records': len(train_df),
            'time_range': (train_df['week_id'].min(), train_df['week_id'].max())
        },
        'val_stats': {
            'n_students': val_df['anon_student_id'].nunique(),
            'n_records': len(val_df),
            'time_range': (val_df['week_id'].min(), val_df['week_id'].max())
        }
    }
    
    with open(output_dir / 'split_info.json', 'w') as f:
        json.dump(split_info, f, indent=2)
    
    print(f"Split info saved to: {output_dir / 'split_info.json'}")


def main():
    parser = argparse.ArgumentParser(description='Create train/validation split for time series data')
    parser.add_argument('--input-file', type=str, 
                       default='../data-analysis/steve_dang_100_window5.csv',
                       help='Input CSV file')
    parser.add_argument('--output-dir', type=str, default='data_splits_steve_dang_100_window5',
                       help='Output directory for split files')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                       help='Proportion of students for training')
    parser.add_argument('--split-method', type=str, default='random',
                       choices=['random', 'stratified', 'temporal'],
                       help='Method for splitting students')
    parser.add_argument('--stratify-by', type=str, default=None,
                       help='Column to use for stratification (if using stratified)')
    parser.add_argument('--min-records', type=int, default=5,
                       help='Minimum records required per student')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("="*80)
    print("CREATING TRAIN/VALIDATION SPLIT")
    print("="*80)
    
    # Load data
    print(f"Loading data from: {args.input_file}")
    df = pd.read_csv(args.input_file)
    print(f"Loaded {len(df)} records")
    
    # Analyze dataset
    analysis = analyze_dataset(df, 'anon_student_id', 'week_id')
    
    # Create split based on method
    print(f"\nCreating {args.split_method} split...")
    
    if args.split_method == 'random':
        train_df, val_df = create_student_level_split(
            df, 'anon_student_id', 
            train_ratio=args.train_ratio,
            min_records_per_student=args.min_records,
            random_state=args.random_state
        )
    elif args.split_method == 'stratified':
        if not args.stratify_by:
            args.stratify_by = 'avg_proficiency'  # Default
        train_df, val_df = create_student_level_split(
            df, 'anon_student_id',
            train_ratio=args.train_ratio,
            stratify_by=args.stratify_by,
            min_records_per_student=args.min_records,
            random_state=args.random_state
        )
    elif args.split_method == 'temporal':
        train_df, val_df = create_temporal_split(
            df, 'anon_student_id', 'week_id',
            train_ratio=args.train_ratio
        )
    
    # Save splits
    train_file = output_dir / 'train.csv'
    val_file = output_dir / 'val.csv'
    
    train_df.to_csv(train_file, index=False)
    val_df.to_csv(val_file, index=False)
    
    print(f"\nSplit files saved:")
    print(f"  Training: {train_file}")
    print(f"  Validation: {val_file}")
    
    # Save split configuration
    config = {
        'input_file': args.input_file,
        'split_method': args.split_method,
        'train_ratio': args.train_ratio,
        'stratify_by': args.stratify_by,
        'min_records': args.min_records,
        'random_state': args.random_state
    }
    
    save_split_info(train_df, val_df, config, output_dir)
    
    print(f"\n✅ Split creation complete!")
    print(f"Use these files in your evaluation scripts:")
    print(f"  --train-data {train_file}")
    print(f"  --val-data {val_file}")


if __name__ == "__main__":
    main()