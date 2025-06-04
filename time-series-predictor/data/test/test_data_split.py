#!/usr/bin/env python3
"""
Test script to demonstrate usage of the chronological split function
"""

import pandas as pd
from data_processing import split_chronologically

def test_chronological_split():
    """
    Test the chronological split function and demonstrate its usage
    """
    print("Testing chronological split function...")
    print("=" * 50)
    
    # Execute the split
    result = split_chronologically()
    
    # Extract the components
    train_data = result['train_data']
    test_data = result['test_data']
    split_info = result['split_info']
    metadata = result['metadata']
    
    print("\n" + "=" * 50)
    print("DETAILED ANALYSIS OF SPLIT RESULTS")
    print("=" * 50)
    
    # Analyze training data
    print("\n📊 TRAINING DATA ANALYSIS:")
    print(f"Shape: {train_data.shape}")
    print(f"Students: {train_data['name'].nunique()}")
    print(f"Week range: {train_data['week'].min()} to {train_data['week'].max()}")
    print(f"Proficient scores range: {train_data['proficient'].min():.1f} to {train_data['proficient'].max():.1f}")
    print(f"Sample of training data:")
    print(train_data.head())
    
    # Analyze test data
    print("\n📊 TEST DATA ANALYSIS:")
    print(f"Shape: {test_data.shape}")
    print(f"Students: {test_data['name'].nunique()}")
    print(f"Week range: {test_data['week'].min()} to {test_data['week'].max()}")
    print(f"Proficient scores range: {test_data['proficient'].min():.1f} to {test_data['proficient'].max():.1f}")
    print(f"Sample of test data:")
    print(test_data.head())
    
    # Analyze split info
    print("\n📊 SPLIT INFO ANALYSIS:")
    print(f"Number of students: {len(split_info)}")
    print(f"Training weeks per student:")
    print(split_info['train_weeks'].describe())
    print(f"Sample split info:")
    print(split_info.head())
    
    # Verify data integrity
    print("\n✅ DATA INTEGRITY CHECKS:")
    
    # Check 1: No overlap between train and test weeks for any student
    overlap_issues = []
    for _, row in split_info.iterrows():
        student = row['student']
        train_weeks = set(range(row['train_week_range'][0], row['train_week_range'][1] + 1))
        test_weeks = set(range(row['test_week_range'][0], row['test_week_range'][1] + 1))
        
        if train_weeks.intersection(test_weeks):
            overlap_issues.append(student)
    
    print(f"Students with train/test week overlap: {len(overlap_issues)}")
    
    # Check 2: All students should have exactly K=3 test weeks
    test_week_counts = test_data.groupby('name').size()
    students_wrong_test_count = test_week_counts[test_week_counts != metadata['K']]
    print(f"Students with wrong test week count: {len(students_wrong_test_count)}")
    
    # Check 3: Total samples should match
    total_samples = len(train_data) + len(test_data)
    expected_samples = metadata['original_samples'] - metadata['students_insufficient_data'] * metadata['K']
    print(f"Sample count verification: {total_samples} vs expected ~{expected_samples}")
    
    # Example of how to use for modeling
    print("\n🔧 EXAMPLE USAGE FOR MODELING:")
    print("# Extract data for modeling")
    print("train_X = train_data[['week']]  # Features")
    print("train_y = train_data['proficient']  # Target")
    print("test_X = test_data[['week']]")
    print("test_y = test_data['proficient']")
    print("\n# For time-series AR model, you'd create lagged features:")
    
    # Demonstrate creating lagged features for one student
    sample_student = train_data['name'].iloc[0]
    student_train_data = train_data[train_data['name'] == sample_student].sort_values('week')
    
    print(f"\nExample for student {sample_student}:")
    print("Original data:")
    print(student_train_data[['week', 'proficient']])
    
    # Create simple lag features
    student_train_data_lag = student_train_data.copy()
    student_train_data_lag['proficient_lag1'] = student_train_data_lag['proficient'].shift(1)
    student_train_data_lag['proficient_lag2'] = student_train_data_lag['proficient'].shift(2)
    
    print("\nWith lag features (for AR model):")
    print(student_train_data_lag[['week', 'proficient', 'proficient_lag1', 'proficient_lag2']].dropna())
    
    return result

if __name__ == "__main__":
    result = test_chronological_split()
    
    print("\n" + "=" * 50)
    print("Split result is now available as 'result' dictionary")
    print("Key components:")
    for key in result.keys():
        if key in ['train_data', 'test_data', 'split_info']:
            print(f"  result['{key}'] - shape: {result[key].shape}")
        else:
            print(f"  result['{key}'] - {type(result[key])}") 