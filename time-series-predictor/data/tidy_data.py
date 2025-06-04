#!/usr/bin/env python3
"""
Minimal-input baseline forecasting pipeline for time-series prediction
Focused on skills-mastered performance using only past performance data

This implements steps 1-2 of the baseline specification:
1. Choose target series: skills-mastered (proficient column)
2. Load & tidy the data
"""

import pandas as pd
import numpy as np
from pathlib import Path

def load_and_tidy_data(csv_path='exp-static-flexible-anon-2025-05-29.csv'):
    """
    Step 1 & 2: Choose target series and load & tidy the data
    
    Target series: skills-mastered (using 'proficient' column as per specification)
    
    Returns:
        pd.DataFrame: Tidied dataset with columns [name, week, proficient]
    """
    print("=" * 60)
    print("STEP 1: Choose the target series")
    print("=" * 60)
    print("Target series: skills-mastered (using 'proficient' column)")
    print()
    
    print("=" * 60)
    print("STEP 2: Load & tidy the data")
    print("=" * 60)
    
    # Parse the CSV
    print("Loading CSV data...")
    df = pd.read_csv(csv_path)
    print(f"Original data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print()
    
    # Keep only the required columns: name, week, proficient (target metric)
    print("Selecting required columns: name, week, proficient")
    columns_to_keep = ['name', 'week', 'proficient']
    df_tidy = df[columns_to_keep].copy()
    print(f"After column selection: {df_tidy.shape}")
    print()
    
    # Sort rows by (name, week) and drop duplicates
    print("Sorting by (name, week) and removing duplicates...")
    df_tidy = df_tidy.sort_values(['name', 'week']).drop_duplicates()
    print(f"After sorting and duplicate removal: {df_tidy.shape}")
    print()
    
    # Display basic statistics about the data
    print("Data overview:")
    print(f"Number of unique students: {df_tidy['name'].nunique()}")
    print(f"Week range: {df_tidy['week'].min()} to {df_tidy['week'].max()}")
    print(f"Target variable (proficient) statistics:")
    print(df_tidy['proficient'].describe())
    print()
    
    # Check for missing values in the target variable
    missing_values = df_tidy['proficient'].isna().sum()
    print(f"Missing values in 'proficient': {missing_values}")
    if missing_values > 0:
        print("Rows with missing proficient values will be handled during imputation.")
    print()
    
    # Analyze missing weeks within student spans
    print("Analyzing missing weeks within student spans...")
    missing_weeks_info = analyze_missing_weeks(df_tidy)
    print(missing_weeks_info)
    print()
    
    # Forward-fill missing weeks within each student's span
    print("Forward-filling missing weeks within each student's span...")
    df_filled = forward_fill_missing_weeks(df_tidy)
    print(f"After forward-filling: {df_filled.shape}")
    print()
    
    # Save the tidied data
    output_path = 'data_tidied.csv'
    print(f"Saving tidied data to: {output_path}")
    df_filled.to_csv(output_path, index=False)
    
    print("=" * 60)
    print("DATA TIDYING COMPLETE")
    print("=" * 60)
    print(f"Final dataset shape: {df_filled.shape}")
    print(f"Columns: {list(df_filled.columns)}")
    print("Sample of final data:")
    print(df_filled.head(10))
    
    return df_filled

def analyze_missing_weeks(df):
    """
    Analyze missing weeks within each student's time span
    """
    results = []
    
    for student in df['name'].unique():
        student_data = df[df['name'] == student].sort_values('week')
        
        if len(student_data) == 0:
            continue
            
        min_week = student_data['week'].min()
        max_week = student_data['week'].max()
        expected_weeks = set(range(min_week, max_week + 1))
        actual_weeks = set(student_data['week'].values)
        missing_weeks = expected_weeks - actual_weeks
        
        if missing_weeks:
            results.append({
                'student': student,
                'min_week': min_week,
                'max_week': max_week,
                'missing_weeks': sorted(missing_weeks),
                'num_missing': len(missing_weeks)
            })
    
    if results:
        total_missing = sum(r['num_missing'] for r in results)
        analysis = f"Found {len(results)} students with missing weeks (total {total_missing} missing week entries)"
        analysis += f"\nExample: Student {results[0]['student']} missing weeks {results[0]['missing_weeks']}"
    else:
        analysis = "No missing weeks found within student spans"
    
    return analysis

def forward_fill_missing_weeks(df):
    """
    Forward-fill missing weeks within each student's span
    Carry forward the last observed value for missing weeks
    """
    filled_data = []
    
    for student in df['name'].unique():
        student_data = df[df['name'] == student].sort_values('week')
        
        if len(student_data) == 0:
            continue
            
        min_week = student_data['week'].min()
        max_week = student_data['week'].max()
        
        # Create complete week range for this student
        complete_weeks = pd.DataFrame({
            'name': [student] * (max_week - min_week + 1),
            'week': range(min_week, max_week + 1)
        })
        
        # Merge with existing data and forward fill
        student_complete = complete_weeks.merge(
            student_data, on=['name', 'week'], how='left'
        )
        
        # Forward fill missing proficient values
        student_complete['proficient'] = student_complete['proficient'].ffill()
        
        # If first value is NaN, fill with 0 (no skills mastered initially)
        student_complete['proficient'] = student_complete['proficient'].fillna(0)
        
        filled_data.append(student_complete)
    
    # Combine all students
    df_filled = pd.concat(filled_data, ignore_index=True)
    return df_filled.sort_values(['name', 'week'])

if __name__ == "__main__":
    # Execute steps 1 and 2
    tidied_data = load_and_tidy_data()
    
    print("\n" + "=" * 60)
    print("NEXT STEPS (Steps 3-6)")
    print("=" * 60)
    print("3. Split chronologically (keep last K weeks as test set)")
    print("4. Add learnable model (AR(p) via linear regression)")
    print("5. Evaluate on held-out weeks (MAE, RMSE, SMAPE)")
    print("6. Package into ready-to-use function")
    print("=" * 60) 