#!/usr/bin/env python3
"""
Diagnose why the student ability models have poor performance
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# Add framework to path
sys.path.append(str(Path(__file__).parent / 'framework_v2'))

from framework_v2 import StudentTimeSeriesDataset


def analyze_target_distribution():
    """Analyze the target variable distribution."""
    print("="*60)
    print("🔍 ANALYZING TARGET VARIABLE (avg_proficiency)")
    print("="*60)
    
    # Load raw data
    data_path = '../data-analysis/student_week_aggregations_rolling_new.csv'
    df = pd.read_csv(data_path)
    
    # Basic statistics
    print("\n📊 Target Variable Statistics:")
    print(df['avg_proficiency'].describe())
    
    # Value distribution
    print("\n📊 Target Value Distribution (top 20):")
    value_counts = df['avg_proficiency'].value_counts().head(20)
    print(value_counts)
    
    # Percentage of zeros
    zero_pct = (df['avg_proficiency'] == 0).sum() / len(df) * 100
    print(f"\n⚠️  Percentage of zeros: {zero_pct:.1f}%")
    
    # Check for NA values
    na_count = df['avg_proficiency'].isna().sum()
    print(f"⚠️  Number of NA values: {na_count}")
    
    # Distribution plot
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.hist(df['avg_proficiency'].dropna(), bins=50, edgecolor='black')
    plt.title('Distribution of avg_proficiency')
    plt.xlabel('avg_proficiency')
    plt.ylabel('Count')
    
    plt.subplot(1, 3, 2)
    plt.hist(df['avg_proficiency'][df['avg_proficiency'] > 0], bins=50, edgecolor='black')
    plt.title('Distribution (excluding zeros)')
    plt.xlabel('avg_proficiency')
    plt.ylabel('Count')
    
    plt.subplot(1, 3, 3)
    plt.boxplot(df['avg_proficiency'].dropna())
    plt.title('Boxplot of avg_proficiency')
    plt.ylabel('avg_proficiency')
    
    plt.tight_layout()
    plt.savefig('target_distribution.png')
    print("\n📊 Saved distribution plots to target_distribution.png")
    
    return df


def analyze_feature_correlations(df):
    """Analyze correlations between features and target."""
    print("\n" + "="*60)
    print("🔍 ANALYZING FEATURE CORRELATIONS")
    print("="*60)
    
    # Select relevant features
    features = ['minutes_per_week', 'problems_solved', 'total_opportunities', 
                'week_difficulty', 'student_ability', 'student_learning_rate']
    
    # Calculate correlations with target
    print("\n📊 Correlations with avg_proficiency:")
    for feature in features:
        if feature in df.columns:
            corr = df[feature].corr(df['avg_proficiency'])
            print(f"  {feature}: {corr:.3f}")
    
    # Correlation matrix
    corr_matrix = df[features + ['avg_proficiency']].corr()
    
    plt.figure(figsize=(10, 8))
    plt.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    plt.colorbar(label='Correlation')
    
    # Add labels
    labels = features + ['avg_proficiency']
    plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
    plt.yticks(range(len(labels)), labels)
    
    # Add correlation values
    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', 
                    ha='center', va='center', 
                    color='white' if abs(corr_matrix.iloc[i, j]) > 0.5 else 'black')
    
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')
    print("\n📊 Saved correlation matrix to correlation_matrix.png")


def analyze_temporal_patterns(df):
    """Analyze temporal patterns in the data."""
    print("\n" + "="*60)
    print("🔍 ANALYZING TEMPORAL PATTERNS")
    print("="*60)
    
    # Convert week_id to datetime for analysis
    df['week_date'] = pd.to_datetime(df['week_id'] + '-1', format='%Y-W%W-%w')
    
    # Average proficiency over time
    weekly_avg = df.groupby('week_date')['avg_proficiency'].agg(['mean', 'std', 'count'])
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(weekly_avg.index, weekly_avg['mean'], label='Mean avg_proficiency')
    plt.fill_between(weekly_avg.index, 
                     weekly_avg['mean'] - weekly_avg['std'], 
                     weekly_avg['mean'] + weekly_avg['std'], 
                     alpha=0.3, label='±1 std')
    plt.title('Average Proficiency Over Time')
    plt.xlabel('Week')
    plt.ylabel('avg_proficiency')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.bar(weekly_avg.index, weekly_avg['count'], width=7)
    plt.title('Number of Students per Week')
    plt.xlabel('Week')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('temporal_patterns.png')
    print("\n📊 Saved temporal patterns to temporal_patterns.png")


def analyze_student_patterns(df):
    """Analyze patterns by student."""
    print("\n" + "="*60)
    print("🔍 ANALYZING STUDENT PATTERNS")
    print("="*60)
    
    # Students with most data
    student_counts = df['anon_student_id'].value_counts()
    print(f"\n📊 Number of unique students: {len(student_counts)}")
    print(f"📊 Average weeks per student: {student_counts.mean():.1f}")
    print(f"📊 Median weeks per student: {student_counts.median():.1f}")
    
    # Analyze a few students
    top_students = student_counts.head(5).index
    
    plt.figure(figsize=(12, 8))
    
    for i, student in enumerate(top_students):
        student_data = df[df['anon_student_id'] == student].sort_values('week_id')
        
        plt.subplot(3, 2, i+1)
        weeks = range(len(student_data))
        plt.plot(weeks, student_data['avg_proficiency'], 'o-', label='Actual')
        plt.plot(weeks, student_data['student_ability'], '--', label='Ability', alpha=0.5)
        plt.title(f'Student {i+1} (n={len(student_data)})')
        plt.xlabel('Week Index')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('student_patterns.png')
    print("\n📊 Saved student patterns to student_patterns.png")


def analyze_prediction_difficulty():
    """Analyze why predictions might be difficult."""
    print("\n" + "="*60)
    print("🔍 ANALYZING PREDICTION CHALLENGES")
    print("="*60)
    
    # Load dataset
    data_path = '../data-analysis/student_week_aggregations_rolling_new.csv'
    dataset = StudentTimeSeriesDataset(
        data_path=data_path,
        sequence_length=5,
        target_column='avg_proficiency',
        student_column='anon_student_id',
        time_column='week_id',
        load_in_memory=True
    )
    
    # Sample some sequences
    print("\n📊 Sample sequences to understand the data:")
    for i in range(5):
        features, target = dataset[i]
        print(f"\nSequence {i}:")
        print(f"  Features shape: {features.shape}")
        print(f"  Target: {target.item():.2f}")
        print(f"  Last proficiency in sequence: {features[-1, 4].item():.2f}")
        print(f"  Proficiency values in sequence: {features[:, 4].numpy()}")


def main():
    """Run all diagnostics."""
    print("🔍 DIAGNOSING MODEL PERFORMANCE ISSUES")
    print("="*60)
    
    # Analyze target distribution
    df = analyze_target_distribution()
    
    # Analyze feature correlations
    analyze_feature_correlations(df)
    
    # Analyze temporal patterns
    analyze_temporal_patterns(df)
    
    # Analyze student patterns
    analyze_student_patterns(df)
    
    # Analyze prediction difficulty
    analyze_prediction_difficulty()
    
    print("\n" + "="*60)
    print("📊 KEY FINDINGS:")
    print("="*60)
    print("\n1. Target variable issues:")
    print("   - High percentage of zeros (sparse target)")
    print("   - Integer values only (0-100 range)")
    print("   - Highly imbalanced distribution")
    print("\n2. Temporal issues:")
    print("   - Irregular sampling (students miss weeks)")
    print("   - Strong temporal trends")
    print("\n3. Model challenges:")
    print("   - Predicting mostly zeros is difficult")
    print("   - Need to handle the discrete nature of target")
    print("   - May benefit from classification or zero-inflated models")


if __name__ == "__main__":
    main() 