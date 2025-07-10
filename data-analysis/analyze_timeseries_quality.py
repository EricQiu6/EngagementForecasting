import pandas as pd
import numpy as np
from datetime import datetime

# Load the data
df = pd.read_csv('student_week_aggregations_rolling_new.csv')

print("=== DATASET OVERVIEW ===")
print(f"Total rows: {len(df)}")
print(f"Total columns: {len(df.columns)}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nUnique students: {df['anon_student_id'].nunique()}")
print(f"Unique weeks: {df['week_id'].nunique()}")

# Convert week_id to datetime for proper ordering
df['week_date'] = pd.to_datetime(df['week_id'] + '-1', format='%Y-W%W-%w')
df = df.sort_values(['anon_student_id', 'week_date'])

print("\n=== TEMPORAL CHARACTERISTICS ===")
print(f"Date range: {df['week_id'].min()} to {df['week_id'].max()}")

# Check sequence length per student
sequence_lengths = df.groupby('anon_student_id').size()
print(f"\nSequence length per student:")
print(f"  Mean: {sequence_lengths.mean():.1f}")
print(f"  Median: {sequence_lengths.median():.1f}")
print(f"  Min: {sequence_lengths.min()}")
print(f"  Max: {sequence_lengths.max()}")
print(f"  Students with >= 10 weeks: {(sequence_lengths >= 10).sum()} ({(sequence_lengths >= 10).sum() / len(sequence_lengths) * 100:.1f}%)")
print(f"  Students with >= 20 weeks: {(sequence_lengths >= 20).sum()} ({(sequence_lengths >= 20).sum() / len(sequence_lengths) * 100:.1f}%)")

# Check for gaps in sequences
print("\n=== TEMPORAL GAPS ===")
gap_info = []
for student in df['anon_student_id'].unique()[:100]:  # Sample first 100 students
    student_data = df[df['anon_student_id'] == student].sort_values('week_date')
    if len(student_data) > 1:
        date_diffs = student_data['week_date'].diff().dt.days
        gaps = date_diffs[date_diffs > 7]
        if len(gaps) > 0:
            gap_info.append({
                'student': student,
                'n_gaps': len(gaps),
                'max_gap_days': gaps.max()
            })

if gap_info:
    print(f"Students with gaps (sample of 100): {len(gap_info)}")
    print(f"Average gaps per student with gaps: {np.mean([g['n_gaps'] for g in gap_info]):.1f}")
    print(f"Max gap observed: {max([g['max_gap_days'] for g in gap_info])} days")

print("\n=== MISSING DATA ANALYSIS ===")
for col in df.columns:
    missing_pct = (df[col].isna().sum() / len(df)) * 100
    if missing_pct > 0:
        print(f"{col}: {missing_pct:.1f}% missing")

print("\n=== TARGET VARIABLE (minutes_per_week) ===")
target = df['minutes_per_week']
print(f"Mean: {target.mean():.2f}")
print(f"Median: {target.median():.2f}")
print(f"Std: {target.std():.2f}")
print(f"Min: {target.min():.2f}")
print(f"Max: {target.max():.2f}")
print(f"Skewness: {target.skew():.2f}")
print(f"% of zeros: {(target == 0).sum() / len(target) * 100:.1f}%")

print("\n=== FEATURE CORRELATIONS WITH TARGET ===")
numeric_cols = ['avg_proficiency', 'problems_solved', 'total_opportunities', 
                'week_difficulty', 'student_ability', 'student_learning_rate']
for col in numeric_cols:
    if col in df.columns:
        corr = df[col].corr(df['minutes_per_week'])
        print(f"{col}: {corr:.3f}")

print("\n=== STATIONARITY CHECK (sample) ===")
# Check a few students for trends
sample_students = df['anon_student_id'].unique()[:5]
for student in sample_students:
    student_data = df[df['anon_student_id'] == student].sort_values('week_date')
    if len(student_data) >= 10:
        trend = np.polyfit(range(len(student_data)), student_data['minutes_per_week'], 1)[0]
        print(f"Student {student}: {len(student_data)} weeks, trend: {trend:.2f} min/week")
