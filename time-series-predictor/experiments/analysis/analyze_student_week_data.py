import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Load the dataset
print("Loading student_week_aggregations_rolling.csv...")
df = pd.read_csv('../data-analysis/student_week_aggregations_rolling.csv')

print("\n=== Dataset Overview ===")
print(f"Shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nData types:\n{df.dtypes}")

print("\n=== Missing Values ===")
print(df.isnull().sum())

print("\n=== Basic Statistics ===")
print(df.describe())

# Parse week_id to get temporal information
print("\n=== Temporal Analysis ===")
df['week_date'] = pd.to_datetime(df['week_id'] + '-1', format='%Y-W%W-%w')
df['year'] = df['week_date'].dt.year
df['week_num'] = df['week_date'].dt.isocalendar().week

# Check date range
print(f"Date range: {df['week_date'].min()} to {df['week_date'].max()}")

# Student-level analysis
print("\n=== Student Analysis ===")
student_stats = df.groupby('anon_student_id').agg({
    'week_id': 'count',
    'minutes_per_week': ['mean', 'std', 'sum'],
    'problems_solved': ['mean', 'sum'],
    'avg_proficiency': 'mean'
}).round(2)
print(f"Number of unique students: {df['anon_student_id'].nunique()}")
print(f"Average weeks per student: {student_stats['week_id']['count'].mean():.1f}")
print(f"Min weeks per student: {student_stats['week_id']['count'].min()}")
print(f"Max weeks per student: {student_stats['week_id']['count'].max()}")

# Check data continuity
print("\n=== Data Continuity Check ===")
# Sample a few students to check their time series continuity
sample_students = df['anon_student_id'].unique()[:5]
for student in sample_students:
    student_data = df[df['anon_student_id'] == student].sort_values('week_date')
    weeks = student_data['week_date'].tolist()
    gaps = []
    for i in range(1, len(weeks)):
        diff = (weeks[i] - weeks[i-1]).days
        if diff > 7:
            gaps.append((weeks[i-1].strftime('%Y-%m-%d'), weeks[i].strftime('%Y-%m-%d'), diff//7))
    
    print(f"\nStudent {student}:")
    print(f"  Total weeks: {len(weeks)}")
    print(f"  Date range: {weeks[0].strftime('%Y-%m-%d')} to {weeks[-1].strftime('%Y-%m-%d')}")
    if gaps:
        print(f"  Gaps in data: {len(gaps)}")
        for start, end, gap_weeks in gaps[:3]:  # Show first 3 gaps
            print(f"    {start} to {end} ({gap_weeks} weeks)")

# Visualizations
print("\n=== Creating Visualizations ===")

# 1. Distribution plots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Minutes per week distribution
axes[0, 0].hist(df['minutes_per_week'], bins=50, edgecolor='black')
axes[0, 0].set_title('Distribution of Minutes per Week')
axes[0, 0].set_xlabel('Minutes')
axes[0, 0].set_ylabel('Frequency')

# Problems solved distribution
axes[0, 1].hist(df['problems_solved'], bins=50, edgecolor='black')
axes[0, 1].set_title('Distribution of Problems Solved per Week')
axes[0, 1].set_xlabel('Problems Solved')
axes[0, 1].set_ylabel('Frequency')

# Average proficiency distribution
axes[1, 0].hist(df['avg_proficiency'], bins=50, edgecolor='black')
axes[1, 0].set_title('Distribution of Average Proficiency')
axes[1, 0].set_xlabel('Average Proficiency')
axes[1, 0].set_ylabel('Frequency')

# Weekly activity over time
weekly_activity = df.groupby('week_date')['anon_student_id'].count()
axes[1, 1].plot(weekly_activity.index, weekly_activity.values)
axes[1, 1].set_title('Number of Active Students per Week')
axes[1, 1].set_xlabel('Week')
axes[1, 1].set_ylabel('Number of Students')
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('student_week_data_distributions.png', dpi=300, bbox_inches='tight')
print("Saved distribution plots to student_week_data_distributions.png")

# 2. Sample student time series
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# Select a student with good data coverage
student_weeks = df.groupby('anon_student_id')['week_id'].count()
good_students = student_weeks[student_weeks > 30].index.tolist()[:3]

for student in good_students[:1]:  # Plot one student
    student_data = df[df['anon_student_id'] == student].sort_values('week_date')
    
    # Minutes per week
    axes[0].plot(student_data['week_date'], student_data['minutes_per_week'], marker='o', label=student)
    axes[0].set_title(f'Time Series for Student {student}')
    axes[0].set_ylabel('Minutes per Week')
    axes[0].grid(True, alpha=0.3)
    
    # Problems solved
    axes[1].plot(student_data['week_date'], student_data['problems_solved'], marker='o', color='orange')
    axes[1].set_ylabel('Problems Solved')
    axes[1].grid(True, alpha=0.3)
    
    # Average proficiency
    axes[2].plot(student_data['week_date'], student_data['avg_proficiency'], marker='o', color='green')
    axes[2].set_ylabel('Average Proficiency')
    axes[2].set_xlabel('Week')
    axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('student_time_series_example.png', dpi=300, bbox_inches='tight')
print("Saved time series example to student_time_series_example.png")

# 3. Correlation analysis
print("\n=== Correlation Analysis ===")
numeric_cols = ['minutes_per_week', 'problems_solved', 'total_opportunities', 'avg_proficiency']
corr_matrix = df[numeric_cols].corr()
print(corr_matrix)

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix of Student Metrics')
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
print("Saved correlation matrix to correlation_matrix.png")

# Predictive modeling considerations
print("\n=== Predictive Modeling Considerations ===")
print("\n1. TARGET VARIABLES:")
print("   - Next week's minutes_per_week (engagement prediction)")
print("   - Next week's problems_solved (activity prediction)")
print("   - Next week's avg_proficiency (performance prediction)")
print("   - Student dropout (binary: will student be active next week?)")

print("\n2. FEATURES:")
print("   - Historical values (lagged features) of all metrics")
print("   - Rolling statistics (mean, std, trend) over past N weeks")
print("   - Week-of-year effects (seasonality)")
print("   - Student-specific features (overall performance level)")

print("\n3. CHALLENGES:")
print("   - Irregular time series (gaps in student activity)")
print("   - Different students have different active periods")
print("   - Need to handle missing weeks appropriately")
print("   - Cold start problem for new students")

print("\n4. MODELING APPROACHES:")
print("   - Time series models per student (if enough data)")
print("   - Panel data models (all students together)")
print("   - Sequence models (LSTM, GRU) for variable-length sequences")
print("   - Traditional ML with engineered features")

# Create a sample dataset for predictive modeling
print("\n=== Creating Sample Predictive Dataset ===")

# Select students with at least 20 weeks of data
active_students = student_weeks[student_weeks >= 20].index.tolist()
print(f"Students with >= 20 weeks of data: {len(active_students)}")

# Create lagged features for one student as an example
if active_students:
    sample_student = active_students[0]
    student_df = df[df['anon_student_id'] == sample_student].sort_values('week_date').copy()
    
    # Create lagged features
    lag_features = ['minutes_per_week', 'problems_solved', 'avg_proficiency']
    for col in lag_features:
        for lag in [1, 2, 3, 4]:  # Past 4 weeks
            student_df[f'{col}_lag{lag}'] = student_df[col].shift(lag)
    
    # Create rolling statistics
    for col in lag_features:
        student_df[f'{col}_rolling_mean_4w'] = student_df[col].rolling(window=4).mean()
        student_df[f'{col}_rolling_std_4w'] = student_df[col].rolling(window=4).std()
    
    # Drop rows with NaN values (due to lagging)
    student_df_clean = student_df.dropna()
    
    print(f"\nSample predictive dataset for student {sample_student}:")
    print(f"Original rows: {len(student_df)}")
    print(f"Rows after feature engineering: {len(student_df_clean)}")
    print(f"\nFeature columns created:")
    feature_cols = [col for col in student_df_clean.columns if 'lag' in col or 'rolling' in col]
    for col in feature_cols:
        print(f"  - {col}")
    
    # Save sample
    student_df_clean.to_csv('sample_predictive_dataset.csv', index=False)
    print("\nSaved sample predictive dataset to sample_predictive_dataset.csv")

print("\n=== Analysis Complete ===") 