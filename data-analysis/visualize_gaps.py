import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the processed data
df = pd.read_csv('student_week_aggregations_gap_processed.csv')
gap_summary = pd.read_csv('gap_analysis_summary.csv')

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Temporal Gap Analysis in Student Learning Data', fontsize=16)

# 1. Gap distribution
ax1 = axes[0, 0]
gap_counts = gap_summary['gap_type'].value_counts()
gap_counts.plot(kind='bar', ax=ax1, color=['green', 'yellow', 'orange', 'red'])
ax1.set_title('Distribution of Gap Types')
ax1.set_xlabel('Gap Type')
ax1.set_ylabel('Count')
ax1.tick_params(axis='x', rotation=45)

# 2. Gap length distribution
ax2 = axes[0, 1]
gap_summary['gap_weeks'].hist(bins=20, ax=ax2, color='skyblue', edgecolor='black')
ax2.set_title('Distribution of Gap Lengths (weeks)')
ax2.set_xlabel('Gap Length (weeks)')
ax2.set_ylabel('Frequency')
ax2.axvline(x=2, color='green', linestyle='--', label='Short threshold')
ax2.axvline(x=4, color='orange', linestyle='--', label='Medium threshold')
ax2.axvline(x=8, color='red', linestyle='--', label='Long threshold')
ax2.legend()

# 3. Minutes before vs after gap
ax3 = axes[1, 0]
colors = {'short': 'green', 'medium': 'yellow', 'long': 'orange', 'very_long': 'red'}
for gap_type in gap_summary['gap_type'].unique():
    gap_data = gap_summary[gap_summary['gap_type'] == gap_type]
    ax3.scatter(gap_data['minutes_before'], gap_data['minutes_after'], 
                label=gap_type, alpha=0.6, color=colors[gap_type])
ax3.plot([0, 100], [0, 100], 'k--', alpha=0.3)  # y=x line
ax3.set_xlabel('Minutes Before Gap')
ax3.set_ylabel('Minutes After Gap')
ax3.set_title('Engagement Before vs After Gaps')
ax3.legend()
ax3.set_xlim(0, 100)
ax3.set_ylim(0, 100)

# 4. Recovery rates by gap type
ax4 = axes[1, 1]
recovery_data = []
for _, gap in gap_summary.iterrows():
    if gap['minutes_before'] > 0:
        recovery_rate = gap['minutes_after'] / gap['minutes_before']
        recovery_data.append({
            'gap_type': gap['gap_type'],
            'recovery_rate': min(recovery_rate, 3)  # Cap at 300% for visualization
        })

recovery_df = pd.DataFrame(recovery_data)
recovery_df.boxplot(column='recovery_rate', by='gap_type', ax=ax4)
ax4.set_title('Recovery Rates by Gap Type')
ax4.set_xlabel('Gap Type')
ax4.set_ylabel('Recovery Rate (post/pre)')
ax4.axhline(y=1, color='red', linestyle='--', alpha=0.5)
plt.sca(ax4)
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('gap_analysis_visualization.png', dpi=300)
plt.show()

# Additional analysis: Students with most gaps
print("\n=== Students with Most Gaps ===")
student_gap_counts = gap_summary['student'].value_counts().head(10)
print(student_gap_counts)

# Gap patterns over time
print("\n=== Gap Patterns Over Time ===")
gap_summary['gap_start_date'] = pd.to_datetime(gap_summary['gap_start'] + '-1', format='%Y-W%W-%w')
gap_summary['month'] = gap_summary['gap_start_date'].dt.to_period('M')
monthly_gaps = gap_summary.groupby('month').size()
print(monthly_gaps.head(10))

# Create timeline visualization for sample students
fig2, ax = plt.subplots(figsize=(15, 8))

# Select 5 students with different gap patterns
sample_students = gap_summary.groupby('student')['gap_weeks'].sum().nlargest(5).index

for i, student in enumerate(sample_students):
    student_data = df[df['anon_student_id'] == student]
    
    # Plot active weeks
    y_pos = i
    for _, week in student_data.iterrows():
        color = 'blue' if week['gap_weeks'] == 0 else 'red'
        ax.scatter(week['week_date'], y_pos, c=color, s=50, alpha=0.7)
    
    # Connect with lines showing gaps
    dates = pd.to_datetime(student_data['week_date'])
    ax.plot(dates, [y_pos] * len(dates), 'k-', alpha=0.3, linewidth=0.5)

ax.set_yticks(range(len(sample_students)))
ax.set_yticklabels(sample_students)
ax.set_xlabel('Date')
ax.set_title('Timeline of Student Activity (Blue=Active, Red=Return from Gap)')
ax.grid(True, axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('student_timelines.png', dpi=300)
plt.show() 