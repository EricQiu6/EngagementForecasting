"""
Prediction Error Analysis Plan - Investigating systematic biases
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 80)
print("PREDICTION ERROR ANALYSIS - INVESTIGATION APPROACHES")
print("=" * 80)

# First, let's understand the target variable distribution
df = pd.read_csv('../data-analysis/student_week_aggregations_rolling_new.csv')
target = 'minutes_per_week'

print("\n1. TARGET VARIABLE DISTRIBUTION ANALYSIS")
print("-" * 50)
print(f"Target statistics:")
print(f"Mean: {df[target].mean():.2f}")
print(f"Std: {df[target].std():.2f}")
print(f"Min: {df[target].min():.2f}")
print(f"Max: {df[target].max():.2f}")
print(f"\nPercentiles:")
percentiles = [0, 10, 25, 50, 75, 90, 95, 99, 100]
for p in percentiles:
    val = df[target].quantile(p/100)
    print(f"{p:3d}%: {val:6.2f}")

print("\n\n2. SUGGESTED ANALYSIS APPROACHES")
print("-" * 50)

approaches = [
    {
        "name": "1. Residual Analysis by Target Range",
        "description": "Bin target values and analyze prediction errors for each bin",
        "methods": [
            "- Create bins: [0, 5), [5, 15), [15, 30), [30, 50), [50+]",
            "- Calculate MAE, bias, and variance for each bin",
            "- Identify if errors are larger for extreme values"
        ],
        "visualizations": ["Residual plots", "Error distribution by bin"]
    },
    {
        "name": "2. Quantile-based Error Analysis",
        "description": "Analyze errors by target value quantiles",
        "methods": [
            "- Split data by target quantiles (e.g., deciles)",
            "- Compare actual vs predicted for each quantile",
            "- Calculate quantile-specific metrics"
        ],
        "visualizations": ["Q-Q plots", "Quantile error heatmap"]
    },
    {
        "name": "3. Error Pattern Detection",
        "description": "Identify systematic over/under prediction patterns",
        "methods": [
            "- Plot predicted vs actual with diagonal reference",
            "- Calculate bias (mean error) vs target value",
            "- Identify heteroscedasticity patterns"
        ],
        "visualizations": ["Scatter plot with LOESS smoothing", "Bias curves"]
    },
    {
        "name": "4. Extreme Value Analysis",
        "description": "Focus on model performance at distribution tails",
        "methods": [
            "- Separate analysis for bottom 10% and top 10%",
            "- Compare error metrics for normal vs extreme ranges",
            "- Analyze feature patterns for mispredicted extremes"
        ],
        "visualizations": ["Tail error distributions", "Feature importance by range"]
    },
    {
        "name": "5. Zero-inflation Analysis",
        "description": "Special handling of zero minutes (no engagement)",
        "methods": [
            "- Separate zero vs non-zero predictions",
            "- Analyze false positive/negative rates for zeros",
            "- Model tendency to predict near-zero values"
        ],
        "visualizations": ["Confusion matrix for zero detection", "Near-zero prediction analysis"]
    },
    {
        "name": "6. Temporal Error Patterns",
        "description": "Check if errors correlate with time patterns",
        "methods": [
            "- Analyze errors by week number",
            "- Check for seasonal patterns in errors",
            "- Identify if certain weeks are harder to predict"
        ],
        "visualizations": ["Error timeline", "Seasonal decomposition"]
    }
]

for approach in approaches:
    print(f"\n{approach['name']}")
    print(f"Description: {approach['description']}")
    print("Methods:")
    for method in approach['methods']:
        print(f"  {method}")
    print("Visualizations:", ", ".join(approach['visualizations']))

print("\n\n3. RECOMMENDED STARTING POINTS")
print("-" * 50)
print("1. Start with residual analysis by target range (most direct)")
print("2. Create predicted vs actual scatter plots for visual inspection")
print("3. Calculate error metrics by target value bins")
print("4. If patterns found, dive deeper into specific ranges")

print("\n\n4. HYPOTHESIS TO TEST")
print("-" * 50)
hypotheses = [
    "H1: Models underpredict high engagement (>50 minutes)",
    "H2: Models struggle with zero/near-zero values",
    "H3: Errors are larger at distribution extremes",
    "H4: Models have systematic bias towards the mean",
    "H5: Certain student types have consistently higher errors"
]
for h in hypotheses:
    print(f"- {h}")

# Create a simple visualization of target distribution
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.hist(df[target], bins=50, alpha=0.7, edgecolor='black')
plt.axvline(df[target].mean(), color='red', linestyle='--', label=f'Mean: {df[target].mean():.1f}')
plt.axvline(df[target].median(), color='green', linestyle='--', label=f'Median: {df[target].median():.1f}')
plt.xlabel('Minutes per Week')
plt.ylabel('Frequency')
plt.title('Target Variable Distribution')
plt.legend()

plt.subplot(1, 3, 2)
plt.hist(df[target], bins=50, alpha=0.7, cumulative=True, density=True, edgecolor='black')
plt.xlabel('Minutes per Week')
plt.ylabel('Cumulative Probability')
plt.title('Cumulative Distribution')
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
# Log scale to see the tail better
plt.hist(df[target][df[target] > 0], bins=50, alpha=0.7, edgecolor='black')
plt.yscale('log')
plt.xlabel('Minutes per Week')
plt.ylabel('Frequency (log scale)')
plt.title('Distribution (Log Scale, Non-zero)')

plt.tight_layout()
plt.savefig('target_distribution_analysis.png', dpi=150, bbox_inches='tight')
print("\n\nTarget distribution saved as 'target_distribution_analysis.png'")

