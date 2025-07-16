# Understanding Additional Features for Goal-Setting Recommendation Algorithm

## Context
Your current model uses a minimal feature set focused primarily on individual student performance (`proficient` - skills mastered per week). The comment suggests adding richer features based on:

1. **Past week class/individual performance (mean etc.)**
2. **Class averages**

## Current Feature Limitations

Based on your codebase analysis, your current features are:
- Individual student's `proficient` values (skills mastered per week)
- Lag features (student's past performance: lag1, lag2, lag3, lag4, lag5)
- Basic rolling statistics on individual level

## Recommended Additional Features

### 1. **Class-Level Performance Features**

#### Class Contextual Features:
```python
# Weekly class statistics
- class_mean_proficient_current_week    # How the class is performing this week
- class_median_proficient_current_week  # Class median performance
- class_std_proficient_current_week     # Class performance variability
- class_percentile_rank                 # Student's rank within class (0-100)

# Historical class trends
- class_mean_proficient_lag1           # Class average last week
- class_mean_proficient_lag2           # Class average 2 weeks ago
- class_improvement_trend              # Is class performance improving/declining?
```

#### Peer Comparison Features:
```python
# Relative performance indicators
- performance_vs_class_mean            # (student_score - class_mean)
- performance_percentile_in_class      # Where student ranks in class distribution
- above_class_median                   # Boolean: performing above class median
- class_performance_gap                # How far student is from class average
```

### 2. **Enhanced Individual Performance Features**

#### Rolling Statistics (Past Week Performance):
```python
# Individual rolling statistics
- student_mean_last_3_weeks           # Student's 3-week rolling average
- student_mean_last_5_weeks           # Student's 5-week rolling average
- student_median_last_3_weeks         # More robust to outliers
- student_max_last_3_weeks            # Best recent performance
- student_min_last_3_weeks            # Worst recent performance
- student_std_last_3_weeks            # Consistency of performance

# Trend indicators
- student_improvement_rate            # Linear trend slope over past weeks
- student_momentum                    # Recent performance vs historical average
- weeks_since_peak_performance        # How long since best week?
- consecutive_improvement_weeks       # Streak of improving performance
```

#### Performance Patterns:
```python
# Behavioral patterns
- student_consistency_score           # How consistent is student week-to-week?
- student_volatility                  # Standard deviation of recent performance
- student_recovery_ability            # How quickly does student bounce back from low weeks?
```

### 3. **Temporal and Cohort Features**

#### Time-Based Features:
```python
# Calendar effects
- week_in_term                        # Week 1, 2, 3... of academic term
- weeks_since_start                   # How long student has been active
- is_early_term                       # Boolean: first few weeks
- is_mid_term                         # Boolean: middle weeks
- is_late_term                        # Boolean: final weeks
```

#### Cohort Analysis Features:
```python
# Student grouping
- starting_ability_quartile           # Which quartile was student in week 1?
- improvement_trajectory_cluster      # Which improvement pattern does student follow?
- similar_students_avg_performance    # Average of students with similar starting ability
```

### 4. **Interaction Features**

#### Student-Class Interactions:
```python
# Complex relationships
- student_vs_class_trend_divergence   # Is student trending differently than class?
- catch_up_potential                  # How much could student improve to reach class avg?
- class_influence_factor              # How much does class performance predict individual?
```

## Implementation Strategy

### Phase 1: Basic Class Features
Start with simple class statistics:
```python
def add_class_features(df):
    # Group by week and calculate class statistics
    weekly_class_stats = df.groupby('week')['proficient'].agg([
        'mean', 'median', 'std', 'count'
    ]).reset_index()
    
    # Merge back to individual student data
    df = df.merge(weekly_class_stats, on='week', suffixes=('', '_class'))
    
    # Calculate relative performance
    df['performance_vs_class_mean'] = df['proficient'] - df['mean_class']
    df['performance_percentile'] = df.groupby('week')['proficient'].rank(pct=True)
    
    return df
```

### Phase 2: Rolling Window Features
Add temporal patterns:
```python
def add_rolling_features(df, windows=[3, 5]):
    for window in windows:
        # Individual rolling stats
        df[f'student_mean_last_{window}_weeks'] = (
            df.groupby('name')['proficient']
            .rolling(window=window, min_periods=1)
            .mean()
            .reset_index(drop=True)
        )
        
        # Class rolling stats
        df[f'class_mean_last_{window}_weeks'] = (
            df.groupby('week')['proficient'].transform('mean')
            .rolling(window=window, min_periods=1)
            .mean()
        )
    
    return df
```

### Phase 3: Advanced Pattern Features
Add behavioral and trend analysis:
```python
def add_pattern_features(df):
    # Calculate improvement trends
    df['student_improvement_trend'] = (
        df.groupby('name')['proficient']
        .rolling(window=4, min_periods=2)
        .apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
        .reset_index(drop=True)
    )
    
    # Consistency measures
    df['student_consistency'] = (
        1 / (1 + df.groupby('name')['proficient']
             .rolling(window=5, min_periods=2)
             .std()
             .reset_index(drop=True))
    )
    
    return df
```

## Expected Benefits

1. **Better Context**: Understanding how student performs relative to peers
2. **Social Learning**: Incorporating peer influence and class dynamics
3. **Adaptive Goals**: Setting realistic goals based on class performance standards
4. **Trend Recognition**: Identifying students who need intervention vs. those on track
5. **Personalization**: Accounting for different learning patterns and trajectories

## Implementation Priority

1. **High Priority**: Class mean/median features, performance percentiles
2. **Medium Priority**: Rolling windows, trend features
3. **Low Priority**: Complex interaction features, clustering-based features

This approach will give you much richer feature representation while maintaining interpretability for goal recommendation.