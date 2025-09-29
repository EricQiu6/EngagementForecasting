# Student ID Integration Guide

This guide explains how to incorporate student ID as a feature across different model types in the goal-setting recommendation algorithm framework.

## Table of Contents

1. [Overview](#overview)
2. [Why Student ID Matters](#why-student-id-matters)
3. [Integration Approaches](#integration-approaches)
4. [Model-Specific Recommendations](#model-specific-recommendations)
5. [Cross-Validation Considerations](#cross-validation-considerations)
6. [Implementation Examples](#implementation-examples)
7. [Best Practices](#best-practices)

## Overview

Student ID integration allows models to capture individual student differences, leading to more personalized and accurate predictions. The framework provides multiple strategies for incorporating student IDs based on the model type and data characteristics.

## Why Student ID Matters

### Student Heterogeneity

- **Individual Differences**: Students have different learning abilities, engagement patterns, and baseline performance levels
- **Temporal Consistency**: Individual students tend to be consistent in their behavior over time
- **Variance Decomposition**: Often 20-40% of variance in student performance is between students rather than within students

### When Student ID is Beneficial

- **High Intraclass Correlation (ICC > 0.2)**: When students are consistently different from each other
- **Sufficient Data per Student**: When you have multiple observations per student
- **Temporal Consistency**: When students maintain similar patterns over time

## Integration Approaches

### 1. **None** (Baseline)

Don't use student ID as a feature.

**Use when:**

- Student effects are minimal (ICC < 0.1)
- Very few observations per student
- Baseline models that don't benefit from individual effects

### 2. **One-Hot Encoding**

Create binary features for each student (1 if student X, 0 otherwise).

```python
# Example: student_A, student_B, student_C columns
df_encoded = pd.get_dummies(df['student_id'], prefix='student')
```

**Pros:**

- Simple and interpretable
- Direct capture of individual effects
- Works well with tree-based models

**Cons:**

- High dimensionality (one feature per student)
- Doesn't generalize to new students
- Can cause overfitting with many students

**Use when:**

- Small number of students (< 100)
- Tree-based models
- Sufficient regularization for linear models

### 3. **Target Encoding**

Replace student ID with student-specific statistics.

```python
# Example features created:
# - student_target_mean: Historical average for this student
# - student_target_std: Historical standard deviation
# - student_target_count: Number of observations
# - student_target_median: Historical median
# - student_target_deviation: Current value - historical mean
```

**Pros:**

- Low dimensionality (few features)
- Captures student-specific patterns
- Generalizes to new students (using global statistics)
- Works with any model type

**Cons:**

- Risk of data leakage if not implemented carefully
- May not capture complex interactions
- Requires careful cross-validation

**Use when:**

- Large number of students (> 100)
- Any model type
- Need to generalize to new students

### 4. **Embeddings** (Neural Networks)

Learn dense vector representations of students.

```python
# Example: Map student IDs to integers, then use embedding layer
student_mapping = {student: idx for idx, student in enumerate(unique_students)}
df['student_id_numeric'] = df['student_id'].map(student_mapping)
```

**Pros:**

- Learns compact representations
- Can capture complex student patterns
- Handles new students with default embeddings

**Cons:**

- Only works with neural networks
- Requires sufficient data per student
- Less interpretable

**Use when:**

- Neural network models
- Sufficient data per student
- Many students (> 100)

### 5. **Mixed Effects**

Treat student ID as random effects in hierarchical models.

**Pros:**

- Statistically principled approach
- Handles hierarchical data structure naturally
- Provides uncertainty estimates
- Works with limited data per student

**Cons:**

- Computationally expensive
- May not capture complex non-linear patterns
- Requires specialized implementation

**Use when:**

- Hierarchical/multilevel modeling
- Want to model student heterogeneity explicitly
- Statistical inference is important

## Model-Specific Recommendations

### Linear Models (Ridge, Lasso, Linear Regression)

- **Few students (< 50)**: One-hot encoding with regularization
- **Many students (> 50)**: Target encoding
- **Rationale**: Linear models struggle with high-dimensional sparse features

### Tree-Based Models (Random Forest, XGBoost, Gradient Boosting)

- **Few students (< 500)**: One-hot encoding
- **Many students (> 500)**: Target encoding
- **Rationale**: Tree models handle sparse features well but can overfit with too many

### Neural Networks (MLP, LSTM, Transformer)

- **Few students (< 100)**: One-hot encoding or target encoding
- **Many students (> 100)**: Embeddings
- **Rationale**: Neural networks can learn complex representations through embeddings

### Mixed Effects Models

- **Always**: Mixed effects approach
- **Rationale**: Designed specifically for hierarchical data

### Baseline Models (Mean, Median)

- **Always**: None
- **Rationale**: Simple baselines don't benefit from individual effects

## Cross-Validation Considerations

### Current Approach: TimeSeriesSplit

- **Splits by time**: Older weeks for training, newer weeks for testing
- **Students appear in both train and test**
- **Good for**: Predicting future performance of known students

### Alternative: GroupKFold

- **Splits by student**: Some students in train, others in test
- **Tests generalization to new students**
- **Good for**: Evaluating generalization capabilities

### Recommendation

- **Use TimeSeriesSplit** for temporal prediction tasks (current approach is correct)
- **Use GroupKFold** for student generalization experiments
- **Consider both** for comprehensive evaluation

### Data Leakage Prevention

When using target encoding:

- **Use only past data** for each prediction
- **Implement rolling windows** to update statistics over time
- **Stratify folds** to ensure balanced student representation

## Implementation Examples

### 1. Quick Analysis

```python
from src.framework.utils.student_id_utils import StudentAwareModelRecommender

# Analyze your dataset
recommender = StudentAwareModelRecommender(
    df=df,
    student_column='anon_student_id',
    target_column='minutes_per_week',
    time_column='week_id'
)

# Get recommendations
recommender.print_summary()
```

### 2. Create Student-Aware Schema

```python
from src.framework.core.schema import get_schema
from src.framework.utils.student_id_utils import create_student_aware_schema

# Base schema
base_schema = get_schema('time_goal_extended')

# Create student-aware schema for random forest
forest_schema = create_student_aware_schema(
    base_schema,
    model_type='forest',
    data_path='your_data.csv'
)
```

### 3. Compare Models

```python
from sklearn.ensemble import RandomForestRegressor
from src.framework.adapters.sklearn_adapter import SchemaBasedSKLearnAdapter

# Model without student ID
model_baseline = SchemaBasedSKLearnAdapter(
    sklearn_model=RandomForestRegressor(),
    schema=base_schema,
    lag_window=5
)

# Model with student ID
model_with_student = SchemaBasedSKLearnAdapter(
    sklearn_model=RandomForestRegressor(),
    schema=forest_schema,
    lag_window=5
)

# Cross-validate both and compare
```

### 4. Neural Network with Embeddings

```python
import torch.nn as nn

class StudentEmbeddingModel(nn.Module):
    def __init__(self, n_students, embedding_dim, other_features):
        super().__init__()
        self.student_embedding = nn.Embedding(n_students, embedding_dim)
        self.main_net = nn.Linear(embedding_dim + other_features, 1)

    def forward(self, student_ids, other_features):
        student_emb = self.student_embedding(student_ids)
        combined = torch.cat([student_emb, other_features], dim=-1)
        return self.main_net(combined)
```

## Best Practices

### 1. Data Analysis First

- Always analyze your dataset before choosing an approach
- Check intraclass correlation (ICC) to assess student heterogeneity
- Examine temporal consistency within students

### 2. Start Simple

- Begin with target encoding for most models
- Use one-hot encoding only with few students
- Reserve embeddings for neural networks with many students

### 3. Prevent Data Leakage

- Use only past data for target encoding
- Implement proper cross-validation
- Be careful with rolling window statistics

### 4. Monitor Performance

- Compare models with and without student ID
- Track both overall performance and student-specific metrics
- Consider computational cost vs. benefit

### 5. Generalization Testing

- Test on new students if relevant to your use case
- Use appropriate cross-validation strategies
- Consider both temporal and student-level generalization

## Decision Tree

```
Do you have student ID data?
├── No → Use baseline approach
└── Yes → Analyze student heterogeneity
    ├── Low ICC (< 0.1) → Consider skipping student ID
    └── High ICC (> 0.2) → Choose strategy based on model:
        ├── Linear models → Target encoding (if many students) or One-hot (if few)
        ├── Tree models → One-hot (if < 500 students) or Target encoding
        ├── Neural networks → Embeddings (if many students) or Target encoding
        └── Mixed effects → Use mixed effects approach
```

## Expected Benefits

Based on the analysis of your dataset, incorporating student ID features can provide:

- **5-15% improvement** in prediction accuracy for models with high student heterogeneity
- **Better personalization** for individual students
- **More stable predictions** over time for the same student
- **Reduced bias** in predictions across different student populations

The exact benefit depends on your specific dataset characteristics and the chosen integration approach.

## Troubleshooting

### Poor Performance with Student ID

- Check for data leakage in target encoding
- Verify cross-validation setup
- Consider if you have enough data per student
- Try different integration strategies

### Overfitting Issues

- Increase regularization for linear models
- Use target encoding instead of one-hot
- Reduce model complexity
- Ensure proper cross-validation

### Generalization Problems

- Test with GroupKFold cross-validation
- Use target encoding with global fallbacks
- Consider mixed effects models for better generalization

## Conclusion

Student ID integration is a powerful technique for improving model performance in educational data. The key is choosing the right approach based on your model type, data characteristics, and generalization requirements. The framework provides tools to analyze your data, choose appropriate strategies, and implement them seamlessly across different model types.
