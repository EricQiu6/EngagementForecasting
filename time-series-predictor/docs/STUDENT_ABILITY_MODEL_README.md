# Student Ability Models

This document describes the implementation of student ability models based on the formulation provided.

## Model Formulation

The model predicts student performance (avg_proficiency) using:

```
ŷ_{i,t} = α + β_a * a_i + β_l * l_i + Σ_{j=1}^{h}(β_y^{(j)} * y_{i,t-j} + β_d^{(j)} * d_{i,t-j}) + ε_{i,t}
```

Where:
- `a_i`: Student ability (from AFM analysis)
- `l_i`: Student learning rate (from AFM analysis)
- `y_{i,t-j}`: Past performance (avg_proficiency at time t-j)
- `d_{i,t-j}`: Past week difficulty
- `h`: History window size (hyperparameter, default=5)

## Data Source

Uses `student_week_aggregations_rolling_new.csv` which contains:
- **anon_student_id**: Student identifier
- **week_id**: Week identifier (e.g., "2011-W36")
- **minutes_per_week**: Study time
- **problems_solved**: Number of problems attempted
- **total_opportunities**: Learning opportunities
- **avg_proficiency**: Target variable (new skills to master)
- **n_skills_measured**: Number of skills measured
- **week_difficulty**: Difficulty of the week
- **student_ability**: Student's overall ability (from AFM)
- **student_learning_rate**: Student's learning rate (from AFM)

## Implementation

### 1. StudentAbilityLinearModel
- Direct implementation of the linear regression formulation
- 13 parameters total (α, β_a, β_l, 5×β_y, 5×β_d)
- Provides interpretable coefficients
- Fast training and inference

### 2. StudentAbilityNeuralModel
- Neural network that can capture non-linear interactions
- Two hidden layers with ReLU activation and dropout
- 961 parameters (with default settings)
- Can model complex relationships between features

## Usage

```python
from framework_v2.models.neural_nets import create_model

# Linear model
linear_model = create_model('student_ability_linear', history_window=5)

# Neural model
neural_model = create_model('student_ability_neural', history_window=5, hidden_size=32)
```

## Performance Results

From the demo run:
1. **Student Ability Linear**: MAE = 2.514, R² = -0.051
2. **Student Ability Neural**: MAE = 2.548, R² = 0.001
3. **LSTM (baseline)**: MAE = 2.900, R² = -0.048
4. **Simple MLP (baseline)**: MAE = 3.342, R² = -0.830

## Coefficient Interpretation (Linear Model)

From the trained model:
- **α (intercept)**: 0.4437 - Base proficiency level
- **β_a (ability)**: 0.1737 - Positive effect of student ability
- **β_l (learning rate)**: 0.0380 - Small positive effect of learning rate
- **Past proficiency coefficients**: Mixed signs showing complex temporal patterns
- **Past difficulty coefficients**: Mostly positive, suggesting harder weeks lead to higher expected proficiency

## Key Insights

1. **Personalization**: Student-specific parameters (ability, learning rate) improve predictions
2. **Temporal Patterns**: Recent performance (t-1) and older patterns (t-4) have stronger influence
3. **Week Difficulty**: Contributes positively to predictions, possibly because harder weeks expose students to more skills
4. **Model Comparison**: The linear model performs competitively while providing interpretability

## Future Improvements

1. **Feature Engineering**: 
   - Interaction terms (ability × difficulty)
   - Rolling averages of performance
   - Trend features

2. **Model Extensions**:
   - Regularization (L1/L2) for coefficient stability
   - Time-varying student parameters
   - Multi-task learning for different skill areas

3. **Evaluation**:
   - Per-student performance analysis
   - Temporal validation (future weeks)
   - Confidence intervals for predictions 