"""
Student Week Predictive Modeling
================================

This script demonstrates how to use the student_week_aggregations_rolling dataset 
for various predictive modeling tasks using the existing framework.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import sys
import os

# Add framework to path
sys.path.append(os.path.dirname(__file__))

# Import sklearn models
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

# Create a comprehensive model factory
def create_model(model_type: str, **kwargs):
    """
    Factory function to create both sklearn and neural network models.
    
    Args:
        model_type: Type of model 
        **kwargs: Model-specific parameters
        
    Returns:
        Model instance
    """
    # Traditional ML models
    if model_type == 'linear':
        return LinearRegression()
    elif model_type == 'ridge':
        return Ridge(alpha=kwargs.get('alpha', 1.0))
    elif model_type == 'lasso':
        return Lasso(alpha=kwargs.get('alpha', 1.0))
    elif model_type == 'random_forest':
        return RandomForestRegressor(
            n_estimators=kwargs.get('n_estimators', 100),
            max_depth=kwargs.get('max_depth', None),
            random_state=kwargs.get('random_state', 42)
        )
    elif model_type == 'svr':
        return SVR(kernel=kwargs.get('kernel', 'rbf'))
    elif model_type == 'xgboost' and HAS_XGBOOST:
        return xgb.XGBRegressor(
            n_estimators=kwargs.get('n_estimators', 100),
            max_depth=kwargs.get('max_depth', 6),
            learning_rate=kwargs.get('learning_rate', 0.1),
            random_state=kwargs.get('random_state', 42)
        )
    else:
        # Try to import neural network models
        try:
            from src.framework.models.neural_nets import create_model as create_nn_model
            return create_nn_model(model_type, **kwargs)
        except Exception:
            raise ValueError(f"Unknown model type: {model_type}")

# Import metrics from sklearn
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math

# Simple evaluator function
def evaluate_predictions(y_true, y_pred):
    """Simple evaluation function for time series predictions"""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    
    # Calculate SMAPE
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    smape = np.mean(np.divide(np.abs(y_true - y_pred), denominator, 
                              out=np.zeros_like(denominator), 
                              where=denominator!=0)) * 100
    
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'smape': smape
    }

# Load the dataset
print("Loading student_week_aggregations_rolling.csv...")
df = pd.read_csv('../../../data-analysis/student_week_aggregations_rolling_new.csv')

# Parse week dates
df['week_date'] = pd.to_datetime(df['week_id'] + '-1', format='%Y-W%W-%w')
df = df.sort_values(['anon_student_id', 'week_date'])

print(f"Dataset shape: {df.shape}")
print(f"Number of students: {df['anon_student_id'].nunique()}")
print(f"Date range: {df['week_date'].min()} to {df['week_date'].max()}")

# ========================================================================
# APPROACH 1: Panel Data Model (All Students Together)
# ========================================================================

print("\n" + "="*70)
print("APPROACH 1: Panel Data Model (All Students Together)")
print("="*70)

# Create lagged features for all students
lag_features = ['minutes_per_week', 'problems_solved', 'avg_proficiency', 'total_opportunities']
lags = [1, 2, 3, 4]

print("\nCreating lagged features...")
df_panel = df.copy()

# Create lagged features per student
for col in lag_features:
    for lag in lags:
        df_panel[f'{col}_lag{lag}'] = df_panel.groupby('anon_student_id')[col].shift(lag)

# Create rolling statistics
window_size = 4
for col in lag_features:
    df_panel[f'{col}_rolling_mean'] = df_panel.groupby('anon_student_id')[col].transform(
        lambda x: x.rolling(window=window_size, min_periods=1).mean()
    ).shift(1)  # Shift to avoid data leakage
    
    df_panel[f'{col}_rolling_std'] = df_panel.groupby('anon_student_id')[col].transform(
        lambda x: x.rolling(window=window_size, min_periods=2).std()
    ).shift(1)

# Add temporal features
df_panel['week_of_year'] = df_panel['week_date'].dt.isocalendar().week
df_panel['month'] = df_panel['week_date'].dt.month

# Drop rows with NaN values in lagged features
df_panel_clean = df_panel.dropna(subset=[f'{col}_lag{lag}' for col in lag_features for lag in lags])

print(f"Clean dataset shape: {df_panel_clean.shape}")
print(f"Students with sufficient data: {df_panel_clean['anon_student_id'].nunique()}")

# Prepare features and target
feature_cols = [col for col in df_panel_clean.columns if any(x in col for x in ['lag', 'rolling', 'week_of_year', 'month'])]
X = df_panel_clean[feature_cols].values
y = df_panel_clean['minutes_per_week'].values  # Predicting next week's engagement

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Split data temporally (last 20% of time as test)
split_date = df_panel_clean['week_date'].quantile(0.8)
train_mask = df_panel_clean['week_date'] < split_date
X_train, X_test = X[train_mask], X[~train_mask]
y_train, y_test = y[train_mask], y[~train_mask]

print(f"\nTrain set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train and evaluate different models
models_to_test = ['linear', 'ridge', 'random_forest']
if HAS_XGBOOST:
    models_to_test.append('xgboost')
results = {}

print("\n" + "-"*50)
print("Training models on panel data...")
print("-"*50)

for model_name in models_to_test:
    print(f"\nTraining {model_name}...")
    
    # Create and train model
    model = create_model(model_name, input_dim=X_train_scaled.shape[1])
    
    if hasattr(model, 'fit'):
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        # For neural networks (if we add them later)
        pass
    
    # Evaluate
    metrics = evaluate_predictions(y_test, y_pred)
    results[model_name] = metrics
    
    print(f"{model_name} - MAE: {metrics['mae']:.2f}, RMSE: {metrics['rmse']:.2f}")

# ========================================================================
# APPROACH 2: Individual Student Models (For Students with Enough Data)
# ========================================================================

print("\n" + "="*70)
print("APPROACH 2: Individual Student Models")
print("="*70)

# Select students with sufficient data
min_weeks = 30
student_counts = df.groupby('anon_student_id').size()
active_students = student_counts[student_counts >= min_weeks].index.tolist()

print(f"\nStudents with >= {min_weeks} weeks of data: {len(active_students)}")

# Example: Train model for one student
if active_students:
    sample_student = active_students[0]
    print(f"\nTraining model for student: {sample_student}")
    
    # Get student data
    student_df = df_panel_clean[df_panel_clean['anon_student_id'] == sample_student].copy()
    
    # Prepare features and target
    X_student = student_df[feature_cols].values
    y_student = student_df['minutes_per_week'].values
    
    # Split data (80/20)
    split_idx = int(len(X_student) * 0.8)
    X_train_s, X_test_s = X_student[:split_idx], X_student[split_idx:]
    y_train_s, y_test_s = y_student[:split_idx], y_student[split_idx:]
    
    print(f"Student train size: {len(X_train_s)}, test size: {len(X_test_s)}")
    
    if len(X_train_s) > 5 and len(X_test_s) > 0:  # Ensure we have enough data
        # Scale
        scaler_s = StandardScaler()
        X_train_s_scaled = scaler_s.fit_transform(X_train_s)
        X_test_s_scaled = scaler_s.transform(X_test_s)
        
        # Train model
        model_s = create_model('ridge', input_dim=X_train_s_scaled.shape[1])
        model_s.fit(X_train_s_scaled, y_train_s)
        y_pred_s = model_s.predict(X_test_s_scaled)
        
        # Evaluate
        metrics_s = evaluate_predictions(y_test_s, y_pred_s)
        
        print(f"Individual model - MAE: {metrics_s['mae']:.2f}, RMSE: {metrics_s['rmse']:.2f}")
        
        # Plot predictions
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(y_test_s)), y_test_s, label='Actual', marker='o')
        plt.plot(range(len(y_pred_s)), y_pred_s, label='Predicted', marker='s', alpha=0.7)
        plt.xlabel('Week')
        plt.ylabel('Minutes per Week')
        plt.title(f'Predictions for Student {sample_student}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('../outputs/individual_student_predictions.png', dpi=300)
        print("Saved individual student predictions plot to outputs/")

# ========================================================================
# APPROACH 3: Sequence-to-Sequence Prediction (Using Our Framework)
# ========================================================================

print("\n" + "="*70)
print("APPROACH 3: Sequence-to-Sequence Prediction")
print("="*70)

def create_sequences(df, sequence_length=8, prediction_horizon=4):
    """Create sequences for sequence-to-sequence prediction"""
    sequences = []
    targets = []
    student_ids = []
    
    for student_id in df['anon_student_id'].unique():
        student_df = df[df['anon_student_id'] == student_id].sort_values('week_date')
        
        if len(student_df) >= sequence_length + prediction_horizon:
            # Use multiple features as input
            features = ['minutes_per_week', 'problems_solved', 'avg_proficiency']
            student_data = student_df[features].fillna(0).values
            
            for i in range(len(student_data) - sequence_length - prediction_horizon + 1):
                seq = student_data[i:i+sequence_length]
                target = student_data[i+sequence_length:i+sequence_length+prediction_horizon, 0]  # Predict minutes_per_week
                
                sequences.append(seq)
                targets.append(target)
                student_ids.append(student_id)
    
    return np.array(sequences), np.array(targets), student_ids

# Create sequences
print("\nCreating sequences...")
X_seq, y_seq, seq_student_ids = create_sequences(df, sequence_length=8, prediction_horizon=4)
print(f"Number of sequences created: {len(X_seq)}")
print(f"Sequence shape: {X_seq.shape}")
print(f"Target shape: {y_seq.shape}")

if len(X_seq) > 0:
    # Split data
    n_train = int(len(X_seq) * 0.8)
    X_train_seq, X_test_seq = X_seq[:n_train], X_seq[n_train:]
    y_train_seq, y_test_seq = y_seq[:n_train], y_seq[n_train:]
    
    # For demonstration, we'll use the first feature (minutes_per_week) only
    # to match our existing LSTM framework
    X_train_seq_1d = X_train_seq[:, :, 0:1]  # Shape: (samples, seq_len, 1)
    X_test_seq_1d = X_test_seq[:, :, 0:1]
    
    # Use only the last value of the target sequence for evaluation
    y_train_last = y_train_seq[:, -1]
    y_test_last = y_test_seq[:, -1]
    
    print(f"\nTraining LSTM on sequences...")
    print(f"Train sequences: {len(X_train_seq_1d)}, Test sequences: {len(X_test_seq_1d)}")
    
    # Note: To use LSTM, you would need to uncomment and adapt this:
    # lstm_model = create_model('lstm', 
    #                          input_dim=1, 
    #                          sequence_length=8,
    #                          hidden_size=32)
    # lstm_model.fit(X_train_seq_1d, y_train_last, epochs=50, batch_size=32, verbose=0)
    # y_pred_lstm = lstm_model.predict(X_test_seq_1d)
    # 
    # evaluator = TimeSeriesEvaluator()
    # metrics_lstm = evaluator.evaluate(y_test_last, y_pred_lstm)
    # print(f"LSTM - MAE: {metrics_lstm['mae']:.2f}, RMSE: {metrics_lstm['rmse']:.2f}")

# ========================================================================
# APPROACH 4: Dropout Prediction (Binary Classification)
# ========================================================================

print("\n" + "="*70)
print("APPROACH 4: Dropout Prediction")
print("="*70)

# Create dropout labels (1 if student doesn't appear next week, 0 otherwise)
df_dropout = df.copy()
df_dropout['next_week_exists'] = df_dropout.groupby('anon_student_id')['week_date'].transform(
    lambda x: x.shift(-1).notna()
)
df_dropout['dropout'] = (~df_dropout['next_week_exists']).astype(int)

# Remove last week for each student (can't determine dropout)
df_dropout = df_dropout[df_dropout['next_week_exists'].notna()]

print(f"\nDropout rate: {df_dropout['dropout'].mean():.2%}")

# Create features for dropout prediction
df_dropout_features = df_panel_clean.merge(
    df_dropout[['anon_student_id', 'week_date', 'dropout']], 
    on=['anon_student_id', 'week_date'],
    how='inner'
)

X_dropout = df_dropout_features[feature_cols].values
y_dropout = df_dropout_features['dropout'].values

# Split data
X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(
    X_dropout, y_dropout, test_size=0.2, random_state=42, stratify=y_dropout
)

# Scale
X_train_d_scaled = scaler.fit_transform(X_train_d)
X_test_d_scaled = scaler.transform(X_test_d)

print(f"\nTraining dropout prediction model...")
print(f"Train size: {len(X_train_d)}, Test size: {len(X_test_d)}")

# Train logistic regression for dropout prediction
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

dropout_model = LogisticRegression(random_state=42, max_iter=1000)
dropout_model.fit(X_train_d_scaled, y_train_d)
y_pred_d = dropout_model.predict(X_test_d_scaled)
y_pred_d_proba = dropout_model.predict_proba(X_test_d_scaled)[:, 1]

print("\nDropout Prediction Results:")
print(classification_report(y_test_d, y_pred_d))
print(f"ROC-AUC Score: {roc_auc_score(y_test_d, y_pred_d_proba):.3f}")

# ========================================================================
# Summary and Recommendations
# ========================================================================

print("\n" + "="*70)
print("SUMMARY AND RECOMMENDATIONS")
print("="*70)

print("""
The student_week_aggregations_rolling dataset can be used for several predictive tasks:

1. **Engagement Prediction** (minutes_per_week)
   - Predict how much time a student will spend next week
   - Useful for identifying students who might disengage

2. **Activity Prediction** (problems_solved)
   - Predict how many problems a student will solve
   - Helps in content recommendation and pacing

3. **Performance Prediction** (avg_proficiency)
   - Predict student's proficiency level
   - Useful for adaptive learning and intervention

4. **Dropout Prediction** (binary)
   - Predict if a student will be active next week
   - Critical for early intervention

**Key Considerations:**
- Handle irregular time series (gaps in student activity)
- Use panel data models for students with limited data
- Individual models for students with sufficient history
- Consider seasonality (week of year, semester effects)
- Feature engineering is crucial (lags, rolling stats, trends)

**Integration with Existing Framework:**
- Use create_model() for standard ML models
- Adapt data preparation for sequence models (LSTM, GRU)
- Use TimeSeriesEvaluator for consistent evaluation
- Can extend to multi-step prediction
""")

# Save results summary
results_df = pd.DataFrame(results).T
results_df.to_csv('../outputs/panel_model_results.csv')
print("\nSaved model results to outputs/panel_model_results.csv")

print("\n=== Predictive Modeling Complete ===") 