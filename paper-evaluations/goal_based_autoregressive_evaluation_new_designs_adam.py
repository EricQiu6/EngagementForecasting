"""
Teacher Forcing Growing Window Evaluation for Goal-Based Predictors
================================================================

Evaluates goal-based predictors alongside XGBoost and LASSO using teacher forcing
with growing window sizes. Models always see ground truth historical data but 
window size increases from 1 week to N weeks.

Separate predictors for minutes_per_week and avg_proficiency targets.
"""

import numpy as np
from pathlib import Path
import argparse
import csv
from typing import Tuple, List, Dict, Any
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, RegressorMixin


def load_and_prepare_data(csv_path: str) -> List[Dict]:
    """Load CSV and prepare data, excluding n_skills_measured column."""
    data = []
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric columns
            for key in ['minutes_per_week', 'problems_solved', 'total_opportunities', 
                       'avg_proficiency', 'week_difficulty', 'student_ability', 'student_learning_rate']:
                if key in row and row[key] != '' and row[key] != 'NaN':
                    try:
                        row[key] = float(row[key])
                    except (ValueError, TypeError):
                        row[key] = np.nan
                else:
                    row[key] = np.nan
            
            # Skip n_skills_measured if present
            if 'n_skills_measured' in row:
                del row['n_skills_measured']
                
            data.append(row)
    
    return data


def get_student_sequences(data: List[Dict], target_col: str) -> Dict[str, List[Dict]]:
    """Group data by student and sort by week."""
    students = {}
    for row in data:
        student_id = row['anon_student_id']
        if student_id not in students:
            students[student_id] = []
        students[student_id].append(row)
    
    # Sort each student's data by week_id
    for student_id in students:
        students[student_id].sort(key=lambda x: x['week_id'])
    
    return students


def engineer_features(historical_data: List[Dict], target_col: str, lag_window: int = 10) -> np.ndarray:
    """
    Engineer features from historical sequence, adapted from simple_evaluation.py.
    Modified to work with target-specific lag features.
    """
    features = []
    seq_len = len(historical_data)
    
    if seq_len == 0:
        # Return zeros if no data
        return np.zeros(50)  # Approximate feature count
    
    # Feature columns (excluding student_id, week_id)
    feature_cols = ['minutes_per_week', 'problems_solved', 'total_opportunities', 
                   'avg_proficiency', 'week_difficulty', 'student_ability', 'student_learning_rate']
    
    # 1. Current values (from last timestep)
    last_row = historical_data[-1]
    for col in feature_cols:
        val = last_row[col]
        features.append(val if not np.isnan(val) else 0.0)
    
    # 2. Lag features for important variables (including target-specific lags)
    lag_vars = ['avg_proficiency', 'minutes_per_week', 'problems_solved', 'total_opportunities']
    
    for var in lag_vars:
        values = np.array([row[var] if not np.isnan(row[var]) else 0.0 for row in historical_data])
        
        # Get last lag_window values (or pad with zeros if sequence too short)
        if seq_len >= lag_window:
            lags = values[-lag_window:]
        else:
            # Pad with zeros at the beginning
            lags = np.pad(values, (lag_window - seq_len, 0), mode='constant', constant_values=0)
        
        features.extend(lags)
    
    # 3. Change/trend features
    change_vars = ['avg_proficiency', 'minutes_per_week', 'problems_solved']
    
    for var in change_vars:
        if seq_len > 1:
            values = np.array([row[var] if not np.isnan(row[var]) else 0.0 for row in historical_data])
            
            # Recent change (last - previous)
            recent_change = values[-1] - values[-2]
            features.append(recent_change)
            
            # Average change over sequence
            changes = np.diff(values)
            avg_change = np.mean(changes)
            features.append(avg_change)
        else:
            features.extend([0.0, 0.0])
    
    # 4. Statistical features over sequence
    
    # Minutes per week statistics
    values = np.array([row['minutes_per_week'] if not np.isnan(row['minutes_per_week']) else 0.0 for row in historical_data])
    features.extend([
        np.mean(values),  # Average engagement
        np.std(values),   # Variability
        np.max(values) - np.min(values),  # Range
        np.percentile(values, 75) - np.percentile(values, 25)  # IQR
    ])
    
    # Problems solved statistics
    values = np.array([row['problems_solved'] if not np.isnan(row['problems_solved']) else 0.0 for row in historical_data])
    features.extend([
        np.mean(values),  # Average practice
        np.sum(values),   # Total practice
        np.std(values)    # Practice consistency
    ])
    
    # Proficiency trend (linear slope)
    if seq_len >= 2:
        values = np.array([row['avg_proficiency'] if not np.isnan(row['avg_proficiency']) else 0.0 for row in historical_data])
        x = np.arange(len(values))
        try:
            slope = np.polyfit(x, values, 1)[0]
            features.append(np.nan_to_num(slope, nan=0.0, posinf=0.0, neginf=0.0))
        except:
            features.append(0.0)
        
        # Quadratic acceleration (second derivative)
        if seq_len > 2:
            try:
                accel = np.polyfit(x, values, 2)[0] * 2
                features.append(np.nan_to_num(accel, nan=0.0, posinf=0.0, neginf=0.0))
            except:
                features.append(0.0)
        else:
            features.append(0.0)
    else:
        features.extend([0.0, 0.0])
    
    # 6. Gap features - Learning gaps (periods with no activity)
    minutes_values = np.array([row['minutes_per_week'] if not np.isnan(row['minutes_per_week']) else 0.0 for row in historical_data])
    
    # Identify gaps (weeks with 0 minutes)
    is_gap = (minutes_values == 0).astype(float)
    
    # Feature 1: has_recent_gap - Was there a gap in the last 3 weeks?
    recent_window = min(3, seq_len)
    has_recent_gap = float(np.any(is_gap[-recent_window:]))
    features.append(has_recent_gap)
    
    # Feature 2: weeks_since_last_gap - How many weeks since the last gap?
    gap_positions = np.where(is_gap == 1)[0]
    if len(gap_positions) > 0:
        # Find the most recent gap
        last_gap_pos = gap_positions[-1]
        weeks_since_gap = seq_len - 1 - last_gap_pos
    else:
        # No gaps in sequence
        weeks_since_gap = seq_len  # All weeks had activity
    features.append(float(weeks_since_gap))
    
    # Feature 3: gap_count - Total number of gaps in the sequence
    gap_count = np.sum(is_gap)
    features.append(float(gap_count))
    
    # 8. Prior achievement features
    if seq_len >= 2:
        proficiency_values = np.array([row['avg_proficiency'] if not np.isnan(row['avg_proficiency']) else 0.0 for row in historical_data])
        
        # Feature 1: Starting ability quartile (based on first 2 weeks)
        early_weeks = min(2, seq_len)
        early_proficiency = proficiency_values[:early_weeks]
        early_avg = np.mean(early_proficiency)
        
        # Assign quartile based on typical proficiency ranges (0-100 scale)
        if early_avg <= 25:
            starting_ability_quartile = 1.0
        elif early_avg <= 50:
            starting_ability_quartile = 2.0
        elif early_avg <= 75:
            starting_ability_quartile = 3.0
        else:
            starting_ability_quartile = 4.0
        features.append(starting_ability_quartile)
        
        # Feature 2: Performance consistency score (coefficient of variation)
        if seq_len >= 3:
            prof_std = np.std(proficiency_values)
            prof_mean = np.mean(proficiency_values)
            if prof_mean > 0:
                consistency_score = prof_std / prof_mean  # Lower = more consistent
            else:
                consistency_score = 0.0
            features.append(consistency_score)
        else:
            features.append(0.0)  # Default for short sequences
            
        # Feature 3: Early vs late performance comparison
        if seq_len >= 4:
            early_half = proficiency_values[:seq_len//2]
            late_half = proficiency_values[seq_len//2:]
            early_mean = np.mean(early_half)
            late_mean = np.mean(late_half)
            improvement = late_mean - early_mean
            features.append(improvement)
        else:
            features.append(0.0)
    else:
        features.extend([0.0, 0.0, 0.0])
    
    # 9. Enhanced features using student_ability, student_learning_rate, week_difficulty
    student_ability = last_row['student_ability'] if not np.isnan(last_row['student_ability']) else 2.2  # Use mean as default
    
    # Ability-difficulty mismatch (normalized scales)
    difficulty = last_row['week_difficulty'] if not np.isnan(last_row['week_difficulty']) else 0.6  # Use mean as default
            
    # Normalize both to z-scores before comparing
    difficulty_z = (difficulty - 0.6) / 1.5
    ability_z = (student_ability - 2.2) / 0.7
    ability_difficulty_mismatch = difficulty_z - ability_z
    features.append(ability_difficulty_mismatch)
        
    # High/low ability indicator
    high_ability = 1.0 if student_ability > 2.2 else 0.0  # Above mean ability
    features.append(high_ability)
    
    # Ability quartile
    if student_ability < 1.8:  # ~25th percentile  
        ability_quartile = 1.0
    elif student_ability < 2.2:  # ~50th percentile
        ability_quartile = 2.0
    elif student_ability < 2.7:  # ~75th percentile
        ability_quartile = 3.0
    else:
        ability_quartile = 4.0
    features.append(ability_quartile)
    
    # Learning rate dynamics
    learning_rates = np.array([row['student_learning_rate'] if not np.isnan(row['student_learning_rate']) else 0.004 for row in historical_data])
    
    # Current learning rate (scaled to meaningful range)
    current_lr = learning_rates[-1]
    current_lr_scaled = current_lr * 1000  # Scale for interpretability
    features.append(current_lr_scaled)
    
    # Learning rate direction (positive vs negative learner)
    positive_learner = 1.0 if current_lr > 0.004 else 0.0  # Above mean learning rate
    features.append(positive_learner)
    
    # Learning rate volatility (scaled)
    lr_volatility = np.std(learning_rates) * 1000  # Scale for interpretability
    features.append(lr_volatility)
    
    # Learning momentum (scaled trend)
    if seq_len >= 3:
        recent_lr_trend = (learning_rates[-1] - learning_rates[-3]) * 1000  # Scale the trend
        features.append(recent_lr_trend)
    else:
        features.append(0.0)
    
    # Engagement-adjusted learning (avoid problematic multiplication)
    minutes = last_row['minutes_per_week'] if not np.isnan(last_row['minutes_per_week']) else 15.8  # Use mean as default
    
    # Instead of multiplication, use categorical interaction
    high_lr_high_engagement = 1.0 if (current_lr > 0.004 and minutes > 15.8) else 0.0
    features.append(high_lr_high_engagement)
    
    # Convert to numpy array and handle any remaining NaNs
    feature_array = np.array(features, dtype=float)
    feature_array = np.nan_to_num(feature_array, nan=0.0, posinf=0.0, neginf=0.0)
    
    return feature_array


class newAdamsMinutesPredictor(BaseEstimator, RegressorMixin):
    """
    new's design of Adams predictor for minutes_per_week prediction.
    
    Algorithm:
    - Week 1: prediction = class_average
    - Week 2: prediction = student_week1 + (class_60th - class_50th)  
    - Week 3-8: prediction = previous_actual + 0.6 * (student_max - previous_actual)
    - Week 9+: prediction = percentile(last_9_values, [50/60/70])
    """
    
    def __init__(self, 
                 prediction_percentile=50,
                 class_average=None,
                 class_50th_percentile=None, 
                 class_60th_percentile=None,
                 random_state=None):
        self.prediction_percentile = prediction_percentile
        self.class_average = class_average
        self.class_50th_percentile = class_50th_percentile
        self.class_60th_percentile = class_60th_percentile
        self.random_state = random_state
        
    def fit(self, X, y):
        """Calculate class statistics if not provided."""
        if self.class_average is None:
            self.class_average = np.mean(y) if len(y) > 0 else 15.0
        if self.class_50th_percentile is None:
            self.class_50th_percentile = np.percentile(y, 50) if len(y) > 0 else 15.0
        if self.class_60th_percentile is None:  
            self.class_60th_percentile = np.percentile(y, 60) if len(y) > 0 else 16.0
        
        
        self.training_median_ = np.median(y) if len(y) > 0 else self.class_average
        return self
    
    def predict_new(self, week_number, student_history):
        """
        Make prediction using new's algorithm.
        
        Args:
            week_number: Current week being predicted (1-based)
            student_history: List of student's actual performance values from previous weeks
        
        Returns:
            Single prediction value
        """
        if week_number == 1:
            # Week 1: prediction = class_average
            return self.class_average
            
        elif week_number == 2:
            # Week 2: prediction = student_week1 + (class_60th - class_50th)
            student_week1 = student_history[0] if student_history else self.class_average
            gap = self.class_60th_percentile - self.class_50th_percentile
            return student_week1 + gap
            
        elif 3 <= week_number <= 8:
            # Week 3-8: prediction = previous_actual + 0.6 * (student_max - previous_actual)
            if not student_history:
                return self.class_average
                
            previous_actual = student_history[-1]  # Most recent actual performance
            student_max = max(student_history)     # Historical maximum
            return previous_actual + 0.6 * (student_max - previous_actual)
            
        else:
            # Week 9+: prediction = percentile(last_9_values, [50/60/70])
            if len(student_history) < 9:
                # Not enough history yet, use all available
                values = student_history
            else:
                # Use last 9 values
                values = student_history[-9:]
                
            if not values:
                return self.class_average
                
            return np.percentile(values, self.prediction_percentile)
    
    def predict(self, X, week_number=None, student_history=None):
        """Compatibility wrapper for sklearn interface."""
        if week_number is not None and student_history is not None:
            return np.array([self.predict_new(week_number, student_history)])
        else:
            # Fallback for standard sklearn usage
            n_samples = len(X) if hasattr(X, '__len__') else 1
            return np.full(n_samples, self.class_average)


class newAdamsProficiencyPredictor(BaseEstimator, RegressorMixin):
    """
    new's design of Adams predictor for avg_proficiency prediction.
    
    Algorithm:
    - Week 1: prediction = class_average
    - Week 2: prediction = student_week1 + (class_60th - class_50th)
    - Week 3-8: prediction = previous_actual + 0.6 * (student_max - previous_actual) 
    - Week 9+: prediction = percentile(last_9_values, [50/60/70])
    """
    
    def __init__(self, 
                 prediction_percentile=50,
                 class_average=None,
                 class_50th_percentile=None,
                 class_60th_percentile=None, 
                 random_state=None):
        self.prediction_percentile = prediction_percentile
        self.class_average = class_average
        self.class_50th_percentile = class_50th_percentile
        self.class_60th_percentile = class_60th_percentile
        self.random_state = random_state
        
    def fit(self, X, y):
        """Calculate class statistics if not provided."""
        if self.class_average is None:
            self.class_average = np.mean(y) if len(y) > 0 else 25.0
        if self.class_50th_percentile is None:
            self.class_50th_percentile = np.percentile(y, 50) if len(y) > 0 else 25.0
        if self.class_60th_percentile is None:
            self.class_60th_percentile = np.percentile(y, 60) if len(y) > 0 else 30.0
            
        self.training_median_ = np.median(y) if len(y) > 0 else self.class_average
        return self
    
    def predict_new(self, week_number, student_history):
        """
        Make prediction using new's algorithm.
        
        Args:
            week_number: Current week being predicted (1-based)
            student_history: List of student's actual performance values from previous weeks
        
        Returns:
            Single prediction value
        """
        if week_number == 1:
            # Week 1: prediction = class_average
            return self.class_average
            
        elif week_number == 2:
            # Week 2: prediction = student_week1 + (class_60th - class_50th)
            student_week1 = student_history[0] if student_history else self.class_average
            gap = self.class_60th_percentile - self.class_50th_percentile
            return student_week1 + gap
            
        elif 3 <= week_number <= 8:
            # Week 3-8: prediction = previous_actual + 0.6 * (student_max - previous_actual)
            if not student_history:
                return self.class_average
                
            previous_actual = student_history[-1]  # Most recent actual performance
            student_max = max(student_history)     # Historical maximum
            return previous_actual + 0.6 * (student_max - previous_actual)
            
        else:
            # Week 9+: prediction = percentile(last_9_values, [50/60/70])
            if len(student_history) < 9:
                # Not enough history yet, use all available
                values = student_history
            else:
                # Use last 9 values
                values = student_history[-9:]
                
            if not values:
                return self.class_average
                
            return np.percentile(values, self.prediction_percentile)
    
    def predict(self, X, week_number=None, student_history=None):
        """Compatibility wrapper for sklearn interface."""
        if week_number is not None and student_history is not None:
            return np.array([self.predict_new(week_number, student_history)])
        else:
            # Fallback for standard sklearn usage
            n_samples = len(X) if hasattr(X, '__len__') else 1
            return np.full(n_samples, self.class_average)


def create_training_sequences(students: Dict[str, List[Dict]], target_col: str, sequence_length: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """Create training sequences for model fitting."""
    X_train = []
    y_train = []
    
    for student_id, student_data in students.items():
        # Create sequences for this student (need at least sequence_length + 1 rows)
        for i in range(sequence_length, len(student_data)):
            # Historical data: rows [i-sequence_length : i]
            # Target: row i
            historical_data = student_data[i-sequence_length:i]
            target = student_data[i][target_col]
            
            if not np.isnan(target):
                features = engineer_features(historical_data, target_col)
                X_train.append(features)
                y_train.append(target)
    
    return np.array(X_train), np.array(y_train)


def evaluate_teacher_forcing_growing_window(models: Dict[str, Any], students: Dict[str, List[Dict]], 
                                          target_col: str, max_weeks: int = 20) -> Dict[str, Any]:
    """
    Evaluate models using teacher forcing with growing window sizes.
    
    Args:
        models: Dictionary of {model_name: fitted_model}
        students: Dictionary of {student_id: [sorted_week_data]}
        target_col: Target column name ('minutes_per_week' or 'avg_proficiency')
        max_weeks: Maximum number of weeks to predict
    
    Returns:
        Dictionary with results for plotting
    """
    results = {
        'ground_truth': [],
        'ground_truth_std': [],
        'models': {},
        'diagnostics': {'per_week': []}
    }
    
    # Initialize results for each model
    for model_name in models.keys():
        results['models'][model_name] = {
            'predictions': [],
            'mae_per_week': [],
            'rmse_per_week': []
        }
    
    # For each prediction week (growing window size)
    for week_idx in range(max_weeks):
        window_size = week_idx + 1  # Window grows: 1, 2, 3, ...
        
        week_ground_truths = []
        week_predictions = {model_name: [] for model_name in models.keys()}
        
        # Evaluate on each student
        for student_id, student_data in students.items():
            if len(student_data) <= week_idx:
                continue  # Student doesn't have enough data
            
            # Ground truth historical sequence (weeks 0 to week_idx-1)
            if week_idx == 0:
                # First week: no history, use empty or minimal features
                historical_data = [student_data[0]]  # Use first week as context
                if len(student_data) <= 1:
                    continue  # Need at least 2 weeks (history + target)
                target_week = student_data[1]
            else:
                historical_data = student_data[:week_idx]
                if len(student_data) <= week_idx:
                    continue  # Not enough data for target
                target_week = student_data[week_idx]
            
            # Get ground truth target
            true_value = target_week[target_col]
            if np.isnan(true_value):
                continue  # Skip if target is missing
            
            week_ground_truths.append(true_value)
            
            # Engineer features from ground truth historical sequence
            features = engineer_features(historical_data, target_col)
            
            # Get predictions from all models
            for model_name, model in models.items():
                try:
                    if 'new' in model_name:
                        # For new's Adams predictors, build student history
                        student_history = []
                        for i in range(week_idx):
                            if i < len(student_data):
                                actual_value = student_data[i][target_col]
                                if not np.isnan(actual_value):
                                    student_history.append(actual_value)
                                    
                        # Use new's algorithm
                        prediction = model.predict_new(week_idx + 1, student_history)
                        
                    elif model_name == 'XGBoost':
                        prediction = model.predict(np.array([features]))[0]
                    elif model_name == 'LASSO':
                        prediction = model.predict(np.array([features]))[0]
                    else:
                        prediction = model.predict([features])[0]
                        
                    week_predictions[model_name].append(prediction)
                except Exception as e:
                    print(f"Prediction error for {model_name} at week {week_idx}: {e}")
                    week_predictions[model_name].append(np.nan)
        
        # Prepare week diagnostics
        week_diag = {
            'week': week_idx + 1,
            'window_size': window_size,
            'students_total': len(students),
            'students_with_enough_data': len(week_ground_truths),
            'model_diagnostics': {}
        }

        # Store averaged results for this week (always append to maintain length = max_weeks)
        if week_ground_truths:
            results['ground_truth'].append(np.mean(week_ground_truths))
            results['ground_truth_std'].append(np.std(week_ground_truths))
        else:
            results['ground_truth'].append(np.nan)
            results['ground_truth_std'].append(np.nan)
            print(f"[DIAG] Week {week_idx + 1}: No ground truth values available; appending NaN to ground_truth.")
        
        for model_name in models.keys():
            preds = week_predictions[model_name]
            valid_preds = [p for p in preds if not np.isnan(p)]
            valid_ground_truths = [week_ground_truths[i] for i, p in enumerate(preds) if not np.isnan(p)]
            
            if week_ground_truths and valid_preds and valid_ground_truths:
                results['models'][model_name]['predictions'].append(np.mean(valid_preds))
                results['models'][model_name]['mae_per_week'].append(
                    mean_absolute_error(valid_ground_truths, valid_preds)
                )
                results['models'][model_name]['rmse_per_week'].append(
                    np.sqrt(mean_squared_error(valid_ground_truths, valid_preds))
                )
                week_diag['model_diagnostics'][model_name] = {
                    'used_nan': False,
                    'reason': '',
                    'n_preds': len(preds),
                    'n_valid_preds': len(valid_preds),
                    'n_ground_truths': len(week_ground_truths)
                }
            else:
                # No ground truth or no valid predictions for this week
                results['models'][model_name]['predictions'].append(np.nan)
                results['models'][model_name]['mae_per_week'].append(np.nan)
                results['models'][model_name]['rmse_per_week'].append(np.nan)
                reason = 'no_ground_truth' if not week_ground_truths else 'no_valid_predictions'
                print(
                    f"[DIAG] Week {week_idx + 1}: Appending NaN for {model_name} "
                    f"(reason={reason}; ground_truths={len(week_ground_truths)}, "
                    f"preds={len(preds)}, valid_preds={len(valid_preds)})."
                )
                week_diag['model_diagnostics'][model_name] = {
                    'used_nan': True,
                    'reason': reason,
                    'n_preds': len(preds),
                    'n_valid_preds': len(valid_preds),
                    'n_ground_truths': len(week_ground_truths)
                }

        # Save diagnostics for this week
        results['diagnostics']['per_week'].append(week_diag)
        
        print(f"Week {week_idx + 1} (window size {window_size}): {len(week_ground_truths)} students evaluated")
    
    return results


def plot_results(results_minutes: Dict[str, Any], results_proficiency: Dict[str, Any], 
                max_weeks: int, save_path: str = None):
    """Plot results for all 5 models."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Teacher Forcing Growing Window Evaluation - All Models', fontsize=16)
    
    weeks = list(range(1, max_weeks + 1))
    colors = {'new-P50': 'blue', 'new-P60': 'green', 'new-P70': 'red', 
              'Adams-P50': 'blue', 'Adams-P60': 'green', 'Adams-P70': 'red',
              'XGBoost': 'orange', 'LASSO': 'purple'}
    
    # Map model names to display names for legend
    display_names = {'new-P50': 'Adams-P50', 'new-P60': 'Adams-P60', 'new-P70': 'Adams-P70',
                     'XGBoost': 'XGBoost', 'LASSO': 'LASSO'}
    
    # Top row: side-by-side predictions vs ground truth (Minutes left, Proficiency right)
    ax1, ax2 = axes[0]

    # Plot 1 (top-left): Minutes predictions vs ground truth
    gt_mean = np.array(results_minutes['ground_truth'])
    gt_std = np.array(results_minutes['ground_truth_std'])
    ax1.plot(weeks, gt_mean, 'k-', linewidth=3, label='Ground Truth')
    # Add standard deviation bands
    valid_mask = ~(np.isnan(gt_mean) | np.isnan(gt_std))
    if np.any(valid_mask):
        ax1.fill_between(weeks, gt_mean - gt_std, gt_mean + gt_std, 
                        color='gray', alpha=0.2, label='±1 Std Dev')
    
    for model_name, model_results in results_minutes['models'].items():
        color = colors.get(model_name, 'gray')
        style = '--' if 'Adams' in model_name or 'new' in model_name else '-'
        display_name = display_names.get(model_name, model_name)
        ax1.plot(weeks, model_results['predictions'], style, color=color, label=display_name)
    
    ax1.set_xlabel('Week')
    ax1.set_ylabel('Average Minutes per Week')
    ax1.set_title('Minutes: Predictions vs Ground Truth')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2 (top-right): Proficiency predictions vs ground truth
    gt_mean_prof = np.array(results_proficiency['ground_truth'])
    gt_std_prof = np.array(results_proficiency['ground_truth_std'])
    ax2.plot(weeks, gt_mean_prof, 'k-', linewidth=3, label='Ground Truth')
    # Add standard deviation bands
    valid_mask_prof = ~(np.isnan(gt_mean_prof) | np.isnan(gt_std_prof))
    if np.any(valid_mask_prof):
        ax2.fill_between(weeks, gt_mean_prof - gt_std_prof, gt_mean_prof + gt_std_prof, 
                        color='gray', alpha=0.2, label='±1 Std Dev')
    for model_name, model_results in results_proficiency['models'].items():
        color = colors.get(model_name, 'gray')
        style = '--' if 'Adams' in model_name or 'new' in model_name else '-'
        display_name = display_names.get(model_name, model_name)
        ax2.plot(weeks, model_results['predictions'], style, color=color, label=display_name)
    ax2.set_xlabel('Week')
    ax2.set_ylabel('Average Proficiency')
    ax2.set_title('Proficiency: Predictions vs Ground Truth')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Bottom row: Minutes MAE (left), Proficiency MAE (right)
    ax3, ax4 = axes[1]
    
    # Plot 3 (bottom-left): Minutes MAE over time
    for model_name, model_results in results_minutes['models'].items():
        color = colors.get(model_name, 'gray')
        display_name = display_names.get(model_name, model_name)
        ax3.plot(weeks, model_results['mae_per_week'], '-o', color=color, label=display_name)
    ax3.set_xlabel('Week')
    ax3.set_ylabel('MAE')
    ax3.set_title('Minutes: Model Performance Over Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4 (bottom-right): Proficiency MAE over time
    for model_name, model_results in results_proficiency['models'].items():
        color = colors.get(model_name, 'gray')
        display_name = display_names.get(model_name, model_name)
        ax4.plot(weeks, model_results['mae_per_week'], '-o', color=color, label=display_name)
    
    ax4.set_xlabel('Week')
    ax4.set_ylabel('MAE')
    ax4.set_title('Proficiency: Model Performance Over Time')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plots saved to {save_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Teacher forcing growing window evaluation of goal-based predictors"
    )
    parser.add_argument("--csv", required=True, help="Path to train.csv file")
    parser.add_argument("--max_weeks", type=int, default=30, help="Maximum weeks to evaluate")
    parser.add_argument("--train_window", type=int, default=5, help="Training sequence window size")
    parser.add_argument("--save_plots", help="Path to save plots (optional)")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    print(f"Loading data from {args.csv}")
    print(f"Max weeks: {args.max_weeks}, Training window: {args.train_window}")
    print()
    
    # Load and prepare data
    data = load_and_prepare_data(args.csv)
    print(f"Loaded {len(data)} rows")
    
    # Group by student
    students_minutes = get_student_sequences(data, 'minutes_per_week')
    students_proficiency = get_student_sequences(data, 'avg_proficiency')
    
    # Filter students with enough data for evaluation (need max_weeks + 1 for weeks 0 to max_weeks)
    min_weeks_needed_eval = args.max_weeks + 1  # e.g., for max_weeks=30, need weeks 0-30 = 31 weeks
    students_minutes = {sid: sdata for sid, sdata in students_minutes.items() 
                       if len(sdata) >= min_weeks_needed_eval}
    students_proficiency = {sid: sdata for sid, sdata in students_proficiency.items() 
                           if len(sdata) >= min_weeks_needed_eval}
    
    print(f"Students with enough data for minutes: {len(students_minutes)}")
    print(f"Students with enough data for proficiency: {len(students_proficiency)}")
    print()
    
    # === CALCULATE GLOBAL CLASS STATISTICS ===
    print("Calculating global class statistics...")
    
    # Calculate global statistics for minutes
    all_minutes_values = []
    for student_data in students_minutes.values():
        for week_data in student_data:
            if not np.isnan(week_data['minutes_per_week']):
                all_minutes_values.append(week_data['minutes_per_week'])
    
    global_minutes_avg = np.mean(all_minutes_values)
    global_minutes_50th = np.percentile(all_minutes_values, 50)
    global_minutes_60th = np.percentile(all_minutes_values, 60)
    
    print(f"Minutes - Global stats: avg={global_minutes_avg:.1f}, 50th={global_minutes_50th:.1f}, 60th={global_minutes_60th:.1f}")
    
    # Calculate global statistics for proficiency
    all_proficiency_values = []
    for student_data in students_proficiency.values():
        for week_data in student_data:
            if not np.isnan(week_data['avg_proficiency']):
                all_proficiency_values.append(week_data['avg_proficiency'])
    
    global_proficiency_avg = np.mean(all_proficiency_values)
    global_proficiency_50th = np.percentile(all_proficiency_values, 50)
    global_proficiency_60th = np.percentile(all_proficiency_values, 60)
    
    print(f"Proficiency - Global stats: avg={global_proficiency_avg:.1f}, 50th={global_proficiency_50th:.1f}, 60th={global_proficiency_60th:.1f}")
    print()
    
    # === MINUTES EVALUATION ===
    print("=" * 60)
    print("TRAINING MODELS FOR MINUTES PREDICTION")
    print("=" * 60)
    
    # Create training data for minutes
    X_train_minutes, y_train_minutes = create_training_sequences(
        students_minutes, 'minutes_per_week', args.train_window
    )
    print(f"Minutes training sequences: {len(X_train_minutes)}")
    
    # Train all 5 models for minutes
    models_minutes = {}
    
    # new Adams predictors for different percentiles
    for percentile in [50, 60, 70]:
        models_minutes[f'new-P{percentile}'] = newAdamsMinutesPredictor(
            prediction_percentile=percentile, 
            class_average=global_minutes_avg,
            class_50th_percentile=global_minutes_50th,
            class_60th_percentile=global_minutes_60th,
            random_state=args.random_state
        )
        models_minutes[f'new-P{percentile}'].fit(X_train_minutes, y_train_minutes)
    
    # XGBoost for minutes
    models_minutes['XGBoost'] = xgb.XGBRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.05,
        objective="reg:absoluteerror", random_state=args.random_state
    )
    models_minutes['XGBoost'].fit(X_train_minutes, y_train_minutes)
    
    # LASSO for minutes - commented out to remove from evaluation
    # models_minutes['LASSO'] = Lasso(alpha=0.1, max_iter=2000, random_state=args.random_state)
    # models_minutes['LASSO'].fit(X_train_minutes, y_train_minutes)
    
    print(f"Trained {len(models_minutes)} models for minutes prediction")
    
    # Evaluate all minutes models together
    print("Evaluating all minutes models with teacher forcing...")
    results_minutes = evaluate_teacher_forcing_growing_window(
        models_minutes, students_minutes, 'minutes_per_week', args.max_weeks
    )
    
    # === PROFICIENCY EVALUATION ===
    print("\n" + "=" * 60)
    print("TRAINING MODELS FOR PROFICIENCY PREDICTION")
    print("=" * 60)
    
    # Create training data for proficiency
    X_train_proficiency, y_train_proficiency = create_training_sequences(
        students_proficiency, 'avg_proficiency', args.train_window
    )
    print(f"Proficiency training sequences: {len(X_train_proficiency)}")
    
    # Train all 5 models for proficiency
    models_proficiency = {}
    
    # new Adams predictors for different percentiles
    for percentile in [50, 60, 70]:
        models_proficiency[f'new-P{percentile}'] = newAdamsProficiencyPredictor(
            prediction_percentile=percentile,
            class_average=global_proficiency_avg,
            class_50th_percentile=global_proficiency_50th,
            class_60th_percentile=global_proficiency_60th,
            random_state=args.random_state
        )
        models_proficiency[f'new-P{percentile}'].fit(X_train_proficiency, y_train_proficiency)
    
    # XGBoost for proficiency
    models_proficiency['XGBoost'] = xgb.XGBRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.05,
        objective="reg:absoluteerror", random_state=args.random_state
    )
    models_proficiency['XGBoost'].fit(X_train_proficiency, y_train_proficiency)
    
    # LASSO for proficiency - commented out to remove from evaluation
    # models_proficiency['LASSO'] = Lasso(alpha=0.1, max_iter=2000, random_state=args.random_state)
    # models_proficiency['LASSO'].fit(X_train_proficiency, y_train_proficiency)
    
    print(f"Trained {len(models_proficiency)} models for proficiency prediction")
    
    # Evaluate all proficiency models together
    print("Evaluating all proficiency models with teacher forcing...")
    results_proficiency = evaluate_teacher_forcing_growing_window(
        models_proficiency, students_proficiency, 'avg_proficiency', args.max_weeks
    )
    
    # === RESULTS ===
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    print("Minutes Models - Final Week Performance:")
    for model_name, model_results in results_minutes['models'].items():
        if model_results['mae_per_week']:
            final_mae = model_results['mae_per_week'][-1]
            print(f"  {model_name}: MAE = {final_mae:.4f}")
    
    print("\nProficiency Models - Final Week Performance:")
    for model_name, model_results in results_proficiency['models'].items():
        if model_results['mae_per_week']:
            final_mae = model_results['mae_per_week'][-1]
            print(f"  {model_name}: MAE = {final_mae:.4f}")
    
    print("\nMinutes Models - Average MAE (Entire Sequence):")
    for model_name, model_results in results_minutes['models'].items():
        if model_results['mae_per_week']:
            avg_mae = np.mean(model_results['mae_per_week'])
            print(f"  {model_name}: Average MAE = {avg_mae:.4f}")
    
    print("\nProficiency Models - Average MAE (Entire Sequence):")
    for model_name, model_results in results_proficiency['models'].items():
        if model_results['mae_per_week']:
            avg_mae = np.mean(model_results['mae_per_week'])
            print(f"  {model_name}: Average MAE = {avg_mae:.4f}")
    
    print("\nMinutes Models - Average MAE (Weeks 1-8):")
    for model_name, model_results in results_minutes['models'].items():
        if model_results['mae_per_week'] and len(model_results['mae_per_week']) >= 8:
            avg_mae_week1to8 = np.mean(model_results['mae_per_week'][:8])  # Weeks 1-8
            print(f"  {model_name}: Average MAE (Weeks 1-8) = {avg_mae_week1to8:.4f}")
    
    print("\nMinutes Models - Average MAE (Starting Week 9):")
    for model_name, model_results in results_minutes['models'].items():
        if model_results['mae_per_week'] and len(model_results['mae_per_week']) >= 9:
            avg_mae_week9plus = np.mean(model_results['mae_per_week'][8:])  # Week 9 is index 8
            print(f"  {model_name}: Average MAE (Week 9+) = {avg_mae_week9plus:.4f}")
    
    print("\nProficiency Models - Average MAE (Weeks 1-8):")
    for model_name, model_results in results_proficiency['models'].items():
        if model_results['mae_per_week'] and len(model_results['mae_per_week']) >= 8:
            avg_mae_week1to8 = np.mean(model_results['mae_per_week'][:8])  # Weeks 1-8
            print(f"  {model_name}: Average MAE (Weeks 1-8) = {avg_mae_week1to8:.4f}")
    
    print("\nProficiency Models - Average MAE (Starting Week 9):")
    for model_name, model_results in results_proficiency['models'].items():
        if model_results['mae_per_week'] and len(model_results['mae_per_week']) >= 9:
            avg_mae_week9plus = np.mean(model_results['mae_per_week'][8:])  # Week 9 is index 8
            print(f"  {model_name}: Average MAE (Week 9+) = {avg_mae_week9plus:.4f}")
    
    # Standard deviation analysis
    print("\nGround Truth Standard Deviation Analysis:")
    
    # Minutes standard deviation analysis
    minutes_std = np.array(results_minutes['ground_truth_std'])
    valid_std_minutes = minutes_std[~np.isnan(minutes_std)]
    if len(valid_std_minutes) >= 8:
        early_std_minutes = np.mean(valid_std_minutes[:8])  # First 8 weeks
        if len(valid_std_minutes) > 8:
            late_std_minutes = np.mean(valid_std_minutes[8:])  # Week 9+
            print(f"  Minutes - Average Std Dev (Weeks 1-8): {early_std_minutes:.4f}")
            print(f"  Minutes - Average Std Dev (Week 9+): {late_std_minutes:.4f}")
        else:
            print(f"  Minutes - Average Std Dev (Weeks 1-8): {early_std_minutes:.4f}")
            print(f"  Minutes - Average Std Dev (Week 9+): No data")
    
    # Proficiency standard deviation analysis  
    proficiency_std = np.array(results_proficiency['ground_truth_std'])
    valid_std_prof = proficiency_std[~np.isnan(proficiency_std)]
    if len(valid_std_prof) >= 8:
        early_std_prof = np.mean(valid_std_prof[:8])  # First 8 weeks
        if len(valid_std_prof) > 8:
            late_std_prof = np.mean(valid_std_prof[8:])  # Week 9+
            print(f"  Proficiency - Average Std Dev (Weeks 1-8): {early_std_prof:.4f}")
            print(f"  Proficiency - Average Std Dev (Week 9+): {late_std_prof:.4f}")
        else:
            print(f"  Proficiency - Average Std Dev (Weeks 1-8): {early_std_prof:.4f}")
            print(f"  Proficiency - Average Std Dev (Week 9+): No data")
    
    # Plot results
    plot_results(results_minutes, results_proficiency, args.max_weeks, args.save_plots)


if __name__ == "__main__":
    main()