#!/usr/bin/env python3
"""
Analyze why SMAPE is over 100%
"""

import pandas as pd
import numpy as np
from model.framework import TimeSeriesFramework
from sklearn.linear_model import LinearRegression
from data.data_processing import split_chronologically

def analyze_high_smape():
    print("=== ANALYSIS OF HIGH SMAPE ===")
    print()
    
    # Train a model and get predictions
    framework = TimeSeriesFramework(LinearRegression())
    framework.train()
    
    # Get test data and predictions
    split_result = split_chronologically(data_path='data/data_tidied.csv', K=3)
    test_data = split_result['test_data']
    X_test, y_test = framework._prepare_features(test_data)
    y_pred = framework.predictor.predict(X_test)
    
    print("📊 TARGET VARIABLE ANALYSIS:")
    print(f"  Mean: {y_test.mean():.2f} skills per week")
    print(f"  Std: {y_test.std():.2f}")
    print(f"  Range: {y_test.min():.0f} to {y_test.max():.0f}")
    print(f"  Zeros: {(y_test == 0).sum()} / {len(y_test)} ({(y_test == 0).mean()*100:.1f}%)")
    print(f"  Values ≤ 2: {(y_test <= 2).sum()} / {len(y_test)} ({(y_test <= 2).mean()*100:.1f}%)")
    
    print(f"\n📈 PREDICTION ANALYSIS:")
    print(f"  Mean: {y_pred.mean():.2f}")
    print(f"  Std: {y_pred.std():.2f}")
    print(f"  Range: {y_pred.min():.2f} to {y_pred.max():.2f}")
    
    # Calculate SMAPE per sample
    smape_per_sample = 200 * np.abs(y_test - y_pred) / (np.abs(y_test) + np.abs(y_pred) + 1e-8)
    
    print(f"\n🎯 SMAPE BREAKDOWN:")
    print(f"  Overall SMAPE: {smape_per_sample.mean():.1f}%")
    print(f"  SMAPE > 100%: {(smape_per_sample > 100).sum()} / {len(smape_per_sample)} ({(smape_per_sample > 100).mean()*100:.1f}%)")
    print(f"  SMAPE > 200%: {(smape_per_sample > 200).sum()} / {len(smape_per_sample)} ({(smape_per_sample > 200).mean()*100:.1f}%)")
    
    print(f"\n💡 WHY SMAPE IS HIGH:")
    print("1. Many target values are 0, 1, or 2 (small numbers)")
    print("2. When both true and predicted values are small, SMAPE explodes")
    print("3. SMAPE = 200 * |true - pred| / (|true| + |pred|)")
    print("4. Example: true=1, pred=3 → SMAPE = 200*2/(1+3) = 100%")
    print("5. Example: true=0, pred=2 → SMAPE = 200*2/(0+2) = 200%")
    
    print(f"\n🔍 WORST PREDICTION EXAMPLES:")
    worst_indices = np.argsort(smape_per_sample)[-8:]
    
    for i in worst_indices:
        true_val = y_test[i]
        pred_val = y_pred[i]
        smape_val = smape_per_sample[i]
        error = abs(true_val - pred_val)
        print(f"  True: {true_val:.0f}, Pred: {pred_val:.1f}, Error: {error:.1f}, SMAPE: {smape_val:.0f}%")
    
    print(f"\n✅ IS THIS NORMAL?")
    print("Yes! SMAPE > 100% is mathematically possible and indicates:")
    print("  • Poor prediction accuracy (which is expected for this baseline)")
    print("  • Target variable has many small values (0-2 skills per week)")
    print("  • Linear AR(2) model may not capture student learning patterns well")
    
    print(f"\n🎯 BETTER METRICS FOR THIS PROBLEM:")
    mae = np.mean(np.abs(y_test - y_pred))
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    print(f"  MAE: {mae:.2f} skills (more interpretable)")
    print(f"  RMSE: {rmse:.2f} skills")
    print(f"  Mean baseline error: {np.mean(np.abs(y_test - y_test.mean())):.2f}")
    print(f"  → Model vs baseline: {mae:.2f} vs {np.mean(np.abs(y_test - y_test.mean())):.2f}")

if __name__ == "__main__":
    analyze_high_smape() 