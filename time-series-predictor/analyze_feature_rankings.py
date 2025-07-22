#!/usr/bin/env python3
"""
Script to analyze feature importance rankings across all windows
"""

import pandas as pd
import os
import numpy as np

def analyze_feature_rankings():
    """Analyze feature importance rankings from cross-window analysis"""
    
    # Read the feature importance trends
    trends_file = "cross_window_analysis_results/feature_importance_trends_20250718_042435.csv"
    
    if not os.path.exists(trends_file):
        print(f"❌ Trends file not found: {trends_file}")
        return
    
    # Load the trends data
    trends_df = pd.read_csv(trends_file)
    
    # Sort by mean importance (descending)
    trends_df = trends_df.sort_values('mean_importance', ascending=False).reset_index(drop=True)
    
    print("🎯 FEATURE IMPORTANCE RANKINGS")
    print("=" * 80)
    print()
    
    # Top 5 features
    print("🏆 TOP 5 MOST IMPORTANT FEATURES:")
    print("-" * 50)
    for i, row in trends_df.head(5).iterrows():
        rank = i + 1
        feature = row['feature']
        importance = row['mean_importance']
        std = row['importance_std']
        windows = row['n_windows']
        trend = "↗️" if row['trend_slope'] > 0 else "↘️" if row['trend_slope'] < 0 else "→"
        
        print(f"{rank:2d}. {feature:<35} {importance:6.3f} ±{std:5.3f} {trend} ({windows} windows)")
    
    print()
    
    # Top 25 features (or all if less than 25)
    top_n = min(25, len(trends_df))
    print(f"📊 TOP {top_n} MOST IMPORTANT FEATURES:")
    print("-" * 80)
    print(f"{'Rank':<4} {'Feature':<40} {'Importance':<12} {'Std':<8} {'Trend':<6} {'Windows'}")
    print("-" * 80)
    
    for i, row in trends_df.head(top_n).iterrows():
        rank = i + 1
        feature = row['feature']
        importance = row['mean_importance']
        std = row['importance_std']
        windows = row['n_windows']
        trend_slope = row['trend_slope']
        
        if trend_slope > 0.01:
            trend = "↗️↗️"  # Strong increase
        elif trend_slope > 0:
            trend = "↗️"    # Mild increase
        elif trend_slope < -0.01:
            trend = "↘️↘️"  # Strong decrease
        elif trend_slope < 0:
            trend = "↘️"    # Mild decrease
        else:
            trend = "→"     # Stable
        
        print(f"{rank:<4} {feature:<40} {importance:<12.3f} ±{std:<7.3f} {trend:<6} {windows}")
    
    print()
    
    # Analyze our key features of interest
    print("🎯 KEY FEATURES WE CARED ABOUT - WHERE DO THEY RANK?")
    print("-" * 60)
    
    # Define the key features we were interested in
    key_features = [
        # Class-level features
        'class_percentile_rank_prof',
        'performance_vs_class_mean_prof',
        
        # Prior achievement features  
        'starting_ability_quartile',
        'learning_acceleration_capacity',
        'performance_consistency_score',
        
        # Student ability features
        'current_student_ability',
        'current_avg_proficiency',
        
        # Time and difficulty features
        'current_week_difficulty',
        'gap_count',
        'minutes_std',
        'proficiency_trend'
    ]
    
    key_features_found = []
    key_features_missing = []
    
    for feature in key_features:
        feature_row = trends_df[trends_df['feature'] == feature]
        if not feature_row.empty:
            rank = feature_row.index[0] + 1
            importance = feature_row['mean_importance'].iloc[0]
            trend_slope = feature_row['trend_slope'].iloc[0]
            
            # Trend indicator
            if trend_slope > 0.01:
                trend = "↗️↗️ Strong increase"
            elif trend_slope > 0:
                trend = "↗️ Increasing"
            elif trend_slope < -0.01:
                trend = "↘️↘️ Strong decrease"
            elif trend_slope < 0:
                trend = "↘️ Decreasing"
            else:
                trend = "→ Stable"
            
            key_features_found.append((rank, feature, importance, trend))
        else:
            key_features_missing.append(feature)
    
    # Sort by rank
    key_features_found.sort(key=lambda x: x[0])
    
    print("✅ FOUND IN RANKINGS:")
    for rank, feature, importance, trend in key_features_found:
        in_top5 = "🏆 TOP 5" if rank <= 5 else ""
        in_top25 = "📊 TOP 25" if rank <= 25 else ""
        badge = in_top5 or in_top25 or ""
        
        print(f"  Rank {rank:2d}: {feature:<35} {importance:6.3f} - {trend} {badge}")
    
    if key_features_missing:
        print()
        print("❌ NOT FOUND (possibly filtered out):")
        for feature in key_features_missing:
            print(f"  - {feature}")
    
    print()
    
    # Summary for key features in top rankings
    top5_key = [f for r, f, i, t in key_features_found if r <= 5]
    top25_key = [f for r, f, i, t in key_features_found if r <= 25]
    
    print("📈 SUMMARY FOR KEY FEATURES:")
    print(f"  🏆 Key features in TOP 5: {len(top5_key)}/5")
    if top5_key:
        print(f"      {', '.join(top5_key)}")
    
    print(f"  📊 Key features in TOP 25: {len(top25_key)}/{min(25, len(trends_df))}")
    if top25_key:
        print(f"      {', '.join(top25_key)}")
    
    print()
    print("🎉 Analysis complete!")

if __name__ == "__main__":
    analyze_feature_rankings() 