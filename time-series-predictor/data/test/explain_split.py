#!/usr/bin/env python3
"""
Script to explain how the chronological split works with concrete examples
"""

from data_processing import split_chronologically
import pandas as pd

def explain_chronological_split():
    """
    Explain the chronological split with concrete examples
    """
    print("=" * 60)
    print("HOW THE CHRONOLOGICAL SPLIT WORKS")
    print("=" * 60)
    
    # Get the split result
    result = split_chronologically()
    
    train_data = result['train_data']
    test_data = result['test_data']
    K = result['metadata']['K']
    
    print(f"\n📋 SPLIT STRATEGY:")
    print(f"• For each student individually:")
    print(f"  - Take their complete time series (all weeks)")
    print(f"  - Last {K} weeks → TEST SET")
    print(f"  - Everything before → TRAINING SET")
    print(f"• This ensures chronological order: we never use future data to predict the past")
    print(f"• No temporal leakage: model only sees past to predict future")
    
    # Load original data to show complete picture
    original_data = pd.read_csv('data_tidied.csv')
    
    # Show examples for 4 students
    students_to_show = original_data['name'].unique()[:4]
    
    print(f"\n" + "=" * 60)
    print("CONCRETE EXAMPLES")
    print("=" * 60)
    
    for i, student in enumerate(students_to_show):
        print(f"\n--- STUDENT {i+1}: {student[:12]}... ---")
        
        # Get all data for this student
        student_all = original_data[original_data['name'] == student].sort_values('week')
        student_train = train_data[train_data['name'] == student].sort_values('week')
        student_test = test_data[test_data['name'] == student].sort_values('week')
        
        total_weeks = len(student_all)
        
        print(f"Complete timeline ({total_weeks} weeks):")
        weeks_str = " ".join([f"W{w:2d}" for w in student_all['week']])
        perf_str = " ".join([f"{p:3.0f}" for p in student_all['proficient']])
        print(f"  Weeks: {weeks_str}")
        print(f"  Skills: {perf_str}")
        
        if len(student_train) > 0 and len(student_test) > 0:
            print(f"\nAfter split:")
            print(f"  TRAINING → Weeks {student_train['week'].min():2d} to {student_train['week'].max():2d} ({len(student_train)} weeks)")
            print(f"  TEST     → Weeks {student_test['week'].min():2d} to {student_test['week'].max():2d} ({len(student_test)} weeks)")
            print(f"  Split point: {student_train['week'].max()} | {student_test['week'].min()}")
        else:
            print(f"  ⚠️  Insufficient data for split (≤{K} weeks total)")
        
        # Show visual representation
        if len(student_train) > 0 and len(student_test) > 0:
            visual = ""
            for week in student_all['week']:
                if week in student_train['week'].values:
                    visual += "T "
                elif week in student_test['week'].values:
                    visual += "E "
                else:
                    visual += "? "
            print(f"  Visual:   {visual}(T=Train, E=tEst)")
    
    print(f"\n" + "=" * 60)
    print("WHY THIS SPLIT MAKES SENSE")
    print("=" * 60)
    
    print("✅ ADVANTAGES:")
    print("  1. REALISTIC: Model sees past → predicts future (like real usage)")
    print("  2. NO LEAKAGE: Future data never contaminates training")
    print("  3. FAIR COMPARISON: All students get same test period length")
    print("  4. SUFFICIENT TRAINING: Most students get 7-8 weeks for training")
    
    print(f"\n📊 SPLIT STATISTICS:")
    print(f"  • Total students: {result['metadata']['total_students']}")
    print(f"  • Students processed: {result['metadata']['students_processed']}")
    print(f"  • Students with insufficient data: {result['metadata']['students_insufficient_data']}")
    print(f"  • Training samples: {result['metadata']['train_samples']}")
    print(f"  • Test samples: {result['metadata']['test_samples']}")
    
    print(f"\n🎯 FOR AR MODEL:")
    print("  • Training data: Use weeks -2 to 5 to learn patterns")
    print("  • Test data: Predict weeks 6, 7, 8 using learned patterns")
    print("  • Features: Create lags (e.g., proficient_lag1, proficient_lag2)")
    print("  • Target: Current week's proficient score")
    
    # Show a lag feature example
    print(f"\n" + "=" * 60)
    print("EXAMPLE: CREATING AR FEATURES")
    print("=" * 60)
    
    sample_student = students_to_show[0]
    student_train = train_data[train_data['name'] == sample_student].sort_values('week')
    
    if len(student_train) > 2:
        print(f"For student {sample_student[:12]}... training data:")
        
        # Create lag features
        student_lag = student_train.copy()
        student_lag['proficient_lag1'] = student_lag['proficient'].shift(1)
        student_lag['proficient_lag2'] = student_lag['proficient'].shift(2)
        
        print("\nOriginal data:")
        print(student_train[['week', 'proficient']].to_string(index=False))
        
        print("\nWith AR features (for model training):")
        ar_features = student_lag[['week', 'proficient', 'proficient_lag1', 'proficient_lag2']].dropna()
        print(ar_features.to_string(index=False))
        
        print("\nInterpretation:")
        print("• Each row = one training example")
        print("• Features = [week, proficient_lag1, proficient_lag2]")
        print("• Target = proficient (current week)")
        print("• Model learns: current_performance = f(past_performance)")

if __name__ == "__main__":
    explain_chronological_split() 