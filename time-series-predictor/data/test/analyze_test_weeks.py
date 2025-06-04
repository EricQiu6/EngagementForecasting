#!/usr/bin/env python3
"""
Analyze why test weeks range from 2 to 8
"""

import pandas as pd
from data_processing import split_chronologically

def analyze_test_week_range():
    print("=== WHY TEST WEEKS RANGE FROM 2 TO 8 ===")
    print()
    
    # Get the split data
    result = split_chronologically()
    test_data = result['test_data']
    
    # Load original data to see student timelines
    original_data = pd.read_csv('data_tidied.csv')
    
    # Analyze when different students end
    student_timelines = original_data.groupby('name')['week'].agg(['min', 'max']).reset_index()
    student_timelines.columns = ['name', 'first_week', 'last_week']
    
    print("🎯 THE KEY INSIGHT:")
    print("Not all students have the same timeline!")
    print()
    
    # Show distribution of ending weeks
    ending_week_counts = student_timelines['last_week'].value_counts().sort_index()
    print("📊 Students by their ENDING week:")
    for week, count in ending_week_counts.items():
        print(f"  {count:3d} students end at week {week}")
    
    print()
    print("💡 Since we take the LAST 3 weeks per student:")
    print("  • Students ending at week 5 → test weeks: 3, 4, 5")
    print("  • Students ending at week 6 → test weeks: 4, 5, 6") 
    print("  • Students ending at week 7 → test weeks: 5, 6, 7")
    print("  • Students ending at week 8 → test weeks: 6, 7, 8")
    
    print()
    print("🔍 CONCRETE EXAMPLES:")
    
    # Show a few examples
    examples = []
    for ending_week in sorted(ending_week_counts.index):
        student_example = student_timelines[student_timelines['last_week'] == ending_week].iloc[0]
        student_name = student_example['name']
        
        # Get this student's test weeks
        student_test_data = test_data[test_data['name'] == student_name].sort_values('week')
        test_weeks = student_test_data['week'].tolist()
        
        examples.append({
            'student': student_name[:12] + '...',
            'ends_at': ending_week,
            'test_weeks': test_weeks
        })
    
    for ex in examples:
        print(f"  Student {ex['student']} ends at week {ex['ends_at']} → test weeks: {ex['test_weeks']}")
    
    print()
    print("📈 COMBINED RESULT:")
    test_week_dist = test_data['week'].value_counts().sort_index()
    print("Test samples per week:")
    for week, count in test_week_dist.items():
        print(f"  Week {week:2d}: {count:3d} test samples")
    
    print()
    print(f"📋 SUMMARY:")
    print(f"  • Earliest test week: {test_data['week'].min()}")
    print(f"  • Latest test week: {test_data['week'].max()}")
    print(f"  • Range: {test_data['week'].min()} to {test_data['week'].max()}")
    print(f"  • This is because students have different ending weeks!")
    
    print()
    print("✅ WHY THIS MAKES SENSE:")
    print("  1. Each student gets their PERSONAL last 3 weeks as test data")
    print("  2. Students joined the study at different times")
    print("  3. Some students have longer participation than others")
    print("  4. The 'last 3 weeks' is relative to each student's timeline")
    print("  5. This preserves the chronological split principle per student")

if __name__ == "__main__":
    analyze_test_week_range() 