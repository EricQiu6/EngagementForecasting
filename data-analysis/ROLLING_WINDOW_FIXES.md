# Rolling Window Pipeline Fixes

## Issues Identified

1. **avg_proficiency showing 0 everywhere**
   - With `PROFICIENCY_METHOD = "new_skills_mastered"`, weeks without models were defaulting to 0
   - Fixed: Now shows NA for weeks without models

2. **NA values in AFM-derived columns** (week_difficulty, student_ability, student_learning_rate)
   - Only 1 model was successfully fitted (week 2012-W22) out of 39 weeks
   - Joins were failing for all other weeks

3. **week_easiness appearing in output**
   - This is an intermediate calculation that shouldn't be in final output
   - Fixed: Now removed before saving

## Root Cause

The model fitting thresholds were too restrictive for small samples:
- Original: Required 100+ rows, 5+ students, 3+ skills
- With 5% sample, most weeks couldn't meet these requirements

## Fixes Applied

### 1. Adaptive Thresholds
```r
# Now scales with sample size:
# 5-10% sample: 30 rows, 3 students, 2 skills
# 10-50% sample: 50 rows, 4 students, 2 skills  
# 50-100% sample: 100 rows, 5 students, 3 skills
```

### 2. Better Handling of Missing Data
- Proficiency now returns NA (not 0) for weeks without models
- Empty data structures properly initialized
- NaN values converted to NA

### 3. Enhanced Debugging
- Shows which weeks have models
- Reports join success rates
- Displays data insufficiency reasons in DEBUG mode

### 4. Data Cleaning
- Removed intermediate `week_easiness` column from output
- Better handling of empty difficulty calculations

## To Run Successfully

1. **For small samples (5-10%)**:
   ```r
   # Run in DEBUG mode to see which weeks are skipped
   Sys.setenv(DEBUG_MODE = "TRUE")
   Sys.setenv(STUDENT_SAMPLE_PERCENT = "10")
   ```

2. **For better coverage**:
   ```r
   # Use at least 30% sample for most weeks to have models
   Sys.setenv(STUDENT_SAMPLE_PERCENT = "30")
   ```

3. **Check results**:
   - Look for "Successfully processed N weeks" in output
   - Check model_metadata.csv to see which weeks converged
   - Verify student_abilities_by_week.csv has data for multiple weeks

## Expected Behavior

- **With 5% sample**: Expect only a few weeks to have enough data
- **With 30% sample**: Most weeks should have models
- **With 100% sample**: Nearly all weeks should have models

Columns will show NA (not 0) for weeks without sufficient data to fit models. 