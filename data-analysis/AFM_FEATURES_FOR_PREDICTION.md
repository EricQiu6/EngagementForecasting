# AFM Features for Predictive Modeling

This document describes the features extracted from the AFM models that can be used for predicting student proficiency in future weeks.

## Static Pipeline Output (combined.Rmd)

### Main Output File: `student_week_aggregations_what.csv`

Contains per student-week:

- **Behavioral metrics:**
  - `minutes_per_week` - Time spent learning
  - `problems_solved` - Number of unique problems attempted
  - `total_opportunities` - Total problem step attempts
- **Proficiency metrics:**
  - `avg_proficiency` - Current week proficiency (mean or new skills mastered)
  - `n_skills_measured` - Number of skills measured
- **Difficulty metric:**
  - `week_difficulty` - Weighted difficulty of skills encountered this week
- **Student parameters (NEW):**
  - `student_ability` - Student's total ability (global intercept + random effect)
  - `student_learning_rate` - Student's total learning rate (global rate + random effect)

### Additional Feature Files in `afm_outputs_combined/`:

- `student_ability.csv` - Student parameters (both deviations and total effects)
  - `ability_deviation` - Random effect (deviation from population)
  - `learning_rate_deviation` - Random effect (deviation from population)
  - `ability` - Total ability (global + deviation)
  - `learning_rate` - Total learning rate (global + deviation)
- `skill_easyness.csv` - Skill difficulty parameters
- `global_learning_rate.csv` - Population-level learning rate
- `intercept.csv` - Model intercept
- `afm_model.rds` - Complete model object

## Rolling Window Pipeline Output (combined_rolling_window.Rmd)

### Main Output File: `student_week_aggregations_rolling.csv`

Same structure as static pipeline, but with week-specific parameters:

- All behavioral, proficiency, and difficulty metrics (same as above)
- `student_ability` - Student ability _from that week's model_
- `student_learning_rate` - Learning rate _from that week's model_

### Additional Feature Files in `afm_outputs_rolling/`:

- `student_abilities_by_week.csv` - Student parameters for each week
  - `ability_deviation`, `learning_rate_deviation` - Random effects (deviations)
  - `ability`, `learning_rate` - Total effects (global + deviation)
  - `week_id` - Which week's model produced these parameters
- `skill_easiness_by_week.csv` - Skill parameters for each week
  - Columns: `kc_sub_skills`, `easiness`, `week_id`
- `global_params_by_week.csv` - Global parameters for each week's model
  - Columns: `week_id`, `global_intercept`, `global_learning_rate`
- `model_metadata.csv` - Information about each week's model
- `afm_model_YYYY-WNN.rds` - Individual model for each week

## Important: Understanding AFM Parameters

### Mixed Effects Model Structure

The AFM uses a mixed-effects model:

```r
outcome_bin ~ n_opportunity + (1 + n_opportunity | anon_student_id) + (1 | kc_sub_skills)
```

This creates:

- **Fixed effects (global)**: Population-level intercept and learning rate
- **Random effects (individual)**: Student-specific deviations from population means

### Parameter Interpretation

**Student's True Ability = Global Intercept + Student's Ability Deviation**
**Student's True Learning Rate = Global Learning Rate + Student's Learning Rate Deviation**

The pipelines now save both:

- **Deviations** (`ability_deviation`, `learning_rate_deviation`) - How much each student differs from average
- **Total effects** (`ability`, `learning_rate`) - The actual values to use for prediction

⚠️ **For prediction, use the total effects (ability, learning_rate), not the deviations!**

## Using Features for Prediction

### Predicting Next Week's Proficiency

To predict student proficiency in week t+1, you can use features from week t:

```r
# Example feature set for prediction
features_week_t <- c(
  # Current performance
  "avg_proficiency",           # How well they're doing now
  "n_skills_measured",         # Breadth of practice

  # Effort indicators
  "minutes_per_week",          # Time investment
  "problems_solved",           # Problem variety
  "total_opportunities",       # Practice volume

  # Difficulty context
  "week_difficulty",           # How hard were this week's skills

  # Student characteristics
  "student_ability",           # Baseline ability
  "student_learning_rate"      # How fast they learn
)
```

### Key Differences Between Pipelines

**Static Pipeline:**

- Student parameters are fixed across all weeks
- Assumes stable population and curriculum
- Simpler for prediction (fewer parameters)

**Rolling Window:**

- Student parameters evolve over time
- Captures temporal changes in student population
- Better for non-stationary environments

### Example Predictive Modeling Code

```r
# Load data
data <- read_csv("student_week_aggregations_rolling.csv")

# Create lagged features for next-week prediction
data_lagged <- data %>%
  group_by(anon_student_id) %>%
  arrange(week_id) %>%
  mutate(
    next_week_proficiency = lead(avg_proficiency),
    # Add more lagged features as needed
    prev_week_proficiency = lag(avg_proficiency)
  ) %>%
  filter(!is.na(next_week_proficiency))

# Train prediction model
model <- lm(next_week_proficiency ~ avg_proficiency + student_ability +
            student_learning_rate + week_difficulty + minutes_per_week,
            data = data_lagged)
```

### Advanced Features

For more sophisticated models, you can join additional skill-level features:

- Join `skill_easiness_by_week.csv` to get skill parameters
- Calculate skill mix diversity metrics
- Track which specific skills a student is working on

The AFM parameters provide valuable information about both student characteristics and content difficulty, making them excellent features for predictive modeling.
