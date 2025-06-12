## Student-Week Aggregation Pipeline
## This script creates student-week level aggregations with AFM-based
## proficiency estimates.

## setup
DATA_PATH    <- "All_Data_1884_2013_0215_193821.csv"
AFM_DIR      <- "afm_outputs8"
OUTPUT_FILE  <- "student_week_aggregations2.csv"

# Control how duration is calculated
USE_PROVIDED_DURATION <- TRUE  # TRUE: use step_duration_sec, FALSE: calculate from step_end_time - step_start_time

# Control how proficiency is calculated
PROFICIENCY_METHOD <- "new_skills_mastered"  # "mean": average proficiency across all skills
                              # "new_skills_mastered": count of NEW skills reaching >95% proficiency each week
# the two methods are intended for different use cases

suppressPackageStartupMessages({
  library(readr)      # fast CSV I/O
  library(dplyr)      # data wrangling
  library(tidyr)      # data manipulation
  library(janitor)    # clean_names()
  library(lubridate)  # date manipulation
  library(stringr)    # string manipulation
  library(lme4)       # functions
  library(rlang)      # for symbol handling in helper functions
})

source("data_cleaning_helpers.R")  # Use shared cleaning functions

## 1. Load original data
raw <- read_delim(DATA_PATH, delim = "\t", show_col_types = FALSE) |>
  clean_names()


## 2. Parse dates and create week identifiers
cat("Processing dates and creating week identifiers...\n")

# Parse the step start time and create week indicators
# Using ISO year/week ensures complete weeks (no split weeks across years)
d_weekly <- raw %>%
  mutate(
    # Parse datetime
    step_start_datetime = ymd_hms(step_start_time),
    
    # Calculate duration based on parameter
    duration_sec = if (USE_PROVIDED_DURATION) {
      as.numeric(step_duration_sec)
    } else {
      # Calculate from step_end_time - step_start_time  
      step_end_datetime <- ymd_hms(step_end_time)
      as.numeric(difftime(step_end_datetime, step_start_datetime, units = "secs"))
    },

    # Create week identifier using ISO year and week: YYYY-WNN, 2011-W37
    week_year = isoyear(step_start_datetime),
    week_num = isoweek(step_start_datetime),
    week_id = paste0(week_year, "-W", sprintf("%02d", week_num)),

  ) %>%
  # Apply consistent filtering for valid timestamps
  filter(!is.na(step_start_datetime)) %>%
  log_filter_result(raw, ., "After filtering invalid timestamps")


## 3. Aggregate by student-week
cat("Aggregating by student-week...\n")

student_week_basic <- d_weekly %>%
  group_by(anon_student_id, week_id) %>%
  reframe(
    # Number of minutes per week (convert from seconds, handle NAs)
    minutes_per_week = sum(duration_sec, na.rm = TRUE) / 60,

    # Problems solved (count unique problems)
    problems_solved = n_distinct(problem_name),

    # why counting opportunities is implemented this way:
    # n() counts every row within each student-week group, where each row
    # represents one attempt at a problem step by that student during that
    # specific week.
    # TODO: check if this is correct
    total_opportunities = n(),

    kc_sub_skills = kc_sub_skills
  ) %>%
  # Unnest and clean skills using systematic approach consistent with AFM pipeline
  # This is NEEDED because some students are filtered out through the skill cleaning,
  # and we must ensure that the students match exactly with the AFM outlines.
  separate_longer_delim(kc_sub_skills, delim = "~~") %>%
  clean_skills_systematic("kc_sub_skills", verbose = FALSE) %>%

  # ensure that each unique combination of (student, week, skill) appears only once
  distinct(anon_student_id, week_id, kc_sub_skills, .keep_all = TRUE)

cat("Created basic student-week aggregations for", 
    n_distinct(student_week_basic$anon_student_id), "students and", 
    n_distinct(student_week_basic$week_id), "weeks\n")


## 4. Load AFM outputs
cat("\nLoading AFM outputs...\n")

# Load student abilities
student_abilities <- read_csv(file.path(AFM_DIR, "student_ability.csv"),
                              show_col_types = FALSE)
cat("Loaded abilities for", nrow(student_abilities), "students\n")

# Load skill difficulties (easyness)
skill_easyness <- read_csv(file.path(AFM_DIR, "skill_easyness.csv"),
                           show_col_types = FALSE)
cat("Loaded easyness for", nrow(skill_easyness), "skills\n")

# Load global learning rate
global_learning <- read_csv(file.path(AFM_DIR, "global_learning_rate.csv"),
                            show_col_types = FALSE)
learning_rate <- global_learning$estimate[1]
cat("Global learning rate:", learning_rate, "\n")

# Load model intercept
afm_model <- readRDS(file.path(AFM_DIR, "afm_model.rds"))
intercept <- fixef(afm_model)["(Intercept)"]
cat("Model intercept:", intercept, "\n")

## 5. Calculate proficiency for each student-skill-week
cat("\nCalculating proficiency estimates...\n")

# First, we need to get the last opportunity for each student-skill in each week
last_opportunities <- d_weekly %>%
  separate_longer_delim(c(kc_sub_skills, opportunity_sub_skills), delim = "~~") %>%
  clean_skills_systematic("kc_sub_skills", verbose = TRUE) %>%
  mutate(
    opportunity = as.integer(opportunity_sub_skills)
  ) %>%
  group_by(anon_student_id, week_id, kc_sub_skills) %>%
  summarise(
    last_opportunity = max(opportunity, na.rm = TRUE),
    .groups = "drop"
  )

# Diagnostic: Check skill overlap between pipelines
total_weekly_skills <- n_distinct(last_opportunities$kc_sub_skills)
afm_skills <- n_distinct(skill_easyness$kc_default)

cat("\nSkill overlap analysis:\n")
cat("  Total skills in weekly data:", total_weekly_skills, "\n")
cat("  Total skills in AFM model:", afm_skills, "\n")
# Join with AFM parameters and calculate proficiency
proficiency_estimates <- last_opportunities %>%
  left_join(student_abilities, by = "anon_student_id", relationship = "many-to-one") %>%
  left_join(skill_easyness, by = c("kc_sub_skills" = "kc_default"), relationship = "many-to-one") %>%
  mutate(
    # Handle missing values by using 0 (average)
    ability = replace_na(ability, 0),
    easyness = replace_na(easyness, 0),

    # Calculate proficiency as probability of solving the problem
    logit_p = intercept + ability + easyness + (learning_rate * last_opportunity),
    proficiency = 1 / (1 + exp(-logit_p))
  )

# Calculate proficiency metric based on chosen method
cat("Using proficiency method:", PROFICIENCY_METHOD, "\n")

if (PROFICIENCY_METHOD == "mean") {
  # Method 1: Average proficiency across all skills
  avg_proficiency <- proficiency_estimates %>%
    group_by(anon_student_id, week_id) %>%
    summarise(
      avg_proficiency = mean(proficiency, na.rm = TRUE),
      n_skills_measured = n(),
      .groups = "drop"
    )
  cat("Calculated average proficiency across skills\n")
  
} else if (PROFICIENCY_METHOD == "new_skills_mastered") {
  # Method 2: Count of NEW skills reaching >95% proficiency each week
  
  # Step 1: For each student-skill, find the first week they reached >95% proficiency
  first_mastery_week <- proficiency_estimates %>%
    filter(proficiency > 0.95) %>%
    group_by(anon_student_id, kc_sub_skills) %>%
    summarise(
      first_mastery_week = min(week_id),
      .groups = "drop"
    )
  
  # Step 2: Count how many skills were newly mastered in each week
  new_skills_per_week <- first_mastery_week %>%
    group_by(anon_student_id, first_mastery_week) %>%
    summarise(
      new_skills_mastered = n(),
      .groups = "drop"
    ) %>%
    rename(week_id = first_mastery_week)
  
  # Step 3: Create full student-week grid and fill in zeros for weeks with no new masteries
  all_student_weeks <- proficiency_estimates %>%
    distinct(anon_student_id, week_id)
  
  avg_proficiency <- all_student_weeks %>%
    left_join(new_skills_per_week, by = c("anon_student_id", "week_id")) %>%
    mutate(
      avg_proficiency = replace_na(new_skills_mastered, 0),
      n_skills_measured = NA_integer_  # measure is not applicable for this counting method
    ) %>%
    select(-new_skills_mastered)
  
  cat("Calculated count of newly mastered skills per week\n")
  
} else {
  stop("Invalid PROFICIENCY_METHOD. Use 'mean' or 'new_skills_mastered'")
}


## 6. Combine all metrics
cat("\nCombining all metrics...\n")

# Get unique student-week combinations with basic metrics
final_aggregation <- student_week_basic %>%
  select(anon_student_id, week_id, minutes_per_week, problems_solved, total_opportunities) %>%
  # dropping potentially duplicate rows from having multiple skills in a week
  distinct() %>%
  left_join(avg_proficiency, by = c("anon_student_id", "week_id")) %>%
  # pretty sure I designed week_id to follow chronologically when sorted lexicographically
  arrange(anon_student_id, week_id)

# Summary statistics
cat("\nSummary of student-week aggregations:\n")
cat("  Total student-weeks:", nrow(final_aggregation), "\n")
cat("  Unique students:", n_distinct(final_aggregation$anon_student_id), "\n")
cat("  Date range:", min(final_aggregation$week_id), "to", max(final_aggregation$week_id), "\n")
cat("\nMetric summaries:\n")
print(summary(select(final_aggregation, minutes_per_week, problems_solved, 
                    total_opportunities, avg_proficiency)))


## 7. Save output
write_csv(final_aggregation, OUTPUT_FILE)

cat("\nStudent-week aggregations saved to:", OUTPUT_FILE, "\n")