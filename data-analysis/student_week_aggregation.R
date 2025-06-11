## Student-Week Aggregation Pipeline
## This script creates student-week level aggregations with AFM-based proficiency estimates

## setup
DATA_PATH    <- "All_Data_1884_2013_0215_193821.csv"          
AFM_DIR      <- "afm_outputs7"
OUTPUT_FILE  <- "student_week_aggregations.csv"       

suppressPackageStartupMessages({
  library(readr)      # fast CSV I/O
  library(dplyr)      # data wrangling
  library(tidyr)      # data manipulation
  library(janitor)    # clean_names()
  library(lubridate)  # date manipulation
  library(stringr)    # string manipulation
  library(lme4)       # for fixef() function with mixed-effects models
})

## 1. Load original data
raw <- read_delim(DATA_PATH, delim = "\t", show_col_types = FALSE) |>
         clean_names()

## 2. Parse dates and create week identifiers
cat("Processing dates and creating week identifiers...\n")

# Parse the step start time and create week indicators
# Using ISO year/week ensures complete weeks (no split weeks across years)
# ISO weeks start on Monday and week 1 is the first week with ≥4 days in the new year
  d_weekly <- raw %>%
    mutate(
      # Parse datetime
      step_start_datetime = ymd_hms(step_start_time),
      
      # Create week identifier using ISO year and week for consistency
      week_year = isoyear(step_start_datetime),
      week_num = isoweek(step_start_datetime),
      week_id = paste0(week_year, "-W", sprintf("%02d", week_num)),
      
      # Ensure duration is numeric
      duration_sec = as.numeric(step_duration_sec)
    ) %>%
    filter(!is.na(step_start_datetime))  # Remove rows without valid timestamps

cat("Number of rows removed after parsing dates:", nrow(raw) - nrow(d_weekly), "\n")

cat("Created week identifiers. Sample weeks:", 
    paste(head(unique(d_weekly$week_id), 5), collapse=", "), "\n")

## 3. Aggregate by student-week
cat("Aggregating by student-week...\n")

student_week_basic <- d_weekly %>%
  group_by(anon_student_id, week_id) %>%
  summarise(
    # Number of minutes per week (convert from seconds, handle NAs)
    minutes_per_week = sum(duration_sec, na.rm = TRUE) / 60,
    
    # Problems solved (count unique problems)
    problems_solved = n_distinct(problem_name),
    
    # why opportunnity is implemented this way:
    # n() counts every row within each student-week group, where each row 
    # represents one attempt at a problem step by that student during that 
    # specific week.
    # TODO: check if this is correct
    total_opportunities = n(),
    
    # For joining with AFM outputs later - collect all skills attempted
    # Use consistent approach with AFM pipeline but deduplicate
    kc_sub_skills_list = kc_sub_skills,
    
    .groups = "drop"
  ) %>%
  # Unnest skills using consistent approach with AFM pipeline
  separate_longer_delim(kc_sub_skills_list, delim = "~~") %>%
  
  filter(!is.na(kc_sub_skills_list) & kc_sub_skills_list != "" & trimws(kc_sub_skills_list) != "") %>%
  # ensure that each unique combination of (student, week, skill) appears only once
  distinct(anon_student_id, week_id, kc_sub_skills_list, .keep_all = TRUE) %>%

  # Rename for clarity
  rename(kc_sub_skills = kc_sub_skills_list)

cat("Created basic student-week aggregations for", 
    n_distinct(student_week_basic$anon_student_id), "students and", 
    n_distinct(student_week_basic$week_id), "weeks\n")

## -------- 4. Load AFM outputs ----------------------------------------------
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

# Also need model intercept - load the model
afm_model <- readRDS(file.path(AFM_DIR, "afm_model.rds"))
intercept <- fixef(afm_model)["(Intercept)"]
cat("Model intercept:", intercept, "\n")

## -------- 5. Calculate proficiency for each student-skill-week -------------
cat("\nCalculating proficiency estimates...\n")

# First, we need to get the last opportunity for each student-skill in each week
last_opportunities <- d_weekly %>%
  filter(!is.na(kc_sub_skills) & kc_sub_skills != "") %>%
  separate_longer_delim(c(kc_sub_skills, opportunity_sub_skills), delim = "~~") %>%
  filter(!is.na(kc_sub_skills) & kc_sub_skills != "") %>%
  mutate(
    opportunity = as.integer(opportunity_sub_skills),
    step_datetime = ymd_hms(step_start_time)
  ) %>%
  group_by(anon_student_id, week_id, kc_sub_skills) %>%
  arrange(step_datetime) %>%
  summarise(
    last_opportunity = max(opportunity, na.rm = TRUE),
    .groups = "drop"
  )

# Join with AFM parameters and calculate proficiency
proficiency_estimates <- last_opportunities %>%
  left_join(student_abilities, by = "anon_student_id") %>%
  left_join(skill_easyness, by = c("kc_sub_skills" = "kc_default")) %>%
  mutate(
    # Handle missing values by using 0 (average)
    ability = replace_na(ability, 0),
    easyness = replace_na(easyness, 0),
    
    # Calculate log-odds
    logit_p = intercept + ability + easyness + (learning_rate * last_opportunity),
    
    # Convert to probability
    proficiency = 1 / (1 + exp(-logit_p))
  )

# Average proficiency by student-week
avg_proficiency <- proficiency_estimates %>%
  group_by(anon_student_id, week_id) %>%
  summarise(
    avg_proficiency = mean(proficiency, na.rm = TRUE),
    n_skills_measured = n(),
    .groups = "drop"
  )

## -------- 6. Combine all metrics ------------------------------------------
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

## -------- 7. Save output --------------------------------------------------
write_csv(final_aggregation, OUTPUT_FILE)

cat("\n✓ Student-week aggregations saved to:", OUTPUT_FILE, "\n")
cat("  Columns: anon_student_id, week_id, week_year, week_num,\n")
cat("           minutes_per_week, problems_solved, total_opportunities,\n")
cat("           avg_proficiency, n_skills_measured\n") 