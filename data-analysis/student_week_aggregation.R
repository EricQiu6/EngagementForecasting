## Student-Week Aggregation Pipeline
## This script creates student-week level aggregations with AFM-based
## proficiency estimates.

## setup
DATA_PATH    <- "All_Data_1884_2013_0215_193821.csv"
AFM_DIR      <- "afm_outputs7"
OUTPUT_FILE  <- "student_week_aggregations2.csv"

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

## These functions provide consistent data cleaning across the pipeline

#' Check if a value is valid (not NA, not empty, not whitespace-only)
#'
#' This is the strictest validation - use for required string fields
#' @param x vector to check
#' @return logical vector
#' @examples
#' is_valid(c("hello", "", NA, "  ", "world"))  # TRUE, FALSE, FALSE, FALSE, TRUE
is_valid <- function(x) {
  !is.na(x) & x != "" & trimws(x) != ""
}

#' Log filtering results for transparency
#'
#' Helper to consistently log how many rows were removed by filtering
#' @param data_before data frame before filtering
#' @param data_after data frame after filtering
#' @param description description of the filtering step
#' @return data_after (for piping)
log_filter_result <- function(data_before, data_after, description) {
  removed <- nrow(data_before) - nrow(data_after)
  cat(description, ":", nrow(data_after), "rows (removed:", removed, ")\n")
  return(data_after)
}

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

    # Create week identifier using ISO year and week: YYYY-WNN, 2011-W37
    week_year = isoyear(step_start_datetime),
    week_num = isoweek(step_start_datetime),
    week_id = paste0(week_year, "-W", sprintf("%02d", week_num)),

    # Ensure duration is numeric
    duration_sec = as.numeric(step_duration_sec)
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

    # For skills - keep all unique values to unnest later
    kc_sub_skills = unique(kc_sub_skills[!is.na(kc_sub_skills)])
  ) %>%
  # Unnest skills using consistent approach with AFM pipeline
  # TODO: perhaps make a function for skill cleaning
  separate_longer_delim(kc_sub_skills, delim = "~~") %>%
  filter(is_valid(kc_sub_skills)) %>%
  mutate(kc_sub_skills = str_trim(kc_sub_skills)) %>%

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
  filter(is_valid(kc_sub_skills)) %>%
  separate_longer_delim(c(kc_sub_skills, opportunity_sub_skills), delim = "~~") %>%
  filter(is_valid(kc_sub_skills)) %>%
  mutate(
    opportunity = as.integer(opportunity_sub_skills)
  ) %>%
  group_by(anon_student_id, week_id, kc_sub_skills) %>%
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

    # Calculate proficiency as probability of solving the problem
    logit_p = intercept + ability + easyness + (learning_rate * last_opportunity),
    proficiency = 1 / (1 + exp(-logit_p))
  )

# Average estimated proficiency across all skills by student-week
avg_proficiency <- proficiency_estimates %>%
  group_by(anon_student_id, week_id) %>%
  summarise(
    avg_proficiency = mean(proficiency, na.rm = TRUE),
    n_skills_measured = n(),
    .groups = "drop"
  )


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