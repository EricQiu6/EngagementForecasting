## Student-Week Aggregation Pipeline
## This script creates student-week level aggregations with rolling AFM-based
## proficiency estimates.

## setup
DATA_PATH    <- "All_Data_1884_2013_0215_193821.csv"
OUTPUT_FILE  <- "student_week_aggregations_rolling.csv"

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

## 3. Function to fit AFM and extract parameters
fit_rolling_afm <- function(data) {
  # Prepare data for AFM
  afm_data <- data %>%
    separate_longer_delim(c(kc_sub_skills, opportunity_sub_skills), delim = "~~") %>%
    clean_skills_systematic("kc_sub_skills", min_length = 10, verbose = FALSE) %>%
    transmute(
      anon_student_id = anon_student_id,
      kc_default = kc_sub_skills,
      n_opportunity = as.integer(opportunity_sub_skills),
      outcome_bin = case_when(
        first_attempt == "correct" ~ 1L,
        first_attempt == "incorrect" ~ 0L,
        first_attempt == "hint" ~ 0L
      )
    ) %>%
    filter(!is.na(kc_default) & kc_default != "" & !is.na(outcome_bin))
  
  # # Check for bad pairs (opportunities not starting at 1)
  # bad_pairs <- afm_data %>% 
  #   group_by(anon_student_id, kc_default) %>% 
  #   summarise(first_opp = min(n_opportunity), .groups = "drop") %>% 
  #   filter(first_opp != 1)
  
  # if (nrow(bad_pairs))
  #   warning("Some student–skill pairs do not start at n_opportunity = 1. (negligible)")
  
  # Fit AFM model
  model <- glmer(
    outcome_bin ~ n_opportunity + (1 + n_opportunity | anon_student_id) + (1 | kc_default),
    data = afm_data,
    family = binomial,
    nAGQ = 0
  )
  
  # Extract parameters
  student_abilities <- ranef(model)$anon_student_id %>%
    tibble::rownames_to_column("anon_student_id") %>%
    rename(
      ability = `(Intercept)`,
      learning_rate = n_opportunity
    )
  
  skill_easyness <- ranef(model)$kc_default %>%
    tibble::rownames_to_column("kc_default") %>%
    rename(easyness = `(Intercept)`)
  
  # Global parameters
  global_learning_rate <- fixef(model)["n_opportunity"]
  intercept <- fixef(model)["(Intercept)"]
  
  return(list(
    student_abilities = student_abilities,
    skill_easyness = skill_easyness,
    global_learning_rate = global_learning_rate,
    intercept = intercept
  ))
}

## 4. Process each week sequentially
cat("\nProcessing weeks sequentially with rolling AFM...\n")

# Get unique weeks in chronological order
weeks <- sort(unique(d_weekly$week_id))

# Initialize empty data frame for results
rolling_results <- data.frame()

# Process each week
for (week in weeks) {
  cat(sprintf("\nProcessing week %s...\n", week))
  
  # Subset data up to current week
  data_up_to_week <- d_weekly %>%
    filter(week_id <= week)
  
  # Fit AFM on data up to current week
  afm_params <- fit_rolling_afm(data_up_to_week)
  
  # Get current week's data
  week_data <- d_weekly %>%
    filter(week_id == week)
  
  # Get last opportunity for each student-skill in current week
  last_opportunities <- week_data %>%
    separate_longer_delim(c(kc_sub_skills, opportunity_sub_skills), delim = "~~") %>%
    clean_skills_systematic("kc_sub_skills", verbose = FALSE) %>%
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
    left_join(afm_params$student_abilities, by = "anon_student_id", relationship = "many-to-one") %>%
    left_join(afm_params$skill_easyness, by = c("kc_sub_skills" = "kc_default"), relationship = "many-to-one") %>%
    mutate(
      # Handle missing values by using 0 (average)
      ability = replace_na(ability, 0),
      easyness = replace_na(easyness, 0),
      learning_rate = replace_na(learning_rate, 0),

      # Calculate proficiency as probability of solving the problem
      # Now using both global and individual learning rates
      total_learning_rate = afm_params$global_learning_rate + learning_rate,
      logit_p = afm_params$intercept + ability + easyness + (total_learning_rate * last_opportunity),
      proficiency = 1 / (1 + exp(-logit_p))
    )
  
  # Log data quality metrics
  n_students <- n_distinct(proficiency_estimates$anon_student_id)
  n_skills <- n_distinct(proficiency_estimates$kc_sub_skills)
  n_missing_ability <- sum(is.na(proficiency_estimates$ability))
  n_missing_easyness <- sum(is.na(proficiency_estimates$easyness))
  
  cat(sprintf("  Processed %d students and %d skills\n", n_students, n_skills))
  if (n_missing_ability > 0 || n_missing_easyness > 0) {
    cat(sprintf("  Warning: %d missing abilities, %d missing easyness values\n", 
                n_missing_ability, n_missing_easyness))
  }
  
  # Calculate proficiency metric based on chosen method
  if (PROFICIENCY_METHOD == "mean") {
    week_aggregation <- proficiency_estimates %>%
      group_by(anon_student_id, week_id) %>%
      summarise(
        avg_proficiency = mean(proficiency, na.rm = TRUE),
        n_skills_measured = n(),
        .groups = "drop"
      )
  } else if (PROFICIENCY_METHOD == "new_skills_mastered") {
    
    # For new skills mastered, we need to track first mastery across all weeks
    first_mastery_week <- proficiency_estimates %>%
      filter(proficiency > 0.95) %>%
      group_by(anon_student_id, kc_sub_skills) %>%
      summarise(
        first_mastery_week = min(week_id),
        .groups = "drop"
      )
    
    new_skills_per_week <- first_mastery_week %>%
      group_by(anon_student_id, first_mastery_week) %>%
      summarise(
        new_skills_mastered = n(),
        .groups = "drop"
      ) %>%
      rename(week_id = first_mastery_week)
    
    all_student_weeks <- proficiency_estimates %>%
      distinct(anon_student_id, week_id)
    
    week_aggregation <- all_student_weeks %>%
      left_join(new_skills_per_week, by = c("anon_student_id", "week_id")) %>%
      mutate(
        avg_proficiency = replace_na(new_skills_mastered, 0),
        n_skills_measured = NA_integer_
      ) %>%
      select(-new_skills_mastered)
  } else {
    stop("Invalid PROFICIENCY_METHOD. Use 'mean' or 'new_skills_mastered'")
  }
  
  # Append to results
  rolling_results <- bind_rows(rolling_results, week_aggregation)
}

## 5. Combine with basic metrics
cat("\nCombining with basic metrics...\n")

student_week_basic <- d_weekly %>%
  group_by(anon_student_id, week_id) %>%
  summarise(
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
    .groups = "drop"
  )

final_aggregation <- student_week_basic %>%
  left_join(rolling_results, by = c("anon_student_id", "week_id")) %>%
  arrange(anon_student_id, week_id)

## 6. Save output
write_csv(final_aggregation, OUTPUT_FILE)

cat("\nRolling AFM student-week aggregations saved to:", OUTPUT_FILE, "\n")