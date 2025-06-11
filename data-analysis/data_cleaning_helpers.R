## Data Cleaning Utilities
## Consistent data cleaning functions for the entire project
## Source this file in other scripts: source("data_cleaning_helpers.R")

suppressPackageStartupMessages({
  library(dplyr)
  library(stringr)
  library(rlang)
})

## Core Validation Functions -----------------------------------------------

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

#' Apply systematic skill cleaning
#' 
#' Applies the full sequence of skill cleaning adapted from original AFM pipeline
#' @param data data frame
#' @param skill_column name of skill column
#' @param min_length minimum skill name length (default 10)
#' @param verbose whether to print filtering results
#' @return cleaned data frame
clean_skills_systematic <- function(data, skill_column = "kc_sub_skills", 
                                   min_length = 10, verbose = TRUE) {
  skill_sym <- rlang::sym(skill_column)
  
  if (verbose) cat("Starting systematic skill cleaning...\n")
  original_data <- data
  
  # Step 1: Remove NA values
  prev_data <- data
  data <- data %>% filter(!is.na(!!skill_sym))
  if (verbose) data <- log_filter_result(prev_data, data, "After removing NA skills")
  
  # Step 2: Remove empty strings  
  prev_data <- data
  data <- data %>% filter(!!skill_sym != "")
  if (verbose) data <- log_filter_result(prev_data, data, "After removing empty skills")
  
  # Step 3: Remove whitespace-only
  prev_data <- data
  data <- data %>% filter(str_trim(!!skill_sym) != "")
  if (verbose) data <- log_filter_result(prev_data, data, "After removing whitespace-only skills")
  
  # Step 4: Remove short skills (likely fragments)
  prev_data <- data
  data <- data %>% filter(nchar(!!skill_sym) >= min_length)
  if (verbose) data <- log_filter_result(prev_data, data, paste("After removing skills shorter than", min_length, "chars"))
  
  # Step 5: Remove incomplete SkillRule patterns
  prev_data <- data
  data <- data %>% filter(!(str_detect(!!skill_sym, "^\\[SkillRule") & !str_detect(!!skill_sym, "\\]$")))
  if (verbose) data <- log_filter_result(prev_data, data, "After removing incomplete SkillRule skills")
  
  # Step 6: Remove skills with problematic patterns
  prev_data <- data
  data <- data %>% filter(!str_detect(!!skill_sym, "NULL|\\?\\?\\?"))
  if (verbose) data <- log_filter_result(prev_data, data, "After removing NULL/??? skills")
  
  if (verbose) {
    cat("Systematic skill cleaning complete!\n")
    cat("Total rows removed:", nrow(original_data) - nrow(data), "\n")
    cat("Unique skills remaining:", n_distinct(data[[skill_column]]), "\n")
  }
  
  data
}

## Logging Functions -------------------------------------------------------

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
  data_after
}

## Example Usage -----------------------------------------------------------

# For basic string validation:
# data %>% filter(is_valid(column_name))

# For systematic skill cleaning:
# data %>% clean_skills_systematic("kc_sub_skills", verbose = TRUE)

# For custom filtering with logging:
# data %>% 
#   filter(some_condition) %>%
#   log_filter_result(data, ., "After filtering on condition")
