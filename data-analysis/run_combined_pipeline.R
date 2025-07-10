#!/usr/bin/env Rscript
# Script to run the combined AFM + weekly aggregation pipeline

# Load required libraries
library(rmarkdown)
library(tictoc)  # For timing

# Configuration options
args <- commandArgs(trailingOnly = TRUE)

# Set options
USE_CACHE <- ifelse(length(args) > 0 && args[1] == "cache", TRUE, FALSE)
OUTPUT_DIR <- ifelse(length(args) > 1, args[2], "results")
LAST_N_WEEKS <- ifelse(length(args) > 2, as.numeric(args[3]), NULL)
STUDENT_PERCENT <- ifelse(length(args) > 3, as.numeric(args[4]), 100)
VERBOSE <- TRUE

cat("========================================\n")
cat("Combined AFM Pipeline Runner\n")
cat("========================================\n")
cat("Configuration:\n")
cat("  Use cached model:", USE_CACHE, "\n")
cat("  Output directory:", OUTPUT_DIR, "\n")
cat("  Last N weeks:", ifelse(is.null(LAST_N_WEEKS), "All data", LAST_N_WEEKS), "\n")
cat("  Student sample %:", STUDENT_PERCENT, "\n")
cat("  Verbose mode:", VERBOSE, "\n")
cat("========================================\n\n")

# Create output directory if it doesn't exist
if (!dir.exists(OUTPUT_DIR)) {
  dir.create(OUTPUT_DIR, recursive = TRUE)
  cat("Created output directory:", OUTPUT_DIR, "\n")
}

# Option 1: Run with full HTML output
run_with_html <- function() {
  cat("Running pipeline with HTML output...\n")
  tic("Total pipeline time")
  
  # Set environment variable to control caching
  Sys.setenv(USE_CACHED_MODEL = USE_CACHE)
  
  render(
    "combined.Rmd",
    output_dir = OUTPUT_DIR,
    output_file = paste0("combined_analysis_", format(Sys.Date(), "%Y%m%d"), ".html"),
    params = list(
      use_cache = USE_CACHE
    ),
    quiet = !VERBOSE
  )
  
  toc()
  cat("\nHTML report generated in:", OUTPUT_DIR, "\n")
}

# Option 2: Run code only (no HTML)
run_code_only <- function() {
  cat("Running pipeline code only (no HTML output)...\n")
  tic("Total pipeline time")
  
  # Extract R code from Rmd
  temp_r_file <- tempfile(fileext = ".R")
  knitr::purl("combined.Rmd", output = temp_r_file, quiet = TRUE)
  
  # Modify the code to use our settings
  code <- readLines(temp_r_file)
  code <- gsub("USE_CACHED_MODEL <- FALSE", 
               paste0("USE_CACHED_MODEL <- ", USE_CACHE), 
               code)
  code <- gsub("LAST_N_WEEKS <- [0-9]+", 
               paste0("LAST_N_WEEKS <- ", ifelse(is.null(LAST_N_WEEKS), "NULL", LAST_N_WEEKS)), 
               code)
  code <- gsub("STUDENT_SAMPLE_PERCENT <- [0-9]+", 
               paste0("STUDENT_SAMPLE_PERCENT <- ", STUDENT_PERCENT), 
               code)
  writeLines(code, temp_r_file)
  
  # Source the code
  source(temp_r_file, echo = VERBOSE)
  
  # Clean up
  unlink(temp_r_file)
  
  toc()
  cat("\nPipeline completed (code only)\n")
}

# Option 3: Run with progress monitoring
run_with_monitoring <- function() {
  cat("Running pipeline with detailed progress monitoring...\n")
  
  # Set up logging
  log_file <- file.path(OUTPUT_DIR, paste0("pipeline_log_", format(Sys.time(), "%Y%m%d_%H%M%S"), ".txt"))
  
  # Capture all output
  sink(log_file, split = TRUE)  # split = TRUE shows output on console AND saves to file
  
  tic("Total pipeline time")
  
  tryCatch({
    # Run the pipeline
    run_code_only()
    
    cat("\n========================================\n")
    cat("Pipeline completed successfully!\n")
    cat("========================================\n")
    
  }, error = function(e) {
    cat("\n========================================\n")
    cat("ERROR: Pipeline failed!\n")
    cat("Error message:", e$message, "\n")
    cat("========================================\n")
    stop(e)
  }, finally = {
    toc()
    sink()  # Stop logging
    cat("\nLog file saved to:", log_file, "\n")
  })
}

# Interactive menu if no arguments provided
if (interactive() && length(args) == 0) {
  cat("How would you like to run the pipeline?\n")
  cat("1. Generate full HTML report\n")
  cat("2. Run code only (faster, no report)\n")
  cat("3. Run with detailed monitoring and logging\n")
  choice <- readline("Enter choice (1-3): ")
  
  switch(choice,
    "1" = run_with_html(),
    "2" = run_code_only(),
    "3" = run_with_monitoring(),
    cat("Invalid choice\n")
  )
} else {
  # Default to code-only mode for command line
  run_code_only()
}

cat("\nTo run with different options:\n")
cat("  Rscript run_combined_pipeline.R [cache/nocache] [output_dir] [last_n_weeks] [student_percent]\n")
cat("Examples:\n")
cat("  Rscript run_combined_pipeline.R nocache results/                # All data, all students\n")
cat("  Rscript run_combined_pipeline.R cache results/cached/           # Cached model, all data\n")
cat("  Rscript run_combined_pipeline.R nocache results/ 12             # Last 12 weeks, all students\n")
cat("  Rscript run_combined_pipeline.R nocache results/ 12 50          # Last 12 weeks, 50% of students\n")
cat("  Rscript run_combined_pipeline.R cache results/test/ 4 10        # Cached model, last 4 weeks, 10% sample\n") 