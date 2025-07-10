#!/usr/bin/env Rscript

# Run Rolling Window AFM Pipeline
# This script provides different ways to execute the rolling window AFM analysis

cat("=====================================\n")
cat("Rolling Window AFM Pipeline Runner\n")
cat("=====================================\n\n")

# Function to run the pipeline with specific parameters
run_rolling_pipeline <- function(
  rolling_window_size = 4,
  student_sample_percent = 30,
  remove_outliers = TRUE,
  tukey_k = 1.5,
  proficiency_method = "new_skills_mastered",
  generate_html = TRUE,
  n_cores = NULL
) {
  
  # Set up environment variables for parameters
  Sys.setenv(
    ROLLING_WINDOW_SIZE = as.character(rolling_window_size),
    STUDENT_SAMPLE_PERCENT = as.character(student_sample_percent),
    REMOVE_DURATION_OUTLIERS = as.character(remove_outliers),
    TUKEY_K = as.character(tukey_k),
    PROFICIENCY_METHOD = proficiency_method
  )
  
  if (!is.null(n_cores)) {
    Sys.setenv(N_CORES = as.character(n_cores))
  }
  
  cat("Running rolling window pipeline with:\n")
  cat("  Rolling window size:", rolling_window_size, "weeks\n")
  cat("  Student sample:", student_sample_percent, "%\n")
  cat("  Remove outliers:", remove_outliers, "\n")
  if (remove_outliers) {
    cat("  Tukey K:", tukey_k, "\n")
  }
  cat("  Proficiency method:", proficiency_method, "\n")
  cat("  Cores:", ifelse(is.null(n_cores), "auto-detect", n_cores), "\n")
  cat("\n")
  
  start_time <- Sys.time()
  
  if (generate_html) {
    # Generate full HTML report
    cat("Generating HTML report...\n")
    rmarkdown::render(
      "combined_rolling_window.Rmd",
      output_file = paste0("rolling_window_report_", format(Sys.time(), "%Y%m%d_%H%M%S"), ".html"),
      quiet = FALSE
    )
  } else {
    # Run code blocks only (faster)
    cat("Running code blocks only (no HTML output)...\n")
    source("run_rmd_code_only.R")
    run_rmd_code_only("combined_rolling_window.Rmd")
  }
  
  end_time <- Sys.time()
  duration <- difftime(end_time, start_time, units = "mins")
  
  cat("\n✅ Pipeline completed in", round(duration, 1), "minutes\n")
}

# Interactive menu
if (interactive()) {
  cat("Select execution mode:\n")
  cat("1. Quick test (4-week window, 10% sample)\n")
  cat("2. Medium run (4-week window, 30% sample)\n")
  cat("3. Full run (4-week window, 100% sample)\n")
  cat("4. Long window (12-week window, 30% sample)\n")
  cat("5. Custom parameters\n")
  cat("6. Code only - no HTML (faster)\n")
  cat("7. DEBUG MODE - Sequential, 5% sample\n")
  cat("0. Exit\n")
  
  choice <- readline("Enter choice (0-7): ")
  
  switch(choice,
    "1" = run_rolling_pipeline(
      rolling_window_size = 4,
      student_sample_percent = 10,
      generate_html = TRUE
    ),
    "2" = run_rolling_pipeline(
      rolling_window_size = 4,
      student_sample_percent = 30,
      generate_html = TRUE
    ),
    "3" = run_rolling_pipeline(
      rolling_window_size = 4,
      student_sample_percent = 100,
      generate_html = TRUE
    ),
    "4" = run_rolling_pipeline(
      rolling_window_size = 12,
      student_sample_percent = 30,
      generate_html = TRUE
    ),
    "5" = {
      window_size <- as.numeric(readline("Rolling window size (weeks): "))
      sample_pct <- as.numeric(readline("Student sample % (1-100): "))
      outliers <- tolower(readline("Remove duration outliers? (y/n): ")) == "y"
      tukey <- if (outliers) as.numeric(readline("Tukey K (1.5 standard, 3.0 conservative): ")) else 1.5
      prof_method <- readline("Proficiency method (mean/new_skills_mastered): ")
      n_cores <- readline("Number of cores (press Enter for auto): ")
      n_cores <- if (n_cores == "") NULL else as.numeric(n_cores)
      
      run_rolling_pipeline(
        rolling_window_size = window_size,
        student_sample_percent = sample_pct,
        remove_outliers = outliers,
        tukey_k = tukey,
        proficiency_method = prof_method,
        n_cores = n_cores,
        generate_html = TRUE
      )
    },
    "6" = run_rolling_pipeline(
      rolling_window_size = 4,
      student_sample_percent = 30,
      generate_html = FALSE
    ),
    "7" = {
      Sys.setenv(DEBUG_MODE = "TRUE")
      run_rolling_pipeline(
        rolling_window_size = 4,
        student_sample_percent = 5,
        n_cores = 1,
        generate_html = FALSE
      )
    },
    "0" = cat("Exiting...\n"),
    cat("Invalid choice\n")
  )
  
} else {
  # Command line mode - parse arguments
  args <- commandArgs(trailingOnly = TRUE)
  
  if (length(args) == 0) {
    cat("Usage: Rscript run_rolling_window_pipeline.R [window_size] [sample_percent] [html]\n")
    cat("Example: Rscript run_rolling_window_pipeline.R 4 30 TRUE\n")
    cat("\nRunning with defaults: 4-week window, 30% sample, HTML output\n\n")
    run_rolling_pipeline()
  } else {
    window_size <- as.numeric(args[1])
    sample_pct <- if (length(args) >= 2) as.numeric(args[2]) else 30
    html <- if (length(args) >= 3) as.logical(args[3]) else TRUE
    
    run_rolling_pipeline(
      rolling_window_size = window_size,
      student_sample_percent = sample_pct,
      generate_html = html
    )
  }
} 