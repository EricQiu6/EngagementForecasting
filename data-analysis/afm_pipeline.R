## -------- 0. Setup ----------------------------------------------------------
DATA_PATH   <- "All_Data_1884_2013_0215_193821.csv"          
OUTPUT_DIR  <- "afm_outputs4"       
# -----------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(readr)      # fast CSV I/O
  library(dplyr)      # data wrangling
  library(tidyr)      # separate_rows()
  library(janitor)    # clean_names()
  library(lme4)       # glmer()
  library(fs)         # file system helpers
})

dir_create(OUTPUT_DIR)

## -------- 1. Load & glimpse raw data ----------------------------------------
raw <- read_delim(DATA_PATH, delim = "\t", show_col_types = FALSE) |>
         clean_names()

glimpse(raw, width = 120)            # quick sanity check

## -------- 2. Keep FIRST attempts only ---------------------------------------
d_first <- raw %>% 
             filter(problem_view == 1)   # drop later guesses / step replays

## -------- 3. Unnest multiple skills (if any) -------------------------------
# separates multiple KCs by "~~"
d_long <- d_first %>% 
           separate_rows(kc_sub_skills, opportunity_sub_skills, sep = "~~")

## -------- 4. Build AFM variables -------------------------------------------
d_afm <- d_long %>% 
  transmute(
    anon_student_id = anon_student_id,
    kc_default      = kc_sub_skills,
    n_opportunity   = as.integer(opportunity_sub_skills) - 1,  # 0-based
    outcome_bin     = case_when(
      first_attempt == "correct" ~ 1L,
      first_attempt == "incorrect" ~ 0L,
      first_attempt == "hint" ~ 0L
    )
  ) %>% 
  filter(!is.na(kc_default) & kc_default != "" & !is.na(outcome_bin))

# Optional sanity check: every student-skill pair must start at 0
bad_pairs <- d_afm %>% 
  group_by(anon_student_id, kc_default) %>% 
  summarise(first_opp = min(n_opportunity), .groups = "drop") %>% 
  filter(first_opp != 0)

if (nrow(bad_pairs))
  warning("Some student–skill pairs do not start at n_opportunity = 0.")

## -------- 5. Fit the AFM ----------------------------------------------------

m_afm <- glmer(
  outcome_bin ~ n_opportunity + 
    (1 | anon_student_id) + 
    (1 | kc_default),
  data   = d_afm,
  family = binomial,
  nAGQ   = 0        
)

saveRDS(m_afm, file = path(OUTPUT_DIR, "afm_model.rds"))

## -------- 6. Extract latent features ---------------------------------------
ability_df <- ranef(m_afm)$anon_student_id %>% 
                tibble::rownames_to_column("anon_student_id") %>% 
                rename(ability = `(Intercept)`)

difficulty_df <- ranef(m_afm)$kc_default %>% 
                   tibble::rownames_to_column("kc_default") %>% 
                   rename(difficulty = `(Intercept)`)

global_learning <- tibble(
  parameter = "n_opportunity",
  estimate  = fixef(m_afm)["n_opportunity"]
)

## -------- 7. Save outputs ---------------------------------------------------
write_csv(ability_df,     path(OUTPUT_DIR, "student_ability.csv"))
write_csv(difficulty_df,  path(OUTPUT_DIR, "skill_difficulty.csv"))
write_csv(global_learning,path(OUTPUT_DIR, "global_learning_rate.csv"))

## -------- 8. Finished -------------------------------------------------------
cat("\nAFM pipeline complete!\n",
    "  • Model saved to:          ", path(OUTPUT_DIR, "afm_model.rds"), "\n",
    "  • Student abilities:       ", path(OUTPUT_DIR, "student_ability.csv"), "\n",
    "  • Skill difficulties:      ", path(OUTPUT_DIR, "skill_difficulty.csv"), "\n",
    "  • Global learning rate:    ", path(OUTPUT_DIR, "global_learning_rate.csv"), "\n")
