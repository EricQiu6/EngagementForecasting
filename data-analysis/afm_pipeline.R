## setup
DATA_PATH   <- "All_Data_1884_2013_0215_193821.csv"          
OUTPUT_DIR  <- "afm_outputs7"       

suppressPackageStartupMessages({
  library(readr)      # fast CSV I/O
  library(dplyr)      # data wrangling
  library(tidyr)      # separate_rows()
  library(janitor)    # clean_names()
  library(lme4)       # glmer()
  library(fs)         # file system helpers
  library(stringr)    # string manipulation
  library(tibble)     # tibble utilities
})

dir_create(OUTPUT_DIR)

## load data
raw <- read_delim(DATA_PATH, delim = "\t", show_col_types = FALSE) |>
         clean_names()

# glimpse(raw, width = 0)            # quick sanity check

## logs first attempts only already

## separates multiple KCs nested in the same row by "~~"
d_long <- raw %>% 
           separate_longer_delim(c(kc_sub_skills, opportunity_sub_skills), delim = "~~")

## -------- Clean up problematic skills before AFM processing ----------------
# Start with the original long data
cat("Original skills after separation:", nrow(d_long), "\n")
cat("Unique skills:", n_distinct(d_long$kc_sub_skills), "\n")

# 1. Remove NA values
d_step <- d_long %>%
  filter(!is.na(kc_sub_skills))
cat("After filtering NA skills:", nrow(d_step), " (removed:", nrow(d_long) - nrow(d_step), ")\n")

# 2. Remove empty strings
n_before <- nrow(d_step)
d_step <- d_step %>%
  filter(kc_sub_skills != "")
cat("After filtering empty skills:", nrow(d_step), " (removed:", n_before - nrow(d_step), ")\n")

# 3. Remove whitespace-only skills
n_before <- nrow(d_step)
d_step <- d_step %>%
  filter(str_trim(kc_sub_skills) != "")
cat("After filtering whitespace-only skills:", nrow(d_step), " (removed:", n_before - nrow(d_step), ")\n")

# 4. Remove skills that are too short (likely truncated fragments)
n_before <- nrow(d_step)
d_step <- d_step %>%
  filter(nchar(kc_sub_skills) >= 10)
cat("After filtering short skills:", nrow(d_step), " (removed:", n_before - nrow(d_step), ")\n")

# 6. Remove skills that are clearly incomplete (don't end with ] if they start with [SkillRule)
n_before <- nrow(d_step)
d_step <- d_step %>%
  filter(!(str_detect(kc_sub_skills, "^\\[SkillRule") & !str_detect(kc_sub_skills, "\\]$")))
cat("After filtering incomplete SkillRule skills:", nrow(d_step), " (removed:", n_before - nrow(d_step), ")\n")

# 7. Remove skills with weird NULL or ??? patterns
n_before <- nrow(d_step)
d_step <- d_step %>%
  filter(!str_detect(kc_sub_skills, "NULL|\\?\\?\\?"))
cat("After filtering skills with NULL or ??? patterns:", nrow(d_step), " (removed:", n_before - nrow(d_step), ")\n")

# Final cleaned data
d_clean <- d_step

cat("Clean skills after filtering:", nrow(d_clean), "\n")
cat("Total removed problematic skills:", nrow(d_long) - nrow(d_clean), "\n")
cat("Unique skills remaining:", n_distinct(d_clean$kc_sub_skills), "\n")

## build AFM variables
d_afm <- d_clean %>% 
  transmute(
    anon_student_id = anon_student_id,
    kc_default      = kc_sub_skills,
    n_opportunity   = as.integer(opportunity_sub_skills),  # start with opportunity 1
    outcome_bin     = case_when(
      first_attempt == "correct" ~ 1L,
      first_attempt == "incorrect" ~ 0L,
      first_attempt == "hint" ~ 0L
    )
  ) %>% 
  filter(!is.na(kc_default) & kc_default != "" & !is.na(outcome_bin))

# every student-skill pair must start at 1??
bad_pairs <- d_afm %>% 
  group_by(anon_student_id, kc_default) %>% 
  summarise(first_opp = min(n_opportunity), .groups = "drop") %>% 
  filter(first_opp != 1)

if (nrow(bad_pairs))
  warning("Some student–skill pairs do not start at n_opportunity = 1. (negligible)")

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

easyness_df <- ranef(m_afm)$kc_default %>% 
                   tibble::rownames_to_column("kc_default") %>% 
                   rename(easyness = `(Intercept)`)

global_learning <- tibble(
  parameter = "n_opportunity",
  estimate  = fixef(m_afm)["n_opportunity"]
)

## -------- 7. Save outputs ---------------------------------------------------
write_csv(ability_df,     path(OUTPUT_DIR, "student_ability.csv"))
write_csv(easyness_df,  path(OUTPUT_DIR, "skill_easyness.csv"))
write_csv(global_learning,path(OUTPUT_DIR, "global_learning_rate.csv"))

## -------- 8. Finished -------------------------------------------------------
cat("\nAFM pipeline complete!\n",
    "  • Model saved to:          ", path(OUTPUT_DIR, "afm_model.rds"), "\n",
    "  • Student abilities:       ", path(OUTPUT_DIR, "student_ability.csv"), "\n",
    "  • Skill difficulties:      ", path(OUTPUT_DIR, "skill_easyness.csv"), "\n",
    "  • Global learning rate:    ", path(OUTPUT_DIR, "global_learning_rate.csv"), "\n")
