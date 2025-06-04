**Big picture goal of the algorithm**

Your goal-setting algorithm is meant to *learn, from the historical log* , how to choose a weekly practice-goal for each student (focus-time vs. skill-mastery) that focuses on two principles at once:

1. We want the estimated probability of a student achieving the goal to be around 50%. This is a heuristic number derived from the fact that we want appropriately challenging goals for the student.
2. We want to gradually increase the difficulty of the goal so that the student can consistently improve over time.

---

**What data you already have**

The experiment log columns are:

| column                                     | meaning                                                                                        |
| ------------------------------------------ | ---------------------------------------------------------------------------------------------- |
| `school`,`teacher`,`grade`,`name`, | contextual identifiers                                                                         |
| `goal`                                   | goal type set that week:`"focus"`(= minutes) or `"skill"`(= mastered-skills)               |
| `week`                                   | week index in the school year, reference the `experiment_design` document for interpretation |
| `reward`                                 | whether the student earned the tangible reward, defined as achieving the goal                  |
| `focusachieved`,`skillachieved`        | binary flags for hitting the focus or skill goal                                               |
| `practice`                               | integer log of the student's actual performance on focus                                       |
| `proficient`,`mastered`                | consider only proficient column as the log of the student's actual performance on skills      |
| `skillgoal`                              | The integer skill goal that some recommender suggested                                         |
| `week_segment`,`report_week`,`call`  | Not clear what these are, perhaps reference `experiment_design` document for interpretation  |


**Algorithm setup**

Consider a predictor that infers a student's performance next week, given a time-series performance of previous weeks, with perhaps some more contextual data as appropriate. Then the predicted performance, or some tweak of it, might serve as a recommnded goal. 

Let's outline the steps to implement a very baseline algorithm focused on skills-mastered. Input is solely the past performance series without any supporting input like historical goals and other contexts. Optimize on predicted vs. actual performance exactly because we ignore concern for the gradual growth principle for now.

### Minimal-input baseline forecasting pipeline

*(using only each student’s own past weekly performance series, no goals or other context)*

1. **Choose the target series**

   skills-mastered
2. **Load & tidy the data**

   * Parse the CSV.
   * Keep the columns `name`, `week`, and the performance metric.
   * Sort rows by (`name`, `week`) and drop duplicates.
   * Forward-fill or impute any missing weeks **inside a student’s span** (e.g. carry last value, or mark with `NaN` and later mask them).
   * Put in a new csv.
3. **Split chronologically**

   *Per student* , keep the **last K weeks** (e.g.  *K = 3* ) as the test set; everything before is training. Set K as you see appropriate from analyzing the data
4. **Add one simple learnable model**

   *Fit on the stacked train pairs from all students.*

   * **AR(p) via linear regression** : $ŷ_t = β₀ + Σ_{i=1}^h β_i y_{t-i}$.
   * Decide if it is better to round to integer.
   * Optimise MSE (ordinary least-squares).
5. **Evaluate on the held-out weeks**

   Compute per-student metrics then average:

   * **MAE** (robust to outliers)
   * **RMSE** (penalises large errors)
   * **SMAPE** (scale-free interpretability)
6. **Package into a ready-to-use function**

* Save the fitted coefficients / network weights.
* Write a small function: given a student’s last t data points, return $ŷ_{t+1}$

---

**Why this meets your “very baseline” brief**

* Only per-student past performance is read—no historical goals, no demographics.
* Objective is **exact prediction accuracy** ; optimisation step matches train/test MSE.
