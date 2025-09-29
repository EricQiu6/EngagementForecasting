Side note: read and clean up outdated scripts, documents, and directories not related to what we are trying to do

---

### 🔧 Modifications

- [ ] Add student ID to data (are we already doing student-level cross-validation? I think we make use of student id to structure things but not as a part of input—not sure need to verify. In anyway we should add student ID as a feature to all models and correctly to student-level cross validation)
- [ ] Add mixed-effect model with all strength (current one is not really because no student id I think)
- [ ] Add additional mean baselines
  - [ ] median, mean, median-without-0s, mean-without-0
  - [ ] Adam’s method (goal_based_predictor.py ) using 50%, 60%, 70% quantiles of past weeks' performance aggregated, you will understand with more context later.
- [ ] Separate evaluation and plotting afterwards through storing prediction output (evaluation takes a long time)

---

### 📊 Plotting / Evaluation Enhancements

- [ ] Change to aggregate MAE rank by window and architecture
- [ ] Add predicted vs. actual value plots for top models and baselines
- [ ] Include feature importance (non-linear) and feature weights (linear)
- [ ] Conduct bootstrapping / significance testing (I'm not familiar with it, but I somehow need to know if the proposed models' performance improvements over the baselines are significant; we should walk through how this is part of our evaluation pipeline; we follow your implementation)

---

### 🔬 Motivations for Experiments Below

- [ ] "zero" value distribution is very different across proficiency vs. minutes
- [ ] Generalize findings using more data
- [ ] Run ablation study with top features

---

### ✅ Evaluation Setup

Evaluate using:

- [ ] models that's given all targets to fit, vs. only non-zero targets (still evaluate on all data)
- [ ] 30% students of all data first time vs. the remaining 70% of all data; partition randomly and store separately)
- [ ] All features vs. top 5 features (min. goal) vs. top 5 (skill goal) (we'd only know these features after doing feature importance/weights using all features)
- [ ] Use different goal types:
  - [ ] "min_per_week"
  - [ ] "avg_proficiency"
- [ ] Window size range: 0 to 30 (step = 3)
