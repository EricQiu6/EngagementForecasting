"""
Paper artifacts computation for RQ items 2, 3, 5, and 11.

This module loads existing evaluation outputs (no re-evaluation), reconstructs
per-student metrics where needed by mapping saved fold indices back to
`anon_student_id` via the dataset schema, and writes paper-ready JSON/CSV/PDF
artifacts under `artifacts/`.

Implemented items:
- 2) Delta % vs averaged baselines, separately for minutes and skills
- 3) Across-family significance tests (linear vs tree vs neural)
- 5) LASSO MAE variability across windows (SD, IQR)
- 11) Feature top-k rankings per target + Kendall's tau rank stability across families

Constraints respected:
- No changes to evaluation code; analysis-only
- Feature importance normalization: sum-to-one per model at load-time only
- Matplotlib-only figures, saved as vector PDFs; also save source CSVs

Usage example:
    from paper_artifacts_rq import main
    main(
        base_dir_minutes="time-series-predictor/evaluation_outputs_with_features/rolling_new_minutes_w11_all_standard_all",
        base_dir_skills="time-series-predictor/evaluation_outputs_with_features/rolling_new_minutes_w11_all_standard_all",  # replace
        window_sizes=[1, 6, 11, 16, 21, 26]
    )

Note:
- Cold-start (weeks 1–8) is intentionally not implemented here given current CV
  setup; these artifacts are valid for the evaluated late-stage weeks.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.stats import kendalltau, wilcoxon
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

# Local imports (analysis helpers and schema/dataset)
from comprehensive_evaluation_analysis import PredictionAnalyzer  # type: ignore
from src.framework.core.schema import get_schema
from src.framework.core.data import SchemaBasedTimeSeriesDataset
from comprehensive_evaluation_with_saved_predictions import run_evaluation_with_predictions, DEFAULT_EXPERIMENT_CONFIG, create_all_models


# ---------------------------- Utilities ------------------------------------


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def save_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def bootstrap_percentile_ci(values: np.ndarray, n_bootstrap: int = 10000, alpha: float = 0.05,
                            random_state: int = 42) -> Tuple[float, float, float]:
    """Return (mean, ci_lo, ci_hi) using percentile CIs.

    Args:
        values: 1D array of per-unit statistics (e.g., per-student deltas)
        n_bootstrap: Number of resamples
        alpha: CI alpha (0.05 => 95% CI)
        random_state: Seed for reproducibility
    """
    rng = np.random.default_rng(random_state)
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    means = np.empty(n_bootstrap, dtype=float)
    n = values.size
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        means[i] = float(np.mean(values[idx]))
    lo = float(np.percentile(means, 100 * (alpha / 2)))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return float(np.mean(values)), lo, hi


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Cliff's delta (effect size) for two paired samples.

    For paired data we compute sign comparisons over pairs.
    Returns delta in [-1, 1].
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]
    if x.size == 0:
        return float("nan")
    diffs = x - y
    n_pos = int(np.sum(diffs > 0))
    n_neg = int(np.sum(diffs < 0))
    n_total = diffs.size
    return float((n_pos - n_neg) / n_total) if n_total > 0 else float("nan")


def model_name_to_family(model_name: str) -> str:
    """Map model name to family consistent with existing analysis heuristics."""
    ml = model_name.lower()
    if any(term in ml for term in ["linear", "ridge", "lasso", "elastic"]):
        return "linear"
    if any(term in ml for term in ["forest", "tree", "xgb", "gradient"]):
        return "tree"
    if any(term in ml for term in ["mlp", "lstm", "neural", "dlinear"]):
        return "neural"
    if any(term in ml for term in ["mixed", "hierarchical"]):
        return "mixed_effects"
    if any(term in ml for term in ["baseline", "naive", "mean", "median"]):
        return "baseline"
    if "adams" in ml or "goal_based" in ml:
        return "goal_based"
    return "other"


def discover_window_dir(base_dir: Path, window_size: int) -> Optional[Path]:
    """Find a window directory under base_dir, trying several naming patterns."""
    patterns = [
        f"*w{window_size}_*",
        f"*window{window_size}_*",
        f"*_{window_size}_*",
    ]
    for pat in patterns:
        matches = list(base_dir.glob(pat))
        if matches:
            return matches[0]
    return None


def load_prediction_analyzer(window_dir: Path) -> PredictionAnalyzer:
    return PredictionAnalyzer(str(window_dir))


def resolve_dataset(window_dir: Path) -> SchemaBasedTimeSeriesDataset:
    """Recreate the dataset used in an evaluation directory to map indices -> student_id."""
    config_path = window_dir / "evaluation_config.json"
    config = read_json(config_path)
    # Resolve data path relative to window dir
    data_path = Path(config["dataset_path"])
    # Optional overrides via environment variables for robustness
    env_abs = os.environ.get("PAPER_DATASET_PATH")
    env_root = os.environ.get("PAPER_DATASET_ROOT")
    if env_abs:
        env_abs_path = Path(env_abs).expanduser().resolve()
        if not env_abs_path.exists():
            raise FileNotFoundError(f"PAPER_DATASET_PATH='{env_abs_path}' does not exist")
        resolved_data_path = env_abs_path
    elif env_root:
        resolved_data_path = Path(env_root).expanduser().resolve() / data_path
    else:
        resolved_data_path = None
    if resolved_data_path is not None:
        data_path = resolved_data_path
    elif not data_path.is_absolute():
        # Try resolving relative to several likely roots
        candidate_bases: List[Path] = [
            window_dir,
            window_dir.parent,
            window_dir.parent.parent if window_dir.parent.parent else window_dir.parent,
            window_dir.parent.parent.parent if window_dir.parent.parent else window_dir.parent,
            Path(__file__).resolve().parent,  # time-series-predictor/
            Path(__file__).resolve().parent.parent,  # repo root
        ]
        resolved: Optional[Path] = None
        for base in candidate_bases:
            candidate = (base / data_path).resolve()
            if candidate.exists():
                resolved = candidate
                break
        if resolved is None:
            # Last resort: if path starts with ../, try from script dir
            candidate = (Path(__file__).resolve().parent / data_path).resolve()
            if candidate.exists():
                resolved = candidate
        if resolved is None:
            raise FileNotFoundError(f"Could not resolve dataset path '{data_path}' from '{window_dir}'. Tried bases: "
                                    f"{[str(b) for b in candidate_bases]}")
        data_path = resolved
    schema = get_schema(config["schema_name"])
    sequence_length = int(config["window_size"])  # required by dataset as sequence_length
    ds = SchemaBasedTimeSeriesDataset(
        data_path=str(data_path),
        schema=schema,
        sequence_length=sequence_length,
        load_in_memory=True,
        validate_data=False,
    )
    return ds


def map_index_to_student(window_dir: Path) -> Dict[int, str]:
    """Return mapping from dataset sequence index (seq_idx) -> anon_student_id."""
    ds = resolve_dataset(window_dir)
    mapping: Dict[int, str] = {}
    for i, seq in enumerate(ds.sequence_index):
        mapping[i] = str(seq["student"])  # anon_student_id
    return mapping


def load_fold_predictions_for_model(window_dir: Path, model_name: str,
                                    idx2student: Mapping[int, str]) -> pd.DataFrame:
    """Load all fold predictions for a model and attach student_id.

    Returns a DataFrame with columns: student_id, fold, abs_error
    """
    model_dir = window_dir / model_name
    if not model_dir.exists():
        return pd.DataFrame(columns=["student_id", "fold", "abs_error"])  # empty
    rows: List[Dict[str, Any]] = []
    for f in model_dir.glob("fold_*_predictions.json"):
        try:
                data = read_json(f)
        except Exception:
                continue
        y_true = np.array(data.get("y_true", []), dtype=float)
        y_pred = np.array(data.get("y_pred", []), dtype=float)
        indices = list(map(int, data.get("indices", [])))
        fold_idx = int(data.get("fold_idx", -1))
        if len(y_true) != len(y_pred) or len(y_true) != len(indices):
            # skip malformed
            continue
        abs_err = np.abs(y_true - y_pred)
        for i, seq_idx in enumerate(indices):
            student = idx2student.get(seq_idx)
            if student is None:
                continue
            rows.append({
                "student_id": student,
                "fold": fold_idx,
                "abs_error": float(abs_err[i])
            })
    if not rows:
        return pd.DataFrame(columns=["student_id", "fold", "abs_error"])  # empty
    return pd.DataFrame(rows)


def per_student_mae_from_predictions(pred_df: pd.DataFrame) -> pd.Series:
    """Compute per-student MAE given per-sample abs_error rows."""
    if pred_df.empty:
        return pd.Series(dtype=float)
    return pred_df.groupby("student_id")["abs_error"].mean()


def collect_per_student_mae_for_models(window_dir: Path, model_names: Sequence[str]) -> Dict[str, pd.Series]:
    """Return mapping model_name -> per-student MAE series for this window.
    Missing models are returned as empty series.
    """
    idx2student = map_index_to_student(window_dir)
    result: Dict[str, pd.Series] = {}
    for m in model_names:
        df = load_fold_predictions_for_model(window_dir, m, idx2student)
        result[m] = per_student_mae_from_predictions(df)
    return result


def average_across_windows_student_mae(
    per_window_mae: Mapping[int, pd.Series]
) -> pd.Series:
    """Average per-student MAE across windows, aligning by student and ignoring missing windows.
    Returns a Series indexed by student_id.
    """
    if not per_window_mae:
        return pd.Series(dtype=float)
    # Concatenate into DataFrame with columns per window
    df = pd.DataFrame({w: s for w, s in per_window_mae.items()})
    return df.mean(axis=1, skipna=True)


def get_baseline_model_names() -> List[str]:
    return [
        "average_all",
        "naive_forecast",  # last-week
        "median_all",
        "median_no_zeros",
        "mean_no_zeros",
        "adams_baseline_50",
        "adams_baseline_60",
        "adams_baseline_70",
    ]


def compute_baseline_average_mae(
    maes_by_model: Mapping[str, pd.Series]
) -> pd.Series:
    """Compute per-student average MAE across available baselines.

    Only baselines present in the mapping are used. If a student is missing from
    some baseline, the average is taken over available values for that student.
    """
    present = {m: s for m, s in maes_by_model.items() if not s.empty}
    if not present:
        return pd.Series(dtype=float)
    df = pd.DataFrame(present)
    return df.mean(axis=1, skipna=True)


def select_top_models_by_family_across_windows(
    window_dirs: Sequence[Path]
) -> Tuple[Dict[str, str], pd.DataFrame]:
    """Return mapping family->top_model_name based on lowest avg MAE across windows.

    Also returns the concatenated performance summaries for inspection.
    """
    summaries: List[pd.DataFrame] = []
    for wd in window_dirs:
        try:
            analyzer = load_prediction_analyzer(wd)
            df = analyzer.create_performance_summary()
            df["window_dir"] = str(wd)
            summaries.append(df)
        except Exception:
            continue
    if not summaries:
        return {}, pd.DataFrame()
    all_df = pd.concat(summaries, ignore_index=True)
    all_df["family"] = all_df["model"].apply(model_name_to_family)
    # Compute average MAE per model across windows
    avg_mae = all_df.groupby(["model", "family"])['mae_mean'].mean().reset_index()
    top_by_family: Dict[str, str] = {}
    for fam in ["linear", "tree", "neural"]:
        fam_df = avg_mae[avg_mae["family"] == fam]
        if fam_df.empty:
            continue
        best_row = fam_df.sort_values("mae_mean").iloc[0]
        top_by_family[fam] = str(best_row["model"])  # model name
    return top_by_family, all_df


def select_top_model_overall_across_windows(window_dirs: Sequence[Path]) -> Optional[str]:
    """Select a single top model (lowest avg MAE across provided windows)."""
    summaries: List[pd.DataFrame] = []
    for wd in window_dirs:
        try:
            analyzer = load_prediction_analyzer(wd)
            df = analyzer.create_performance_summary()
            summaries.append(df)
        except Exception:
            continue
    if not summaries:
        return None
    all_df = pd.concat(summaries, ignore_index=True)
    avg_mae = all_df.groupby('model')['mae_mean'].mean().reset_index()
    best_row = avg_mae.sort_values('mae_mean').iloc[0]
    return str(best_row['model'])


def select_top_nonbaseline_model_overall_across_windows(window_dirs: Sequence[Path]) -> Optional[str]:
    """Select top model excluding baselines/goal_based across provided windows."""
    summaries: List[pd.DataFrame] = []
    for wd in window_dirs:
        try:
            analyzer = load_prediction_analyzer(wd)
            df = analyzer.create_performance_summary()
            summaries.append(df)
        except Exception:
            continue
    if not summaries:
        return None
    all_df = pd.concat(summaries, ignore_index=True)
    all_df['family'] = all_df['model'].apply(model_name_to_family)
    filt = all_df[~all_df['family'].isin(['baseline', 'goal_based'])]
    if filt.empty:
        return None
    avg_mae = filt.groupby('model')['mae_mean'].mean().reset_index()
    best_row = avg_mae.sort_values('mae_mean').iloc[0]
    return str(best_row['model'])


# ---------------------------- Item 5 ----------------------------------------


def compute_lasso_window_variability(window_dirs: Sequence[Path]) -> Dict[str, Any]:
    """Item 5: SD and IQR of LASSO MAE across windows."""
    maes: List[float] = []
    for wd in window_dirs:
        try:
            analyzer = load_prediction_analyzer(wd)
            df = analyzer.create_performance_summary()
            lasso_rows = df[df['model'].str.contains('lasso', case=False, regex=False)]
            if lasso_rows.empty:
                continue
            # If multiple lasso variants, average within window first
            maes.append(float(lasso_rows['mae_mean'].mean()))
        except Exception:
            continue
    if not maes:
        return {"lasso_window_variability": {"sd": None, "iqr": None}}
    arr = np.array(maes, dtype=float)
    sd = float(np.std(arr))
    iqr = float(np.percentile(arr, 75) - np.percentile(arr, 25))
    return {"lasso_window_variability": {"sd": sd, "iqr": iqr}}


# ---------------------------- Item 2 ----------------------------------------


def compute_item2_delta(
    window_dirs: Sequence[Path]
) -> Dict[str, Any]:
    """Item 2: Δ% vs averaged baselines separately for minutes and skills.

    Returns mapping with keys minutes and skills; each includes mean_delta_pct and CI.
    """
    baseline_names = set(get_baseline_model_names())
    by_target_dirs: Dict[str, List[Path]] = {"minutes": [], "skills": []}
    for wd in window_dirs:
        cfg = read_json(wd / "evaluation_config.json")
        goal_type = str(cfg.get("goal_type", "")).lower()
        if "minute" in goal_type:
            by_target_dirs["minutes"].append(wd)
        else:
            by_target_dirs["skills"].append(wd)

    results: Dict[str, Any] = {}
    for target, dirs in by_target_dirs.items():
        if not dirs:
            results[target] = {"mean_delta_pct": None, "ci": [None, None], "top_model": None, "top_mae_mean": None, "baseline_mae_mean": None, "n_students": 0}
            continue

        # Select a single top model overall across these windows
        top_model = select_top_model_overall_across_windows(dirs)
        if top_model is None:
            results[target] = {"mean_delta_pct": None, "ci": [None, None], "top_model": None, "top_mae_mean": None, "baseline_mae_mean": None, "n_students": 0}
            continue

        # For each window, compute per-student MAE for top model and for each baseline
        model_mae_by_window: Dict[int, pd.Series] = {}
        baseline_avg_by_window: Dict[int, pd.Series] = {}
        for wd in dirs:
            cfg = read_json(wd / "evaluation_config.json")
            w = int(cfg.get("window_size", -1))
            # list models in this window folder to know availability
            model_dirs = [p.name for p in wd.iterdir() if p.is_dir()]
            available_baselines = [m for m in baseline_names if m in model_dirs]

            maes_all = collect_per_student_mae_for_models(wd, [top_model] + available_baselines)
            model_mae = maes_all.get(top_model, pd.Series(dtype=float))
            baseline_avg = compute_baseline_average_mae({k: v for k, v in maes_all.items() if k in available_baselines})

            if not model_mae.empty:
                model_mae_by_window[w] = model_mae
            if not baseline_avg.empty:
                baseline_avg_by_window[w] = baseline_avg

        # Aggregate across windows per student (mean over available windows)
        agg_model_mae = average_across_windows_student_mae(model_mae_by_window)
        agg_base_mae = average_across_windows_student_mae(baseline_avg_by_window)
        # Align students and compute delta percent where baseline > 0
        df_join = pd.concat([agg_model_mae.rename("model"), agg_base_mae.rename("baseline")], axis=1).dropna()
        df_join = df_join[df_join["baseline"] > 0]
        if df_join.empty:
            results[target] = {"mean_delta_pct": None, "ci": [None, None], "top_model": top_model, "top_mae_mean": None, "baseline_mae_mean": None, "n_students": 0}
            continue
        deltas = (df_join["baseline"] - df_join["model"]) / df_join["baseline"] * 100.0
        mean_delta, lo, hi = bootstrap_percentile_ci(deltas.values, n_bootstrap=10000, alpha=0.05)
        results[target] = {
            "mean_delta_pct": round(float(mean_delta), 3),
            "ci": [round(float(lo), 3), round(float(hi), 3)],
            "top_model": top_model,
            "top_mae_mean": round(float(df_join["model"].mean()), 6),
            "baseline_mae_mean": round(float(df_join["baseline"].mean()), 6),
            "n_students": int(df_join.shape[0])
        }

    return results


# ---------------------------- Item 3 ----------------------------------------


def compute_item3_family_tests(window_dirs: Sequence[Path]) -> Dict[str, Any]:
    """Item 3: Across-family significance tests (linear vs tree vs neural), per target.

    Returns dict with keys minutes and skills, each containing pairwise results.
    """
    by_target_dirs: Dict[str, List[Path]] = {"minutes": [], "skills": []}
    for wd in window_dirs:
        cfg = read_json(wd / "evaluation_config.json")
        goal_type = str(cfg.get("goal_type", "")).lower()
        if "minute" in goal_type:
            by_target_dirs["minutes"].append(wd)
        else:
            by_target_dirs["skills"].append(wd)

    out: Dict[str, Any] = {}
    for target, dirs in by_target_dirs.items():
        if not dirs:
            out[target] = {}
            continue
        # Select top model within each family
        top_by_family, _ = select_top_models_by_family_across_windows(dirs)
        fam_models = {k: v for k, v in top_by_family.items() if k in {"linear", "tree", "neural"}}
        # Prepare aggregated per-student MAE vectors per selected model
        maes_agg_by_model: Dict[str, pd.Series] = {}
        for fam, model_name in fam_models.items():
            per_window: Dict[int, pd.Series] = {}
            for wd in dirs:
                cfg = read_json(wd / "evaluation_config.json")
                w = int(cfg.get("window_size", -1))
                maes = collect_per_student_mae_for_models(wd, [model_name])
                s = maes.get(model_name, pd.Series(dtype=float))
                if not s.empty:
                    per_window[w] = s
            maes_agg_by_model[model_name] = average_across_windows_student_mae(per_window)

        # Pairwise tests
        pairs = [("linear", "tree"), ("linear", "neural"), ("tree", "neural")]
        res_pairs: Dict[str, Any] = {}
        for a, b in pairs:
            if a not in fam_models or b not in fam_models:
                continue
            ma = maes_agg_by_model.get(fam_models[a], pd.Series(dtype=float))
            mb = maes_agg_by_model.get(fam_models[b], pd.Series(dtype=float))
            df = pd.concat([ma.rename("A"), mb.rename("B")], axis=1).dropna()
            if df.empty:
                continue
            diffs = (df["A"].values - df["B"].values).astype(float)
            mean_diff, lo, hi = bootstrap_percentile_ci(diffs, n_bootstrap=10000, alpha=0.05)
            cd = cliffs_delta(df["A"].values, df["B"].values)
            p_wil = None
            if _HAS_SCIPY:
                try:
                    stat = wilcoxon(diffs)
                    p_wil = float(stat.pvalue)
                except Exception:
                    p_wil = None
            res_pairs[f"{a}_vs_{b}"] = {
                "mean_diff": round(float(mean_diff), 6),
                "ci": [round(float(lo), 6), round(float(hi), 6)],
                "cliffs_delta": None if (cd is None or math.isnan(cd)) else round(float(cd), 6),
                "p_wilcoxon": p_wil,
            }
        out[target] = res_pairs
    return out


# ------------------ Additional family performance JSON (new) --------------


def compute_family_mae_new(window_dirs: Sequence[Path]) -> Dict[str, Any]:
    """Aggregate per-architecture MAE for minutes and skills.

    For each target (minutes, skills) and family (linear, tree, neural, mixed_effects),
    we select the best model per family within each window (lowest mae_mean),
    then report the average of those per-window MAEs. Also include per-window breakdown.

    Returns a JSON-serializable dict.
    """
    by_target_dirs: Dict[str, List[Path]] = {"minutes": [], "skills": []}
    for wd in window_dirs:
        cfg = read_json(wd / "evaluation_config.json")
        goal_type = str(cfg.get("goal_type", "")).lower()
        if "minute" in goal_type:
            by_target_dirs["minutes"].append(wd)
        else:
            by_target_dirs["skills"].append(wd)

    out: Dict[str, Any] = {}
    families = ["linear", "tree", "neural", "mixed_effects"]
    for target, dirs in by_target_dirs.items():
        fam_to_window_entries: Dict[str, List[Dict[str, Any]]] = {f: [] for f in families}
        for wd in dirs:
            try:
                cfg = read_json(wd / "evaluation_config.json")
                w = int(cfg.get("window_size", -1))
                analyzer = load_prediction_analyzer(wd)
                perf = analyzer.create_performance_summary()
                perf["family"] = perf["model"].apply(model_name_to_family)
                # pick best per family for this window
                best = perf.sort_values("mae_mean").groupby("family", as_index=False).first()
                for _, row in best.iterrows():
                    fam = str(row["family"]).lower()
                    if fam in fam_to_window_entries:
                        fam_to_window_entries[fam].append({
                            "window": w,
                            "model": str(row["model"]),
                            "mae_mean": float(row["mae_mean"])
                        })
            except Exception:
                continue

        # summarize per family
        fam_summary: Dict[str, Any] = {}
        for fam in families:
            rows = fam_to_window_entries.get(fam, [])
            if not rows:
                fam_summary[fam] = {
                    "mae_mean": None,
                    "n_windows": 0,
                    "per_window": []
                }
                continue
            mae_vals = [r["mae_mean"] for r in rows]
            fam_summary[fam] = {
                "mae_mean": float(np.mean(mae_vals)),
                "n_windows": int(len(rows)),
                "per_window": rows
            }
        out[target] = fam_summary
    return out


def compute_model_breakdown(window_dirs: Sequence[Path]) -> Dict[str, Any]:
    """Compute per-model average MAE across windows and Δ% vs averaged baselines with CI.

    Returns a dict with keys minutes and skills, each mapping to a list of entries:
      {model, family, mae_mean, n_windows, delta_pct_mean, delta_pct_ci, n_students}
    """
    baseline_names = set(get_baseline_model_names())
    by_target_dirs: Dict[str, List[Path]] = {"minutes": [], "skills": []}
    for wd in window_dirs:
        cfg = read_json(wd / "evaluation_config.json")
        goal_type = str(cfg.get("goal_type", "")).lower()
        if "minute" in goal_type:
            by_target_dirs["minutes"].append(wd)
        else:
            by_target_dirs["skills"].append(wd)

    out: Dict[str, Any] = {}
    for target, dirs in by_target_dirs.items():
        if not dirs:
            out[target] = []
            continue

        # 1) Collect performance summaries to compute mae_mean across windows per model
        perf_rows: List[pd.DataFrame] = []
        # Also track per-window mae_mean per model
        model_window_mae: Dict[Tuple[str, int], float] = {}
        for wd in dirs:
            try:
                analyzer = load_prediction_analyzer(wd)
                df_full = analyzer.create_performance_summary().copy()
                df = df_full[["model", "mae_mean"]].copy()
                df["window_dir"] = str(wd)
                # resolve window number
                cfg_w = read_json(wd / "evaluation_config.json")
                wnum = int(cfg_w.get("window_size", -1))
                for _, r in df_full.iterrows():
                    model_window_mae[(str(r["model"]), wnum)] = float(r["mae_mean"])
                perf_rows.append(df)
            except Exception:
                continue
        if not perf_rows:
            out[target] = []
            continue
        perf_all = pd.concat(perf_rows, ignore_index=True)
        # Average MAE across windows (only where present)
        avg_mae = perf_all.groupby("model")["mae_mean"].mean().reset_index()

        # 2) For Δ% vs baselines, build per-student aggregated MAE for each model and for baseline-average
        #    across available windows for that target.
        # Build list of all models observed
        all_models = sorted(avg_mae["model"].unique().tolist())

        # Prepare baseline per-window aggregated Series upfront to reuse
        baseline_avg_by_window: Dict[int, pd.Series] = {}
        for wd in dirs:
            cfg = read_json(wd / "evaluation_config.json")
            w = int(cfg.get("window_size", -1))
            model_dirs = [p.name for p in wd.iterdir() if p.is_dir()]
            available_baselines = [m for m in baseline_names if m in model_dirs]
            if not available_baselines:
                continue
            maes_all = collect_per_student_mae_for_models(wd, available_baselines)
            base_avg = compute_baseline_average_mae(maes_all)
            if not base_avg.empty:
                baseline_avg_by_window[w] = base_avg

        entries: List[Dict[str, Any]] = []
        for model_name in all_models:
            family = model_name_to_family(model_name)
            # Average MAE across windows for this model (from perf_all)
            mae_mean_val = float(avg_mae.loc[avg_mae["model"] == model_name, "mae_mean"].values[0])

            # Build per-student aggregated vectors across windows for this model
            per_window_mae: Dict[int, pd.Series] = {}
            per_window_records: List[Dict[str, Any]] = []
            for wd in dirs:
                cfg = read_json(wd / "evaluation_config.json")
                w = int(cfg.get("window_size", -1))
                maes = collect_per_student_mae_for_models(wd, [model_name])
                s = maes.get(model_name, pd.Series(dtype=float))
                if not s.empty:
                    per_window_mae[w] = s
                # window-specific baseline and delta% computation
                model_mae_series = s
                # compute baseline average for this window (reuse if available)
                if w in baseline_avg_by_window:
                    base_series = baseline_avg_by_window[w]
                else:
                    model_dirs = [p.name for p in wd.iterdir() if p.is_dir()]
                    available_baselines = [m for m in baseline_names if m in model_dirs]
                    if available_baselines:
                        maes_all = collect_per_student_mae_for_models(wd, available_baselines)
                        base_series = compute_baseline_average_mae(maes_all)
                        if not base_series.empty:
                            baseline_avg_by_window[w] = base_series
                    else:
                        base_series = pd.Series(dtype=float)

                # join and compute window delta if both present
                win_delta_mean = None
                win_ci = [None, None]
                n_students_win = 0
                if not model_mae_series.empty and not base_series.empty:
                    dfw = pd.concat([
                        model_mae_series.rename("model"),
                        base_series.rename("baseline")
                    ], axis=1).dropna()
                    dfw = dfw[dfw["baseline"] > 0]
                    if not dfw.empty:
                        deltas_w = (dfw["baseline"] - dfw["model"]) / dfw["baseline"] * 100.0
                        d_mean, d_lo, d_hi = bootstrap_percentile_ci(deltas_w.values, n_bootstrap=10000, alpha=0.05)
                        win_delta_mean = round(float(d_mean), 6)
                        win_ci = [round(float(d_lo), 6), round(float(d_hi), 6)]
                        n_students_win = int(dfw.shape[0])

                per_window_records.append({
                    "window": w,
                    "mae_mean": model_window_mae.get((model_name, w), None),
                    "delta_pct_mean": win_delta_mean,
                    "delta_pct_ci": win_ci,
                    "n_students": n_students_win
                })

            agg_model_mae = average_across_windows_student_mae(per_window_mae)
            agg_base_mae = average_across_windows_student_mae(baseline_avg_by_window)
            df_join = pd.concat([agg_model_mae.rename("model"), agg_base_mae.rename("baseline")], axis=1).dropna()
            df_join = df_join[df_join["baseline"] > 0]
            if df_join.empty:
                entries.append({
                    "model": model_name,
                    "family": family,
                    "mae_mean": mae_mean_val,
                    "n_windows": int(perf_all[perf_all["model"] == model_name]["window_dir"].nunique()),
                    "delta_pct_mean": None,
                    "delta_pct_ci": [None, None],
                    "n_students": 0,
                    "per_window": per_window_records
                })
                continue
            deltas = (df_join["baseline"] - df_join["model"]) / df_join["baseline"] * 100.0
            mean_delta, lo, hi = bootstrap_percentile_ci(deltas.values, n_bootstrap=10000, alpha=0.05)
            entries.append({
                "model": model_name,
                "family": family,
                "mae_mean": mae_mean_val,
                "n_windows": int(perf_all[perf_all["model"] == model_name]["window_dir"].nunique()),
                "delta_pct_mean": round(float(mean_delta), 6),
                "delta_pct_ci": [round(float(lo), 6), round(float(hi), 6)],
                "n_students": int(df_join.shape[0]),
                "per_window": per_window_records
            })

        # Sort entries by mae_mean asc
        entries = sorted(entries, key=lambda x: (float("inf") if x["mae_mean"] is None else x["mae_mean"]))
        out[target] = entries
    return out


# ---------------------------- Item 11 ---------------------------------------


def normalize_importance(imp: Mapping[str, float]) -> Dict[str, float]:
    arr = np.array(list(imp.values()), dtype=float)
    s = float(np.sum(arr))
    if s <= 0:
        # avoid division by zero; return zeros
        return {k: 0.0 for k in imp.keys()}
    return {k: float(v) / s for k, v in imp.items()}


def load_feature_importances(window_dir: Path) -> List[Tuple[str, str, Dict[str, float]]]:
    """Return list of (model_name, family, normalized_importance) for models with importance."""
    out: List[Tuple[str, str, Dict[str, float]]] = []
    for model_dir in [p for p in window_dir.iterdir() if p.is_dir()]:
        summary_path = model_dir / "summary.json"
        if not summary_path.exists():
            continue
        try:
            summary = read_json(summary_path)
        except Exception:
            continue
        imp = summary.get("feature_importance")
        if not imp:
            continue
        model_name = str(summary.get("model_name", model_dir.name))
        family = model_name_to_family(model_name)
        norm = normalize_importance({str(k): float(v) for k, v in imp.items()})
        out.append((model_name, family, norm))
    return out


def compute_item11_feature_ranks_and_tau(window_dirs: Sequence[Path],
                                         k_top: int = 15) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Item 11: Top-k feature ranks per target + Kendall's tau across families.

    Returns: (minutes_ranks_df, skills_ranks_df, tau_metrics_json)
    """
    by_target_dirs: Dict[str, List[Path]] = {"minutes": [], "skills": []}
    for wd in window_dirs:
        cfg = read_json(wd / "evaluation_config.json")
        goal_type = str(cfg.get("goal_type", "")).lower()
        if "minute" in goal_type:
            by_target_dirs["minutes"].append(wd)
        else:
            by_target_dirs["skills"].append(wd)

    out_tau: Dict[str, Any] = {}
    ranks_by_target: Dict[str, pd.DataFrame] = {}

    def feature_to_category(name: str) -> str:
        n = name.lower()
        # Prior achievement group
        if any(key in n for key in ["starting_ability", "starting_ability_quartile", "consistency", "performance_consistency", "learning_acceleration_capacity"]):
            return "prior_achievement"
        # Gaps group
        if any(key in n for key in ["gap", "missing", "temporal_gap"]):
            return "gaps"
        # AFM-induced group (ability, learning rate, difficulty)
        if any(key in n for key in ["ability", "learning_rate", "difficulty", "afm"]):
            return "afm_induced"
        # Lags / stats (default catch-all including current, trends, aggregates)
        if any(key in n for key in ["lag", "std", "mean", "median", "iqr", "range", "trend", "avg_change", "rolling", "percentile", "current_"]):
            return "lags_stats"
        # Fallback
        return "lags_stats"

    for target, dirs in by_target_dirs.items():
        if not dirs:
            ranks_by_target[target] = pd.DataFrame(columns=["rank", "feature", "linear_importance", "tree_importance", "neural_importance", "overall_importance"])
            out_tau[target] = {"median_tau": None, "pairwise": {}, "category_level": {"median_tau": None, "pairwise": {}, "category_importance_by_family": {}}}
            continue

        # Accumulate normalized importance per model across windows, then average per family
        family_to_feature_to_vals: Dict[str, Dict[str, List[float]]] = {}
        for wd in dirs:
            for model_name, family, norm_imp in load_feature_importances(wd):
                if family not in {"linear", "tree", "neural"}:
                    continue
                fam_map = family_to_feature_to_vals.setdefault(family, {})
                for feat, val in norm_imp.items():
                    fam_map.setdefault(feat, []).append(float(val))

        # Compute average importance per family
        all_features: List[str] = sorted({f for fam in family_to_feature_to_vals.values() for f in fam.keys()})
        family_importance_avg: Dict[str, Dict[str, float]] = {}
        for fam, fmap in family_to_feature_to_vals.items():
            family_importance_avg[fam] = {feat: float(np.mean(fmap.get(feat, [0.0]))) for feat in all_features}

        # Overall importance as the mean across available families
        overall = {feat: float(np.mean([family_importance_avg.get(fam, {}).get(feat, 0.0) for fam in ["linear", "tree", "neural"]])) for feat in all_features}
        # Build ranks DataFrame
        df = pd.DataFrame({
            "feature": all_features,
            "linear_importance": [family_importance_avg.get("linear", {}).get(f, 0.0) for f in all_features],
            "tree_importance": [family_importance_avg.get("tree", {}).get(f, 0.0) for f in all_features],
            "neural_importance": [family_importance_avg.get("neural", {}).get(f, 0.0) for f in all_features],
            "overall_importance": [overall.get(f, 0.0) for f in all_features],
        })
        df = df.sort_values("overall_importance", ascending=False).reset_index(drop=True)
        df["rank"] = np.arange(1, len(df) + 1)
        ranks_by_target[target] = df[["rank", "feature", "linear_importance", "tree_importance", "neural_importance", "overall_importance"]].head(k_top)

        # Kendall's tau between family rank vectors across all shared features (per-feature)
        tau_pairs: Dict[str, Optional[float]] = {}
        fams = ["linear", "tree", "neural"]
        # Convert importances to ranks (higher importance => lower rank number)
        fam_ranks: Dict[str, pd.Series] = {}
        for fam in fams:
            if fam in family_importance_avg and family_importance_avg[fam]:
                series = pd.Series(family_importance_avg[fam])
                # Rank descending importance: top importance gets rank 1
                fam_ranks[fam] = series.rank(ascending=False, method="average")
        pair_list = [("linear", "tree"), ("linear", "neural"), ("tree", "neural")]
        tau_vals: List[float] = []
        for a, b in pair_list:
            if a not in fam_ranks or b not in fam_ranks:
                tau_pairs[f"{a}_{b}"] = None
                continue
            joined = pd.concat([fam_ranks[a].rename("A"), fam_ranks[b].rename("B")], axis=1).dropna()
            if joined.empty or not _HAS_SCIPY:
                tau_pairs[f"{a}_{b}"] = None
                continue
            try:
                tau, _ = kendalltau(joined["A"].values, joined["B"].values)
                tau_pairs[f"{a}_{b}"] = None if (tau is None or math.isnan(tau)) else float(tau)
                if tau is not None and not math.isnan(tau):
                    tau_vals.append(float(tau))
            except Exception:
                tau_pairs[f"{a}_{b}"] = None
        median_tau = float(np.median(tau_vals)) if tau_vals else None
        # Category-level aggregation and Kendall's tau
        # Compute family -> category -> mean importance
        categories = ["afm_induced", "gaps", "prior_achievement", "lags_stats"]
        fam_cat_means: Dict[str, Dict[str, float]] = {}
        for fam in fams:
            if fam not in family_importance_avg:
                continue
            feat_map = family_importance_avg.get(fam, {})
            # Group feature importances into categories
            cat_to_vals: Dict[str, List[float]] = {c: [] for c in categories}
            for feat, val in feat_map.items():
                cat = feature_to_category(feat)
                if cat not in cat_to_vals:
                    cat_to_vals[cat] = []
                cat_to_vals[cat].append(float(val))
            fam_cat_means[fam] = {c: (float(np.mean(v)) if len(v) > 0 else 0.0) for c, v in cat_to_vals.items()}

        # Build rank vectors per family over categories
        cat_tau_pairs: Dict[str, Optional[float]] = {}
        cat_tau_vals: List[float] = []
        fam_cat_ranks: Dict[str, pd.Series] = {}
        for fam, cmap in fam_cat_means.items():
            s = pd.Series({c: cmap.get(c, 0.0) for c in categories})
            fam_cat_ranks[fam] = s.rank(ascending=False, method="average")

        for a, b in pair_list:
            if a not in fam_cat_ranks or b not in fam_cat_ranks or not _HAS_SCIPY:
                cat_tau_pairs[f"{a}_{b}"] = None
                continue
            try:
                cat_tau, _ = kendalltau(fam_cat_ranks[a].values, fam_cat_ranks[b].values)
                cat_tau_pairs[f"{a}_{b}"] = None if (cat_tau is None or math.isnan(cat_tau)) else float(cat_tau)
                if cat_tau is not None and not math.isnan(cat_tau):
                    cat_tau_vals.append(float(cat_tau))
            except Exception:
                cat_tau_pairs[f"{a}_{b}"] = None
        median_cat_tau = float(np.median(cat_tau_vals)) if cat_tau_vals else None

        out_tau[target] = {
            "median_tau": median_tau,
            "pairwise": tau_pairs,
            "category_level": {
                "median_tau": median_cat_tau,
                "pairwise": cat_tau_pairs,
                "category_importance_by_family": fam_cat_means,
            },
        }

    return ranks_by_target.get("minutes", pd.DataFrame()), ranks_by_target.get("skills", pd.DataFrame()), out_tau


def plot_item11_topk(minutes_df: pd.DataFrame, skills_df: pd.DataFrame, out_pdf: Path,
                     source_minutes_csv: Path, source_skills_csv: Path) -> None:
    """Horizontal bar charts for aggregated feature importance rankings (top-k), per target."""
    source_minutes_csv.parent.mkdir(parents=True, exist_ok=True)
    source_skills_csv.parent.mkdir(parents=True, exist_ok=True)
    save_csv(source_minutes_csv, minutes_df)
    save_csv(source_skills_csv, skills_df)

    # Two-panel figure, Matplotlib-only
    plt.rcParams.update({'font.size': 12})
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))

    def _plot(ax, df: pd.DataFrame, title: str) -> None:
        if df.empty:
            ax.set_title(f"{title} (no data)")
            ax.axis('off')
            return
        plot_df = df.copy()
        plot_df = plot_df.iloc[::-1]  # reverse for horizontal bars (rank k at bottom)
        ax.barh(plot_df['feature'], plot_df['overall_importance'])
        ax.set_xlabel('Normalized importance (sum=1 per model then averaged)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    _plot(axes[0], minutes_df, 'Top-k features (minutes)')
    _plot(axes[1], skills_df, 'Top-k features (skills)')
    plt.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_pdf, format='pdf', bbox_inches='tight')
    plt.close()


def compute_item11_full_category_importance(window_dirs: Sequence[Path]) -> Dict[str, Any]:
    """Aggregate normalized feature importances into the 8 adapter-defined categories and verify sums.

    Categories (from sklearn_adapter.py sections):
      - current, lags, changes, stats, interactions, gaps, class_peer, prior_achievement
    Returns JSON-serializable dict per target with per-family category importances, their sums, and overall means.
    """
    categories = ["current", "lags", "changes", "stats", "interactions", "gaps", "class_peer", "prior_achievement"]

    def feature_to_category8(name: str) -> str:
        n = name.lower()
        if n.startswith("current_"):
            return "current"
        if "_lag" in n:
            return "lags"
        if n.endswith("_recent_change") or n.endswith("_avg_change"):
            return "changes"
        if n in {"minutes_x_difficulty"}:
            return "interactions"
        if n in {"has_recent_gap", "weeks_since_last_gap", "gap_count"} or "gap" in n:
            return "gaps"
        if n in {
            "performance_vs_class_mean_prof", "class_percentile_rank_prof", "class_improvement_trend_prof",
            "performance_vs_class_mean_mins", "class_percentile_rank_mins", "class_improvement_trend_mins"
        }:
            return "class_peer"
        if n in {"starting_ability_quartile", "performance_consistency_score", "learning_acceleration_capacity"}:
            return "prior_achievement"
        # Statistical features per adapter's section (includes trends/acceleration placed there in adapter)
        if n in {
            "minutes_mean", "minutes_std", "minutes_range", "minutes_iqr",
            "problems_mean", "problems_sum", "problems_std",
            "proficiency_trend", "proficiency_acceleration"
        }:
            return "stats"
        # Fallback: stats (ensures partition covers all features)
        return "stats"

    # Reuse same loading as feature ranks
    by_target_dirs: Dict[str, List[Path]] = {"minutes": [], "skills": []}
    for wd in window_dirs:
        cfg = read_json(wd / "evaluation_config.json")
        goal_type = str(cfg.get("goal_type", "")).lower()
        if "minute" in goal_type:
            by_target_dirs["minutes"].append(wd)
        else:
            by_target_dirs["skills"].append(wd)

    result: Dict[str, Any] = {}
    for target, dirs in by_target_dirs.items():
        if not dirs:
            result[target] = {}
            continue
        # family -> feature -> mean importance (across models and windows)
        family_to_feature_to_vals: Dict[str, Dict[str, List[float]]] = {}
        for wd in dirs:
            for model_name, family, norm_imp in load_feature_importances(wd):
                if family not in {"linear", "tree", "neural"}:
                    continue
                fam_map = family_to_feature_to_vals.setdefault(family, {})
                for feat, val in norm_imp.items():
                    fam_map.setdefault(feat, []).append(float(val))
        family_importance_avg: Dict[str, Dict[str, float]] = {}
        for fam, fmap in family_to_feature_to_vals.items():
            family_importance_avg[fam] = {feat: float(np.mean(fmap.get(feat, [0.0]))) for feat in fmap.keys()}

        # Convert to category importances per family by summing features
        family_category_importance: Dict[str, Dict[str, float]] = {}
        for fam, fmap in family_importance_avg.items():
            cat_sums: Dict[str, float] = {c: 0.0 for c in categories}
            for feat, imp in fmap.items():
                cat = feature_to_category8(feat)
                cat_sums[cat] = cat_sums.get(cat, 0.0) + float(imp)
            # Normalize lightly to ensure minor fp drift sums to 1
            total = float(sum(cat_sums.values()))
            if total > 0:
                cat_sums = {k: float(v) / total for k, v in cat_sums.items()}
            family_category_importance[fam] = cat_sums

        # Overall category importances as mean across available families
        fams_present = list(family_category_importance.keys())
        overall_cat: Dict[str, float] = {}
        if fams_present:
            for c in categories:
                vals = [family_category_importance[fam][c] for fam in fams_present]
                overall_cat[c] = float(np.mean(vals))
            # Re-normalize overall for exactness
            total_overall = float(sum(overall_cat.values()))
            if total_overall > 0:
                overall_cat = {k: float(v) / total_overall for k, v in overall_cat.items()}

        # Build sums/checks
        sums_by_family = {fam: float(sum(cat.values())) for fam, cat in family_category_importance.items()}
        result[target] = {
            "categories": categories,
            "category_importance_by_family": family_category_importance,
            "sums_by_family": sums_by_family,
            "overall_category_importance": overall_cat,
            "overall_sum": float(sum(overall_cat.values())) if overall_cat else None,
        }

    return result


def compute_and_save_ablation_artifacts(artifacts_root_path: Path, windows: Optional[List[int]] = None,
                                        base_dir_minutes: Optional[str] = None,
                                        base_dir_skills: Optional[str] = None) -> None:
    """Re-evaluate top models with ablations and write rq3_ablations.json and fig.

    Ablations:
      - remove_afm_prior: exclude categories ["stats" (AFM-induced subset not separable here), "prior_achievement", "interactions" if AFM], but practically we exclude ["prior_achievement", "afm_induced"] captured via proxies (ability/learning_rate/difficulty in engineered features go to stats/interactions). Since adapter sections label AFM as ability/learning_rate/difficulty, we exclude those via feature name mapping inside FeatureCategoryAblationAdapter.
      - remove_gaps: exclude ["gaps"].

    Note: This function assumes availability of DEFAULT_EXPERIMENT_CONFIG and run_evaluation_with_predictions.
    """
    mets_dir = artifacts_root_path / "metrics"
    figs_dir = artifacts_root_path / "figures"
    ensure_dir(mets_dir)
    ensure_dir(figs_dir)

    # Windows to evaluate
    if windows is None:
        windows = [1, 6, 11, 16, 21, 26]

    # Determine top non-baseline models per target from existing outputs (if provided)
    minutes_top = None
    skills_top = None
    if base_dir_minutes:
        min_dirs = [d for d in (discover_window_dir(Path(base_dir_minutes), w) for w in windows) if d]
        minutes_top = select_top_nonbaseline_model_overall_across_windows(min_dirs)
    if base_dir_skills:
        skl_dirs = [d for d in (discover_window_dir(Path(base_dir_skills), w) for w in windows) if d]
        skills_top = select_top_nonbaseline_model_overall_across_windows(skl_dirs)

    # Fallback to earlier artifact if selection failed
    try:
        r = read_json(artifacts_root_path / 'metrics' / 'rq1_delta_breakdown.json')
        if not minutes_top:
            minutes_top = r.get('minutes', {}).get('top_model')
        if not skills_top:
            # ensure non-baseline
            tm = r.get('skills', {}).get('top_model')
            if tm and model_name_to_family(tm) not in ['baseline', 'goal_based']:
                skills_top = tm
    except Exception:
        pass

    # Ensure models exist in our registry
    model_registry = create_all_models()
    def confirm_model(name: Optional[str]) -> Optional[str]:
        if name and name in model_registry:
            fam = model_name_to_family(name)
            if fam not in ['baseline', 'goal_based']:
                return name
        # pick a reasonable default
        for cand in ['random_forest', 'xgboost', 'lasso', 'ridge']:
            if cand in model_registry:
                return cand
        return None

    minutes_top = confirm_model(minutes_top)
    skills_top = confirm_model(skills_top)

    # Helper to build experiment config
    def build_exp(goal: str, w: int) -> Dict[str, Any]:
        return {
            'dataset_name': 'rolling_new',
            'goal_type': goal,
            'window_size': w,
            'feature_set': 'all',
            'cv_config': 'standard',
            'model_set': 'all'
        }

    # Collect per-student deltas across windows
    def run_ablation_series(goal: str, top_model: str, excluded: List[str]) -> np.ndarray:
        deltas: List[float] = []
        for w in windows:
            exp = build_exp(goal, w)
            # run full and ablated for top model only
            res_full, _ = run_evaluation_with_predictions(exp, analysis_mode='single_config', models_override={top_model: model_registry[top_model]})
            res_abl, _ = run_evaluation_with_predictions(exp, analysis_mode='single_config', models_override={top_model: model_registry[top_model]}, ablation_excluded_categories=excluded)
            # reconstruct window dirs
            base_dir = DEFAULT_EXPERIMENT_CONFIG.output_base_dir
            exp_name = DEFAULT_EXPERIMENT_CONFIG.get_experiment_name('rolling_new', goal, w, 'all', 'standard', 'all')
            full_dir = Path(base_dir) / exp_name
            abl_dir = Path(base_dir + '_ablation') / f"{exp_name}_exclude_{'-'.join(sorted(excluded))}"
            # per-student MAE
            full_mae = collect_per_student_mae_for_models(full_dir, [top_model]).get(top_model, pd.Series(dtype=float))
            abl_mae = collect_per_student_mae_for_models(abl_dir, [top_model]).get(top_model, pd.Series(dtype=float))
            df = pd.concat([full_mae.rename('full'), abl_mae.rename('abl')], axis=1).dropna()
            if df.empty:
                continue
            d = (df['abl'] - df['full']) / df['full'] * 100.0
            deltas.extend(list(d.values))
        return np.array(deltas, dtype=float)

    minutes_deltas_afm = run_ablation_series('minutes', minutes_top, ['prior_achievement']) if minutes_top else np.array([])
    minutes_deltas_gaps = run_ablation_series('minutes', minutes_top, ['gaps']) if minutes_top else np.array([])
    skills_deltas_afm = run_ablation_series('proficiency', skills_top, ['prior_achievement']) if skills_top else np.array([])
    skills_deltas_gaps = run_ablation_series('proficiency', skills_top, ['gaps']) if skills_top else np.array([])

    def summarize(deltas: np.ndarray) -> Dict[str, Any]:
        if deltas.size == 0:
            return {'delta_pct': None, 'ci': [None, None]}
        mean_delta, lo, hi = bootstrap_percentile_ci(deltas, n_bootstrap=10000, alpha=0.05)
        return {'delta_pct': round(float(mean_delta), 3), 'ci': [round(float(lo), 3), round(float(hi), 3)]}

    out = {
        'remove_afm_prior': {
            'minutes': summarize(minutes_deltas_afm),
            'skills': summarize(skills_deltas_afm),
        },
        'remove_gaps': {
            'minutes': summarize(minutes_deltas_gaps),
            'skills': summarize(skills_deltas_gaps),
        }
    }
    save_json(mets_dir / 'rq3_ablations.json', out)
    # Figure: two groups with minutes/skills bars
    labels = ['remove_afm_prior (minutes)', 'remove_afm_prior (skills)', 'remove_gaps (minutes)', 'remove_gaps (skills)']
    vals = [out['remove_afm_prior']['minutes']['delta_pct'], out['remove_afm_prior']['skills']['delta_pct'], out['remove_gaps']['minutes']['delta_pct'], out['remove_gaps']['skills']['delta_pct']]
    plt.figure(figsize=(8, 5))
    plt.barh(labels, vals)
    plt.xlabel('ΔMAE% (ablation vs full)')
    plt.tight_layout()
    plt.savefig(figs_dir / 'fig_rq3_ablation.pdf', format='pdf', bbox_inches='tight')
    plt.close()


# ---------------------------- Orchestration ---------------------------------


@dataclass
class PaperArtifactsConfig:
    base_dir_minutes: Optional[str]
    base_dir_skills: Optional[str]
    window_sizes: List[int]
    artifacts_root: str = "artifacts"


def main(base_dir_minutes: Optional[str], base_dir_skills: Optional[str],
         window_sizes: List[int], artifacts_root: str = "artifacts") -> None:
    """Compute paper artifacts for RQ items 2, 3, 5, and 11.

    Args:
        base_dir_minutes: Base directory containing minutes experiments for multiple windows
        base_dir_skills: Base directory containing skills/proficiency experiments
        window_sizes: List of window sizes to include
        artifacts_root: Root output directory for artifacts
    """
    artifacts_root_path = Path(artifacts_root)
    figs_dir = artifacts_root_path / "figures"
    tabs_dir = artifacts_root_path / "tables"
    mets_dir = artifacts_root_path / "metrics"
    for p in [figs_dir, tabs_dir, mets_dir]:
        ensure_dir(p)

    window_dirs: List[Path] = []
    # Discover window dirs for minutes and skills separately (if provided)
    def _discover_all(base: Optional[str]) -> List[Path]:
        if not base:
            return []
        base_path = Path(base)
        out: List[Path] = []
        for w in window_sizes:
            wd = discover_window_dir(base_path, w)
            if wd is not None and wd.exists():
                out.append(wd)
        return out

    window_dirs.extend(_discover_all(base_dir_minutes))
    window_dirs.extend(_discover_all(base_dir_skills))
    # Deduplicate
    window_dirs = sorted(set(window_dirs))
    if len(window_dirs) == 0:
        print("No window directories found. Nothing to do.")
        return

    # Item 5
    lasso_var = compute_lasso_window_variability(window_dirs)
    save_json(mets_dir / "lasso_window_variability.json", lasso_var)

    # Item 2
    item2 = compute_item2_delta(window_dirs)
    save_json(mets_dir / "rq1_delta_breakdown.json", item2)

    # Item 3
    item3 = compute_item3_family_tests(window_dirs)
    save_json(mets_dir / "rq1_family_tests.json", item3)

    # Item 11
    minutes_rank_df, skills_rank_df, tau_json = compute_item11_feature_ranks_and_tau(window_dirs, k_top=15)
    save_csv(tabs_dir / "feature_ranks_minutes.csv", minutes_rank_df)
    save_csv(tabs_dir / "feature_ranks_skills.csv", skills_rank_df)
    save_json(mets_dir / "rq3_rank_stability.json", tau_json)
    plot_item11_topk(
        minutes_rank_df,
        skills_rank_df,
        figs_dir / "fig_rq3_topk.pdf",
        tabs_dir / "fig_rq3_topk_minutes_source.csv",
        tabs_dir / "fig_rq3_topk_skills_source.csv",
    )

    # Additional artifact: full 8-category importances with sum checks
    cat8 = compute_item11_full_category_importance(window_dirs)
    save_json(mets_dir / "rq3_category_importance_8cat.json", cat8)

    # New: Family MAE summaries (new)
    fam_new = compute_family_mae_new(window_dirs)
    save_json(mets_dir / "rq1_family_tests_new.json", fam_new)

    # New: Model breakdown with avg MAE and delta% vs baselines (with CI)
    model_breakdown = compute_model_breakdown(window_dirs)
    save_json(mets_dir / "rq1_insane_model_breakdown.json", model_breakdown)

    # Optional ablation pipeline (disabled by default):
    # Uncomment to run ablations for top models per target. Requires re-evaluation.
    # compute_and_save_ablation_artifacts(artifacts_root_path)

    print("✅ Paper artifacts generated:")
    print(f"  - {mets_dir / 'rq1_delta_breakdown.json'}")
    print(f"  - {mets_dir / 'rq1_family_tests.json'}")
    print(f"  - {mets_dir / 'lasso_window_variability.json'}")
    print(f"  - {mets_dir / 'rq3_rank_stability.json'}")
    print(f"  - {tabs_dir / 'feature_ranks_minutes.csv'}")
    print(f"  - {tabs_dir / 'feature_ranks_skills.csv'}")
    print(f"  - {figs_dir / 'fig_rq3_topk.pdf'}")
    print(f"  - {mets_dir / 'rq1_family_tests_new.json'}")
    print(f"  - {mets_dir / 'rq1_insane_model_breakdown.json'}")


if __name__ == "__main__":
    # Example invocation: set base dirs via env vars for quick testing
    minutes_dir = os.environ.get("PAPER_BASE_DIR_MINUTES")
    skills_dir = os.environ.get("PAPER_BASE_DIR_SKILLS")
    windows_env = os.environ.get("PAPER_WINDOW_SIZES", "1,6,11,16,21,26")
    windows = [int(x.strip()) for x in windows_env.split(",") if x.strip()]
    main(minutes_dir, skills_dir, windows)


