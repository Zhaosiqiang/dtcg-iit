"""Reproducible multi-dataset experiments for the IEEE TII manuscript.

The code evaluates a drift-triggered conformal quality gate (DTCG) under a
strict chronological protocol.  The two public manufacturing datasets are
used as proxies for an aerospace-oriented quality/maintenance workflow; no
Airbus or proprietary data are used.  Test labels are used only for reporting
metrics after decisions have been made, never for fitting or tuning.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"
OUT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

ALPHA = 0.10
GAMMA = 0.65
DEFAULT_BATCH_SIZE = 50
SEEDS = [7, 17, 27, 37, 47]
MAX_RECENT_LABELS = 200


@dataclass
class PreparedData:
    name: str
    x: np.ndarray
    y: np.ndarray
    train_end: int
    cal_end: int
    n_raw_features: int
    retained_features: np.ndarray
    missing_values_before_imputation: int
    source: str
    task: str


@dataclass
class ModelArtifact:
    data: PreparedData
    seed: int
    model: RandomForestClassifier
    x_cal: np.ndarray
    y_cal: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    p_cal: np.ndarray
    p_test: np.ndarray
    test_auc: float
    test_average_precision: float
    drift_records: pd.DataFrame


def load_secom() -> PreparedData:
    data_dir = DATA_DIR / "secom"
    x_raw = pd.read_csv(data_dir / "secom.data", sep=" ", header=None, na_values="NaN")
    labels = pd.read_csv(data_dir / "secom_labels.data", sep=" ", header=None)
    y = labels.iloc[:, 0].map({-1: 0, 1: 1}).to_numpy(dtype=int)
    x_all = x_raw.to_numpy(dtype=float)
    train_end, cal_end = 400, 800
    keep = np.isfinite(x_all[:train_end]).any(axis=0)
    x_all = x_all[:, keep]
    medians = np.nanmedian(x_all[:train_end], axis=0)
    x_all = np.where(np.isfinite(x_all), x_all, medians)
    return PreparedData(
        name="SECOM",
        x=x_all,
        y=y,
        train_end=train_end,
        cal_end=cal_end,
        n_raw_features=x_raw.shape[1],
        retained_features=np.flatnonzero(keep),
        missing_values_before_imputation=int(x_raw.isna().sum().sum()),
        source="UCI Machine Learning Repository, DOI: 10.24432/C54305",
        task="semiconductor manufacturing pass/failure classification",
    )


def load_ai4i() -> PreparedData:
    path = DATA_DIR / "ai4i" / "ai4i2020.csv"
    frame = pd.read_csv(path, encoding="utf-8-sig")
    y = frame["Machine failure"].astype(int).to_numpy()
    # Product identifiers and failure-mode subtype columns are excluded to
    # prevent target leakage.  Type is represented by fixed one-hot columns.
    type_one_hot = pd.get_dummies(frame["Type"], prefix="Type").reindex(
        columns=["Type_L", "Type_M", "Type_H"], fill_value=False
    )
    numeric_columns = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ]
    x_frame = pd.concat([frame[numeric_columns], type_one_hot.astype(float)], axis=1)
    x_all = x_frame.to_numpy(dtype=float)
    train_end, cal_end = 6000, 8000
    imputer = SimpleImputer(strategy="median")
    imputer.fit(x_all[:train_end])
    x_all = imputer.transform(x_all)
    return PreparedData(
        name="AI4I",
        x=x_all,
        y=y,
        train_end=train_end,
        cal_end=cal_end,
        n_raw_features=14,
        retained_features=np.arange(x_frame.shape[1]),
        missing_values_before_imputation=int(frame.isna().sum().sum()),
        source="UCI Machine Learning Repository, DOI: 10.24432/C52G8C",
        task="predictive maintenance machine-failure classification",
    )


def load_datasets() -> list[PreparedData]:
    return [load_secom(), load_ai4i()]


def conformal_quantile(scores: np.ndarray, alpha: float = ALPHA) -> float:
    scores = np.asarray(scores, dtype=float)
    if len(scores) == 0:
        return 1.0
    order = np.sort(scores)
    rank = int(np.ceil((len(order) + 1) * (1 - alpha)))
    return float(order[min(rank - 1, len(order) - 1)])


def weighted_quantile(scores: np.ndarray, weights: np.ndarray, alpha: float = ALPHA) -> float:
    scores = np.asarray(scores, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if len(scores) == 0 or np.sum(weights) <= 0:
        return 1.0
    order = np.argsort(scores)
    s = scores[order]
    w = np.maximum(weights[order], 1e-8)
    target = (1 - alpha) * np.sum(w)
    index = np.searchsorted(np.cumsum(w), target, side="left")
    return float(s[np.clip(index, 0, len(s) - 1)])


def mondrian_threshold(p: np.ndarray, y: np.ndarray, alpha: float = ALPHA) -> float:
    """One-sided auto-pass threshold from class-conditional CP scores."""
    pass_scores = p[y == 0]
    fail_scores = 1.0 - p[y == 1]
    q_pass = conformal_quantile(pass_scores, alpha)
    q_fail = conformal_quantile(fail_scores, alpha)
    return float(min(q_pass, 1.0 - q_fail))


def global_threshold(p: np.ndarray, y: np.ndarray, alpha: float = ALPHA) -> float:
    scores = np.where(y == 1, 1.0 - p, p)
    q = conformal_quantile(scores, alpha)
    return float(min(q, 1.0 - q))


def weighted_mondrian_threshold(
    p_cal: np.ndarray,
    y_cal: np.ndarray,
    x_cal: np.ndarray,
    x_batch: np.ndarray,
    alpha: float = ALPHA,
) -> tuple[float, dict[str, float]]:
    """Estimate a covariate-shift weight using only unlabeled features."""
    domain_x = np.vstack([x_cal, x_batch])
    domain_y = np.r_[np.zeros(len(x_cal), dtype=int), np.ones(len(x_batch), dtype=int)]
    detector = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.1, max_iter=2000, solver="liblinear"),
    )
    detector.fit(domain_x, domain_y)
    prob = np.clip(detector.predict_proba(x_cal)[:, 1], 1e-4, 1 - 1e-4)
    ratio = (prob / (1 - prob)) * (len(x_cal) / max(len(x_batch), 1))
    ratio = np.clip(ratio, 0.05, 20.0)
    q_pass = weighted_quantile(p_cal[y_cal == 0], ratio[y_cal == 0], alpha)
    q_fail = weighted_quantile(1.0 - p_cal[y_cal == 1], ratio[y_cal == 1], alpha)
    diagnostics = {
        "mean_density_weight": float(np.mean(ratio)),
        "min_density_weight": float(np.min(ratio)),
        "median_density_weight": float(np.median(ratio)),
        "max_density_weight": float(np.max(ratio)),
        "low_clip_fraction": float(np.mean(ratio <= 0.0500001)),
        "high_clip_fraction": float(np.mean(ratio >= 19.999999)),
    }
    return float(min(q_pass, 1.0 - q_fail)), diagnostics


def drift_auc(x_reference: np.ndarray, x_batch: np.ndarray, seed: int) -> float:
    """Cross-fitted domain-classification AUC; no batch labels are used."""
    x = np.vstack([x_reference, x_batch])
    d = np.r_[np.zeros(len(x_reference), dtype=int), np.ones(len(x_batch), dtype=int)]
    detector = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.1, max_iter=2000, solver="liblinear"),
    )
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    try:
        values = cross_val_score(detector, x, d, cv=cv, scoring="roc_auc", n_jobs=1)
        return float(np.mean(values))
    except ValueError:
        return 0.5


def fit_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    class_weight: dict[int, int] | None = None,
) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=500,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight=class_weight or {0: 1, 1: 8},
        n_jobs=-1,
        random_state=seed,
    )
    model.fit(x_train, y_train)
    return model


def decision_metrics(p: np.ndarray, y: np.ndarray, threshold: float) -> dict[str, float]:
    auto = p < threshold
    missed = auto & (y == 1)
    review = ~auto
    return {
        "n": float(len(y)),
        "fail_n": float(np.sum(y == 1)),
        "auto_release_rate": float(np.mean(auto)),
        "inspection_rate": float(np.mean(review)),
        "missed_failure_rate": float(np.sum(missed) / max(np.sum(y == 1), 1)),
        "false_accepts": float(np.sum(missed)),
        "inspected_failure_count": float(np.sum(review & (y == 1))),
        "review_fail_capture": float(np.sum(review & (y == 1)) / max(np.sum(y == 1), 1)),
    }


def add_metric_row(
    rows: list[dict],
    dataset: str,
    seed: int,
    method: str,
    batch: int,
    p: np.ndarray,
    y: np.ndarray,
    threshold: float,
    test_auc: float,
    test_average_precision: float,
    **extra,
) -> None:
    row = {
        "dataset": dataset,
        "seed": seed,
        "method": method,
        "batch": batch,
        "threshold": threshold,
        "test_auc": test_auc,
        "test_average_precision": test_average_precision,
    }
    row.update(decision_metrics(p, y, threshold))
    row.update(extra)
    rows.append(row)


def batch_records(
    x_cal: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    batch_size: int,
    cal_end: int,
) -> pd.DataFrame:
    records = []
    for batch_id, start in enumerate(range(0, len(x_test), batch_size)):
        stop = min(start + batch_size, len(x_test))
        xb = x_test[start:stop]
        auc = drift_auc(x_cal, xb, seed + batch_id)
        records.append(
            {
                "batch": batch_id,
                "start_index": cal_end + start,
                "end_index": cal_end + stop - 1,
                "n": stop - start,
                "failure_count": int(np.sum(y_test[start:stop])),
                "failure_rate": float(np.mean(y_test[start:stop])),
                "domain_auc": auc,
                "shift_flag": int(auc >= GAMMA),
            }
        )
    return pd.DataFrame(records)


def rolling_calibration_window(
    p_cal: np.ndarray,
    y_cal: np.ndarray,
    recent_p: list[np.ndarray],
    recent_y: list[np.ndarray],
    max_recent: int = MAX_RECENT_LABELS,
) -> tuple[np.ndarray, np.ndarray]:
    if not recent_p or not recent_y:
        return p_cal, y_cal
    p_recent = np.concatenate(recent_p)[-max_recent:]
    y_recent = np.concatenate(recent_y)[-max_recent:]
    return np.concatenate([p_cal, p_recent]), np.concatenate([y_cal, y_recent])


def batch_views(artifact: ModelArtifact, batch_size: int):
    drift = artifact.drift_records
    if len(drift) != int(np.ceil(len(artifact.x_test) / batch_size)):
        drift = batch_records(
            artifact.x_cal,
            artifact.x_test,
            artifact.y_test,
            artifact.seed,
            batch_size,
            artifact.data.cal_end,
        )
    for batch_id, start in enumerate(range(0, len(artifact.x_test), batch_size)):
        stop = min(start + batch_size, len(artifact.x_test))
        row = drift.iloc[batch_id]
        yield (
            batch_id,
            artifact.x_test[start:stop],
            artifact.y_test[start:stop],
            artifact.p_test[start:stop],
            float(row.domain_auc),
            int(row.shift_flag),
        )


def evaluate_static_method(
    artifact: ModelArtifact,
    method: str,
    threshold: float,
    batch_size: int,
) -> list[dict]:
    rows: list[dict] = []
    for batch_id, _xb, yb, pb, auc, shift_flag in batch_views(artifact, batch_size):
        add_metric_row(
            rows,
            artifact.data.name,
            artifact.seed,
            method,
            batch_id,
            pb,
            yb,
            threshold,
            artifact.test_auc,
            artifact.test_average_precision,
            domain_auc=auc,
            shift_flag=shift_flag,
        )
    return rows


def evaluate_weighted_method(
    artifact: ModelArtifact,
    batch_size: int,
    alpha: float = ALPHA,
    diagnostics_rows: list[dict] | None = None,
) -> list[dict]:
    rows: list[dict] = []
    for batch_id, xb, yb, pb, auc, shift_flag in batch_views(artifact, batch_size):
        tau, diagnostics = weighted_mondrian_threshold(
            artifact.p_cal, artifact.y_cal, artifact.x_cal, xb, alpha
        )
        if diagnostics_rows is not None:
            diagnostics_rows.append(
                {
                    "dataset": artifact.data.name,
                    "seed": artifact.seed,
                    "batch": batch_id,
                    "domain_auc": auc,
                    "threshold": tau,
                    **diagnostics,
                }
            )
        add_metric_row(
            rows,
            artifact.data.name,
            artifact.seed,
            "Weighted-Mondrian",
            batch_id,
            pb,
            yb,
            tau,
            artifact.test_auc,
            artifact.test_average_precision,
            domain_auc=auc,
            shift_flag=shift_flag,
            **diagnostics,
        )
    return rows


def evaluate_dynamic_policy(
    artifact: ModelArtifact,
    method: str,
    batch_size: int,
    alpha: float = ALPHA,
    gamma: float = GAMMA,
    trigger: bool = True,
    first_shift_hold: bool = True,
    rolling: bool = True,
    minimum_failures: int = 5,
) -> list[dict]:
    base_tau = mondrian_threshold(artifact.p_cal, artifact.y_cal, alpha)
    rows: list[dict] = []
    recent_p: list[np.ndarray] = []
    recent_y: list[np.ndarray] = []
    seen_shift = False
    for batch_id, _xb, yb, pb, auc, _detector_shift_flag in batch_views(artifact, batch_size):
        shifted = bool(auc >= gamma)
        prior_failures = int(np.sum(np.concatenate(recent_y))) if recent_y else 0
        tau = base_tau
        mode = "stable"
        if trigger and shifted:
            if first_shift_hold and not seen_shift:
                tau = 0.0
                mode = "first-shift-hold"
                seen_shift = True
            elif rolling and prior_failures >= minimum_failures:
                rolling_p, rolling_y = rolling_calibration_window(
                    artifact.p_cal, artifact.y_cal, recent_p, recent_y
                )
                tau = min(base_tau, mondrian_threshold(rolling_p, rolling_y, alpha))
                mode = "shifted-rolling-calibration"
            else:
                mode = "shifted-base-calibration"
        elif not trigger and rolling and prior_failures >= minimum_failures:
            rolling_p, rolling_y = rolling_calibration_window(
                artifact.p_cal, artifact.y_cal, recent_p, recent_y
            )
            tau = mondrian_threshold(rolling_p, rolling_y, alpha)
            mode = "rolling-calibration"
        add_metric_row(
            rows,
            artifact.data.name,
            artifact.seed,
            method,
            batch_id,
            pb,
            yb,
            tau,
            artifact.test_auc,
            artifact.test_average_precision,
            domain_auc=auc,
            shift_flag=int(shifted),
            gate_mode=mode,
            minimum_failures=minimum_failures,
        )
        # Only labels from inspected units become available after the current
        # batch's decisions; auto-released units cannot update calibration.
        inspected = pb >= tau
        if np.any(inspected):
            recent_p.append(pb[inspected])
            recent_y.append(yb[inspected])
        if sum(len(a) for a in recent_y) > MAX_RECENT_LABELS:
            all_p = np.concatenate(recent_p)[-MAX_RECENT_LABELS:]
            all_y = np.concatenate(recent_y)[-MAX_RECENT_LABELS:]
            recent_p = [all_p]
            recent_y = [all_y]
    return rows


def fit_artifact(
    data: PreparedData,
    seed: int,
    class_weight: dict[int, int] | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> ModelArtifact:
    x_train, y_train = data.x[: data.train_end], data.y[: data.train_end]
    x_cal, y_cal = data.x[data.train_end : data.cal_end], data.y[data.train_end : data.cal_end]
    x_test, y_test = data.x[data.cal_end :], data.y[data.cal_end :]
    model = fit_model(x_train, y_train, seed, class_weight)
    p_cal = model.predict_proba(x_cal)[:, 1]
    p_test = model.predict_proba(x_test)[:, 1]
    try:
        test_auc = float(roc_auc_score(y_test, p_test))
        test_ap = float(average_precision_score(y_test, p_test))
    except ValueError:
        test_auc, test_ap = float("nan"), float("nan")
    artifact = ModelArtifact(
        data=data,
        seed=seed,
        model=model,
        x_cal=x_cal,
        y_cal=y_cal,
        x_test=x_test,
        y_test=y_test,
        p_cal=p_cal,
        p_test=p_test,
        test_auc=test_auc,
        test_average_precision=test_ap,
        drift_records=pd.DataFrame(),
    )
    artifact.drift_records = batch_records(
        x_cal, x_test, y_test, seed, batch_size, data.cal_end
    )
    return artifact


def main_rows_for_artifact(
    artifact: ModelArtifact,
    weighted_diagnostics: list[dict],
) -> list[dict]:
    base_global = global_threshold(artifact.p_cal, artifact.y_cal, ALPHA)
    base_mondrian = mondrian_threshold(artifact.p_cal, artifact.y_cal, ALPHA)
    rows: list[dict] = []
    rows.extend(evaluate_static_method(artifact, "Probability-0.5", 0.5, DEFAULT_BATCH_SIZE))
    rows.extend(evaluate_static_method(artifact, "Global-CP", base_global, DEFAULT_BATCH_SIZE))
    rows.extend(evaluate_static_method(artifact, "Mondrian-CP", base_mondrian, DEFAULT_BATCH_SIZE))
    rows.extend(evaluate_weighted_method(artifact, DEFAULT_BATCH_SIZE, ALPHA, weighted_diagnostics))
    rows.extend(
        evaluate_dynamic_policy(
            artifact,
            "Periodic-Rolling-CP",
            DEFAULT_BATCH_SIZE,
            ALPHA,
            GAMMA,
            trigger=False,
            first_shift_hold=False,
            rolling=True,
            minimum_failures=5,
        )
    )
    rows.extend(
        evaluate_dynamic_policy(
            artifact,
            "DTCG-Proposed",
            DEFAULT_BATCH_SIZE,
            ALPHA,
            GAMMA,
            trigger=True,
            first_shift_hold=True,
            rolling=True,
            minimum_failures=5,
        )
    )
    return rows


def summarize(rows: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    metric_cols = [
        "auto_release_rate",
        "inspection_rate",
        "missed_failure_rate",
        "review_fail_capture",
        "threshold",
        "test_auc",
        "test_average_precision",
    ]
    summary = rows.groupby(group_columns, as_index=False)[metric_cols].agg(["mean", "std"])
    summary.columns = [
        "_".join(str(part) for part in column if part).rstrip("_")
        for column in summary.columns.to_flat_index()
    ]
    return summary


def decision_trace_from_main_frame(main_frame: pd.DataFrame) -> pd.DataFrame:
    """Export one auditable DTCG batch event for every seed and dataset."""
    frame = main_frame[main_frame["method"] == "DTCG-Proposed"].copy()
    frame = frame.sort_values(["dataset", "seed", "batch"]).reset_index(drop=True)
    frame["lot_id"] = frame.apply(
        lambda row: f"public-{str(row['dataset']).lower()}-seed-{int(row['seed']):02d}",
        axis=1,
    )
    frame["model_version"] = "rf-v1"
    frame["calibration_version"] = "mondrian-alpha-0.10"
    frame["policy_version"] = "dtcg-v1"
    frame["auto_release_count"] = np.rint(
        frame["auto_release_rate"] * frame["n"]
    ).astype(int)
    frame["inspection_count"] = (
        frame["n"].round().astype(int) - frame["auto_release_count"]
    )
    frame["failure_count_observed_after_inspection"] = (
        frame["inspected_failure_count"].round().astype(int)
    )
    columns = [
        "lot_id",
        "seed",
        "batch",
        "model_version",
        "calibration_version",
        "policy_version",
        "domain_auc",
        "gate_mode",
        "threshold",
        "auto_release_count",
        "inspection_count",
        "failure_count_observed_after_inspection",
        "shift_flag",
    ]
    return frame[columns]


def ablation_experiments(artifacts: dict[tuple[str, int], ModelArtifact]) -> pd.DataFrame:
    variants = [
        ("DTCG-no-first-shift-hold", True, False, True, 5),
        ("DTCG-no-rolling", True, True, False, 5),
        ("DTCG-always-roll", False, False, True, 5),
        ("DTCG-min-failures-1", True, True, True, 1),
        ("DTCG-min-failures-10", True, True, True, 10),
    ]
    rows: list[dict] = []
    for artifact in artifacts.values():
        for method, trigger, hold, roll, min_fail in variants:
            rows.extend(
                evaluate_dynamic_policy(
                    artifact,
                    method,
                    DEFAULT_BATCH_SIZE,
                    ALPHA,
                    GAMMA,
                    trigger=trigger,
                    first_shift_hold=hold,
                    rolling=roll,
                    minimum_failures=min_fail,
                )
            )
    return pd.DataFrame(rows)


def first_five_summary(ablation: pd.DataFrame) -> pd.DataFrame:
    subset = ablation[ablation["batch"] < 5].copy()
    grouped = subset.groupby(["dataset", "method"], as_index=False).agg(
        first5_missed_failures=("false_accepts", "sum"),
        first5_inspection_rate=("inspection_rate", "mean"),
        first5_missed_failure_rate=("missed_failure_rate", "mean"),
    )
    return grouped


def first_batch_summary(ablation: pd.DataFrame) -> pd.DataFrame:
    subset = ablation[ablation["batch"] == 0].copy()
    return subset.groupby(["dataset", "method"], as_index=False).agg(
        first_batch_inspection_rate=("inspection_rate", "mean"),
        first_batch_missed_failures=("false_accepts", "sum"),
        first_batch_shift_flag=("shift_flag", "mean"),
    )


def statistical_tests(main_rows: pd.DataFrame) -> pd.DataFrame:
    comparisons = [
        ("DTCG-Proposed", "Mondrian-CP"),
        ("DTCG-Proposed", "Periodic-Rolling-CP"),
        ("Mondrian-CP", "Global-CP"),
        ("Mondrian-CP", "Probability-0.5"),
    ]
    records = []
    for dataset in sorted(main_rows["dataset"].unique()):
        subset = main_rows[main_rows["dataset"] == dataset]
        for left, right in comparisons:
            left_df = subset[subset.method == left].set_index(["seed", "batch"])
            right_df = subset[subset.method == right].set_index(["seed", "batch"])
            paired = left_df.join(right_df, lsuffix="_left", rsuffix="_right", how="inner")
            for metric in ["auto_release_rate", "missed_failure_rate"]:
                difference = (
                    paired[f"{metric}_left"] - paired[f"{metric}_right"]
                ).to_numpy(dtype=float)
                difference = difference[np.isfinite(difference)]
                if len(difference) == 0 or np.allclose(difference, 0):
                    statistic, p_value = 0.0, 1.0
                else:
                    result = wilcoxon(
                        difference,
                        zero_method="wilcox",
                        alternative="two-sided",
                        method="auto",
                    )
                    statistic, p_value = float(result.statistic), float(result.pvalue)
                records.append(
                    {
                        "dataset": dataset,
                        "comparison": f"{left} vs {right}",
                        "metric": metric,
                        "n_pairs": int(len(difference)),
                        "wilcoxon_statistic": statistic,
                        "p_value": p_value,
                        "significant_at_0.05": int(p_value < 0.05),
                    }
                )
    return pd.DataFrame(records)


def sensitivity_experiments(
    artifacts: dict[tuple[str, int], ModelArtifact],
) -> dict[str, pd.DataFrame]:
    outputs: dict[str, list[dict]] = {
        "alpha": [],
        "gamma": [],
        "batch_size": [],
        "class_weight": [],
    }
    alpha_values = [0.05, 0.10, 0.15, 0.20]
    gamma_values = [0.55, 0.60, 0.65, 0.70, 0.75]
    batch_sizes = [25, 50, 75, 100]
    class_weights = [{0: 1, 1: 4}, {0: 1, 1: 8}, {0: 1, 1: 16}, {0: 1, 1: 32}]

    def append_aggregate(
        bucket: list[dict],
        parameter: str,
        value: str | float | int,
        rows: list[dict],
        dataset: str,
    ):
        frame = pd.DataFrame(rows)
        bucket.append(
            {
                "dataset": dataset,
                "parameter": parameter,
                "value": value,
                "auto_release_rate_mean": float(frame.auto_release_rate.mean()),
                "inspection_rate_mean": float(frame.inspection_rate.mean()),
                "missed_failure_rate_mean": float(frame.missed_failure_rate.mean()),
                "missed_failure_rate_std": float(frame.missed_failure_rate.std()),
            }
        )

    for artifact in artifacts.values():
        for value in alpha_values:
            rows = evaluate_dynamic_policy(
                artifact,
                "DTCG-Proposed",
                DEFAULT_BATCH_SIZE,
                alpha=value,
                gamma=GAMMA,
                trigger=True,
                first_shift_hold=True,
                rolling=True,
                minimum_failures=5,
            )
            append_aggregate(outputs["alpha"], "alpha", value, rows, artifact.data.name)
        for value in gamma_values:
            rows = evaluate_dynamic_policy(
                artifact,
                "DTCG-Proposed",
                DEFAULT_BATCH_SIZE,
                alpha=ALPHA,
                gamma=value,
                trigger=True,
                first_shift_hold=True,
                rolling=True,
                minimum_failures=5,
            )
            append_aggregate(outputs["gamma"], "gamma", value, rows, artifact.data.name)
        for value in batch_sizes:
            altered = ModelArtifact(**{**artifact.__dict__})
            altered.drift_records = batch_records(
                artifact.x_cal,
                artifact.x_test,
                artifact.y_test,
                artifact.seed,
                value,
                artifact.data.cal_end,
            )
            rows = evaluate_dynamic_policy(
                altered,
                "DTCG-Proposed",
                value,
                alpha=ALPHA,
                gamma=GAMMA,
                trigger=True,
                first_shift_hold=True,
                rolling=True,
                minimum_failures=5,
            )
            append_aggregate(outputs["batch_size"], "batch_size", value, rows, artifact.data.name)

    datasets_by_name = {artifact.data.name: artifact.data for artifact in artifacts.values()}
    for data_name in sorted(datasets_by_name):
        data = datasets_by_name[data_name]
        for weight in class_weights:
            for seed in SEEDS:
                base = artifacts[(data.name, seed)]
                weighted_artifact = fit_artifact(data, seed, weight, DEFAULT_BATCH_SIZE)
                weighted_artifact.drift_records = base.drift_records.copy()
                rows = evaluate_dynamic_policy(
                    weighted_artifact,
                    "DTCG-Proposed",
                    DEFAULT_BATCH_SIZE,
                    alpha=ALPHA,
                    gamma=GAMMA,
                    trigger=True,
                    first_shift_hold=True,
                    rolling=True,
                    minimum_failures=5,
                )
                append_aggregate(
                    outputs["class_weight"],
                    "class_weight",
                    f"1:{weight[1]}",
                    rows,
                    data.name,
                )
    grouped_outputs: dict[str, pd.DataFrame] = {}
    for key, value in outputs.items():
        frame = pd.DataFrame(value)
        grouped_outputs[key] = (
            frame.groupby(["dataset", "parameter", "value"], as_index=False)
            .agg(
                auto_release_rate_mean=("auto_release_rate_mean", "mean"),
                inspection_rate_mean=("inspection_rate_mean", "mean"),
                missed_failure_rate_mean=("missed_failure_rate_mean", "mean"),
                missed_failure_rate_std=("missed_failure_rate_mean", "std"),
            )
        )
    return grouped_outputs


def stable_prefix_analysis(artifact: ModelArtifact) -> pd.DataFrame:
    """Detector-only stress test: stable calibration-like prefix then a jump."""
    rng = np.random.default_rng(20260910 + artifact.seed)
    prefix_batches = 4
    prefix_size = prefix_batches * DEFAULT_BATCH_SIZE
    stable_features = artifact.x_cal[rng.integers(0, len(artifact.x_cal), size=prefix_size)]
    sudden_features = artifact.x_test[: 8 * DEFAULT_BATCH_SIZE]
    scenario = np.vstack([stable_features, sudden_features])
    records = []
    for batch_id, start in enumerate(range(0, len(scenario), DEFAULT_BATCH_SIZE)):
        stop = min(start + DEFAULT_BATCH_SIZE, len(scenario))
        auc = drift_auc(artifact.x_cal, scenario[start:stop], artifact.seed + 1000 + batch_id)
        records.append(
            {
                "dataset": artifact.data.name,
                "seed": artifact.seed,
                "batch": batch_id,
                "phase": "stable-prefix" if batch_id < prefix_batches else "post-shift",
                "domain_auc": auc,
                "shift_flag": int(auc >= GAMMA),
            }
        )
    return pd.DataFrame(records)


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def make_figures(
    main_rows: pd.DataFrame,
    drift: pd.DataFrame,
    sensitivity: dict[str, pd.DataFrame],
    stable: pd.DataFrame,
) -> None:
    configure_plot_style()

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=False)
    for ax, dataset in zip(axes, sorted(drift.dataset.unique())):
        subset = drift[drift.dataset == dataset]
        mean = subset.groupby("batch", as_index=False).agg(
            domain_auc=("domain_auc", "mean"),
            auc_std=("domain_auc", "std"),
            failure_rate=("failure_rate", "mean"),
        )
        ax.plot(mean.batch, mean.domain_auc, marker="o", ms=3, color="#0b4f6c", label="Domain AUC")
        ax.fill_between(
            mean.batch,
            mean.domain_auc - mean.auc_std.fillna(0),
            mean.domain_auc + mean.auc_std.fillna(0),
            color="#0b4f6c",
            alpha=0.15,
        )
        ax.axhline(GAMMA, color="#b23a48", ls="--", lw=0.9, label="Shift threshold")
        ax.set_title(dataset)
        ax.set_xlabel("Chronological batch")
        ax.set_ylabel("Domain AUC")
        ax.set_ylim(0.45, 1.02)
        ax.grid(axis="y", alpha=0.18)
    axes[0].legend(frameon=False, loc="lower right")
    fig.suptitle("Unlabeled drift monitoring across public manufacturing streams", y=1.02)
    fig.tight_layout()
    save_figure(fig, "drift_monitoring")

    performance = main_rows.groupby(["dataset", "method"], as_index=False).agg(
        inspection_rate=("inspection_rate", "mean"),
        missed_failure_rate=("missed_failure_rate", "mean"),
    )
    methods = [
        "Probability-0.5",
        "Global-CP",
        "Weighted-Mondrian",
        "Periodic-Rolling-CP",
        "Mondrian-CP",
        "DTCG-Proposed",
    ]
    short_methods = [
        "Prob. 0.5",
        "Global CP",
        "Weighted CP",
        "Periodic roll.",
        "Mondrian CP",
        "DTCG",
    ]
    colors = {"SECOM": "#4e79a7", "AI4I": "#f28e2b"}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.45), sharex=True, sharey=False)
    x = np.arange(len(methods))
    width = 0.36
    for ax, metric, ylabel, title in [
        (axes[0], "inspection_rate", "Inspection rate", "Inspection"),
        (axes[1], "missed_failure_rate", "Missed-failure rate", "Missed failure"),
    ]:
        for offset, dataset in [(-width / 2, "SECOM"), (width / 2, "AI4I")]:
            subset = performance[performance.dataset == dataset].set_index("method").reindex(methods)
            ax.bar(
                x + offset,
                subset[metric],
                width=width,
                label=dataset,
                color=colors[dataset],
            )
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(short_methods, rotation=28, ha="right")
        ax.set_ylim(0, 1.02)
        ax.grid(axis="y", alpha=0.18)
    axes[0].legend(frameon=False, loc="upper left", ncol=2)
    fig.tight_layout(w_pad=1.5)
    save_figure(fig, "gate_performance")

    dtcg = main_rows[main_rows.method == "DTCG-Proposed"]
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    secom = dtcg[dtcg.dataset == "SECOM"].groupby("batch", as_index=False).threshold.mean()
    ax.plot(secom.batch, secom.threshold, marker="o", ms=3, label="DTCG")
    mondrian = main_rows[
        (main_rows.dataset == "SECOM") & (main_rows.method == "Mondrian-CP")
    ].groupby("batch", as_index=False).threshold.mean()
    ax.plot(mondrian.batch, mondrian.threshold, marker="o", ms=3, label="Mondrian-CP")
    ax.set_xlabel("Chronological batch")
    ax.set_ylabel("Auto-release threshold")
    ax.set_title("SECOM threshold trace")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.18)
    save_figure(fig, "threshold_trace")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.05), sharey=True)
    for ax, key in zip(axes, ["alpha", "gamma"]):
        frame = sensitivity[key]
        for dataset, group in frame.groupby("dataset"):
            group = group.sort_values("value")
            ax.plot(group.value, group.missed_failure_rate_mean, marker="o", ms=3, label=dataset)
        ax.axhline(ALPHA, color="#777777", ls=":", lw=0.9)
        ax.set_xlabel("$\\alpha$" if key == "alpha" else "$\\gamma$")
        ax.set_ylabel("Missed-failure rate")
        ax.set_title(f"Sensitivity to {key}")
        ax.grid(axis="y", alpha=0.18)
    axes[1].legend(frameon=False, loc="upper left", ncol=2)
    fig.tight_layout(w_pad=1.5)
    save_figure(fig, "sensitivity")

    stable_mean = stable.groupby(["batch", "phase"], as_index=False).domain_auc.mean()
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    colors = stable_mean.phase.map({"stable-prefix": "#4e79a7", "post-shift": "#f28e2b"})
    ax.scatter(stable_mean.batch, stable_mean.domain_auc, c=colors, s=20)
    ax.axhline(GAMMA, color="#b23a48", ls="--", lw=0.9)
    ax.axvline(3.5, color="#555555", ls=":", lw=0.9)
    ax.set_xlabel("Scenario batch")
    ax.set_ylabel("Domain AUC")
    ax.set_title("Stable-prefix detector stress test")
    ax.grid(axis="y", alpha=0.18)
    save_figure(fig, "stable_prefix_detector")


def write_manifest(datasets: list[PreparedData]) -> None:
    manifest = {
        "datasets": [
            {
                "name": data.name,
                "task": data.task,
                "source": data.source,
                "n_instances": int(len(data.y)),
                "n_raw_features": int(data.n_raw_features),
                "n_retained_features": int(len(data.retained_features)),
                "missing_values_before_imputation": int(data.missing_values_before_imputation),
                "positive_failures": int(np.sum(data.y == 1)),
                "train_range": [0, data.train_end - 1],
                "calibration_range": [data.train_end, data.cal_end - 1],
                "test_range": [data.cal_end, len(data.y) - 1],
            }
            for data in datasets
        ],
        "default_batch_size": DEFAULT_BATCH_SIZE,
        "alpha": ALPHA,
        "gamma": GAMMA,
        "seeds": SEEDS,
        "max_recent_inspected_labels": MAX_RECENT_LABELS,
        "test_labels_used_for_tuning": False,
        "sensitivity_aggregation": "each parameter value is evaluated for all five fixed seeds; reported standard deviations are across seed-level mean rates",
        "stable_prefix_analysis": "calibration-like bootstrap prefix followed by the original test feature stream",
    }
    (OUT_DIR / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    datasets = load_datasets()
    artifacts: dict[tuple[str, int], ModelArtifact] = {}
    main_rows: list[dict] = []
    drift_rows: list[pd.DataFrame] = []
    weighted_diagnostics: list[dict] = []
    stable_rows: list[pd.DataFrame] = []

    for data in datasets:
        for seed in SEEDS:
            artifact = fit_artifact(data, seed)
            artifacts[(data.name, seed)] = artifact
            main_rows.extend(main_rows_for_artifact(artifact, weighted_diagnostics))
            drift_frame = artifact.drift_records.copy()
            drift_frame.insert(0, "seed", seed)
            drift_frame.insert(0, "dataset", data.name)
            drift_rows.append(drift_frame)
            stable_rows.append(stable_prefix_analysis(artifact))

    main_frame = pd.DataFrame(main_rows)
    drift_frame = pd.concat(drift_rows, ignore_index=True)
    stable_frame = pd.concat(stable_rows, ignore_index=True)
    main_frame.to_csv(OUT_DIR / "seed_batch_metrics.csv", index=False)
    decision_trace_from_main_frame(main_frame).to_csv(
        OUT_DIR / "decision_trace.csv", index=False
    )
    drift_frame.to_csv(OUT_DIR / "drift_by_batch.csv", index=False)
    stable_frame.to_csv(OUT_DIR / "stable_prefix_analysis.csv", index=False)
    pd.DataFrame(weighted_diagnostics).to_csv(OUT_DIR / "weighted_diagnostics.csv", index=False)

    method_summary = summarize(main_frame, ["dataset", "method"])
    method_summary.to_csv(OUT_DIR / "method_summary.csv", index=False)
    for dataset in datasets:
        method_summary[method_summary.dataset == dataset.name].to_csv(
            OUT_DIR / f"{dataset.name.lower()}_method_summary.csv", index=False
        )

    ablation = ablation_experiments(artifacts)
    ablation.to_csv(OUT_DIR / "ablation_seed_batch_metrics.csv", index=False)
    ablation_summary = summarize(ablation, ["dataset", "method"])
    ablation_summary.to_csv(OUT_DIR / "ablation_summary.csv", index=False)
    ablation_with_proposed = pd.concat(
        [main_frame[main_frame.method == "DTCG-Proposed"], ablation],
        ignore_index=True,
    )
    first_five_summary(ablation_with_proposed).to_csv(
        OUT_DIR / "ablation_first5.csv", index=False
    )
    first_batch_summary(ablation_with_proposed).to_csv(
        OUT_DIR / "ablation_firstbatch.csv", index=False
    )

    tests = statistical_tests(main_frame)
    tests.to_csv(OUT_DIR / "statistical_tests.csv", index=False)

    sensitivity = sensitivity_experiments(artifacts)
    for key, frame in sensitivity.items():
        frame.to_csv(OUT_DIR / f"sensitivity_{key}.csv", index=False)

    make_figures(main_frame, drift_frame, sensitivity, stable_frame)
    write_manifest(datasets)

    print("=== Main method summary ===")
    print(method_summary.to_string(index=False))
    print("\n=== Ablation summary ===")
    print(ablation_summary.to_string(index=False))
    print("\n=== Statistical tests ===")
    print(tests.to_string(index=False))
    print("\nSaved results to", OUT_DIR)


if __name__ == "__main__":
    main()
