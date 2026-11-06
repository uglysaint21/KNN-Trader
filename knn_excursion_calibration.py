#!/usr/bin/env python3
"""Chronological post-hoc probability calibration for the KNN excursion model.

The calibration layer never changes the five-dimensional KNN features, neighbour
selection, target definitions, or OHLC first-hit labels. Calibration is fitted
only on validation forecasts and evaluated on the later untouched test period.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit, softmax
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

TARGETS_PIPS = list(range(1, 11))
FIRST_HIT_PAIRS = [(2, 2), (3, 2), (3, 3), (4, 3), (4, 4), (5, 3), (5, 4), (5, 5)]
EPS = 1e-6


@dataclass
class BinaryCalibrator:
    method: str
    model: Any = None
    constant: float | None = None

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        p = np.asarray(probabilities, dtype=float)
        clipped = np.clip(p, EPS, 1 - EPS)
        if self.constant is not None:
            return np.full(p.shape, self.constant, dtype=float)
        if self.method == "sigmoid":
            out = self.model.predict_proba(logit(clipped).reshape(-1, 1))[:, 1]
        elif self.method == "isotonic":
            out = self.model.predict(clipped)
        else:
            raise ValueError(f"Unknown binary calibration method: {self.method}")
        return np.clip(np.asarray(out, dtype=float), 0.0, 1.0)


def calibrate_binary_probability(probabilities: np.ndarray, labels: np.ndarray,
                                 method: str = "sigmoid") -> BinaryCalibrator:
    """Fit sigmoid/Platt or isotonic calibration to a probability series."""
    p = np.asarray(probabilities, dtype=float).reshape(-1)
    y = np.asarray(labels, dtype=int).reshape(-1)
    valid = np.isfinite(p) & np.isfinite(y)
    p, y = np.clip(p[valid], EPS, 1 - EPS), y[valid]
    if len(p) == 0:
        raise ValueError("No valid observations available to fit calibrator")
    if method not in {"sigmoid", "isotonic"}:
        raise ValueError(f"Unsupported method: {method}")
    if np.unique(y).size < 2:
        return BinaryCalibrator(method=method, constant=float(np.mean(y)))
    if method == "sigmoid":
        model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
        model.fit(logit(p).reshape(-1, 1), y)
        return BinaryCalibrator(method=method, model=model)
    model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    model.fit(p, y)
    return BinaryCalibrator(method=method, model=model)


def enforce_threshold_monotonicity(probabilities: np.ndarray) -> np.ndarray:
    """Project rows to non-increasing probabilities across ascending pip targets.

    Columns must be ordered by increasing barrier size. Cumulative minimum is a
    simple, deterministic projection that preserves [0,1] bounds.
    """
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 2:
        raise ValueError("Expected a 2-D array [rows, ascending thresholds]")
    if not np.isfinite(p).all():
        raise ValueError("Probabilities must be finite")
    return np.minimum.accumulate(np.clip(p, 0.0, 1.0), axis=1)


def _normalise_multiclass(probabilities: np.ndarray) -> np.ndarray:
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 2 or p.shape[1] < 2:
        raise ValueError("Multiclass probabilities must be a 2-D matrix with at least 2 classes")
    p = np.clip(p, EPS, None)
    sums = p.sum(axis=1, keepdims=True)
    return p / sums


def temperature_scale_multiclass(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    """Apply multiclass temperature scaling while preserving row sums of one."""
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite")
    p = _normalise_multiclass(probabilities)
    return softmax(np.log(p) / float(temperature), axis=1)


def fit_temperature(probabilities: np.ndarray, labels: np.ndarray) -> float:
    p = _normalise_multiclass(probabilities)
    y = np.asarray(labels, dtype=int).reshape(-1)
    valid = np.isfinite(y) & (y >= 0) & (y < p.shape[1])
    p, y = p[valid], y[valid]
    if len(y) < 20 or np.unique(y).size < 2:
        return 1.0
    def objective(log_t: float) -> float:
        q = temperature_scale_multiclass(p, float(np.exp(log_t)))
        return multiclass_metrics(q, y)["log_loss"]
    result = minimize_scalar(objective, bounds=(math.log(0.05), math.log(20.0)), method="bounded")
    return float(np.exp(result.x)) if result.success and np.isfinite(result.x) else 1.0


def binary_metrics(probabilities: np.ndarray, labels: np.ndarray, bins: int = 10) -> Dict[str, float]:
    p = np.asarray(probabilities, dtype=float).reshape(-1)
    y = np.asarray(labels, dtype=float).reshape(-1)
    valid = np.isfinite(p) & np.isfinite(y)
    p, y = np.clip(p[valid], EPS, 1 - EPS), y[valid]
    if len(y) == 0:
        return {"n": 0, "brier": math.nan, "log_loss": math.nan, "ece_10": math.nan, "event_rate": math.nan}
    brier = float(np.mean((p - y) ** 2))
    ll = float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))
    # Fixed-width reliability bins; empty bins contribute zero to ECE.
    edges = np.linspace(0.0, 1.0, bins + 1)
    bin_ids = np.minimum(np.digitize(p, edges[1:-1]), bins - 1)
    ece = 0.0
    for b in range(bins):
        mask = bin_ids == b
        if mask.any():
            ece += float(mask.mean()) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return {"n": int(len(y)), "brier": brier, "log_loss": ll, "ece_10": float(ece), "event_rate": float(y.mean()), "mean_probability": float(p.mean())}


def multiclass_metrics(probabilities: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    p = _normalise_multiclass(probabilities)
    y = np.asarray(labels, dtype=int).reshape(-1)
    valid = np.isfinite(y) & (y >= 0) & (y < p.shape[1])
    p, y = p[valid], y[valid]
    if len(y) == 0:
        return {"n": 0, "brier": math.nan, "log_loss": math.nan, "row_sum_min": math.nan, "row_sum_max": math.nan}
    onehot = np.eye(p.shape[1])[y]
    chosen = np.clip(p[np.arange(len(y)), y], EPS, 1)
    return {"n": int(len(y)), "brier": float(np.mean(np.sum((p - onehot) ** 2, axis=1))),
            "log_loss": float(-np.mean(np.log(chosen))), "row_sum_min": float(p.sum(axis=1).min()),
            "row_sum_max": float(p.sum(axis=1).max())}


def _binary_columns() -> List[Tuple[str, str, str]]:
    # Include direction of the five-candle ending close as well as all nested
    # high/low excursion events.
    cols = [("p_end_close_up", "actual_end_close_up", "end_close_up")]
    cols.extend((f"p_hit_{d}_{a}pip", f"actual_hit_{d}_{a}pip", f"{d}_{a}pip")
                for a in TARGETS_PIPS for d in ("up", "down"))
    return cols


def _multiclass_columns(tp: int, sl: int) -> List[str]:
    return [f"p_tp{tp}_before_sl{sl}", f"p_sl{sl}_before_tp{tp}",
            f"p_neither_tp{tp}_sl{sl}", f"p_ambiguous_tp{tp}_sl{sl}"]


def _evaluate_binary_matrix(df: pd.DataFrame, suffix: str = "") -> Dict[str, float]:
    scores = []
    logloss = []
    eces = []
    for pcol, ycol, name in _binary_columns():
        col = pcol if not suffix else f"{pcol}_{suffix}"
        if col not in df or ycol not in df:
            continue
        m = binary_metrics(df[col].to_numpy(float), df[ycol].to_numpy(float))
        scores.append(m["brier"]); logloss.append(m["log_loss"]); eces.append(m["ece_10"])
    return {"mean_brier": float(np.nanmean(scores)) if scores else math.nan,
            "mean_log_loss": float(np.nanmean(logloss)) if logloss else math.nan,
            "mean_ece_10": float(np.nanmean(eces)) if eces else math.nan,
            "events_evaluated": int(len(scores))}


def calibrate_files(validation_csv: str | Path, test_csv: str | Path, out_dir: str | Path,
                    fit_fraction: float = 0.60, min_isotonic_each_class: int = 50) -> Dict[str, Any]:
    """Compare calibration on a time-ordered validation sub-split, then test once."""
    val = pd.read_csv(validation_csv)
    test = pd.read_csv(test_csv)
    if len(val) < 100 or len(test) < 100:
        raise ValueError("At least 100 validation and 100 test forecasts are required")
    if not 0.2 <= fit_fraction <= 0.8:
        raise ValueError("fit_fraction must be between 0.2 and 0.8")
    # Sort by timestamp before splitting, regardless of source file order.
    val = val.sort_values("row_index").reset_index(drop=True)
    test = test.sort_values("row_index").reset_index(drop=True)
    split = int(len(val) * fit_fraction)
    fit, tune = val.iloc[:split].copy(), val.iloc[split:].copy()
    if len(tune) < 30:
        raise ValueError("Calibration-selection segment is too small")

    # Compare global sigmoid and isotonic approaches on the later validation block.
    comparison: Dict[str, Dict[str, Any]] = {}
    for method in ("raw", "sigmoid", "isotonic"):
        scores, losses, eces = [], [], []
        for pcol, ycol, _ in _binary_columns():
            if pcol not in val or ycol not in val:
                continue
            if method == "raw":
                p_tune = tune[pcol].to_numpy(float)
            else:
                yfit = fit[ycol].to_numpy(int)
                if method == "isotonic" and (int(yfit.sum()) < min_isotonic_each_class or int((1 - yfit).sum()) < min_isotonic_each_class):
                    # Isotonic has too little support for this event; use sigmoid fallback.
                    used = "sigmoid"
                else:
                    used = method
                cal = calibrate_binary_probability(fit[pcol].to_numpy(float), yfit, used)
                p_tune = cal.predict(tune[pcol].to_numpy(float))
            met = binary_metrics(p_tune, tune[ycol].to_numpy(float))
            scores.append(met["brier"]); losses.append(met["log_loss"]); eces.append(met["ece_10"])
        comparison[method] = {"mean_brier": float(np.nanmean(scores)), "mean_log_loss": float(np.nanmean(losses)),
                              "mean_ece_10": float(np.nanmean(eces)), "events_evaluated": len(scores)}

    # Isotonic is selected only if it yields a practical Brier gain and does not
    # materially worsen log loss; otherwise prefer the lower-variance sigmoid.
    sig = comparison["sigmoid"]
    iso = comparison["isotonic"]
    selected_binary_method = "isotonic" if (iso["mean_brier"] < sig["mean_brier"] - 0.001 and iso["mean_log_loss"] <= sig["mean_log_loss"] + 0.02) else "sigmoid"

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    calibrated_test = test.copy()
    # Fit final binary calibrators on the full validation period only.
    binary_details = []
    for pcol, ycol, name in _binary_columns():
        if pcol not in val or ycol not in val or pcol not in test:
            continue
        yfit = val[ycol].to_numpy(int)
        method = selected_binary_method
        if method == "isotonic" and (int(yfit.sum()) < min_isotonic_each_class or int((1-yfit).sum()) < min_isotonic_each_class):
            method = "sigmoid"
        cal = calibrate_binary_probability(val[pcol].to_numpy(float), yfit, method)
        calibrated_test[f"{pcol}_calibrated_raw"] = cal.predict(test[pcol].to_numpy(float))
        calibrated_test[f"{pcol}_calibrated"] = calibrated_test[f"{pcol}_calibrated_raw"]
        binary_details.append({"event": name, "method": method, "validation_positives": int(yfit.sum()), "validation_negatives": int((1-yfit).sum())})

    # Enforce the nested-event invariant P(hit >= k+1) <= P(hit >= k) per direction.
    for direction in ("up", "down"):
        cols = [f"p_hit_{direction}_{a}pip_calibrated_raw" for a in TARGETS_PIPS]
        if all(c in calibrated_test for c in cols):
            projected = enforce_threshold_monotonicity(calibrated_test[cols].to_numpy(float))
            for j, a in enumerate(TARGETS_PIPS):
                calibrated_test[f"p_hit_{direction}_{a}pip_calibrated"] = projected[:, j]

    # Multiclass first-hit: temperature scaling is the low-parameter candidate.
    first_hit_details = []
    for tp, sl in FIRST_HIT_PAIRS:
        cols = _multiclass_columns(tp, sl)
        actual_col = f"actual_first_hit_tp{tp}_sl{sl}"
        if not all(c in val.columns and c in test.columns for c in [*cols, actual_col]):
            continue
        pfit = val[cols].to_numpy(float); yfit = val[actual_col].to_numpy(int)
        p_tune = tune[cols].to_numpy(float); y_tune = tune[actual_col].to_numpy(int)
        temperature = fit_temperature(fit[cols].to_numpy(float), fit[actual_col].to_numpy(int))
        raw_tune_metrics = multiclass_metrics(p_tune, y_tune)
        temp_tune_metrics = multiclass_metrics(temperature_scale_multiclass(p_tune, temperature), y_tune)
        use_temp = (temp_tune_metrics["brier"] < raw_tune_metrics["brier"] and
                    temp_tune_metrics["log_loss"] <= raw_tune_metrics["log_loss"] + 0.02)
        final_temperature = fit_temperature(pfit, yfit) if use_temp else 1.0
        calibrated_test[[f"{c}_calibrated" for c in cols]] = temperature_scale_multiclass(test[cols].to_numpy(float), final_temperature)
        first_hit_details.append({"tp_pips": tp, "sl_pips": sl, "temperature_candidate": temperature,
                                  "selected_temperature": final_temperature, "selected": bool(use_temp),
                                  "tune_raw": raw_tune_metrics, "tune_temperature": temp_tune_metrics})

    # Save reliability diagnostics and per-event metrics. Original raw columns remain untouched.
    event_metrics = []
    for pcol, ycol, name in _binary_columns():
        if pcol not in test or ycol not in test:
            continue
        # The event name already includes the pip suffix, e.g. up_1pip.
        calibrated_col = f"{pcol}_calibrated"
        for variant, col in [("raw", pcol), ("calibrated", calibrated_col)]:
            if col not in calibrated_test:
                continue
            met = binary_metrics(calibrated_test[col].to_numpy(float), calibrated_test[ycol].to_numpy(float))
            event_metrics.append({"event": name, "variant": variant, "prediction_column": col,
                                  "actual_column": ycol, **met})
    pd.DataFrame(event_metrics).to_csv(out_dir / "calibration_binary_test_metrics.csv", index=False)
    calibrated_test.to_csv(out_dir / "test_predictions_filtered_adaptive_calibrated.csv", index=False)
    # Save both pre-calibration and test reliability bin data for the pooled event forecasts.
    rel_rows = []
    for event in event_metrics:
        name, variant = event["event"], event["variant"]
        pcol = event["prediction_column"]
        ycol = event["actual_column"]
        p = calibrated_test[pcol].to_numpy(float); y = calibrated_test[ycol].to_numpy(int)
        ids = np.minimum((np.clip(p, 0, 1) * 10).astype(int), 9)
        for b in range(10):
            mask = ids == b
            if mask.any():
                rel_rows.append({"event": name, "variant": variant, "bin_lower": b/10, "bin_upper": (b+1)/10,
                                 "n": int(mask.sum()), "mean_probability": float(p[mask].mean()), "observed_rate": float(y[mask].mean())})
    pd.DataFrame(rel_rows).to_csv(out_dir / "calibration_reliability_bins.csv", index=False)

    # Compare aggregated test diagnostics for raw versus calibrated excursion probabilities.
    raw_test_summary = _evaluate_binary_matrix(calibrated_test, "")
    cal_test_summary = _evaluate_binary_matrix(calibrated_test, "calibrated")
    first_hit_test = {}
    for tp, sl in FIRST_HIT_PAIRS:
        cols = _multiclass_columns(tp, sl)
        cal_cols = [f"{c}_calibrated" for c in cols]
        ycol = f"actual_first_hit_tp{tp}_sl{sl}"
        if all(c in calibrated_test.columns for c in [*cols, *cal_cols, ycol]):
            first_hit_test[f"tp{tp}_sl{sl}"] = {"raw": multiclass_metrics(calibrated_test[cols].to_numpy(float), calibrated_test[ycol].to_numpy(int)),
                                                "calibrated": multiclass_metrics(calibrated_test[cal_cols].to_numpy(float), calibrated_test[ycol].to_numpy(int))}

    first_hit_test_mean = {}
    for variant in ("raw", "calibrated"):
        scores = [v[variant]["brier"] for v in first_hit_test.values() if np.isfinite(v[variant]["brier"])]
        losses = [v[variant]["log_loss"] for v in first_hit_test.values() if np.isfinite(v[variant]["log_loss"])]
        first_hit_test_mean[variant] = {"mean_brier": float(np.mean(scores)) if scores else math.nan,
                                        "mean_log_loss": float(np.mean(losses)) if losses else math.nan,
                                        "pairs_evaluated": len(scores)}

    summary = {
        "split": {"validation_forecasts": len(val), "calibrator_fit_rows": len(fit), "method_selection_rows": len(tune),
                  "test_forecasts": len(test), "validation_fit_fraction": fit_fraction,
                  "calibrator_fitted_on": "validation forecasts only", "final_test_used_for_selection": False},
        "binary_method_comparison_on_later_validation_segment": comparison,
        "selected_binary_method": selected_binary_method,
        "binary_test_raw": raw_test_summary,
        "binary_test_calibrated": cal_test_summary,
        "binary_test_event_details": binary_details,
        "first_hit_test": first_hit_test,
        "first_hit_test_mean": first_hit_test_mean,
        "first_hit_calibration_details": first_hit_details,
        "interpretation": [
            "Calibration uses only chronological validation forecasts; the final test labels are not used for fitting or method selection.",
            "Binary methods: Platt-style sigmoid and isotonic regression are compared on a later validation segment for ending-close direction plus all 20 pip-excursion events. Isotonic is chosen only for a practical Brier improvement without material log-loss degradation; events with too few validation examples fall back to sigmoid.",
            "First-hit probabilities use a one-parameter temperature-scaling candidate selected on the later validation segment; output category probabilities sum to one.",
            "Calibrated probabilities for the nested +1..+10 pip and -1..-10 pip hit events are projected to be non-increasing as the barrier increases.",
            "Calibration improves the interpretation of forecast probabilities; it does not establish a profitable trading edge, and costs/fills are not modeled here."
        ]
    }
    with open(out_dir / "calibration_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, allow_nan=False, default=lambda x: None if isinstance(x, float) and not math.isfinite(x) else x)
    _write_report(summary, out_dir / "KNN_CALIBRATION_REPORT.md")
    return summary


def _write_report(summary: Dict[str, Any], path: Path) -> None:
    split = summary["split"]
    raw = summary["binary_test_raw"]
    cal = summary["binary_test_calibrated"]
    lines = ["# KNN Price-Excursion Model — Probability Calibration", "",
             "## Calibration protocol", "",
             f"The validation file contained {split['validation_forecasts']:,} forecasts. The first {split['calibrator_fit_rows']:,} were used to fit calibrators, the later {split['method_selection_rows']:,} to select calibration methods, and the final {split['test_forecasts']:,} test forecasts were evaluated once.", "",
             "The five-dimensional KNN, neighbour screening, volatility filter, temporal de-clustering, adaptive neighbour count, OHLC event definitions, and ambiguous first-hit category are unchanged.", "",
             "## Binary excursion probabilities — untouched test set", "",
             "| Metric | Raw | Calibrated | Change (calibrated − raw) |", "|---|---:|---:|---:|"]
    for metric in ("mean_brier", "mean_log_loss", "mean_ece_10"):
        a, b = raw.get(metric, math.nan), cal.get(metric, math.nan)
        delta = b - a if np.isfinite(a) and np.isfinite(b) else math.nan
        lines.append(f"| {metric} | {a:.6f} | {b:.6f} | {delta:+.6f} |")
    lines += ["", f"**Selected binary method:** {summary['selected_binary_method']}", "",
              "Lower Brier, log loss, and expected calibration error (ECE) are better. A calibration can improve some metrics and worsen others, so inspect all three instead of declaring success from one number.", "",
              "## Binary calibration method comparison", "", "| Method | Mean Brier | Mean log loss | Mean ECE (10 bins) |", "|---|---:|---:|---:|"]
    for method, m in summary["binary_method_comparison_on_later_validation_segment"].items():
        lines.append(f"| {method} | {m['mean_brier']:.6f} | {m['mean_log_loss']:.6f} | {m['mean_ece_10']:.6f} |")
    first_raw = summary["first_hit_test_mean"]["raw"]
    first_cal = summary["first_hit_test_mean"]["calibrated"]
    lines += ["", "## First-hit probabilities", "", "For each target/stop pair, four outcomes are retained: target first, stop first, neither within five bars, and ambiguous same-candle double touch. Temperature scaling is accepted only if the later validation segment improves Brier score without materially worsening log loss.", "", f"Across {first_raw['pairs_evaluated']} TP/SL pairs, mean multiclass Brier was {first_raw['mean_brier']:.6f} raw versus {first_cal['mean_brier']:.6f} calibrated; mean log loss was {first_raw['mean_log_loss']:.6f} raw versus {first_cal['mean_log_loss']:.6f} calibrated.", "", "| Pair | Raw Brier | Calibrated Brier | Raw log loss | Calibrated log loss | Temperature |", "|---|---:|---:|---:|---:|---:|"]
    selected_temp = {(x['tp_pips'], x['sl_pips']): x['selected_temperature'] for x in summary['first_hit_calibration_details']}
    for key, v in summary["first_hit_test"].items():
        tp, sl = map(int, key.replace("tp", "").split("_sl"))
        lines.append(f"| TP {tp} / SL {sl} | {v['raw']['brier']:.6f} | {v['calibrated']['brier']:.6f} | {v['raw']['log_loss']:.6f} | {v['calibrated']['log_loss']:.6f} | {selected_temp[(tp, sl)]:.4f} |")
    lines += ["", "## Output files", "", "- `test_predictions_filtered_adaptive_calibrated.csv`: original raw predictions plus calibrated probability columns; the original columns are retained unchanged.", "- `calibration_summary.json`: method selection and test metrics.", "- `calibration_binary_test_metrics.csv`: Brier, log loss, ECE, event rate, and mean forecast probability by event and raw/calibrated variant.", "- `calibration_reliability_bins.csv`: observed event rate versus mean forecast probability by reliability bin.", "", "## Limitations", "", "The dataset supplied covers 2026-03-10 through 2026-06-15 in the file, with 98,851 valid rows; the file name is not treated as proof of longer historical coverage. Validation calibration may still be affected by temporal/regime drift between validation and test. Results are gross of spread, commission, slippage and execution latency; this is not evidence of a profitable strategy or a live-trading recommendation.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--validation", required=True, help="validation_predictions_filtered_adaptive.csv")
    ap.add_argument("--test", required=True, help="test_predictions_filtered_adaptive.csv")
    ap.add_argument("--out-dir", default="calibrated_results", help="Where calibrated outputs are saved")
    ap.add_argument("--fit-fraction", type=float, default=0.60, help="Initial validation fraction used to fit calibration methods")
    args = ap.parse_args()
    summary = calibrate_files(args.validation, args.test, args.out_dir, args.fit_fraction)
    print(json.dumps({"selected_binary_method": summary["selected_binary_method"],
                      "binary_test_raw": summary["binary_test_raw"],
                      "binary_test_calibrated": summary["binary_test_calibrated"]}, indent=2))
    print(f"Report: {Path(args.out_dir) / 'KNN_CALIBRATION_REPORT.md'}")


if __name__ == "__main__":
    main()