#!/usr/bin/env python3
"""Probabilistic KNN model for five-minute EUR/USD price excursions.

Input features: five consecutive M1 open-to-close log returns, normalized by
an as-of rolling volatility scale. Outcomes: the next five candles' OHLC path.

The script compares:
  1) a distance-weighted KNN baseline; and
  2) the proposed volatility-filtered, temporally de-clustered, adaptive-radius KNN.

Predictions are made every five bars for evaluation so evaluation outcome windows
are non-overlapping. The script uses a chronological 70%/15%/15% split:
- validation predictions use only examples whose five-bar outcomes are complete
  before the validation period;
- final test predictions use examples whose outcomes are complete before the
  final test period begins.

This is research software, not an automated trading system.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

HORIZON = 5
FEATURE_DIM = 5
PIP_SIZE = 0.0001  # EUR/USD conventional pip
TARGETS_PIPS = list(range(1, 11))
FIRST_HIT_PAIRS = [(2, 2), (3, 2), (3, 3), (4, 3), (4, 4), (5, 3), (5, 4), (5, 5)]
INITIAL_POOL = 1000
MAX_FINAL_NEIGHBORS = 120
MIN_ADAPTIVE_NEIGHBORS = 30
ELBOW_RATIO_THRESHOLD = 1.35
VOL_RATIO_LIMIT = 2.0
DECLUSTER_GAP_BARS = 10
VOL_WINDOW = 60


def load_ohlc(path: str) -> pd.DataFrame:
    """Load common MetaTrader tab-delimited OHLC export or a CSV equivalent."""
    try:
        df = pd.read_csv(path, sep="\t")
        if len(df.columns) <= 1:
            df = pd.read_csv(path)
    except Exception:
        df = pd.read_csv(path)
    df.columns = [str(c).strip().replace("<", "").replace(">", "").upper() for c in df.columns]

    # Join MetaTrader DATE + TIME columns when present.
    if "DATETIME" not in df.columns:
        if "DATE" in df.columns and "TIME" in df.columns:
            df["DATETIME"] = df["DATE"].astype(str).str.strip() + " " + df["TIME"].astype(str).str.strip()
        elif "DATE" in df.columns:
            df["DATETIME"] = df["DATE"].astype(str).str.strip()
        else:
            # accept already-indexed files with a timestamp-like first column
            first = df.columns[0]
            parsed = pd.to_datetime(df[first], errors="coerce")
            if parsed.notna().mean() > 0.9:
                df["DATETIME"] = parsed
            else:
                raise ValueError(f"Could not find DATE/TIME columns. Columns: {list(df.columns)}")

    # MetaTrader date format: YYYY.MM.DD HH:MM:SS
    df["DATETIME"] = pd.to_datetime(df["DATETIME"], errors="coerce")
    required = ["OPEN", "HIGH", "LOW", "CLOSE"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required OHLC columns: {missing}. Found: {list(df.columns)}")
    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["DATETIME", *required]).sort_values("DATETIME")
    df = df.drop_duplicates("DATETIME", keep="last").reset_index(drop=True)
    valid = (
        (df["OPEN"] > 0) & (df["HIGH"] > 0) & (df["LOW"] > 0) & (df["CLOSE"] > 0)
        & (df["HIGH"] >= df[["OPEN", "CLOSE", "LOW"]].max(axis=1))
        & (df["LOW"] <= df[["OPEN", "CLOSE", "HIGH"]].min(axis=1))
    )
    invalid_count = int((~valid).sum())
    df = df.loc[valid].reset_index(drop=True)
    if len(df) < 1000:
        raise ValueError(f"Only {len(df)} valid OHLC rows; at least 1,000 are needed for a useful prototype.")
    df.attrs["invalid_rows_removed"] = invalid_count
    return df


def build_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return sample rows, normalized 5D features, volatility scale, and raw returns."""
    op = df["OPEN"].to_numpy(dtype=float)
    hi = df["HIGH"].to_numpy(dtype=float)
    lo = df["LOW"].to_numpy(dtype=float)
    cl = df["CLOSE"].to_numpy(dtype=float)
    n = len(df)
    body = np.log(cl / op)
    cc_ret = np.zeros(n, dtype=float)
    cc_ret[1:] = np.log(cl[1:] / cl[:-1])
    # RMS of recent close-to-close returns, available as of candle close.
    scale = pd.Series(cc_ret * cc_ret).rolling(VOL_WINDOW, min_periods=max(30, VOL_WINDOW // 2)).mean().pow(0.5).to_numpy()
    # Avoid degenerate scales; retain the current data's observable past only.
    finite_positive = scale[np.isfinite(scale) & (scale > 0)]
    fallback = float(np.nanmedian(finite_positive)) if len(finite_positive) else 1e-6
    scale[~np.isfinite(scale) | (scale <= 1e-12)] = fallback

    rows = np.arange(max(FEATURE_DIM - 1, VOL_WINDOW - 1), n - HORIZON, dtype=np.int64)
    offsets = np.arange(FEATURE_DIM - 1, -1, -1, dtype=np.int64)
    body_windows = body[rows[:, None] - offsets[None, :]]
    X = body_windows / scale[rows, None]
    return rows, X.astype(np.float64), scale, cc_ret


def get_neighbour_indices(train_rows: np.ndarray, query_rows: np.ndarray, all_features: np.ndarray,
                          row_to_sample: np.ndarray, n_neighbors: int = INITIAL_POOL):
    """Fit KD-tree on chronological training candidates and retrieve large pools."""
    train_sample_positions = row_to_sample[train_rows]
    query_sample_positions = row_to_sample[query_rows]
    if np.any(train_sample_positions < 0) or np.any(query_sample_positions < 0):
        raise ValueError("Internal row-to-feature mapping error.")
    k = min(n_neighbors, len(train_rows))
    model = NearestNeighbors(n_neighbors=k, algorithm="kd_tree", leaf_size=40, n_jobs=-1)
    model.fit(all_features[train_sample_positions])
    distances, indices = model.kneighbors(all_features[query_sample_positions], return_distance=True)
    # Convert positions in train_rows to original candle row indices.
    candidate_rows = train_rows[indices]
    return distances, candidate_rows


def adaptive_count(distances: np.ndarray, min_neighbors: int = MIN_ADAPTIVE_NEIGHBORS,
                   max_neighbors: int = MAX_FINAL_NEIGHBORS,
                   ratio_threshold: float = ELBOW_RATIO_THRESHOLD) -> int:
    """Choose a local neighbour count using a distance-gap/elbow heuristic."""
    n = len(distances)
    if n <= min_neighbors:
        return n
    top = min(max_neighbors, n)
    if top <= min_neighbors:
        return top
    ds = np.maximum(np.asarray(distances[:top], dtype=float), 1e-12)
    # Look for a conspicuous multiplicative gap after at least min_neighbors.
    ratios = ds[1:] / ds[:-1]
    start_idx = min_neighbors - 1
    eligible = np.arange(len(ratios)) >= start_idx
    if not np.any(eligible):
        return top
    candidate_positions = np.where(eligible)[0]
    best_pos = candidate_positions[np.argmax(ratios[eligible])]
    if ratios[best_pos] >= ratio_threshold:
        return max(min_neighbors, int(best_pos + 1))
    return top


def select_neighbours(candidate_rows: np.ndarray, distances: np.ndarray, current_scale: float,
                      historical_scale_by_row: np.ndarray, mode: str,
                      vol_ratio_limit: float = VOL_RATIO_LIMIT,
                      decluster_gap: int = DECLUSTER_GAP_BARS) -> Tuple[np.ndarray, np.ndarray, float]:
    """Filter the ordered KNN pool and return selected rows, distances, bandwidth."""
    candidate_rows = np.asarray(candidate_rows, dtype=np.int64)
    distances = np.asarray(distances, dtype=float)
    if mode == "baseline":
        count = min(80, len(candidate_rows))
        selected_rows = candidate_rows[:count]
        selected_distances = distances[:count]
        if len(selected_distances) == 0:
            return selected_rows, selected_distances, 1.0
        h = max(float(selected_distances[-1]) * 1.05, 1e-12)
        return selected_rows, selected_distances, h

    # First screen by volatility similarity. A ratio in [1/2, 2] is the
    # initial, explicit rule; validation/test metrics determine whether it helps.
    ratios = historical_scale_by_row[candidate_rows] / max(current_scale, 1e-12)
    eligible = (ratios >= 1.0 / vol_ratio_limit) & (ratios <= vol_ratio_limit)
    screened_rows = candidate_rows[eligible]
    screened_distances = distances[eligible]

    # Greedily de-cluster candidates by candle index, preserving distance order.
    chosen_rows: List[int] = []
    chosen_distances: List[float] = []
    forbidden_rows = set()
    for r, d in zip(screened_rows, screened_distances):
        r_int = int(r)
        if r_int not in forbidden_rows:
            chosen_rows.append(r_int)
            chosen_distances.append(float(d))
            # Mark every candle index close enough to overlap the candidate's
            # combined input+future window. This avoids an O(K^2) Python scan.
            forbidden_rows.update(range(r_int - decluster_gap + 1, r_int + decluster_gap))
            if len(chosen_rows) >= MAX_FINAL_NEIGHBORS:
                break

    if not chosen_rows:
        # Graceful fallback: the closest raw candidates, still distance-weighted.
        count = min(30, len(candidate_rows))
        selected_rows = candidate_rows[:count]
        selected_distances = distances[:count]
        h = max(float(selected_distances[-1]) * 1.05, 1e-12) if len(selected_distances) else 1.0
        return selected_rows, selected_distances, h

    selected_rows = np.asarray(chosen_rows, dtype=np.int64)
    selected_distances = np.asarray(chosen_distances, dtype=float)
    count = adaptive_count(selected_distances)
    selected_rows = selected_rows[:count]
    selected_distances = selected_distances[:count]
    h = max(float(selected_distances[-1]) * 1.05, 1e-12)
    return selected_rows, selected_distances, h


def kernel_weights(distances: np.ndarray, bandwidth: float) -> np.ndarray:
    d = np.asarray(distances, dtype=float)
    h = max(float(bandwidth), 1e-12)
    z = d / h
    w = np.square(np.maximum(0.0, 1.0 - z * z))
    if not np.isfinite(w).all() or w.sum() <= 1e-15:
        # Stable fallback to inverse-distance weighting.
        w = 1.0 / np.maximum(d, 1e-9)
    return w / w.sum()


def get_future_arrays(rows: np.ndarray, op: np.ndarray, hi: np.ndarray, lo: np.ndarray, cl: np.ndarray):
    offsets = np.arange(1, HORIZON + 1, dtype=np.int64)
    anchor = cl[rows]
    fut_open = op[rows[:, None] + offsets[None, :]]
    fut_high = hi[rows[:, None] + offsets[None, :]]
    fut_low = lo[rows[:, None] + offsets[None, :]]
    fut_close = cl[rows + HORIZON]
    up_exc = (np.max(fut_high, axis=1) - anchor) / PIP_SIZE
    down_exc = (np.min(fut_low, axis=1) - anchor) / PIP_SIZE
    end_ret = np.log(fut_close / anchor)
    return anchor, fut_open, fut_high, fut_low, fut_close, up_exc, down_exc, end_ret


def first_hit_category(anchor: np.ndarray, fut_open: np.ndarray, fut_high: np.ndarray, fut_low: np.ndarray,
                       tp_pips: int, sl_pips: int) -> np.ndarray:
    """Classify each path as target-first(0), stop-first(1), neither(2), ambiguous(3).

    If the open is already beyond a barrier, it is treated as the first event at
    that minute's open. If both high and low hit their barriers in one candle and
    the open is between the barriers, OHLC cannot reveal the order; mark ambiguous.
    """
    n = len(anchor)
    result = np.full(n, 2, dtype=np.int8)  # default neither
    unresolved = np.ones(n, dtype=bool)
    up_level = anchor + tp_pips * PIP_SIZE
    down_level = anchor - sl_pips * PIP_SIZE
    for j in range(HORIZON):
        ids = np.where(unresolved)[0]
        if len(ids) == 0:
            break
        o = fut_open[ids, j]
        h = fut_high[ids, j]
        l = fut_low[ids, j]
        # Opening gap/order has precedence over intrabar high/low.
        open_up = o >= up_level[ids] - 1e-12
        open_down = o <= down_level[ids] + 1e-12
        up_hit = h >= up_level[ids] - 1e-12
        down_hit = l <= down_level[ids] + 1e-12

        # Open beyond one barrier resolves direction even if the range also
        # reaches the other barrier later in that minute.
        up_first = open_up
        down_first = (~open_up) & open_down
        intrabar_both = (~open_up) & (~open_down) & up_hit & down_hit
        intrabar_up = (~open_up) & (~open_down) & up_hit & (~down_hit)
        intrabar_down = (~open_up) & (~open_down) & down_hit & (~up_hit)

        result[ids[up_first | intrabar_up]] = 0
        result[ids[down_first | intrabar_down]] = 1
        result[ids[intrabar_both]] = 3
        done = up_first | down_first | intrabar_both | intrabar_up | intrabar_down
        unresolved[ids[done]] = False
    return result


def weighted_predictions(query_row: int, selected_rows: np.ndarray, selected_distances: np.ndarray, bandwidth: float,
                         op: np.ndarray, hi: np.ndarray, lo: np.ndarray, cl: np.ndarray) -> Dict[str, float]:
    if len(selected_rows) == 0:
        return {}
    weights = kernel_weights(selected_distances, bandwidth)
    anchor, fop, fhi, flo, fcl, up_exc, down_exc, end_ret = get_future_arrays(selected_rows, op, hi, lo, cl)
    out: Dict[str, float] = {}
    out["neighbor_count"] = float(len(selected_rows))
    out["effective_sample_size"] = float(1.0 / np.sum(weights * weights))
    out["median_neighbor_distance"] = float(np.median(selected_distances))
    out["max_neighbor_distance"] = float(np.max(selected_distances))
    out["p_end_close_up"] = float(weights[end_ret > 0].sum())
    out["expected_end_return_pips"] = float(np.sum(weights * end_ret) * 10000.0)

    for a in TARGETS_PIPS:
        out[f"p_hit_up_{a}pip"] = float(weights[up_exc >= a - 1e-8].sum())
        out[f"p_hit_down_{a}pip"] = float(weights[down_exc <= -a + 1e-8].sum())

    for tp, sl in FIRST_HIT_PAIRS:
        cats = first_hit_category(anchor, fop, fhi, flo, tp, sl)
        out[f"p_tp{tp}_before_sl{sl}"] = float(weights[cats == 0].sum())
        out[f"p_sl{sl}_before_tp{tp}"] = float(weights[cats == 1].sum())
        out[f"p_neither_tp{tp}_sl{sl}"] = float(weights[cats == 2].sum())
        out[f"p_ambiguous_tp{tp}_sl{sl}"] = float(weights[cats == 3].sum())
    return out


def actual_outcome(query_row: int, op: np.ndarray, hi: np.ndarray, lo: np.ndarray, cl: np.ndarray) -> Dict[str, float]:
    rows = np.array([query_row], dtype=np.int64)
    anchor, fop, fhi, flo, fcl, up_exc, down_exc, end_ret = get_future_arrays(rows, op, hi, lo, cl)
    out: Dict[str, float] = {
        "actual_up_excursion_pips": float(up_exc[0]),
        "actual_down_excursion_pips": float(down_exc[0]),
        "actual_end_return_pips": float(end_ret[0] * 10000.0),
        "actual_end_close_up": int(end_ret[0] > 0),
    }
    for a in TARGETS_PIPS:
        out[f"actual_hit_up_{a}pip"] = int(up_exc[0] >= a)
        out[f"actual_hit_down_{a}pip"] = int(down_exc[0] <= -a)
    for tp, sl in FIRST_HIT_PAIRS:
        cats = first_hit_category(anchor, fop, fhi, flo, tp, sl)
        out[f"actual_first_hit_tp{tp}_sl{sl}"] = int(cats[0])
    return out


def run_predictions(df: pd.DataFrame, sample_rows: np.ndarray, X: np.ndarray, scale: np.ndarray,
                    row_to_sample: np.ndarray, query_rows: np.ndarray, train_cut_row: int,
                    mode: str, pool_size: int = INITIAL_POOL) -> pd.DataFrame:
    train_rows = sample_rows[sample_rows + HORIZON < train_cut_row]
    if len(train_rows) < 100:
        raise ValueError(f"Insufficient historical samples before row {train_cut_row}: {len(train_rows)}")
    dist_matrix, cand_rows_matrix = get_neighbour_indices(train_rows, query_rows, X, row_to_sample, pool_size)
    op = df["OPEN"].to_numpy(dtype=float)
    hi = df["HIGH"].to_numpy(dtype=float)
    lo = df["LOW"].to_numpy(dtype=float)
    cl = df["CLOSE"].to_numpy(dtype=float)

    result_rows = []
    for q, (dists, cand_rows) in enumerate(zip(dist_matrix, cand_rows_matrix)):
        query_row = int(query_rows[q])
        qscale = scale[query_row]
        sel_rows, sel_dists, bandwidth = select_neighbours(
            cand_rows, dists, qscale, scale, mode=mode,
        )
        pred = weighted_predictions(query_row, sel_rows, sel_dists, bandwidth, op, hi, lo, cl)
        actual = actual_outcome(query_row, op, hi, lo, cl)
        rec = {"datetime": df.loc[query_row, "DATETIME"].isoformat(), "row_index": query_row,
               "model": mode, **pred, **actual}
        result_rows.append(rec)
    return pd.DataFrame(result_rows)


def brier_metrics(pred_df: pd.DataFrame, train_event_rates: Dict[str, float] | None = None) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    briers = []
    base_briers = []
    for a in TARGETS_PIPS:
        for direction in ("up", "down"):
            pcol = f"p_hit_{direction}_{a}pip"
            ycol = f"actual_hit_{direction}_{a}pip"
            if pcol not in pred_df or ycol not in pred_df:
                continue
            p = pred_df[pcol].to_numpy(float)
            y = pred_df[ycol].to_numpy(float)
            m = np.isfinite(p) & np.isfinite(y)
            if not m.any():
                continue
            score = float(np.mean((p[m] - y[m]) ** 2))
            base_rate = float(train_event_rates.get(ycol, np.mean(y[m]))) if train_event_rates else float(np.mean(y[m]))
            base_score = float(np.mean((base_rate - y[m]) ** 2))
            metrics[f"brier_{direction}_{a}pip"] = score
            metrics[f"test_event_rate_{direction}_{a}pip"] = float(np.mean(y[m]))
            metrics[f"train_base_rate_{direction}_{a}pip"] = base_rate
            metrics[f"brier_baseline_{direction}_{a}pip"] = base_score
            metrics[f"brier_skill_{direction}_{a}pip"] = float(1.0 - score / base_score) if base_score > 0 else None
            briers.append(score)
            base_briers.append(base_score)
    metrics["mean_excursion_brier"] = float(np.mean(briers)) if briers else math.nan
    metrics["mean_climatology_brier"] = float(np.mean(base_briers)) if base_briers else math.nan
    metrics["mean_brier_skill"] = float(1.0 - np.mean(briers) / np.mean(base_briers)) if briers and np.mean(base_briers) > 0 else math.nan

    # Multiclass Brier score for each first-hit outcome: target first, stop first,
    # neither within horizon, or intrabar OHLC ambiguity.
    first_scores = []
    for tp, sl in FIRST_HIT_PAIRS:
        actual_col = f"actual_first_hit_tp{tp}_sl{sl}"
        pred_cols = [f"p_tp{tp}_before_sl{sl}", f"p_sl{sl}_before_tp{tp}",
                     f"p_neither_tp{tp}_sl{sl}", f"p_ambiguous_tp{tp}_sl{sl}"]
        if not all(c in pred_df for c in [actual_col, *pred_cols]):
            continue
        p = pred_df[pred_cols].to_numpy(float)
        ycat = pred_df[actual_col].to_numpy(int)
        onehot = np.eye(4)[ycat]
        score = float(np.mean(np.sum((p - onehot) ** 2, axis=1)))
        metrics[f"multiclass_brier_tp{tp}_sl{sl}"] = score
        first_scores.append(score)
    metrics["mean_first_hit_multiclass_brier"] = float(np.mean(first_scores)) if first_scores else math.nan
    return metrics


def event_rates_from_rows(df: pd.DataFrame, rows: np.ndarray) -> Dict[str, float]:
    """Estimate event prevalences using only a chronological historical segment."""
    if len(rows) == 0:
        return {}
    hi = df["HIGH"].to_numpy(float)
    lo = df["LOW"].to_numpy(float)
    cl = df["CLOSE"].to_numpy(float)
    offsets = np.arange(1, HORIZON + 1, dtype=np.int64)
    anchor = cl[rows]
    fut_hi = hi[rows[:, None] + offsets[None, :]]
    fut_lo = lo[rows[:, None] + offsets[None, :]]
    up_exc = (np.max(fut_hi, axis=1) - anchor) / PIP_SIZE
    down_exc = (np.min(fut_lo, axis=1) - anchor) / PIP_SIZE
    rates: Dict[str, float] = {}
    for a in TARGETS_PIPS:
        rates[f"actual_hit_up_{a}pip"] = float(np.mean(up_exc >= a - 1e-8))
        rates[f"actual_hit_down_{a}pip"] = float(np.mean(down_exc <= -a + 1e-8))
    return rates


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, help="Path to M1 OHLC CSV/TSV file")
    ap.add_argument("--out-dir", default="knn_excursion_output", help="Output directory")
    ap.add_argument("--pool-size", type=int, default=INITIAL_POOL, help="Initial nearest-neighbour candidate pool")
    ap.add_argument("--test-step", type=int, default=5, help="Evaluate every N bars; 5 avoids overlapping five-bar test paths")
    args = ap.parse_args()
    if args.pool_size < 100:
        raise ValueError("pool-size should be at least 100 for this design")
    if args.test_step < HORIZON:
        print("Warning: test-step < 5 means test target windows overlap; metrics are more dependent.", file=sys.stderr)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_ohlc(args.data)
    sample_rows, X, scale, _ = build_features(df)
    n = len(df)
    row_to_sample = np.full(n, -1, dtype=np.int64)
    row_to_sample[sample_rows] = np.arange(len(sample_rows))

    train_cut = int(n * 0.70)
    val_cut = int(n * 0.85)
    valid_start = int(sample_rows[0])
    val_rows = np.arange(max(train_cut, valid_start), max(train_cut, val_cut - HORIZON), args.test_step, dtype=np.int64)
    test_rows = np.arange(max(val_cut, valid_start), n - HORIZON, args.test_step, dtype=np.int64)
    # only rows with feature vectors are eligible
    val_rows = val_rows[row_to_sample[val_rows] >= 0]
    test_rows = test_rows[row_to_sample[test_rows] >= 0]
    if len(val_rows) < 100 or len(test_rows) < 100:
        raise ValueError(f"Not enough validation/test predictions: validation={len(val_rows)}, test={len(test_rows)}")

    validation_training_rows = sample_rows[sample_rows + HORIZON < train_cut]
    test_training_rows = sample_rows[sample_rows + HORIZON < val_cut]
    validation_train_rates = event_rates_from_rows(df, validation_training_rows)
    test_train_rates = event_rates_from_rows(df, test_training_rows)

    split_info = {
        "rows": int(n),
        "valid_feature_samples": int(len(sample_rows)),
        "train_cut_row": int(train_cut),
        "validation_cut_row": int(val_cut),
        "validation_predictions": int(len(val_rows)),
        "test_predictions": int(len(test_rows)),
        "first_datetime": df["DATETIME"].min().isoformat(),
        "last_datetime": df["DATETIME"].max().isoformat(),
        "invalid_rows_removed": int(df.attrs.get("invalid_rows_removed", 0)),
        "input_feature_dimensions": FEATURE_DIM,
        "future_path_bars": HORIZON,
        "test_step_bars": int(args.test_step),
        "initial_candidate_pool": int(args.pool_size),
        "decluster_gap_bars": DECLUSTER_GAP_BARS,
        "volatility_ratio_range": [1.0 / VOL_RATIO_LIMIT, VOL_RATIO_LIMIT],
        "adaptive_radius_min_neighbors": MIN_ADAPTIVE_NEIGHBORS,
        "adaptive_radius_max_neighbors": MAX_FINAL_NEIGHBORS,
        "adaptive_radius_gap_ratio": ELBOW_RATIO_THRESHOLD,
        "target_pips": TARGETS_PIPS,
        "first_hit_target_stop_pairs": [list(x) for x in FIRST_HIT_PAIRS],
        "note": "Pip-level OHLC barrier estimates are gross; observed spread/commission/slippage are not deducted. Same-candle dual barrier touches are marked ambiguous.",
    }

    print(f"Loaded {n:,} OHLC rows from {split_info['first_datetime']} to {split_info['last_datetime']}.")
    print(f"Validation forecasts: {len(val_rows):,}; final test forecasts: {len(test_rows):,}.")
    modes = ["baseline", "filtered_adaptive"]
    all_metrics = {}
    all_predictions = {}

    # Validation forecasts use only data strictly before the validation split,
    # including complete five-candle outcomes.
    for mode in modes:
        print(f"Running validation: {mode} ...", flush=True)
        val_pred = run_predictions(df, sample_rows, X, scale, row_to_sample, val_rows, train_cut, mode, args.pool_size)
        val_metrics = brier_metrics(val_pred, validation_train_rates)
        all_metrics[f"validation_{mode}"] = val_metrics
        val_pred.to_csv(out_dir / f"validation_predictions_{mode}.csv", index=False)
        print(f"  {mode}: mean excursion Brier={val_metrics.get('mean_excursion_brier', float('nan')):.6f}; "
              f"mean first-hit Brier={val_metrics.get('mean_first_hit_multiclass_brier', float('nan')):.6f}")

    # Final test forecasts use all earlier examples whose future labels are
    # complete before the test period; parameters are fixed in advance.
    for mode in modes:
        print(f"Running final test: {mode} ...", flush=True)
        test_pred = run_predictions(df, sample_rows, X, scale, row_to_sample, test_rows, val_cut, mode, args.pool_size)
        test_metrics = brier_metrics(test_pred, test_train_rates)
        all_metrics[f"test_{mode}"] = test_metrics
        all_predictions[mode] = test_pred
        test_pred.to_csv(out_dir / f"test_predictions_{mode}.csv", index=False)
        print(f"  {mode}: mean excursion Brier={test_metrics.get('mean_excursion_brier', float('nan')):.6f}; "
              f"mean first-hit Brier={test_metrics.get('mean_first_hit_multiclass_brier', float('nan')):.6f}")

    # Save a concise per-mode summary with the recommended model clearly named.
    summary = {
        "data": split_info,
        "metrics": all_metrics,
        "interpretation": {
            "mean_excursion_brier": "Average binary Brier score across upward/downward hit events for 1-10 pip thresholds; lower is better.",
            "mean_climatology_brier": "Brier score for a baseline that predicts historical training event rates.",
            "mean_brier_skill": "1 - model mean Brier / climatology mean Brier; positive means better than that baseline on these events.",
            "mean_first_hit_multiclass_brier": "Average multiclass Brier score over target-first, stop-first, neither, and ambiguous outcomes for the defined TP/SL pairs; lower is better.",
            "probabilities": "All probabilities are empirical, weighted analog estimates and require calibration on additional unseen data before live use.",
        },
    }
    with open(out_dir / "summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, allow_nan=False, default=lambda x: None if isinstance(x, float) and not math.isfinite(x) else x)

    # Save a user-facing sample of the last 20 test predictions for quick inspection.
    rec_file = out_dir / "latest_test_prediction_sample.csv"
    all_predictions["filtered_adaptive"].tail(20).to_csv(rec_file, index=False)
    with open(out_dir / "README.txt", "w", encoding="utf-8") as f:
        f.write("KNN PRICE-EXCURSION MODEL\n")
        f.write("========================\n\n")
        f.write("Primary model: filtered_adaptive.\n")
        f.write("The input is exactly five consecutive M1 open-to-close log returns, scaled by a 60-bar RMS close-to-close volatility estimate.\n")
        f.write("A candidate pool is retrieved by Euclidean distance, screened by volatility ratio, temporally de-clustered, then reduced by an adaptive distance-gap radius. Surviving neighbours are weighted with a compact-support quartic kernel.\n\n")
        f.write("Outputs include probabilities of touching +1..+10 pip and -1..-10 pip excursions in the next five candles, the five-minute ending-return direction, and first-hit probabilities for upper/lower TP-SL barrier pairs.\n\n")
        f.write("For TP/SL paths, the output categories are target first, stop first, neither reached within five bars, and ambiguous same-candle dual touches. One-minute OHLC cannot recover the order when both barriers are inside one candle's high-low range and the open is between them; those cases are retained as ambiguous rather than guessed.\n\n")
        f.write("The final 15% of the timeline is an untouched chronological test period. Test forecasts are spaced at least five bars apart. Baseline and filtered models are both reported. Brier scores compare probability forecasts with realised binary/multiclass outcomes.\n\n")
        f.write("Important limitations: this dataset's timestamp coverage is whatever is reported in summary_metrics.json; the filename alone is not treated as proof of coverage. Results are gross of trading costs. OHLC barrier hits are not identical to executable bid/ask fills. Do not deploy live without more out-of-sample periods, realistic cost modelling, robustness checks and paper trading.\n")

    print(f"Results written to: {out_dir}")
    print(f"Summary: {out_dir / 'summary_metrics.json'}")


if __name__ == "__main__":
    main()