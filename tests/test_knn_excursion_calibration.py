import numpy as np
import pytest

from knn_excursion_calibration import (
    calibrate_binary_probability,
    enforce_threshold_monotonicity,
    temperature_scale_multiclass,
    binary_metrics,
    multiclass_metrics,
)


def test_sigmoid_calibration_returns_finite_probabilities_in_unit_interval():
    p = np.array([0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.9])
    y = np.array([0, 0, 0, 0, 1, 1, 1])
    model = calibrate_binary_probability(p, y, method="sigmoid")
    out = model.predict(p)
    assert np.isfinite(out).all()
    assert ((out >= 0) & (out <= 1)).all()
    assert np.all(np.diff(out) >= -1e-12)


def test_isotonic_calibration_handles_out_of_range_inputs_by_clipping():
    p = np.linspace(0.01, 0.99, 200)
    y = (p > 0.65).astype(int)
    model = calibrate_binary_probability(p, y, method="isotonic")
    out = model.predict(np.array([-0.2, 0.1, 0.9, 1.2]))
    assert np.isfinite(out).all()
    assert ((out >= 0) & (out <= 1)).all()
    assert np.all(np.diff(out) >= -1e-12)


def test_threshold_projection_makes_nested_hit_probabilities_monotone():
    # Columns are ordered from the nearest to furthest pip threshold.
    p = np.array([[0.7, 0.8, 0.2], [0.6, 0.4, 0.5]])
    out = enforce_threshold_monotonicity(p)
    assert np.all(np.diff(out, axis=1) <= 1e-12)
    assert ((out >= 0) & (out <= 1)).all()
    assert np.allclose(out[0], [0.7, 0.7, 0.2])
    assert np.allclose(out[1], [0.6, 0.4, 0.4])


def test_temperature_scaling_preserves_multiclass_simplex():
    p = np.array([[0.7, 0.1, 0.15, 0.05], [0.1, 0.6, 0.2, 0.1], [0.2, 0.2, 0.5, 0.1]])
    out = temperature_scale_multiclass(p, temperature=1.4)
    assert np.all(np.isfinite(out))
    assert np.all(out >= 0)
    assert np.allclose(out.sum(axis=1), 1.0, atol=1e-10)


def test_temperature_must_be_positive():
    with pytest.raises(ValueError):
        temperature_scale_multiclass(np.array([[0.5, 0.3, 0.1, 0.1]]), temperature=0)


def test_binary_metrics_include_brier_logloss_and_calibration_error():
    got = binary_metrics(np.array([0.1, 0.8, 0.7, 0.2]), np.array([0, 1, 1, 0]))
    assert got["brier"] < 0.1
    assert got["log_loss"] >= 0
    assert got["ece_10"] >= 0


def test_multiclass_metrics_accept_normalized_probabilities():
    p = np.array([[0.8, 0.1, 0.05, 0.05], [0.1, 0.7, 0.1, 0.1]])
    y = np.array([0, 1])
    got = multiclass_metrics(p, y)
    assert got["brier"] >= 0
    assert got["log_loss"] >= 0
    assert np.allclose(got["row_sum_min"], 1.0)
    assert np.allclose(got["row_sum_max"], 1.0)


def _make_synthetic_predictions(n, seed, include_actual=True):
    import pandas as pd
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"row_index": np.arange(n), "datetime": pd.date_range("2025-01-01", periods=n, freq="5min").astype(str), "model": "filtered_adaptive"})
    p_end_up = rng.beta(2.0, 2.0, size=n)
    df["p_end_close_up"] = p_end_up
    if include_actual:
        df["actual_end_close_up"] = rng.binomial(1, p_end_up * 0.8 + 0.1)
    for direction in ("up", "down"):
        base = rng.beta(2, 8, size=n)
        for pip in range(1, 11):
            p = np.clip(base * (1 - (pip - 1) * 0.055) + rng.normal(0, 0.01, n), 0.001, 0.95)
            df[f"p_hit_{direction}_{pip}pip"] = p
            if include_actual:
                df[f"actual_hit_{direction}_{pip}pip"] = rng.binomial(1, p * 0.75 + 0.04)
    for tp, sl in [(2, 2), (3, 2), (3, 3), (4, 3), (4, 4), (5, 3), (5, 4), (5, 5)]:
        ps = rng.dirichlet([2.5, 2.5, 5.0, 0.25], size=n)
        cols = [f"p_tp{tp}_before_sl{sl}", f"p_sl{sl}_before_tp{tp}", f"p_neither_tp{tp}_sl{sl}", f"p_ambiguous_tp{tp}_sl{sl}"]
        for j, col in enumerate(cols):
            df[col] = ps[:, j]
        if include_actual:
            df[f"actual_first_hit_tp{tp}_sl{sl}"] = [rng.choice(4, p=row) for row in ps]
    return df


def test_calibration_pipeline_does_not_use_test_labels_to_change_probabilities(tmp_path):
    import pandas as pd
    from knn_excursion_calibration import calibrate_files

    val = _make_synthetic_predictions(240, seed=11)
    test = _make_synthetic_predictions(140, seed=22)
    val_path, test_path = tmp_path / "val.csv", tmp_path / "test.csv"
    val.to_csv(val_path, index=False)
    test.to_csv(test_path, index=False)
    calibrate_files(val_path, test_path, tmp_path / "out_a")

    altered = test.copy()
    for c in altered.columns:
        if c.startswith("actual_hit_") or c == "actual_end_close_up":
            altered[c] = 1 - altered[c].astype(int)
        elif c.startswith("actual_first_hit_"):
            altered[c] = (altered[c].astype(int) + 1) % 4
    altered_path = tmp_path / "test_altered.csv"
    altered.to_csv(altered_path, index=False)
    calibrate_files(val_path, altered_path, tmp_path / "out_b")

    a = pd.read_csv(tmp_path / "out_a" / "test_predictions_filtered_adaptive_calibrated.csv")
    b = pd.read_csv(tmp_path / "out_b" / "test_predictions_filtered_adaptive_calibrated.csv")
    prob_cols = [c for c in a.columns if c.endswith("_calibrated") or c.endswith("_calibrated_raw")]
    assert prob_cols
    assert "p_end_close_up_calibrated" in a.columns
    assert np.allclose(a[prob_cols].to_numpy(float), b[prob_cols].to_numpy(float))
    for direction in ("up", "down"):
        cols = [f"p_hit_{direction}_{i}pip_calibrated" for i in range(1, 11)]
        assert np.all(np.diff(a[cols].to_numpy(float), axis=1) <= 1e-12)
    for tp, sl in [(2, 2), (3, 2), (3, 3), (4, 3), (4, 4), (5, 3), (5, 4), (5, 5)]:
        cols = [f"p_tp{tp}_before_sl{sl}_calibrated", f"p_sl{sl}_before_tp{tp}_calibrated",
                f"p_neither_tp{tp}_sl{sl}_calibrated", f"p_ambiguous_tp{tp}_sl{sl}_calibrated"]
        assert np.allclose(a[cols].sum(axis=1), 1.0, atol=1e-10)
