# Experiment Log

## 2026-10-26 — Initial KNN excursion model

- Five consecutive M1 open-to-close log returns form the feature vector.
- Forecast horizon: next five observed M1 candles.
- Labels: up/down barrier touches from 1 to 10 pips, ending-close direction/return, and first-hit outcomes for eight TP/SL pairs.
- Neighbour filtering: large initial pool, volatility-ratio guard, temporal de-clustering, adaptive neighbour count, and distance/kernel weights.
- Evaluation: chronological train/validation/test split, with forecast points sampled every five bars.
- Initial run reported that filtered/adaptive neighbours improved aggregate excursion Brier score and first-hit multiclass Brier score versus the baseline.
- Data caveat: 98,851 valid rows from 2026-03-10 to 2026-06-15.

## 2026-10-27 — Calibration iteration

- Compared sigmoid and isotonic methods for binary event forecasts.
- Evaluated temperature scaling for first-hit multiclass distributions.
- Used chronological validation fit/selection segments; held-out test labels were reserved for scoring.
- Added per-event metrics and reliability-bin diagnostics.
- Reported aggregate metrics improved: binary Brier 0.07733 → 0.07659; binary log loss 0.25035 → 0.24614; ECE 0.02030 → 0.00964; first-hit multiclass Brier 0.33303 → 0.33075; first-hit log loss 0.60797 → 0.60411.
- Reported checks: 8 tests passed, Python compilation succeeded, and project ZIP integrity passed.

## Backlog

### Data integrity
- Verify timestamps, timezone, duplicates, missing minute intervals, weekend gaps, and bid/ask conventions.
- Explicitly handle gaps so the model does not silently equate five rows with five clock minutes.
- Save input fingerprint, config, dependency versions, date boundaries, and source commit with each run.

### Model evaluation
- Repeat on longer history and multiple non-overlapping periods.
- Use rolling-origin evaluation and block-bootstrap confidence intervals.
- Compare unconditional event rates, volatility-only models, logistic regression, and simple conditional-frequency baselines.
- Report performance and calibration by volatility/session/regime.

### Calibration
- Track calibration drift with past-only rolling refits.
- Evaluate multiple binning schemes and proper scores.
- Consider beta calibration and minimum class-count safeguards.
- Keep raw and calibrated predictions side by side.

### Trading-policy study (separate from forecast model)
- Define executable entry and exit conventions.
- Include spread, commissions, slippage, latency, missed fills, and ambiguous OHLC-bar treatment.
- Select probability thresholds on validation only.
- Report net expectancy, drawdown, exposure, turnover, risk of ruin, and multiple-testing risk.

## Reproducibility

Never overwrite prior results. Save each experiment in a new folder or tagged commit and record exact source commit, data fingerprint, dependencies, configuration, split dates, and command line.
