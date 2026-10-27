# KNN Trader — EUR/USD M1 Price-Excursion Probability Model

Research software for estimating EUR/USD price-excursion probabilities from a five-dimensional K-nearest-neighbours (KNN) representation of the last five completed one-minute candles. The research includes volatility-regime filtering, temporal de-clustering, first-hit outcomes, chronological validation, and post-hoc probability calibration.

> **Research only — not financial advice.** Probability improvements do not establish profitability. Do not use with live capital without independent reproduction, robust out-of-sample testing, and realistic execution costs.

## Project concept

At each completed M1 candle, construct the vector of the last five open-to-close log returns, find historical analogues, filter candidates by volatility regime and temporal proximity, and use weighted neighbours to estimate the probability of hitting price barriers over the next five observed candles.

The system reports:
- Probabilities of upward and downward barrier touches from 1 to 10 pips.
- Ending-close direction and return estimates.
- First-hit probabilities for target-first, stop-first, neither, and ambiguous same-candle touches.
- Raw and calibrated probabilities, per-event scoring metrics, and reliability-bin diagnostics.

## Reported calibration results

| Metric | Raw | Calibrated |
|---|---:|---:|
| Mean binary Brier score | 0.07733 | 0.07659 |
| Mean binary log loss | 0.25035 | 0.24614 |
| Mean ECE (10 bins) | 0.02030 | 0.00964 |
| Mean first-hit multiclass Brier | 0.33303 | 0.33075 |
| Mean first-hit log loss | 0.60797 | 0.60411 |

Lower is better. Binary Brier improvement is modest (~0.95% relative), while mean ECE fell by ~52.5%. These scores assess probability quality, not trading profitability.

## Data caveat

Although the input filename suggested coverage from January 2024, the processed CSV contained **98,851 valid rows from 2026-03-10 through 2026-06-15**. Verify actual timestamp coverage, timezone, missing minutes, and broker feed before reproducing. The forecast horizon means the next five observed rows, which is not necessarily five wall-clock minutes when there are gaps.

## Mathematical specification

See [Mathematical Model](docs/MATHEMATICAL_MODEL.md), [Calibration Study](docs/CALIBRATION_STUDY.md), and [Experiment Log](docs/EXPERIMENT_LOG.md).

For candle i, (r_i=\ln(C_i/O_i)). The feature vector is
(x_t=[r_{t-4},r_{t-3},r_{t-2},r_{t-1},r_t]\in\mathbb{R}^5).
Historical labels use the next five candles' high/low path and are only eligible once that full horizon has completed.

## Reproducibility protocol

1. Use chronological train/validation/test splits; never shuffle time-series rows.
2. Exclude neighbours whose future label horizon overlaps the forecast timestamp.
3. Fit and select calibration only on validation forecasts; freeze before final test.
4. Report Brier score, log loss, calibration/reliability bins, and first-hit multiclass scores.
5. Include spread, commissions, slippage, latency, missed fills, and conservative handling of ambiguous OHLC barrier touches in any trading backtest.
6. Keep final test periods untouched while iterating on model design.

## Environment

Python 3.10+ recommended. Install dependencies from `requirements.txt` and run tests with `python -m pytest -q`. Check the scripts' `--help` for the exact command-line interface. Store raw market data locally; do not commit private account data or credentials.

## Current status

The working project reported eight passing calibration tests, successful Python compilation, and ZIP integrity verification on 2026-10-27. The original local package included model/calibration scripts, tests, reports, metrics, and prediction exports. Each result should be re-run against the exact input data and recorded with a source commit, data fingerprint, dependency versions, split dates, and configuration before being treated as reproducible.
