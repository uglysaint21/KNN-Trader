# KNN Trader — EUR/USD M1 Price-Excursion Probability Model

Research project implementing a five-dimensional k-nearest-neighbours (KNN) model for EUR/USD price-excursion probabilities, first-hit outcomes, chronological evaluation, and post-hoc probability calibration.

> **Research only — not financial advice.** Probability improvements do not establish profitability. Validate independently before risking capital.

## Project concept

At each completed M1 candle, construct a vector of the last five open-to-close log returns, find historical analogues, filter candidates by volatility regime and temporal proximity, and use weighted neighbours to estimate probabilities of hitting price barriers over the next five observed candles.

The model estimates upward/downward barrier-touch probabilities from 1 to 10 pips, ending-close direction/return, first-hit probabilities for eight TP/SL pairs, and raw/calibrated forecasts with reliability diagnostics.

## Repository contents

- `knn_price_excursion.py` — main KNN model and chronological evaluation.
- `knn_excursion_calibration.py` — binary and multiclass calibration pipeline.
- `tests/test_knn_excursion_calibration.py` — calibration and leakage-regression tests.
- `requirements.txt` — Python dependencies.
- [Mathematical Model](docs/MATHEMATICAL_MODEL.md)
- [Calibration Study](docs/CALIBRATION_STUDY.md)
- [Experiment Log](docs/EXPERIMENT_LOG.md)
- [Synchronization Status](PROJECT_SYNC_STATUS.md)

## Setup

Python 3.10+ is recommended.

```bash
python -m venv .venv
# Windows:
.venv\\Scripts\\activate
# macOS/Linux:
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m pytest -q tests
```

## Run

```bash
python knn_price_excursion.py --data EURUSD_M1_1JAN24_16JUN26.csv --out-dir results
python knn_excursion_calibration.py --validation results/validation_predictions_filtered_adaptive.csv --test results/test_predictions_filtered_adaptive.csv --out-dir results/calibrated
```

The input must contain timestamp and OHLC columns. The script accepts common MetaTrader tab-delimited exports and CSV equivalents.

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

The processed CSV contained **98,851 valid rows from 2026-03-10 through 2026-06-15**, although its filename suggested a longer date range. Verify timestamp coverage, timezone, missing minutes, weekend gaps, and broker feed before reproducing. The horizon means the next five observed rows, not necessarily five wall-clock minutes when there are gaps.

## Reproducibility safeguards

Use chronological train/validation/test splits; exclude neighbours whose future label horizon has not completed; fit calibration only on validation forecasts; keep final test periods untouched; and include spread, commissions, slippage, latency, and conservative handling of ambiguous OHLC barrier touches in any trading backtest.

See [PROJECT_SYNC_STATUS.md](PROJECT_SYNC_STATUS.md) for the remaining reproducibility work and verification status.
