# Project Synchronization Status

Updated: 2026-11-06

## Restored to GitHub

The two missing implementation modules have now been restored from the validated project files in the user's Library:

- `knn_price_excursion.py` — five-dimensional KNN price-excursion model, chronological evaluation, neighbour filtering, weighted predictions, and metrics.
- `knn_excursion_calibration.py` — binary sigmoid/isotonic calibration, multiclass temperature scaling, monotonic projection, reliability diagnostics, and reports.

The repository also contains:

- Project README and setup notes.
- Mathematical model specification.
- Calibration methodology and experiment log.
- Main model and calibration result reports.
- Calibration test suite (`tests/test_knn_excursion_calibration.py`).
- Dependency manifest (`requirements.txt`), ignore rules, and license.

## Verification performed in this synchronization pass

- Both restored Python modules passed `python -m py_compile`.
- The existing test suite and dependency manifest were confirmed present in GitHub.
- The full pytest suite was **not executed** in this pass; do not interpret syntax compilation as a passing test suite.

## Still not uploaded

Generated output files remain outside the repository:

- `results/summary_metrics.json`
- `results/validation_predictions_baseline.csv`
- `results/validation_predictions_filtered_adaptive.csv`
- `results/test_predictions_baseline.csv`
- `results/test_predictions_filtered_adaptive.csv`
- `results/latest_test_prediction_sample.csv`
- `results/calibrated/calibration_summary.json`
- `results/calibrated/calibration_binary_test_metrics.csv`
- `results/calibrated/calibration_reliability_bins.csv`
- `results/calibrated/test_predictions_filtered_adaptive_calibrated.csv`

The prediction CSVs are several megabytes each. Raw OHLC market data should remain outside Git unless its provenance/license and repository-size implications have been checked. Preserve a data fingerprint and exact configuration for reproducibility.

## Research limitations and next steps

The processed input previously used contained 98,851 valid rows from 2026-03-10 through 2026-06-15, despite a filename suggesting a longer period. Verify timestamp coverage, timezone, missing minutes, weekend gaps, duplicates, and bid/ask conventions before reproducing.

Recommended next work:
1. Run the complete pytest suite in a clean environment using `requirements.txt`.
2. Reproduce the model and calibration outputs from the original OHLC input, recording its fingerprint and exact environment/configuration.
3. Expand chronological rolling-origin evaluation across multiple non-overlapping periods and use block-bootstrap uncertainty intervals.
4. Compare against unconditional event-rate, volatility-only, and logistic baselines.
5. Only after forecast robustness is established, evaluate a separate executable trading policy with spread, commissions, slippage, latency, missed fills, conservative ambiguous-bar treatment, drawdown, and risk-of-ruin analysis.

This repository now contains the implementation source, but it is not yet a fully reproducible experiment bundle because the original data and generated prediction outputs are not committed.
