# Project Synchronization Status

Updated: 2026-10-27

## Uploaded to GitHub

- Project README and setup notes.
- Mathematical model specification.
- Calibration methodology and experiment log.
- Main model and calibration result reports.
- Calibration test suite.
- Dependency manifest, ignore rules, and license.
- This synchronization status record.

## Still not uploaded from the validated local ZIP

The implementation files remain missing from this repository:
- `knn_price_excursion.py` — main KNN model and chronological evaluation.
- `knn_excursion_calibration.py` — binary and multiclass calibration pipeline.

Generated output files also remain missing:
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

These were present in the validated local ZIP at `/mnt/data/knn_price_excursion_project.zip`, but the available GitHub connector supports creating repository text blobs rather than transferring files directly from the local filesystem. The prediction CSVs are several megabytes each, so they have not been copied into this repository in this pass.

## Important

This repository is **not yet a fully runnable source backup** because both Python implementation files are still absent. The test suite has been committed, but cannot run from the repository until the calibration script is synchronized. Do not treat the documentation-only model description as a replacement for the implementation.

Raw OHLC market data should remain outside Git unless its provenance/license and repository-size implications have been checked. Preserve a data fingerprint and exact configuration for reproducibility.
