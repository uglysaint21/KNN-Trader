# Project Synchronization Status

Updated: 2026-10-27

## What is currently saved in this GitHub repository

This initial synchronization contains:
- Project overview and reported calibration metrics.
- Mathematical model specification.
- Calibration method and evaluation notes.
- Experiment log and research backlog.
- Dependency manifest, `.gitignore`, and MIT license.

## What remains to be synchronized from the validated local project package

The working package at the time of this note also contained:
- `knn_price_excursion.py` — main KNN model and chronological evaluation.
- `knn_excursion_calibration.py` — binary and multiclass calibration pipeline.
- `tests/test_knn_excursion_calibration.py` — automated tests (reported 8 passing).
- `results/` — original run reports, metrics, validation/test predictions, calibration report, per-event metrics, reliability bins, and calibrated prediction exports.

These implementation and generated-result files have **not** been committed in this initial GitHub synchronization. The project ZIP was available in the ChatGPT working environment as `/mnt/data/knn_price_excursion_project.zip`; that local path is not a public GitHub artifact.

## Before future iterations

Synchronize the exact validated source and tests, then run:

```bash
python -m py_compile knn_price_excursion.py knn_excursion_calibration.py
python -m pytest -q tests
```

Keep raw market data outside version control unless its provenance/license and repository-size implications have been checked. Record a data fingerprint and configuration for reproducibility.
