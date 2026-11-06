# Project Synchronization Status

Updated: 2026-11-06

## Restored to GitHub

The two implementation modules have been restored from the validated project files in the user's Library:

- `knn_price_excursion.py` — five-dimensional KNN price-excursion model and chronological evaluation.
- `knn_excursion_calibration.py` — binary/multiclass calibration, monotonic projection, reliability diagnostics, and reports.

The repository also contains the README, mathematical/calibration documentation, experiment notes, dependency manifest, and calibration tests.

## Stage 1 verification — completed test portion

- `python -m py_compile knn_price_excursion.py knn_excursion_calibration.py`: passed.
- `python -m pytest -q`: **8 passed**.
- Environment and limitations are recorded in `STAGE1_VERIFICATION.md`.
- This was not a clean-room installation test. `pip check` reported an unrelated moviepy/Pillow version conflict in the shared environment.

## Still required for full end-to-end reproducibility

The original EUR/USD M1 OHLC input and matching validation prediction CSV were not located in accessible project files. A test prediction CSV exists in the Library, but this is not enough to reproduce the entire model/calibration pipeline.

Previously documented dataset metadata says 98,851 valid rows spanning 2026-03-10 through 2026-06-15. Verify this against the actual file; do not treat the filename as proof of a longer history.

Next:
1. Recover the exact source OHLC file and matching validation predictions.
2. Record SHA-256 fingerprints, feed/timezone details, data-quality checks, exact configuration, and package environment.
3. Re-run the model and calibration pipeline and compare generated metrics with archived results.
4. Only after reproducibility, expand rolling-origin validation and benchmark comparisons.

Generated prediction files remain outside Git unless specifically needed and size/licensing have been checked. Raw market data should not be committed without provenance/license review.

Passing unit tests verifies software behavior covered by those tests; it does not prove predictive edge or profitability.
