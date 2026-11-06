# Stage 1 — Reproducibility and Test Verification

Updated: 2026-11-06

## Test and syntax results

- Command: `python -m pytest -q`
- Result: **8 passed** (1.35 seconds on the final run).
- The eight tests cover sigmoid and isotonic calibration bounds/monotonicity, nested-threshold projection, multiclass temperature scaling and simplex preservation, positive-temperature validation, binary/multiclass metrics, and a regression test that changes test labels and confirms calibrated probabilities do not change.
- Command: `python -m py_compile knn_price_excursion.py knn_excursion_calibration.py`
- Result: **passed**.

## Environment recorded

- Python: 3.13.5
- NumPy: 2.3.5
- pandas: 2.2.3
- scikit-learn: 1.8.0
- SciPy: 1.17.0
- Matplotlib: 3.10.8
- pytest: 9.0.2
- Platform: Linux x86_64, glibc 2.41

This is a successful test run in the available execution environment, not yet a clean-room installation test. `pip check` also reports an unrelated environment conflict: installed moviepy 2.2.1 requires Pillow <12, while Pillow 12.3.0 is installed. The KNN test suite itself passes.

## Reproducibility status

The source and test suite are syntactically valid, and the calibration unit/regression suite passes. Full end-to-end reproduction is **blocked on source artifacts**:

1. The original EUR/USD M1 OHLC input file is not currently available in the accessible project files.
2. The validation prediction CSV was not located.
3. A test prediction CSV is present in the Library, but without the matching validation predictions and exact original input, it is insufficient to regenerate and independently verify the complete pipeline and reported metrics.

The prior processed-data description records 98,851 valid rows from 2026-03-10 through 2026-06-15. That description is historical metadata, not a substitute for the original file.

## Next required inputs/actions

- Recover the exact original OHLC file, including broker/source, timezone, and export conventions.
- Recover the matching `validation_predictions_filtered_adaptive.csv` and, if available, the baseline prediction files.
- Once present, calculate SHA-256 fingerprints, inspect timestamp/OHLC integrity and gaps, record exact CLI arguments and environment, then rerun model and calibration outputs.

No model parameters or trading rules were changed in Stage 1. Passing these tests is software verification, not evidence of a profitable trading edge.
