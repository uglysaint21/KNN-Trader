# KNN Price-Excursion Model — Probability Calibration

## Calibration protocol

The validation file contained 2,965 forecasts. The first 1,779 were used to fit calibrators, the later 1,186 to select calibration methods, and the final 2,965 test forecasts were evaluated once.

The five-dimensional KNN, neighbour screening, volatility filter, temporal de-clustering, adaptive neighbour count, OHLC event definitions, and ambiguous first-hit category are unchanged.

## Binary excursion probabilities — untouched test set

| Metric | Raw | Calibrated | Change |
|---|---:|---:|---:|
| Mean Brier | 0.077330 | 0.076594 | -0.000737 |
| Mean log loss | 0.250351 | 0.246143 | -0.004207 |
| Mean ECE (10 bins) | 0.020303 | 0.009639 | -0.010664 |

**Selected binary method:** sigmoid.

## Binary method comparison on later validation

| Method | Mean Brier | Mean log loss | Mean ECE |
|---|---:|---:|---:|
| Raw | 0.071194 | 0.229089 | 0.030318 |
| Sigmoid | 0.069629 | 0.223884 | 0.011182 |
| Isotonic | 0.069740 | 0.228903 | 0.010223 |

## First-hit probabilities

Across eight TP/SL pairs, mean multiclass Brier was 0.333029 raw versus 0.330749 calibrated; mean log loss was 0.607971 raw versus 0.604114 calibrated. Temperature scaling is retained only when validation metrics improve under the selection rule. Four outcomes are preserved: target first, stop first, neither, and ambiguous same-candle dual touch.

## Limitations

The input contains 98,851 valid rows from 2026-03-10 through 2026-06-15. Validation calibration can drift between validation and test. Results are gross of spread, commission, slippage and execution latency; this is not evidence of a profitable strategy.
