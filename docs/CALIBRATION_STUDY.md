# Probability Calibration Study

## Goal

KNN produces weighted frequencies from a selected neighbourhood. Those scores may be overconfident or underconfident. Calibration maps raw scores to probabilities whose empirical frequencies more closely match their predictions without changing the underlying feature definition, neighbour retrieval, or future labels.

## Methods

### Sigmoid (Platt-style) binary calibration

For raw probability p, clip away from 0 and 1, transform to log odds, and fit a logistic map:

\[
p_{cal}=\frac{1}{1+\exp[-(a\operatorname{logit}(p)+b)]}.
\]

This low-parameter method can correct systematic confidence bias and may be stable on moderate samples.

### Isotonic regression

Isotonic regression fits a non-decreasing mapping between raw scores and observed event frequencies. It is more flexible but can overfit with sparse positive/negative examples. The project compares it using a separate chronological validation-selection segment and uses a practical gain threshold before accepting it.

### Multiclass temperature scaling

For first-hit probabilities p_k,

\[
p_{k,cal}=\frac{\exp(\log(p_k)/T)}{\sum_j\exp(\log(p_j)/T)}.
\]

The positive temperature T is selected on validation log loss. Softmax normalization preserves a valid probability vector.

## Split protocol

1. Generate raw validation forecasts from historical data.
2. Fit candidate calibrators on the first 60% of validation forecasts.
3. Select methods using the later 40% of validation forecasts.
4. Refit selected calibrators on all validation forecasts.
5. Freeze the calibrators and apply them to final held-out test forecasts.
6. Compute test metrics without tuning on the test labels.

## Metrics

- **Brier score:** mean squared probability error; lower is better.
- **Log loss:** strongly penalizes confident wrong predictions; lower is better.
- **Expected calibration error (ECE):** weighted absolute difference between mean predicted probability and observed frequency across bins. It depends on binning and must be read with bin counts.
- **Multiclass Brier/log loss:** assess first-hit class probability vectors.
- **Reliability bins:** show counts, mean predictions, and empirical frequencies.

## Reported test metrics

| Metric | Raw | Calibrated |
|---|---:|---:|
| Mean binary Brier | 0.07733 | 0.07659 |
| Mean binary log loss | 0.25035 | 0.24614 |
| Mean ECE (10 bins) | 0.02030 | 0.00964 |
| Mean first-hit multiclass Brier | 0.33303 | 0.33075 |
| Mean first-hit log loss | 0.60797 | 0.60411 |

The mean binary Brier improvement is approximately 0.95% relative; ECE fell by approximately 52.5%. The Brier gain is modest and should not be oversold. Inspect per-event metrics and reliability-bin sample counts.

## Invariants and tests

- Probabilities remain finite and within [0,1].
- First-hit class probabilities sum to one.
- Probabilities for larger nested barriers cannot exceed those for nearer barriers in the same direction.
- Changing final test labels must not change frozen calibrators or predictions.

The working project reported 8 passing tests, including probability bounds, monotonicity, class normalization, and a no-test-label-leakage check.

## Limitations and next work

- Processed input had 98,851 valid rows from 2026-03-10 through 2026-06-15, despite a filename suggesting a longer period.
- M1 gaps mean five observed rows are not always five clock minutes.
- Forecast horizons/outcomes are dependent; use block-aware uncertainty intervals.
- These scores assess probability quality, not net returns.
- Repeat across multiple non-overlapping date ranges and data feeds.
- Compare unconditional event frequencies and simpler volatility-aware/logistic baselines.
- Any trading policy must include spread, commissions, slippage, latency, fill assumptions, and conservative ambiguous-bar treatment.
