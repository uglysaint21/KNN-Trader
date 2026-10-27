# KNN Price-Excursion Model — Prototype Results

## What was implemented

The primary `filtered_adaptive` model uses five M1 open-to-close log returns as its feature vector, normalized by a 60-bar RMS close-to-close volatility estimate. It retrieves up to 1,000 historical analogues, filters candidates whose historical/current volatility ratio falls outside [0.5, 2.0], de-clusters nearby candidates, uses an adaptive distance-gap rule to retain up to 120 neighbours, and applies compact-support distance weights.

For each query it estimates upward/downward +1 to +10 pip barrier touches over the next five candles, ending-close direction and return, and first-hit outcomes for eight TP/SL combinations. The filtered model is compared with a distance-weighted KNN baseline.

## Data coverage caveat

The input filename suggested January 2024 to June 2026, but the processed data contained **98,851 valid rows from 10 March 2026 01:20 through 15 June 2026 23:59**. It does not cover the full date range implied by the filename.

## Chronological evaluation

- First 70% of rows: historical candidate pool.
- Next 15%: validation, with 2,965 forecasts sampled every five bars.
- Final 15%: held-out test, with 2,965 forecasts sampled every five bars.
- Test neighbours were restricted to historical examples whose five-bar outcome was complete before the test period began.

## Held-out test results

| Metric | Distance-weighted baseline | Filtered adaptive KNN |
|---|---:|---:|
| Mean Brier across 20 excursion events | 0.07500 | **0.06844** |
| Historical-rate baseline Brier | 0.07325 | 0.07325 |
| Brier skill vs historical-rate baseline | -2.38% | **+6.57%** |
| First-hit multiclass Brier | 0.38127 | **0.33303** |

These results are sample-specific evidence worth investigating, not proof of a durable trading edge.

## Example: 3-pip target versus 3-pip stop

| Outcome | Mean probability | Realized test frequency |
|---|---:|---:|
| Target first | 15.37% | 13.19% |
| Stop first | 16.26% | 14.27% |
| Neither within five bars | 68.27% | 72.51% |
| Ambiguous same-candle order | 0.10% | 0.03% |

## Important limits

OHLC touches are gross of spread, commission, slippage and execution uncertainty. If one M1 candle spans both target and stop while opening between them, OHLC cannot determine order; those cases are marked ambiguous. The history spans about three months, so more periods and regimes must be tested before drawing strategy conclusions.
