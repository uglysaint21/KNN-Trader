KNN PRICE-EXCURSION MODEL + PROBABILITY CALIBRATION

Purpose
Estimate probabilities that EUR/USD reaches various upward/downward price changes within the next five M1 candles using a five-dimensional KNN query. The first-hit layer estimates target first, stop first, neither, or ambiguous same-candle double touch.

Agreed model design
1. Five consecutive M1 open-to-close log returns.
2. Normalize by an as-of 60-bar RMS close-to-close volatility scale.
3. Retrieve up to 1,000 candidates by Euclidean distance.
4. Filter historical/current volatility ratios outside [0.5, 2.0].
5. Temporally de-cluster candidate windows.
6. Adaptively choose 30–120 neighbours.
7. Apply compact-support distance weights.
8. Estimate +/-1 to +/-10 pip hit probabilities over the next five candles.
9. Estimate first-hit outcomes for eight TP/SL pairs.

Calibration
The separate calibration pipeline compares sigmoid/Platt and isotonic methods using an early validation fit segment and later validation selection segment, then refits the selected calibrator on all validation data. First-hit multiclass forecasts are tested with temperature scaling. Test labels are not used for fitting or method choice. Nested hit probabilities are projected to be non-increasing with barrier distance, and first-hit classes sum to one.

Install
Python 3.10+ recommended. Install dependencies from requirements.txt.

Run model:
python knn_price_excursion.py --data EURUSD_M1_1JAN24_16JUN26.csv --out-dir results

Calibrate predictions:
python knn_excursion_calibration.py --validation results/validation_predictions_filtered_adaptive.csv --test results/test_predictions_filtered_adaptive.csv --out-dir results/calibrated

Tests:
python -m pytest -q tests

Interpretation and risks
Lower Brier/log loss and ECE are better probability diagnostics, not proof of a profitable edge. OHLC cannot resolve intrabar order when both barriers are touched; such cases are marked ambiguous. Results are gross of spread, commissions, slippage, latency, and executable fill effects. The actual input covered 2026-03-10 through 2026-06-15, with 98,851 valid rows.
