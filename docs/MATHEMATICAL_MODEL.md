# Mathematical Model: KNN Price Excursion Probabilities

## 1. Objective and information set

At time t, after the current one-minute candle is complete, estimate probabilities of price excursions over the next five observed candles. This is a probabilistic forecasting problem, not a deterministic price prediction.

Let each candle have open O, high H, low L, and close C. Features may use only candles completed by t. Labels may use future candles only during offline training/evaluation, and a historical example is eligible for a query only after its complete outcome horizon has elapsed.

## 2. Five-dimensional feature vector

For candle i, define the open-to-close log return

\[
r_i=\ln(C_i/O_i).
\]

The five-dimensional feature vector is

\[
\mathbf{x}_t=(r_{t-4},r_{t-3},r_{t-2},r_{t-1},r_t)\in\mathbb{R}^5.
\]

Log returns are dimensionless and reduce dependence on the absolute price level. The feature vector deliberately uses candle open/close information rather than future highs, lows, or returns.

### Volatility scaling

Let \(\hat\sigma_t\) be a rolling volatility scale calculated from data available through t. A normalized feature is

\[
z_{t,j}=\frac{r_{t-5+j}}{\hat\sigma_t+\epsilon},\quad j=1,\ldots,5,
\]

where epsilon prevents division by zero. The precise estimator/lookback must be saved with every experiment. Never estimate the scale using future data.

## 3. Future barrier labels

Let P_t be the chosen reference price (normally the query close) and q=0.0001 the EUR/USD pip convention. For distance d pips,

\[
U_{t,d}=P_t+dq,\qquad D_{t,d}=P_t-dq.
\]

The upper-touch label is

\[
Y^{up}_{t,d}=\mathbb{1}\{\max_{1\le k\le5}H_{t+k}\ge U_{t,d}\},
\]

and the lower-touch label is

\[
Y^{down}_{t,d}=\mathbb{1}\{\min_{1\le k\le5}L_{t+k}\le D_{t,d}\}.
\]

These are separate marginal events; both can occur during the same horizon. The model also estimates ending-close direction/return and first-hit outcomes.

## 4. KNN retrieval and filtering

For query \(\mathbf{z}_t\), compute distances to eligible historical vectors, for example Euclidean distance:

\[
d_i=\|\mathbf{z}_t-\mathbf{z}_i\|_2.
\]

The research pipeline uses:
1. A large initial nearest-neighbour candidate pool.
2. Volatility-regime filtering to exclude candidates with materially different recent volatility.
3. Temporal de-clustering to reduce redundant examples from adjacent candles/local episodes.
4. Adaptive neighbour count based on distance profile, within configured limits.
5. Distance/kernel weighting so closer candidates contribute more.

A generic weighted event estimate is

\[
\hat p(Y=1\mid\mathbf{x}_t)=
\frac{\sum_{i\in N(t)}w_iY_i}{\sum_{i\in N(t)}w_i},\quad w_i\ge0.
\]

This is a local empirical frequency, not automatically a calibrated probability. Nonstationarity and dependence among historical examples remain important limitations.

## 5. Nested barriers and monotonicity

For each d=1,...,10 pips, estimate upward and downward touch probabilities. Nested barriers imply

\[
P(\text{hit }d+1\text{ pips})\le P(\text{hit }d\text{ pips}).
\]

Sampling noise or independent calibration may violate this. Apply a monotonic projection across barrier distances and report that operation. Do not treat up/down probabilities as complements.

## 6. First-hit outcomes

For target distance a and stop distance b, define target/stop barriers relative to P_t. Inspect future candles chronologically and classify the outcome as:

- **Target first:** target touched before stop.
- **Stop first:** stop touched before target.
- **Neither:** neither barrier touched during the horizon.
- **Ambiguous:** one OHLC candle spans both barriers and intrabar order is unknowable from OHLC alone.

Use the candle open to detect a gap through a barrier where appropriate. If a single candle's high and low span both barriers, do not invent intrabar ordering; retain ambiguity or perform a conservative sensitivity analysis.

The class probabilities must obey

\[
p_T+p_S+p_N+p_A=1,\quad p_k\in[0,1].
\]

## 7. Chronological validation and leakage prevention

Use ordered training, validation, and test periods. Each historical neighbour's label horizon must be fully complete before the query time. Overlapping forecast horizons cause dependence, so use spaced forecasts and/or block-aware uncertainty estimates.

Calibration method selection uses the earlier part of validation for fitting candidates and a later part for selection. The chosen calibrator is refit on validation forecasts and then applied once to the held-out test predictions. Test labels must not influence calibration.

## 8. Interpretation

A calibrated probability is not the same as a profitable strategy. Expected trading value also depends on payoff, executable bid/ask prices, spread, commission, slippage, latency, fill probability, risk limits, and the policy used to select trades. Choose thresholds on validation data and evaluate them on a final untouched test period.
