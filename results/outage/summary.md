# Cell-outage impact: estimates scored against ground truth

4 simulated cities (seeds differ), 40 outage scenarios each. World 0 was used to choose the method; **every number below comes from the other 3 worlds (120 scenarios)**: 79 single-cell, 30 two-cell, 11 where nobody can reconnect; 40 overlap a scripted demand surge. Truth on average: 50% of displaced traffic lost, 43.5 Mbit/s displaced per outage. Intervals are 95% bootstraps over scenarios.

## 1. Which neighbours absorbed traffic? (material absorber = at least 10% of the rerouted traffic)

| method | precision | recall | F1 | false alarm when nobody reconnects |
|---|---|---|---|---|
| flag every neighbour | 0.52 [0.47, 0.55] | 1.00 [1.00, 1.00] | 0.68 [0.64, 0.71] | 100% |
| handover statistics only (sigma 0.5) | 0.75 [0.68, 0.80] | 0.85 [0.81, 0.89] | 0.79 [0.75, 0.83] | 100% |
| data, level counterfactual | 0.81 [0.75, 0.87] | 0.41 [0.36, 0.48] | 0.55 [0.49, 0.60] | 18% |
| data, control counterfactual | 0.83 [0.77, 0.89] | 0.43 [0.37, 0.50] | 0.57 [0.51, 0.62] | 18% |

## 2. How much was lost and how much reconnected?

Error of the estimated **loss fraction** (share of displaced traffic that found no cell) and of the estimated reconnected rate. Displaced traffic itself is estimated to within a median 9% (control counterfactual).

| estimator | loss-fraction error, pp [95% CI] | reconnected-rate error, Mbit/s [95% CI] | bias, Mbit/s | dev-world error, pp |
|---|---|---|---|---|
| assume everything is lost | 50.3 [45.6, 54.5] | 22.2 [19.5, 25.3] | -22.2 | 51.4 |
| assume half reconnects | 19.7 [17.4, 22.0] | 8.5 [7.3, 9.8] | -1.0 | 23.6 |
| level counterfactual, sum over neighbours | 29.2 [25.8, 32.5] | 16.8 [14.6, 19.4] | -7.1 | 30.0 |
| control counterfactual, sum over neighbours | 22.5 [18.9, 25.9] | 12.1 [10.2, 14.1] | +2.8 | 22.0 |
| control counterfactual, detected neighbours only | 25.6 [21.8, 29.2] | 10.9 [9.2, 12.6] | -2.2 | 26.7 |
| control counterfactual + handover prior (sigma 0.25) | 20.3 [17.1, 23.5] | 7.2 [6.0, 8.5] | +1.0 | 18.8 |
| control counterfactual + handover prior (sigma 0.5) | 21.6 [18.5, 24.6] | 7.7 [6.4, 9.0] | -0.6 | 17.1 |
| control counterfactual + handover prior (sigma 1) | 21.7 [18.5, 24.6] | 8.1 [6.6, 9.6] | -2.3 | 22.7 |
| control counterfactual + handover prior (sigma 2) | 24.4 [21.0, 27.8] | 9.3 [7.9, 10.8] | -5.9 | 21.0 |
| control + true shares as prior (upper bound) | 19.2 [16.5, 22.3] | 6.7 [5.5, 7.8] | +1.6 | 15.7 |

## 3. Who took how much? (total variation distance to the true shares; 0 is perfect, 1 is worst)

| method | distance [95% CI] |
|---|---|
| uniform over neighbours | 0.38 [0.36, 0.41] |
| data only (control counterfactual) | 0.28 [0.25, 0.32] |

Blend of data and handover statistics, by noise in the statistics (rows) and weight on them (columns, 0 = data only, 1 = statistics only):

| handover noise sigma | weight 0 | weight 0.5 | weight 1 |
|---|---|---|---|
| 0.25 | 0.28 | 0.19 | 0.18 |
| 0.5 | 0.28 | 0.20 | 0.23 |
| 1 | 0.28 | 0.23 | 0.30 |
| 2 | 0.28 | 0.27 | 0.42 |

## 4. Degradation in the neighbouring cells

Extra user-seconds per second below the served-fraction threshold, estimated from observed load and users against the cell's own pre-outage rate.

| counterfactual | correlation with truth [95% CI] | mean abs. error | error if predicting zero | true mean |
|---|---|---|---|---|
| level | 0.83 [0.61, 0.92] | 0.77 | 0.95 | 0.95 |
| control | 0.83 [0.61, 0.92] | 0.77 | 0.95 | 0.95 |

## 5. Is the absorber test honest?

False-positive rate of the placebo-in-space test on control cells (leave-one-out), nominal alpha = 0.05:

| counterfactual | observed rate [95% CI] |
|---|---|
| level | 0.059 [0.057, 0.061] |
| control | 0.059 [0.057, 0.061] |

## 6. Where does it break?

| subset | scenarios | loss error, data only (pp) | loss error, with handover prior (pp) | absorber F1 |
|---|---|---|---|---|
| single-cell outage | 79 | 25.1 | 24.9 | 0.54 |
| two-cell (site) outage | 30 | 18.8 | 18.3 | 0.62 |
| nobody can reconnect | 11 | 14.2 | 6.8 | 0.00 |
| outage overlaps a surge | 40 | 28.9 | 28.1 | 0.54 |
| no surge overlap | 80 | 19.3 | 18.3 | 0.58 |
| small outage (displaced < 31 Mbit/s) | 40 | 26.1 | 24.6 | 0.54 |
| medium outage (31-45 Mbit/s) | 40 | 22.7 | 20.5 | 0.52 |
| large outage (> 45 Mbit/s) | 40 | 18.8 | 19.6 | 0.62 |
| short outage (< 643 s) | 39 | 21.2 | 20.2 | 0.54 |
| medium-length outage (643-927 s) | 41 | 25.3 | 23.7 | 0.58 |
| long outage (> 927 s) | 40 | 20.9 | 20.7 | 0.56 |

## 7. How much traffic must a neighbour absorb before the data finds it?

Share of neighbours flagged as absorbers (control counterfactual), by the traffic they truly took over:

| absorbed traffic, Mbit/s | neighbours | found |
|---|---|---|
| 0 to 1 | 226 | 7% |
| 1 to 3 | 120 | 14% |
| 3 to 6 | 123 | 30% |
| 6 or more | 154 | 64% |
