## Data (msData, Open RAN 5G testbed)

3,175,140 per-UE records (median spacing 97 ms, 5th-95th percentile 44-295 ms; not 1 ms samples) -> cell-level load in 1 s bins (sum of all UEs' downlink rate), 702 contiguous streams (silences over 3 s split a stream, streams under 120 s dropped), 44.6 h of cell time, 65,944 decision points. **High-load burst** = cell load reaches 10 Mbit/s within 10 s (a configured level, about the 99.3rd percentile of 1 s load; the cell's capacity is not observed, so this is not verified congestion). **Burst rate 4.0%** (2,627 events).

| mobility pattern | streams | decision points | burst rate |
|---|---|---|---|
| bus | 95 | 7,531 | 4.4% |
| car | 340 | 31,066 | 4.6% |
| pedestrian | 178 | 19,354 | 2.9% |
| static | 54 | 4,582 | 3.3% |
| train | 35 | 3,411 | 3.9% |

Descriptive only (the traffic label is never a model input): burst rate by the traffic class of the heaviest UE in the bin.

| heaviest UE's traffic | decision points | burst rate |
|---|---|---|
| ddos-ripper-C | 11,546 | 1.5% |
| dos-hulk-C | 10,956 | 9.9% |
| iot | 9,992 | 1.0% |
| portscan | 9,847 | 2.4% |
| SIPP | 7,735 | 1.2% |
| youtube | 7,369 | 5.9% |
| Web Browsing | 4,482 | 7.3% |
| slowloris-C | 3,851 | 4.6% |
| none | 166 | 4.8% |

### Burstiness of the cell load

Streams of at least 300 s. Median [IQR]; iid-noise Hurst reference 0.57.

| streams | n | Hurst H | CV (std/mean) | autocorr lag 1 s | autocorr lag 10 s |
|---|---|---|---|---|---|
| all | 119 | 0.88 [0.83, 0.93] | 0.95 [0.77, 1.16] | 0.53 [0.42, 0.60] | 0.38 [0.27, 0.52] |
| bus | 9 | 0.84 [0.72, 0.87] | 0.84 [0.67, 0.90] | 0.26 [0.23, 0.52] | 0.34 [0.27, 0.55] |
| car | 41 | 0.89 [0.83, 0.93] | 0.96 [0.84, 1.18] | 0.55 [0.45, 0.65] | 0.38 [0.29, 0.52] |
| pedestrian | 59 | 0.89 [0.85, 0.94] | 1.01 [0.79, 1.25] | 0.54 [0.45, 0.61] | 0.36 [0.23, 0.47] |
| static | 4 | 0.82 [0.77, 0.87] | 0.75 [0.68, 0.83] | 0.29 [0.24, 0.38] | 0.47 [0.44, 0.52] |
| train | 6 | 0.87 [0.86, 0.88] | 0.72 [0.65, 0.74] | 0.37 [0.31, 0.48] | 0.57 [0.47, 0.63] |

## 1. Imbalance strategies (unseen mobility patterns, raw scores)

| strategy | PR-AUC [95% CI] | macro-F1 [95% CI] | Brier | mean predicted p (true rate 0.040) | train fit s |
|---|---|---|---|---|---|
| none | 0.316 [0.268, 0.361] | 0.672 [0.653, 0.691] | 0.0324 | 0.027 | 1.3 |
| class_weight | 0.334 [0.292, 0.373] | 0.674 [0.653, 0.692] | 0.0528 | 0.103 | 1.0 |
| smote | 0.328 [0.288, 0.369] | 0.677 [0.657, 0.696] | 0.0358 | 0.060 | 1.2 |
| undersample | 0.346 [0.294, 0.388] | 0.683 [0.663, 0.700] | 0.0555 | 0.106 | 0.5 |

## 2. Calibration (unseen mobility patterns)

| strategy | calibration | Brier [95% CI] | ECE [95% CI] | PR-AUC |
|---|---|---|---|---|
| none | raw | 0.0324 [0.0289, 0.0351] | 0.0124 [0.0094, 0.0157] | 0.316 |
| none | sigmoid | 0.0320 [0.0288, 0.0346] | 0.0057 [0.0035, 0.0084] | 0.313 |
| none | isotonic | 0.0322 [0.0289, 0.0347] | 0.0071 [0.0053, 0.0100] | 0.302 |
| class_weight | raw | 0.0528 [0.0491, 0.0559] | 0.0654 [0.0618, 0.0708] | 0.334 |
| class_weight | sigmoid | 0.0316 [0.0284, 0.0341] | 0.0055 [0.0036, 0.0084] | 0.334 |
| class_weight | isotonic | 0.0318 [0.0287, 0.0344] | 0.0060 [0.0041, 0.0090] | 0.321 |
| smote | raw | 0.0358 [0.0324, 0.0380] | 0.0251 [0.0221, 0.0283] | 0.328 |
| smote | sigmoid | 0.0316 [0.0285, 0.0343] | 0.0048 [0.0034, 0.0077] | 0.329 |
| smote | isotonic | 0.0317 [0.0285, 0.0345] | 0.0061 [0.0045, 0.0089] | 0.313 |
| undersample | raw | 0.0555 [0.0518, 0.0589] | 0.0681 [0.0638, 0.0734] | 0.346 |
| undersample | sigmoid | 0.0314 [0.0283, 0.0338] | 0.0052 [0.0033, 0.0082] | 0.336 |
| undersample | isotonic | 0.0316 [0.0285, 0.0340] | 0.0072 [0.0053, 0.0102] | 0.323 |

PR-AUC inside each held-out mobility pattern (class_weight). A monotone calibrator (sigmoid) cannot change it; isotonic can, through tied scores. Pooled PR-AUC above mixes mobility patterns, so it also depends on the per-mobility pattern calibration maps.

| held-out mobility pattern | burst rate | raw | sigmoid | isotonic |
|---|---|---|---|---|
| bus | 4.4% | 0.266 | 0.266 | 0.258 |
| car | 4.6% | 0.394 | 0.394 | 0.365 |
| pedestrian | 2.9% | 0.301 | 0.301 | 0.277 |
| static | 3.3% | 0.198 | 0.198 | 0.185 |
| train | 3.9% | 0.228 | 0.228 | 0.218 |

## 3. Alert policy: what does calibration buy?

Costs: missed burst = 20, false alarm = 1, so the Bayes threshold for a calibrated probability is 0.048. Cost per 1000 decisions (lower is better); never alert = 797, always alert = 960.

| strategy | calibration | cost at Bayes threshold [95% CI] | cost at validation-tuned threshold |
|---|---|---|---|
| none | raw | 399 [352, 438] | 377 |
| none | sigmoid | 377 [343, 411] | 379 |
| none | isotonic | 380 [351, 413] | 381 |
| class_weight | raw | 416 [392, 439] | 376 |
| class_weight | sigmoid | 378 [348, 411] | 379 |
| class_weight | isotonic | 376 [343, 406] | 376 |
| smote | raw | 384 [354, 415] | 382 |
| smote | sigmoid | 379 [345, 410] | 380 |
| smote | isotonic | 381 [346, 412] | 381 |
| undersample | raw | 393 [370, 415] | 377 |
| undersample | sigmoid | 372 [344, 405] | 374 |
| undersample | isotonic | 377 [347, 409] | 377 |

## 4. Does the evaluation protocol flatter the model?

Same model (class_weight, isotonic-calibrated), 2 ways of splitting the data. PR-AUC is only comparable across rows with similar burst prevalence, so prevalence is shown.

| protocol | PR-AUC [95% CI] | macro-F1 | Brier | ECE | burst prevalence in test |
|---|---|---|---|---|---|
| random rows | 0.509 [0.463, 0.551] | 0.758 | 0.0265 | 0.0017 | 0.041 |
| unseen mobility pattern | 0.321 [0.280, 0.361] | 0.674 | 0.0318 | 0.0060 | 0.040 |

## 5. Probabilistic regression: peak load relative to the configured high-load level in the next 10 s (80% intervals)

| interval | coverage on unseen mobility patterns [95% CI] | coverage when a burst occurs | mean width | interval score |
|---|---|---|---|---|
| raw quantile interval | 0.731 [0.724, 0.739] | 0.233 | 0.361 | 0.649 |
| conformalised (CQR) | 0.783 [0.777, 0.790] | 0.252 | 0.385 | 0.644 |

Per held-out mobility pattern coverage (raw / conformalised): bus 0.78/0.81, car 0.69/0.75, pedestrian 0.75/0.82, static 0.83/0.84, train 0.74/0.77.

Median forecast MAE 0.130 vs 0.239 for 'peak = current level'.

## 6. Feature selection vs reduction (unseen mobility patterns)

| method | k | PR-AUC [95% CI] | selection s | fit s | predict ms / 1k rows | fold stability (Jaccard) |
|---|---|---|---|---|---|---|
| all | 36 | 0.334 [0.292, 0.373] | 0.00 | 0.96 | 4.99 |  |
| mutual_info | 3 | 0.280 [0.235, 0.323] | 5.81 | 0.53 | 4.74 | 1.00 |
| mutual_info | 6 | 0.289 [0.241, 0.325] | 5.81 | 0.59 | 5.70 | 0.89 |
| mutual_info | 10 | 0.314 [0.264, 0.351] | 5.81 | 0.65 | 5.22 | 0.87 |
| mutual_info | 16 | 0.311 [0.258, 0.355] | 5.81 | 0.68 | 5.14 | 0.86 |
| l1 | 3 | 0.290 [0.244, 0.327] | 1.05 | 0.53 | 4.52 | 0.37 |
| l1 | 6 | 0.300 [0.242, 0.337] | 1.05 | 0.55 | 4.58 | 0.52 |
| l1 | 10 | 0.293 [0.248, 0.332] | 1.05 | 0.61 | 4.38 | 0.45 |
| l1 | 16 | 0.318 [0.268, 0.359] | 1.05 | 0.66 | 5.19 | 0.51 |
| tree | 3 | 0.238 [0.201, 0.269] | 1.31 | 0.51 | 5.23 | 0.49 |
| tree | 6 | 0.297 [0.253, 0.331] | 1.31 | 0.54 | 5.71 | 0.89 |
| tree | 10 | 0.302 [0.252, 0.336] | 1.31 | 0.63 | 5.14 | 0.85 |
| tree | 16 | 0.310 [0.266, 0.346] | 1.31 | 0.65 | 4.86 | 0.83 |
| pca | 3 | 0.286 [0.238, 0.321] | 0.03 | 0.50 | 4.80 |  |
| pca | 6 | 0.285 [0.238, 0.323] | 0.03 | 0.52 | 4.34 |  |
| pca | 10 | 0.316 [0.270, 0.352] | 0.03 | 0.59 | 4.53 |  |
| pca | 16 | 0.322 [0.280, 0.361] | 0.03 | 0.64 | 4.59 |  |

Features chosen by tree importance in every fold at k=3: util_std_30.
