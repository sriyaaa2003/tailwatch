## Data (client-side 5G production traces, Irish operator)

87 streams, 200,219 s (55.6 h) of 1 Hz phone KPIs, 73% of seconds on 5G, 75,593 decision points (stride 2 s, channel currently good: CQI >= 9). **Channel collapse** = the worst CQI in the next 10 s is <= 5. **Collapse rate 3.8%** (2,848 events).

| mobility / app | streams | seconds | decision points | collapse rate |
|---|---|---|---|---|
| static / prime | 8 | 37,823 | 17,846 | 0.9% |
| static / download | 5 | 17,217 | 7,125 | 2.2% |
| driving / netflix | 23 | 39,280 | 14,693 | 5.2% |
| driving / download | 18 | 28,147 | 9,105 | 7.7% |
| driving / prime | 22 | 40,456 | 12,347 | 5.9% |
| static / netflix | 11 | 37,296 | 14,477 | 2.3% |

## 1. Imbalance strategies (unseen mobility / apps, raw scores)

| strategy | PR-AUC [95% CI] | macro-F1 [95% CI] | Brier | mean predicted p (true rate 0.038) | train fit s |
|---|---|---|---|---|---|
| none | 0.132 [0.118, 0.149] | 0.578 [0.567, 0.590] | 0.0355 | 0.025 | 33.1 |
| class_weight | 0.121 [0.107, 0.139] | 0.577 [0.567, 0.589] | 0.0617 | 0.125 | 32.7 |
| smote | 0.127 [0.114, 0.146] | 0.583 [0.575, 0.594] | 0.0363 | 0.042 | 44.7 |
| undersample | 0.145 [0.130, 0.166] | 0.584 [0.574, 0.596] | 0.0573 | 0.112 | 29.7 |

## 2. Calibration (unseen mobility / apps)

| strategy | calibration | Brier [95% CI] | ECE [95% CI] | PR-AUC |
|---|---|---|---|---|
| none | raw | 0.0355 [0.0325, 0.0384] | 0.0139 [0.0114, 0.0168] | 0.132 |
| none | sigmoid | 0.0345 [0.0318, 0.0372] | 0.0104 [0.0083, 0.0125] | 0.133 |
| none | isotonic | 0.0347 [0.0320, 0.0375] | 0.0096 [0.0085, 0.0126] | 0.127 |
| class_weight | raw | 0.0617 [0.0582, 0.0653] | 0.0914 [0.0877, 0.0952] | 0.121 |
| class_weight | sigmoid | 0.0348 [0.0320, 0.0377] | 0.0078 [0.0060, 0.0104] | 0.119 |
| class_weight | isotonic | 0.0350 [0.0322, 0.0379] | 0.0117 [0.0098, 0.0146] | 0.114 |
| smote | raw | 0.0363 [0.0335, 0.0391] | 0.0136 [0.0115, 0.0163] | 0.127 |
| smote | sigmoid | 0.0346 [0.0319, 0.0374] | 0.0104 [0.0084, 0.0128] | 0.127 |
| smote | isotonic | 0.0347 [0.0320, 0.0375] | 0.0086 [0.0069, 0.0116] | 0.120 |
| undersample | raw | 0.0573 [0.0541, 0.0607] | 0.0771 [0.0739, 0.0813] | 0.145 |
| undersample | sigmoid | 0.0344 [0.0317, 0.0372] | 0.0095 [0.0080, 0.0124] | 0.135 |
| undersample | isotonic | 0.0345 [0.0317, 0.0372] | 0.0108 [0.0085, 0.0132] | 0.134 |

PR-AUC inside each held-out mobility / app (class_weight). A monotone calibrator (sigmoid) cannot change it; isotonic can, through tied scores. Pooled PR-AUC above mixes mobility / apps, so it also depends on the per-mobility / app calibration maps.

| held-out mobility / app | channel collapse rate | raw | sigmoid | isotonic |
|---|---|---|---|---|
| driving / download | 7.7% | 0.133 | 0.133 | 0.121 |
| driving / netflix | 5.2% | 0.138 | 0.138 | 0.124 |
| driving / prime | 5.9% | 0.196 | 0.196 | 0.179 |
| static / download | 2.2% | 0.068 | 0.068 | 0.053 |
| static / netflix | 2.3% | 0.095 | 0.095 | 0.081 |
| static / prime | 0.9% | 0.021 | 0.021 | 0.019 |

## 3. Alert policy: what does calibration buy?

Costs: missed channel collapse = 20, false alarm = 1, so the Bayes threshold for a calibrated probability is 0.048. Cost per 1000 decisions (lower is better); never alert = 754, always alert = 962.

| strategy | calibration | cost at Bayes threshold [95% CI] | cost at validation-tuned threshold |
|---|---|---|---|
| none | raw | 516 [471, 561] | 476 |
| none | sigmoid | 466 [430, 502] | 476 |
| none | isotonic | 485 [447, 527] | 485 |
| class_weight | raw | 643 [618, 672] | 510 |
| class_weight | sigmoid | 505 [469, 548] | 511 |
| class_weight | isotonic | 512 [471, 560] | 511 |
| smote | raw | 489 [448, 527] | 482 |
| smote | sigmoid | 472 [434, 509] | 488 |
| smote | isotonic | 482 [440, 521] | 484 |
| undersample | raw | 559 [529, 589] | 475 |
| undersample | sigmoid | 458 [423, 496] | 475 |
| undersample | isotonic | 475 [436, 512] | 477 |

## 4. Does the evaluation protocol flatter the model?

Same model (class_weight, isotonic-calibrated), 2 ways of splitting the data. PR-AUC is only comparable across rows with similar channel collapse prevalence, so prevalence is shown.

| protocol | PR-AUC [95% CI] | macro-F1 | Brier | ECE | channel collapse prevalence in test |
|---|---|---|---|---|---|
| random rows | 0.268 [0.233, 0.306] | 0.655 | 0.0296 | 0.0026 | 0.036 |
| unseen mobility / app | 0.114 [0.103, 0.130] | 0.577 | 0.0350 | 0.0117 | 0.038 |

## 5. Probabilistic regression: worst channel-quality deficit relative to the event level in the next 10 s (80% intervals)

| interval | coverage on unseen mobility / apps [95% CI] | coverage when a channel collapse occurs | mean width | interval score |
|---|---|---|---|---|
| raw quantile interval | 0.795 [0.786, 0.804] | 0.087 | 0.598 | 0.782 |
| conformalised (CQR) | 0.814 [0.806, 0.823] | 0.087 | 0.598 | 0.782 |

Per held-out mobility / app coverage (raw / conformalised): driving / download 0.79/0.79, driving / netflix 0.82/0.82, driving / prime 0.76/0.76, static / download 0.84/0.84, static / netflix 0.81/0.81, static / prime 0.76/0.84.

Median forecast MAE 0.144 vs 0.196 for 'peak = current level'.

## 6. Feature selection vs reduction (unseen mobility / apps)

| method | k | PR-AUC [95% CI] | selection s | fit s | predict ms / 1k rows | fold stability (Jaccard) |
|---|---|---|---|---|---|---|
| all | 43 | 0.121 [0.107, 0.139] | 0.00 | 7.72 | 13.08 |  |
| mutual_info | 3 | 0.105 [0.094, 0.121] | 13.82 | 3.87 | 10.42 | 0.57 |
| mutual_info | 6 | 0.096 [0.085, 0.112] | 13.82 | 4.04 | 11.07 | 0.60 |
| mutual_info | 10 | 0.093 [0.081, 0.109] | 13.82 | 4.68 | 19.66 | 0.51 |
| mutual_info | 16 | 0.107 [0.095, 0.125] | 13.82 | 4.40 | 12.62 | 0.68 |
| l1 | 3 | 0.058 [0.051, 0.068] | 4.47 | 4.49 | 12.32 | 0.21 |
| l1 | 6 | 0.093 [0.081, 0.107] | 4.47 | 2.86 | 15.59 | 0.28 |
| l1 | 10 | 0.097 [0.085, 0.114] | 4.47 | 4.62 | 14.71 | 0.35 |
| l1 | 16 | 0.111 [0.098, 0.127] | 4.47 | 12.83 | 19.72 | 0.43 |
| tree | 3 | 0.109 [0.097, 0.124] | 1.58 | 19.82 | 24.46 | 1.00 |
| tree | 6 | 0.105 [0.093, 0.120] | 1.58 | 25.60 | 36.34 | 0.62 |
| tree | 10 | 0.116 [0.102, 0.133] | 1.58 | 18.26 | 23.84 | 0.69 |
| tree | 16 | 0.098 [0.086, 0.111] | 1.58 | 17.00 | 15.59 | 0.81 |
| pca | 3 | 0.110 [0.101, 0.125] | 0.08 | 13.60 | 17.52 |  |
| pca | 6 | 0.125 [0.114, 0.142] | 0.08 | 10.36 | 14.43 |  |
| pca | 10 | 0.134 [0.119, 0.152] | 0.08 | 6.44 | 10.79 |  |
| pca | 16 | 0.125 [0.113, 0.143] | 0.08 | 4.00 | 10.89 |  |

Features chosen by tree importance in every fold at k=3: cqi_min_15, cqi_mean_30, cqi_min_30.
