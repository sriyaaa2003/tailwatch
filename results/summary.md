## Data (synthetic digital twin)

691,200 one-second cell observations -> 136,026 decision points (stride 5 s, already-congested rows removed). Storm = congestion (utilisation >= 1.1) begins within 30 s. **Storm rate 5.7%** (7,746 events).

| district | decision points | storm rate |
|---|---|---|
| business | 34,307 | 1.7% |
| leisure | 33,317 | 13.5% |
| residential | 34,321 | 1.3% |
| transit | 34,081 | 6.5% |

Generator check (aggregated-variance Hurst exponent on load with surges and diurnal cycle switched off; iid-noise reference H = 0.53):

| cell | district | H measured | H theory (3-alpha_min)/2 |
|---|---|---|---|
| 0 | residential | 0.76 | 0.85 |
| 2 | business | 0.73 | 0.80 |
| 4 | leisure | 0.77 | 0.90 |
| 6 | transit | 0.82 | 0.88 |

## 1. Imbalance strategies (unseen districts, raw scores)

| strategy | PR-AUC [95% CI] | macro-F1 [95% CI] | Brier | mean predicted p (true rate 0.057) | train fit s |
|---|---|---|---|---|---|
| none | 0.492 [0.436, 0.547] | 0.731 [0.713, 0.749] | 0.0386 | 0.044 | 1.7 |
| class_weight | 0.506 [0.450, 0.559] | 0.727 [0.706, 0.745] | 0.0640 | 0.133 | 1.1 |
| smote | 0.448 [0.396, 0.503] | 0.705 [0.686, 0.725] | 0.0418 | 0.072 | 1.4 |
| undersample | 0.518 [0.459, 0.571] | 0.717 [0.698, 0.734] | 0.0482 | 0.103 | 0.7 |

## 2. Calibration (unseen districts)

| strategy | calibration | Brier [95% CI] | ECE [95% CI] | PR-AUC |
|---|---|---|---|---|
| none | raw | 0.0386 [0.0343, 0.0433] | 0.0125 [0.0083, 0.0165] | 0.492 |
| none | sigmoid | 0.0398 [0.0354, 0.0446] | 0.0113 [0.0065, 0.0160] | 0.461 |
| none | isotonic | 0.0417 [0.0369, 0.0467] | 0.0125 [0.0080, 0.0178] | 0.428 |
| class_weight | raw | 0.0640 [0.0587, 0.0702] | 0.0773 [0.0718, 0.0840] | 0.506 |
| class_weight | sigmoid | 0.0397 [0.0351, 0.0444] | 0.0079 [0.0035, 0.0127] | 0.457 |
| class_weight | isotonic | 0.0403 [0.0357, 0.0451] | 0.0067 [0.0044, 0.0117] | 0.439 |
| smote | raw | 0.0418 [0.0379, 0.0463] | 0.0175 [0.0146, 0.0215] | 0.448 |
| smote | sigmoid | 0.0450 [0.0397, 0.0505] | 0.0320 [0.0261, 0.0379] | 0.426 |
| smote | isotonic | 0.0448 [0.0395, 0.0504] | 0.0320 [0.0263, 0.0379] | 0.411 |
| undersample | raw | 0.0482 [0.0439, 0.0532] | 0.0474 [0.0438, 0.0515] | 0.518 |
| undersample | sigmoid | 0.0398 [0.0352, 0.0446] | 0.0099 [0.0050, 0.0148] | 0.459 |
| undersample | isotonic | 0.0406 [0.0358, 0.0455] | 0.0101 [0.0078, 0.0152] | 0.441 |

PR-AUC inside each held-out district (class_weight). A monotone calibrator (sigmoid) cannot change it; isotonic can, through tied scores. Pooled PR-AUC above mixes districts, so it also depends on the per-district calibration maps.

| held-out district | storm rate | raw | sigmoid | isotonic |
|---|---|---|---|---|
| business | 1.7% | 0.295 | 0.295 | 0.282 |
| leisure | 13.5% | 0.618 | 0.618 | 0.587 |
| residential | 1.3% | 0.118 | 0.118 | 0.110 |
| transit | 6.5% | 0.473 | 0.473 | 0.446 |

## 3. Alert policy: what does calibration buy?

Costs: missed storm = 20, false alarm = 1, so the Bayes threshold for a calibrated probability is 0.048. Cost per 1000 decisions (lower is better); never alert = 1139, always alert = 943.

| strategy | calibration | cost at Bayes threshold [95% CI] | cost at validation-tuned threshold |
|---|---|---|---|
| none | raw | 352 [314, 390] | 334 |
| none | sigmoid | 345 [312, 381] | 333 |
| none | isotonic | 332 [301, 364] | 332 |
| class_weight | raw | 359 [335, 383] | 329 |
| class_weight | sigmoid | 340 [308, 375] | 334 |
| class_weight | isotonic | 330 [304, 362] | 330 |
| smote | raw | 327 [302, 355] | 413 |
| smote | sigmoid | 428 [384, 484] | 410 |
| smote | isotonic | 409 [370, 460] | 409 |
| undersample | raw | 326 [302, 352] | 329 |
| undersample | sigmoid | 341 [310, 376] | 336 |
| undersample | isotonic | 331 [302, 363] | 331 |

## 4. Does the evaluation protocol flatter the model?

Same model (class_weight, isotonic-calibrated), four splits. PR-AUC is only comparable across rows with similar storm prevalence, so prevalence is shown.

| protocol | PR-AUC [95% CI] | macro-F1 | Brier | ECE | storm prevalence in test |
|---|---|---|---|---|---|
| random rows | 0.571 [0.518, 0.624] | 0.765 | 0.0337 | 0.0034 | 0.057 |
| temporal (same cells) | 0.395 [0.275, 0.484] | 0.711 | 0.0144 | 0.0024 | 0.020 |
| unseen district | 0.439 [0.388, 0.488] | 0.727 | 0.0403 | 0.0067 | 0.057 |
| unseen district + future | 0.265 [0.160, 0.371] | 0.701 | 0.0180 | 0.0073 | 0.021 |

## 5. Probabilistic regression: peak utilisation in the next 30 s (80% intervals)

| interval | coverage on unseen districts [95% CI] | coverage when a storm occurs | mean width | interval score |
|---|---|---|---|---|
| raw quantile interval | 0.764 [0.758, 0.770] | 0.352 | 0.283 | 0.445 |
| conformalised (CQR) | 0.797 [0.790, 0.803] | 0.376 | 0.302 | 0.443 |

Per held-out district coverage (raw / conformalised): business 0.84/0.87, leisure 0.68/0.71, residential 0.78/0.81, transit 0.75/0.79.

Median forecast MAE 0.097 vs 0.269 for 'peak = current utilisation'.

## 6. Feature selection vs reduction (unseen districts)

| method | k | PR-AUC [95% CI] | selection s | fit s | predict ms / 1k rows | fold stability (Jaccard) |
|---|---|---|---|---|---|---|
| all | 31 | 0.506 [0.450, 0.559] | 0.00 | 1.15 | 4.01 |  |
| mutual_info | 3 | 0.508 [0.452, 0.564] | 4.99 | 0.64 | 3.81 | 1.00 |
| mutual_info | 6 | 0.524 [0.468, 0.579] | 4.99 | 0.67 | 3.92 | 0.76 |
| mutual_info | 10 | 0.525 [0.470, 0.580] | 4.99 | 0.78 | 3.93 | 0.85 |
| mutual_info | 16 | 0.536 [0.478, 0.589] | 4.99 | 0.84 | 4.07 | 1.00 |
| l1 | 3 | 0.523 [0.471, 0.576] | 1.08 | 0.64 | 3.82 | 0.43 |
| l1 | 6 | 0.503 [0.445, 0.559] | 1.08 | 0.68 | 3.69 | 0.49 |
| l1 | 10 | 0.507 [0.450, 0.564] | 1.08 | 0.78 | 3.61 | 0.60 |
| l1 | 16 | 0.494 [0.440, 0.550] | 1.08 | 0.87 | 3.77 | 0.62 |
| tree | 3 | 0.531 [0.479, 0.585] | 1.16 | 0.64 | 4.08 | 1.00 |
| tree | 6 | 0.537 [0.486, 0.592] | 1.16 | 0.67 | 4.16 | 1.00 |
| tree | 10 | 0.494 [0.443, 0.549] | 1.16 | 0.77 | 3.89 | 0.91 |
| tree | 16 | 0.513 [0.462, 0.568] | 1.16 | 0.85 | 3.91 | 0.83 |
| pca | 3 | 0.543 [0.488, 0.597] | 0.05 | 0.63 | 3.68 |  |
| pca | 6 | 0.524 [0.470, 0.579] | 0.04 | 0.65 | 3.47 |  |
| pca | 10 | 0.520 [0.464, 0.574] | 0.04 | 0.75 | 3.49 |  |
| pca | 16 | 0.523 [0.471, 0.576] | 0.04 | 0.84 | 3.68 |  |

Features chosen by tree importance in every fold at k=3: util_mean_30, util_max_30, util_max_60.

## 7. Scale (Parquet, rolling-window feature job)

| engine | scale | rows | status | wall s | peak MB |
|---|---|---|---|---|---|
| pandas | 1x | 172,800 | ok | 0.08 (0.08-0.15) | 156 |
| pandas | 10x | 1,728,000 | ok | 1.17 (1.13-1.23) | 374 |
| pandas | 100x | 17,280,000 | ok | 11.33 (10.96-11.33) | 2367 |
| polars | 1x | 172,800 | ok | 0.02 (0.01-0.02) | 101 |
| polars | 10x | 1,728,000 | ok | 0.09 (0.07-0.11) | 308 |
| polars | 100x | 17,280,000 | ok | 0.68 (0.63-0.74) | 1120 |
| duckdb | 1x | 172,800 | ok | 0.14 (0.13-0.14) | 141 |
| duckdb | 10x | 1,728,000 | ok | 0.46 (0.45-0.46) | 227 |
| duckdb | 100x | 17,280,000 | ok | 4.43 (4.35-4.53) | 708 |

| scale | rows | Parquet (zstd) MB | bytes/row |
|---|---|---|---|
| 1x | 172,800 | 1.9 | 11.3 |
| 10x | 1,728,000 | 18.5 | 11.3 |
| 100x | 17,280,000 | 185.4 | 11.3 |

1x as CSV: 7.0 MB; Parquet: 1.9 MB (3.8x smaller); in-memory Arrow: 4.0 MB.

Cross-engine agreement on identical job output: 1x yes, 10x yes, 100x yes.
