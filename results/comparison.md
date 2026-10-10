# msData vs client-side 5G traces

Same code, same estimators; each column is its own data set, its own rare event and its own held-out groups (mobility patterns for msData, mobility / app pairs for the 5G traces). Intervals are 95% block-bootstrap CIs. The two columns are different systems and different events, so compare *patterns*, not absolute levels.

| | msData (Open RAN testbed) | client-side 5G traces |
|---|---|---|
| rare-event rate | 4.0% (2,627 events) | 3.8% (2,848 events) |
| PR-AUC, none | 0.316 [0.268, 0.361] | 0.132 [0.118, 0.149] |
| PR-AUC, class_weight | 0.334 [0.292, 0.373] | 0.121 [0.107, 0.139] |
| PR-AUC, smote | 0.328 [0.288, 0.369] | 0.127 [0.114, 0.146] |
| PR-AUC, undersample | 0.346 [0.294, 0.388] | 0.145 [0.130, 0.166] |
| mean predicted probability, class_weight (true rate in row 1) | 0.103 | 0.125 |
| ECE, class_weight, raw -> isotonic | 0.0654 -> 0.0060 | 0.0914 -> 0.0117 |
| Brier, class_weight, raw -> isotonic | 0.0528 -> 0.0318 | 0.0617 -> 0.0350 |
| alert cost per 1000, class_weight, Bayes threshold, raw -> isotonic | 416 -> 376 (tuned threshold: 376) | 643 -> 512 (tuned threshold: 510) |
| PR-AUC, random row split | 0.509 [0.463, 0.551] | 0.268 [0.233, 0.306] |
| PR-AUC, unseen group | 0.321 [0.280, 0.361] | 0.114 [0.103, 0.130] |
| random split inflates PR-AUC by | 59% | 134% |
| 80% interval coverage (conformalised), overall | 0.783 [0.777, 0.790] | 0.814 [0.806, 0.823] |
| ... on the rows where the rare event occurs | 0.252 | 0.087 |
| PR-AUC, all features | 0.334 [0.292, 0.373] (36 features) | 0.121 [0.107, 0.139] (43 features) |
| PR-AUC, mutual_info with 6 features | 0.289 [0.241, 0.325] | 0.096 [0.085, 0.112] |
| PR-AUC, l1 with 6 features | 0.300 [0.242, 0.337] | 0.093 [0.081, 0.107] |
| PR-AUC, tree with 6 features | 0.297 [0.253, 0.331] | 0.105 [0.093, 0.120] |
| PR-AUC, pca with 6 features | 0.285 [0.238, 0.323] | 0.125 [0.114, 0.142] |
