# Twin vs real trace

Same code, same estimators; each column is its own data set and its own held-out groups (districts for the twin, mobility patterns for the real trace). Intervals are 95% block-bootstrap CIs. The two columns are different systems, so compare *patterns*, not absolute levels.

| | digital twin | real trace (msData) |
|---|---|---|
| rare-event rate | 5.7% (7,746 events) | 4.0% (2,627 events) |
| PR-AUC, none | 0.492 [0.436, 0.547] | 0.316 [0.268, 0.361] |
| PR-AUC, class_weight | 0.506 [0.450, 0.559] | 0.334 [0.292, 0.373] |
| PR-AUC, smote | 0.448 [0.396, 0.503] | 0.328 [0.288, 0.369] |
| PR-AUC, undersample | 0.518 [0.459, 0.571] | 0.346 [0.294, 0.388] |
| mean predicted probability, class_weight (true rate in row 1) | 0.133 | 0.103 |
| ECE, class_weight, raw -> isotonic | 0.0773 -> 0.0067 | 0.0654 -> 0.0060 |
| Brier, class_weight, raw -> isotonic | 0.0640 -> 0.0403 | 0.0528 -> 0.0318 |
| alert cost per 1000, class_weight, Bayes threshold, raw -> isotonic | 359 -> 330 (tuned threshold: 329) | 416 -> 376 (tuned threshold: 376) |
| PR-AUC, random row split | 0.571 [0.518, 0.624] | 0.509 [0.463, 0.551] |
| PR-AUC, unseen group | 0.439 [0.388, 0.488] | 0.321 [0.280, 0.361] |
| random split inflates PR-AUC by | 30% | 59% |
| 80% interval coverage (conformalised), overall | 0.797 [0.790, 0.803] | 0.783 [0.777, 0.790] |
| ... on the rows where the rare event occurs | 0.376 | 0.252 |
| PR-AUC, all features | 0.506 [0.450, 0.559] (31 features) | 0.334 [0.292, 0.373] (36 features) |
| PR-AUC, mutual_info with 6 features | 0.524 [0.468, 0.579] | 0.289 [0.241, 0.325] |
| PR-AUC, l1 with 6 features | 0.503 [0.445, 0.559] | 0.300 [0.242, 0.337] |
| PR-AUC, tree with 6 features | 0.537 [0.486, 0.592] | 0.297 [0.253, 0.331] |
| PR-AUC, pca with 6 features | 0.524 [0.470, 0.579] | 0.285 [0.238, 0.323] |
