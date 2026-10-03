# tailwatch: storm radar for the radio network

Congestion in a cellular network is rare, bursty and local. `tailwatch` treats it like severe weather: instead of
a yes/no label it issues a **calibrated probability** ("12% chance this cell congests within 30 s") and a **prediction
interval** for peak load, then asks the questions a meteorologist would. When it says 30%, is it right 30% of the time?
Does that still hold in a district it has never seen? What does an alert policy built on it actually cost?

![reliability](results/reliability.png)

Everything runs on a **synthetic city digital twin** (hex-grid cells, four districts, scripted surge events, heavy-tailed
ON/OFF traffic sources). It is synthetic by design so the whole study is reproducible and I can hold out whole
districts; it is **not** real operator data, and the numbers below say nothing about a real network (see
[What is not measured](#what-is-not-measured)).

## Results at a glance

All figures are out-of-district: every test row is predicted by a model that never saw that row's district. Intervals are
95% block-bootstrap CIs (blocks of 600 s per cell). The full tables are in [`results/summary.md`](results/summary.md),
generated from JSON by `tailwatch report`.

| question | finding |
|---|---|
| Rare class | Storm = congestion onset within 30 s. 5.7% of decision points overall, 1.3% to 13.5% by district. |
| Imbalance handling | **No strategy is a clear winner.** PR-AUC: none 0.49 [0.44, 0.55], class weights 0.51 [0.45, 0.56], under-sampling 0.52 [0.46, 0.57], SMOTE 0.45 [0.40, 0.50]. SMOTE is lowest; the rest overlap. |
| Calibration | Class weights and under-sampling make a model **over-predict** (mean p 0.133 and 0.103 against a true rate of 0.057). Recalibrating on later data fixes it: ECE 0.077 -> 0.007 (class weights), Brier 0.064 -> 0.040. Plain training is already close (ECE 0.013). |
| Does calibration pay? | With costs of 20 per missed storm and 1 per false alarm, a calibrated model can use the textbook threshold 1/21 without tuning (329.7 per 1000 decisions) while the raw class-weighted model at that threshold costs 358.8. A threshold tuned on validation data reaches about the same ~330 for every strategy except SMOTE (~410). So calibration buys *not having to tune*, not a lower floor. |
| Generalisation | A random row split flatters the model: PR-AUC 0.57 [0.52, 0.62] vs 0.44 [0.39, 0.49] on an unseen district. Per district it ranges from 0.12 (residential, 1.3% storms) to 0.62 (leisure, 13.5%). |
| Intervals | Conformalised quantile regression gives 79.7% pooled coverage for an 80% target, but 71% to 87% across districts, and **only 38% coverage on the rows that actually became storms**. Marginal coverage looks fine while the tail, which is the point, is not. |
| Feature selection | 3 features match all 31 (PR-AUC 0.53 vs 0.51, CIs overlap); the top features are the recent peak and mean utilisation. Cost differs more than accuracy: mutual information 5.0 s, tree 1.2 s, L1 1.1 s, PCA 0.05 s. L1 picks unstable features across folds (Jaccard 0.43 to 0.62); tree and MI are stable (0.76 to 1.0). |
| Scale | Same job on Parquet at 1x/10x/100x (172 k to 17.3 M rows): Polars 0.68 s, DuckDB 4.4 s, pandas 11.3 s at 100x, with identical outputs. Parquet is 11.3 bytes/row, 3.8x smaller than CSV. |

## Why these choices

* **Digital twin, validated against its own theory.** Aggregating ON/OFF sources with Pareto durations (shape 1 < a < 2)
  gives long-range-dependent traffic with Hurst exponent H = (3 - a_min)/2. Measured H is 0.73 to 0.82 against
  theoretical 0.80 to 0.90 (iid noise gives ~0.5): clearly self-similar, but the estimator reads low on a 4-hour series.
* **Onset, not persistence.** Cells already congested are dropped; the task is forecasting the transition.
* **Held-out districts differ on purpose** (different source counts, rates, tail indices, utilisation), so the
  geographic test is a real distribution shift.
* **Leakage guards.** Labels look strictly forward, features strictly backward; a unit test compares both against a
  naive implementation. A purge gap separates train and validation in time.
* **Alert policy built on probabilities**, with costs read from config, so the operator's trade-off is an input.

## Run it

```bash
pip install -e ".[dev]"          # add ".[duckdb]" to include the DuckDB engine in the benchmark
tailwatch all                    # simulate -> hurst -> experiments -> bench -> radar -> report (~15 min)
pytest                           # 11 tests
```

Steps can run alone: `tailwatch simulate | hurst | experiments | bench | radar | report`. Every parameter lives in
[`configs/default.yaml`](configs/default.yaml) and is type-checked at load (`--config other.yaml` to override).
Artefacts go to `data/` (Parquet) and `results/` (JSON, PNG, `summary.md`, `radar.html`).

Open `results/radar.html` (self-contained) for the animated hex-map radar of a surge: colour is forecast probability,
a white ring marks a cell where congestion really began within the horizon.

## Layout

```
configs/default.yaml   all tunables (cities, events, thresholds, models, costs, benchmark)
src/tailwatch/
  config.py            typed, validated config; no code-side defaults
  sim.py               digital twin (hex grid, ON/OFF sources, surges)      hurst.py   generator check
  features.py          Polars features + forward labels                     store.py   Parquet / JSON I/O
  core.py              folds, resampling, calibrators, metrics, block bootstrap
  experiments.py       out-of-district predictions; imbalance, calibration, alert policy
  generalization.py    four split protocols        probabilistic.py   quantile + conformal intervals
  selection.py         MI / L1 / tree / PCA        scale.py           Parquet engine benchmark
  radar.py plots.py report.py cli.py
tests/                 leakage, determinism, conformal coverage, calibration, config validation
docs/BLUEPRINT.md      design decisions and failure handling
```

## What is not measured

* **Real traffic.** No real operator data; results describe this generator. Real 4G/5G traces (e.g. public
  measurement datasets, licences to be checked) would need a loader and a re-run of every table.
* **Radio effects.** Signal strength is a function of load plus noise, not a propagation model; handovers, mobility and
  interference between cells are not simulated (neighbour utilisation is the only spatial coupling).
* **Millisecond dynamics.** Resolution is 1 s; sub-second bursts are out of scope.
* **Causal claims** about features. Importances describe predictive use in this simulator.
* **Calibration under shift.** Calibrators are fitted on other districts' later data and only checked on a new
  district; the 12 cells per district give wide per-district uncertainty, so per-district numbers are indicative.
* **Pooled vs per-district metrics.** Pooled PR-AUC mixes districts with 1% to 13% storm rates and depends on the
  per-district calibration maps; per-district values are in `results/summary.md`. The "unseen district + future" row has a
  2% storm rate and a wide CI, so its PR-AUC is not comparable with the others.
* **Coverage guarantees.** Conformal coverage assumes exchangeability; held-out districts violate it, which is what the
  per-district coverage spread shows.
* **Single seed** for the simulated city and the models; CIs reflect resampling of one realisation, not
  re-simulation.
* **Scale test** replicates cells with new ids; it measures engine throughput on one machine, not modelling behaviour at
  scale.
