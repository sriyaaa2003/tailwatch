# tailwatch: calibrated forecasts of rare radio-network events

Rare, bursty events in a cellular network are best treated like severe weather: not a yes/no label but a **calibrated probability**
("4% chance the channel collapses within 10 s") and a **prediction interval** for the coming peak, then the questions a forecaster asks.
When it says 30%, is it right 30% of the time? Does that hold on conditions it has never seen? What does an alert policy built on it cost?

One pipeline, **two real data sets, no simulation**:

| data set | what it is | event (held-out groups) |
|---|---|---|
| [msData](https://huggingface.co/datasets/subinak/Open_RAN_Performance_Measurement_Dataset_with_Traffic_and_Mobility_Labels) | Open RAN 5G testbed, 3.2 M per-UE records, one cell, 44.6 h of cell time | cell load reaches 10 Mbit/s within 10 s (whole mobility patterns held out: car, bus, train, static, pedestrian) |
| [client-side 5G traces](https://github.com/uccmisl/5Gdataset) (Raca et al., MMSys 2020) | 87 sessions, 55.6 h of 1 Hz phone KPIs (RSRP, RSRQ, SNR, CQI, serving cell, rate) on an Irish operator, in a car or at rest, streaming or downloading | the channel (CQI) collapses to 5 or less within 10 s, from a good state (whole mobility / app pairs held out) |

Features look strictly backward over 5, 15 and 30 s windows and labels strictly forward (tested against a naive reimplementation); every
split holds out whole groups; intervals are 95% block bootstraps. The msData burst level is a configured threshold, not a measured
capacity.

## What replicates on both

| finding | msData | 5G traces |
|---|---|---|
| rare-event rate | 4.0% | 3.8% |
| PR-AUC on unseen groups (none / class weights / SMOTE / undersampling) | 0.32 / 0.33 / 0.33 / 0.35, intervals overlap | 0.13 / 0.12 / 0.13 / 0.15, intervals overlap |
| **class weights make the model over-predict**; recalibration repairs it (ECE raw → isotonic) | 0.065 → 0.006 | 0.091 → 0.012 |
| calibrated probabilities need no threshold tuning (alert cost per 1000 decisions, raw → calibrated) | 416 → 376 (tuned 376) | 643 → 512 (tuned 510) |
| **a random split flatters the model** (PR-AUC random rows vs unseen group) | 0.51 vs 0.32 (+59%) | 0.27 vs 0.11 (**+134%**) |
| **80% intervals look right on average but miss the event**: coverage overall / when the event occurs | 0.78 / 0.25 | 0.81 / **0.09** |
| six features match all features? | no (0.29 to 0.30 vs 0.33) | mostly no (0.09 to 0.11 vs 0.12; PCA 0.13) |

Reading it plainly: the models rank events only modestly (PR-AUC about 8 times the base rate on msData and 3 times on the phone
traces, a hard 10-second-ahead problem), so **calibration, honest splits and interval coverage are where the value is**. Random
splitting inflates results by 59% to 134%, and intervals that cover 80% of all rows cover under 25% of the rows that matter. Full tables
with intervals: [`results/msdata/summary.md`](results/msdata/summary.md), [`results/fiveg/summary.md`](results/fiveg/summary.md) and the
side-by-side [`results/comparison.md`](results/comparison.md).

msData's cell load is long-range dependent (Hurst exponent 0.88 [IQR 0.83, 0.93]; iid noise gives 0.57), coefficient of variation 0.95,
which is why short horizons are hard.

## Run

```bash
pip install -e .[dev]
tailwatch run msdata     # download (794 MB), prepare, burstiness, experiments, report
tailwatch run fiveg      # download (2 MB), prepare, experiments, report
tailwatch compare        # results/comparison.md
pytest                   # 16 tests: labels and features against naive forward/backward windows, fold leakage, calibration, conformal coverage
```

Every parameter is in `configs/default.yaml` and type-checked on load; each step reads its inputs from disk and can be re-run alone, runs are
seeded, and every number in the reports is generated from the run outputs. The experiments take about 20 to 40 minutes per data set on a
laptop CPU.

Limits: one testbed cell and one operator's phones; msData records are about 100 ms apart, not 1 ms; the 5G data has two mobility patterns
and three apps. Data: msData (MIT / CC BY 4.0, see its card); Raca et al., "Beyond Throughput, the Next Generation: a 5G Dataset with
Channel and Context Metrics", MMSys 2020. Code: MIT.
