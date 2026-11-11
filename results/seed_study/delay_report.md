# Label delay and end-to-end timing (third review)

## 1. Reference substrate, shifting-axis stream: balanced accuracy vs label delay

Five seeds per cell (mean +/- sd). Paired differences are seed-matched.

| Method | delay 0 | delay 50 | delay 250 | delay 1000 |
|---|---|---|---|---|
| Sliding window (recency) | 0.827 +/- 0.024 | 0.817 +/- 0.023 | 0.776 +/- 0.016 | 0.643 +/- 0.009 |
| Feature similarity | 0.803 +/- 0.015 | 0.794 +/- 0.015 | 0.754 +/- 0.019 | 0.637 +/- 0.037 |
| Class ratio (stratified) | 0.803 +/- 0.021 | 0.793 +/- 0.022 | 0.758 +/- 0.013 | 0.643 +/- 0.007 |
| Learned policy | 0.812 +/- 0.020 | 0.783 +/- 0.018 | 0.755 +/- 0.006 | 0.645 +/- 0.027 |
| Anchored policy (recency vertex) | 0.828 +/- 0.024 | 0.818 +/- 0.023 | 0.777 +/- 0.015 | 0.645 +/- 0.010 |

| Delay | anchored - sliding window | learned policy - best fixed (per seed) | best fixed rule |
|---|---|---|---|
| 0 | +0.001 [0.001, 0.002], 5/5 seeds, t p=0.002 | -0.015 [-0.038, 0.010], 1/5 seeds, t p=0.306 | Sliding window (recency) |
| 50 | +0.002 [0.001, 0.002], 5/5 seeds, t p=0.000 | -0.034 [-0.062, -0.006], 0/5 seeds, t p=0.109 | Sliding window (recency) |
| 250 | +0.002 [0.001, 0.002], 5/5 seeds, t p=0.012 | -0.021 [-0.032, -0.011], 0/5 seeds, t p=0.030 | Sliding window (recency) |
| 1000 | +0.002 [0.000, 0.003], 4/5 seeds, t p=0.060 | -0.010 [-0.017, -0.004], 1/5 seeds, t p=0.050 | Class ratio (stratified) |

## 2. TabPFN, Electricity: delay 0 vs delay 250

Delay-0 values are the existing seed-study rows (fixed selectors seed 0; policy and anchored policy: the seeds that also exist at delay 250). Deterministic selectors: one seed.

| Method | delay 0 | delay 250 | change | seeds |
|---|---|---|---|---|
| Sliding window (recency) | 0.9582 | 0.7252 | -0.2330 | 1 |
| Feature similarity | 0.9586 | 0.7281 | -0.2304 | 1 |
| Class ratio (stratified) | 0.9593 | 0.7379 | -0.2214 | 1 |
| Learned policy | 0.9526 | 0.7250 | -0.2276 | 3 |
| Anchored policy (recency vertex) | 0.9580 | 0.7272 | -0.2308 | 3 |

At delay 250: anchored policy 0.7272 +/- 0.0015 (3 seeds) vs sliding window 0.7252; learned policy 0.7250 +/- 0.0082 (3 seeds).

## 3. End-to-end per-instance timings, TabPFN on Electricity (same procedure for every stage)

Wall time per scored instance measured with `time.perf_counter` around each stage inside the prequential loop of the same run: context selection, frozen-model inference, selector update, and their sum (step). Milliseconds.

| Method | block | stage | mean | p50 | p95 | p99 | max |
|---|---|---|---|---|---|---|---|
| Anchored policy (recency vertex) | D_tab_delay | select | 0.67 | 0.81 | 0.85 | 0.89 | 1.4 |
| Anchored policy (recency vertex) | D_tab_delay | infer | 500.98 | 497.14 | 528.26 | 534.63 | 609.9 |
| Anchored policy (recency vertex) | D_tab_delay | update | 0.02 | 0.02 | 0.02 | 0.02 | 0.0 |
| Anchored policy (recency vertex) | D_tab_delay | step | 501.66 | 497.85 | 528.91 | 535.38 | 610.7 |
| Anchored policy (recency vertex) | D_tab_timing | select | 0.67 | 0.80 | 0.84 | 0.88 | 1.5 |
| Anchored policy (recency vertex) | D_tab_timing | infer | 498.08 | 494.30 | 525.73 | 533.02 | 589.1 |
| Anchored policy (recency vertex) | D_tab_timing | update | 0.04 | 0.04 | 0.04 | 0.05 | 0.2 |
| Anchored policy (recency vertex) | D_tab_timing | step | 498.79 | 495.05 | 526.40 | 533.81 | 590.0 |
| Class ratio (stratified) | D_tab_delay | select | 0.23 | 0.28 | 0.31 | 0.32 | 0.6 |
| Class ratio (stratified) | D_tab_delay | infer | 503.21 | 498.29 | 534.12 | 555.30 | 1008.7 |
| Class ratio (stratified) | D_tab_delay | step | 503.44 | 498.53 | 534.32 | 555.59 | 1009.0 |
| Class ratio (stratified) | D_tab_timing | select | 0.24 | 0.28 | 0.31 | 0.32 | 0.5 |
| Class ratio (stratified) | D_tab_timing | infer | 502.67 | 497.96 | 530.13 | 539.41 | 1429.9 |
| Class ratio (stratified) | D_tab_timing | step | 502.91 | 498.22 | 530.40 | 539.67 | 1430.2 |
| Learned policy | D_tab_delay | select | 0.66 | 0.74 | 0.86 | 0.89 | 1.8 |
| Learned policy | D_tab_delay | infer | 457.65 | 496.40 | 530.54 | 541.88 | 755.2 |
| Learned policy | D_tab_delay | update | 0.02 | 0.02 | 0.02 | 0.02 | 0.2 |
| Learned policy | D_tab_delay | step | 458.31 | 497.05 | 531.29 | 542.66 | 756.4 |
| Learned policy | D_tab_timing | select | 0.65 | 0.76 | 0.81 | 0.85 | 1.4 |
| Learned policy | D_tab_timing | infer | 482.87 | 493.43 | 524.85 | 532.06 | 579.8 |
| Learned policy | D_tab_timing | update | 0.04 | 0.04 | 0.04 | 0.05 | 0.2 |
| Learned policy | D_tab_timing | step | 483.55 | 494.09 | 525.50 | 532.75 | 580.4 |
| Feature similarity | D_tab_delay | select | 0.51 | 0.59 | 0.67 | 0.72 | 1.8 |
| Feature similarity | D_tab_delay | infer | 507.71 | 502.49 | 542.20 | 573.59 | 747.2 |
| Feature similarity | D_tab_delay | step | 508.22 | 502.99 | 542.76 | 574.10 | 747.4 |
| Feature similarity | D_tab_timing | select | 0.49 | 0.58 | 0.61 | 0.64 | 1.5 |
| Feature similarity | D_tab_timing | infer | 499.57 | 493.94 | 529.23 | 555.95 | 661.8 |
| Feature similarity | D_tab_timing | step | 500.06 | 494.42 | 529.62 | 556.12 | 662.0 |
| Sliding window (recency) | D_tab_delay | select | 0.20 | 0.24 | 0.26 | 0.27 | 0.5 |
| Sliding window (recency) | D_tab_delay | infer | 501.93 | 497.85 | 531.17 | 541.43 | 1794.3 |
| Sliding window (recency) | D_tab_delay | step | 502.13 | 498.05 | 531.33 | 541.64 | 1794.3 |
| Sliding window (recency) | D_tab_timing | select | 0.20 | 0.25 | 0.26 | 0.27 | 0.4 |
| Sliding window (recency) | D_tab_timing | infer | 498.76 | 494.86 | 526.44 | 533.63 | 610.9 |
| Sliding window (recency) | D_tab_timing | step | 498.96 | 495.03 | 526.63 | 533.80 | 611.2 |

Share of the end-to-end mean: selection Anchored policy (recency vertex) 0.13%, Anchored policy (recency vertex) 0.13%, Class ratio (stratified) 0.05%, Class ratio (stratified) 0.05%, Learned policy 0.14%, Learned policy 0.13%, Feature similarity 0.10%, Feature similarity 0.10%, Sliding window (recency) 0.04%, Sliding window (recency) 0.04%; inference Anchored policy (recency vertex) 99.9%, Anchored policy (recency vertex) 99.9%, Class ratio (stratified) 100.0%, Class ratio (stratified) 100.0%, Learned policy 99.9%, Learned policy 99.9%, Feature similarity 99.9%, Feature similarity 99.9%, Sliding window (recency) 100.0%, Sliding window (recency) 100.0%.

Reading the tails: the p99 and max of the end-to-end step coincide with the inference stage's tails, not with selection; selection's own maximum stays within a few milliseconds. The second-long outliers therefore come from the frozen model's forward pass (first-call warm-up, occasional context re-tokenisation), under the identical timing procedure.
