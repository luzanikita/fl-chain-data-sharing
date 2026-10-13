# 80 — Extended fixed-precision runs (larger d, logrows-20 series, K=32)

Same harness as `70_fixed_results.md` (`bench_zkp_fixed.py`, input_scale 12, param_scale 25, decomp_legs 3), same
machine (M4 Pro, 24 GB), same budgets (setup 30 min, online 10 min, RSS kill 20 GiB), pk deleted after each config.
9 new rows appended to `eval/results_fixed.csv` (sweeps `S1min_fix`, `S1lr20_fix`, `S2_fix` K=32); existing rows
untouched. Runner: one sequential background job, ~12 min wall total. Disk 91 → 90 GB free.

**All 9 configs: status ok, verify True on all 3 reps, no timeout / budget / OOM.** logrows used = requested.

| sweep | K | d | logrows | num_rows | pk MB | setup s | prove s med [min–max] | verify ms | peak RSS GB | proof MB | rel L2 err |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S1min_fix | 2 | 124,012 | 17 | 496,108 | 1,687.3 | 23.9 | 15.57 [14.76–15.87] | 544 | 5.20 | 19.8 | 0.142 % |
| S1min_fix | 2 | 250,000 | 18 | 1,000,062 | 3,374.6 | 46.6 | 29.28 [29.20–32.10] | 1,115 | 8.75 | 39.8 | 0.141 % |
| S1lr20_fix | 2 | 100 | 20 | 460 | 4,431.5 | 12.4 | 16.67 [16.50–16.92] | 195 | 9.91 | 0.08 | 0.147 % |
| S1lr20_fix | 2 | 1,000 | 20 | 4,060 | 4,431.5 | 11.7 | 16.50 [16.28–16.74] | 200 | 10.40 | 0.2 | 0.136 % |
| S1lr20_fix | 2 | 5,000 | 20 | 20,060 | 4,431.5 | 12.2 | 16.75 [16.72–16.79] | 215 | 10.32 | 0.8 | 0.141 % |
| S1lr20_fix | 2 | 10,000 | 20 | 40,060 | 4,431.5 | 12.9 | 17.15 [17.14–17.30] | 229 | 10.27 | 1.6 | 0.143 % |
| S1lr20_fix | 2 | 20,000 | 20 | 80,060 | 4,431.5 | 14.5 | 17.86 [17.69–18.56] | 273 | 9.66 | 3.2 | 0.142 % |
| S1lr20_fix | 2 | 40,000 | 20 | 160,060 | 4,431.5 | 17.8 | 19.23 [19.05–19.38] | 348 | 10.18 | 6.4 | 0.141 % |
| S2_fix | 32 | 5,000 | 13 | 245,100 | 1,047.6 | 5.1 | 7.22 [7.21–7.25] | 226 | 5.13 | 1.3 | 0.197 % |

(peak RSS = ru_maxrss of the worker. proof MB = proof JSON file incl. pretty public outputs; the halo2 proof bytes
themselves are small, see 90_summary.md. `num_inner_cols` is reported as 2 from run_args for every row; when
num_rows > 2^logrows EZKL widens the layout internally, not reflected in that field.)

## 1. Rows stay exactly linear up to 250k
Fit over all 9 min-valid S1 fixed points (S1_fix d = 100…62,006 plus S1min_fix 124,012 and 250,000):
**num_rows = 4.000·d + 60** (slope 4.000005, max residual 1.7 rows; rows − 4d ∈ {60, 61, 62} at every point).
No change in shape at 124k/250k.

pk on the min-valid rule keeps doubling with each logrows step: lr16 843.6 MB (62k) → lr17 1,687 MB (124k) →
lr18 3,375 MB (250k), i.e. here pk ≈ 13.6 kB per d (both 124k and 250k sit just under their power-of-2 boundary, so
the step function happens to look linear at these points). Prove: 7.48 s → 15.6 s → 29.3 s (≈ ×2 per doubling of d),
setup 11.8 → 23.9 → 46.6 s, verify 276 → 544 → 1,115 ms, RSS 2.9 → 5.2 → 8.8 GB.
At d = 250k the min-valid pk (3.37 GB) is still below the lr20 pk (4.43 GB).

**Largest feasible d ≤ 2^20 − 6 not attempted** (judged not cheap): it would need logrows 20 with ~4.2 M rows; RSS
grew ×1.7 per logrows step (5.2 → 8.75 GB), extrapolating to ~25 GB at lr20 with 4M rows, above both the 20 GiB kill
threshold and the 24 GB physical RAM. [inference from extrapolation, not run]

## 2. logrows 20: pk flat, prove nearly flat
- **pk is byte-identical (4,431,545,883 B) for every d** from 100 to 62,006 (the S3_fix lr20 row too), vk likewise
  constant in size. The 4.4 GB key is purely a function of logrows=20.
- **Prove time at lr20 has a large fixed floor plus a small d-dependent term:** 16.7 s at d = 100 → 19.2 s at 40k →
  20.8 s at 62k (S3_fix). Linear fit over the 7 lr20 points: **t_prove ≈ 16.5 s + 0.069 ms·d** — the floor
  (FFTs/MSMs over 2^20 rows) is ~80 % of the cost even at the paper's d = 62k; d = 100 costs 80 % of d = 62k.
- Setup grows mildly (11.7 → 22.6 s from d=1k to 62k), verify 195 → 461 ms (instance-column/public-output work
  scales with d), peak RSS ~9.7–10.4 GB irrespective of d (8.5 GB at 62k in S3_fix).
- Contrast: at the min-valid logrows the same d=100 config proves in 0.46 s with a 40 MB pk; lr20 makes it ~36× slower
  and ~110× larger.

## 3. K = 32 vs the K trend (d = 5000, logrows 13 throughout)
| K | num_rows | pk MB | prove s | K·t(K=1) s (sequential est.) |
|---|---|---|---|---|
| 1 | 10,058 | 92.8 | 0.50 | 0.50 |
| 2 | 20,060 | 126.9 | 1.04 | 0.99 |
| 4 | 35,064 | 195.1 | 1.58 | 1.99 |
| 8 | 65,072 | 297.4 | 2.27 | 3.97 |
| 16 | 125,079 | 570.2 | 4.05 | 7.95 |
| **32** | **245,100** | **1,047.6** | **7.22** | **15.90** |

- Rows remain linear in K: from K=2 on, each extra client adds 7,500 rows (= 1.5·d) — K 16→32 adds exactly
  120,021 rows (K=1→2 adds 2·d because K=1 is a special case). K 1→32: rows ×24.4, pk ×11.3, prove ×14.5;
  verify 25 → 226 ms; RSS 0.7 → 5.1 GB; error 0.197 % (same as K ≥ 4).
- pk and prove grow sub-linearly in K at fixed logrows 13 (the extra rows go into more columns): K 16→32 gives
  pk ×1.84, prove ×1.78.
- **Sequential estimate vs monolithic:** K·t(K=1) = 32 × 0.50 s = 15.9 s vs **7.2 s measured monolithic** — the
  monolithic circuit is ~2.2× faster in total prover time at K=32 (and already faster from K=4). The sequential
  scheme's advantage is memory/key size: pk(K=1) = **92.8 MB** vs **1,047.6 MB** monolithic (11.3× smaller), and the
  key does not grow with K. [estimate, sequential scheme not implemented]

## Failures
None. The optional "largest d ≤ 2^20−6" run was skipped (see §1).
