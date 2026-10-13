# 70 — Re-run with fixed precision: original vs fixed

Fixed run: `eval/results_fixed.csv` (sweep labels `*_fix`; extra columns `req_input_scale`, `req_param_scale`,
`req_decomp_legs`, `out_nonzero_frac`), settings in `eval/settings/*_fix_*_in12_p25_l3.json`, runs in `eval/runs/`.
Scales: **input_scale 12, param_scale 25, decomp_legs 3** (output scale 49), chosen in `60_precision_fix.md`. Same
harness (`bench_zkp.py` via `bench_zkp_fixed.py`), same machine (M4 Pro, 24 GB), same budgets (setup 30 min, online
10 min), same logrows rule (min valid = max(10, ceil(log2(d+6)))). Original: `eval/results.csv` (unchanged).
All 18 fixed configs: status ok, verify True on all reps, no timeouts/budget hits/OOM. Total wall ≈ 5 min.

Cells are "original → fixed". prove = median of 3 (fixed min–max in brackets), verify = median of 3.
pk is **byte-identical** in every config (vk too), so a single value is shown. logrows used = requested in all rows.

| sweep | K | d | logrows | num_rows | pk MB | prove s | verify ms | peak RSS GB | rel L2 err |
|---|---|---|---|---|---|---|---|---|---|
| S1 | 2 | 100 | 10 | 450 → 460 | 39.9 | 0.46 → 0.46 [0.46–0.48] | 11 → 11 | 0.50 → 0.51 | 100 % → 0.147 % |
| S1 | 2 | 1,000 | 10 | 4,050 → 4,062 | 76.7 | 1.62 → 1.73 [1.73–1.77] | 36 → 36 | 1.05 → 1.06 | 100 % → 0.136 % |
| S1 | 2 | 5,000 | 13 | 20,050 → 20,060 | 126.9 | 1.03 → 1.06 [1.05–1.10] | 26 → 27 | 0.80 → 0.86 | 100 % → 0.141 % |
| S1 | 2 | 10,000 | 14 | 40,050 → 40,060 | 264.4 | 1.88 → 1.86 [1.85–1.90] | 47 → 49 | 1.22 → 1.21 | 100 % → 0.143 % |
| S1 | 2 | 20,000 | 15 | 80,050 → 80,060 | 327.4 | 2.87 → 2.93 [2.88–3.01] | 87 → 91 | 1.29 → 1.28 | 100 % → 0.142 % |
| S1 | 2 | 40,000 | 16 | 160,050 → 160,060 | 654.8 | 5.45 → 5.77 [5.41–5.80] | 172 → 177 | 2.57 → 2.54 | 100 % → 0.141 % |
| S1 | 2 | **62,006** | 16 | 248,074 → 248,085 | **843.6** | **7.48 → 7.48** [7.44–8.15] | 269 → 276 | 3.08 → 2.85 | 100 % → 0.141 % |
| S2 | 1 | 5,000 | 13 | 10,048 → 10,058 | 92.8 | 0.49 → 0.50 [0.48–0.56] | 24 → 25 | 0.69 → 0.71 | 100 % → 0.142 % |
| S2 | 2 | 5,000 | 13 | 20,050 → 20,060 | 126.9 | 1.05 → 1.04 [1.04–1.06] | 26 → 27 | 0.83 → 0.86 | 100 % → 0.141 % |
| S2 | 4 | 5,000 | 13 | 35,054 → 35,064 | 195.1 | 1.60 → 1.58 [1.58–1.61] | 31 → 32 | 1.15 → 1.20 | 100 % → 0.197 % |
| S2 | 8 | 5,000 | 13 | 65,060 → 65,072 | 297.4 | 2.28 → 2.27 [2.26–2.36] | 41 → 41 | 1.42 → 1.42 | 100 % → 0.199 % |
| S2 | 16 | 5,000 | 13 | 125,073 → 125,079 | 570.2 | 3.93 → 4.05 [3.99–4.06] | 86 → 88 | 2.88 → 2.71 | 100 % → 0.198 % |
| S0 | 2 | 1,000 | 10 (fixed) | 4,052 → 4,061 | 76.7 | 1.61 → 1.61 [1.59–1.61] | 35 → 36 | 1.06 → 1.07 | 100 % → 0.136 % |
| S3 | 2 | 10,000 | 14 | 40,050 → 40,060 | 264.4 | 1.84 → 1.86 [1.85–1.86] | 48 → 49 | 1.23 → 1.23 | 100 % → 0.143 % |
| S3 | 2 | 10,000 | 16 | 40,050 → 40,060 | 277.0 | 1.83 → 1.84 [1.81–1.84] | 49 → 50 | 1.11 → 1.11 | 100 % → 0.143 % |
| S3 | 2 | 10,000 | 18 | 40,050 → 40,060 | 1,107.9 | 4.51 → 4.63 [4.50–4.69] | 85 → 85 | 3.43 → 3.29 | 100 % → 0.143 % |
| S3 | 2 | 62,006 | 18 | 248,074 → 248,084 | 1,107.9 | 8.60 → 8.84 [8.50–9.11] | 280 → 285 | 3.66 → 3.55 | 100 % → 0.141 % |
| S3 | 2 | 62,006 | **20** | 248,074 → 248,084 | **4,431.5** | **21.3 → 20.8** [20.8–21.3] | 449 → 461 | 8.18 → 8.46 | 100 % → 0.141 % |

Fixed-run error detail: max abs error 0.9–1.8·10^-4 (true |avg| up to ~0.2); 99.8–100 % of outputs nonzero (the rest
are true values that round to 0 at scale 12). Setup times are unchanged within ±4 % (e.g. 11.8 → 11.8 s for the paper
config at logrows 16, 22.8 → 22.6 s at 20).

## Verdict
- **The fix costs essentially nothing.** num_rows +6…12 (the decomp_legs=3 Recip gadget; +0.05 % at d=5k, +0.004 % at
  62k), no change in minimal logrows, pk and vk byte-identical, prove/verify/setup/RSS within run-to-run noise
  (largest prove deltas: +7 % at d=1000/lr10, +6 % at d=40k, −2 % at lr20; S0 d=1000 repeat shows +0 %, so these are
  noise). The cost numbers in `90_summary.md` stand; the proofs now certify the correct average (rel. L2 0.14–0.20 %).
- **Rows linear:** still exactly linear in d (fit: rows = 4.000·d + 60 for K=2) and linear in K (K 1→16: ×12.4).
- **pk/prove step-like:** unchanged: pk depends only on logrows in the single-block regime (S3: d=10k and d=62k both
  1,107.9 MB at lr18; 4,431.5 MB at lr20), prove jumps at logrows boundaries.
- **5 GB at logrows 20:** reproduces identically with the fix (pk 4.43 GB, prove 20.8 s, peak RSS 8.5 GB). At minimal
  logrows 16 the paper config is 0.84 GB / 7.5 s.
- **Prove time:** "several minutes" still does not reproduce on this machine (≤ 21 s).
- **Surprises / caveats:**
  1. The repo's defaults are not just imprecise but would also **fail at witness time** after a naive scale fix if
     clients hold more data: with decomp_legs 2, Σn·2^input_scale must stay ≲ 2.7·10^8 (input 12 fails at K=4 with
     25k examples each; input 10 at K=16). Keygen succeeds, so this only surfaces during the first proving round.
  2. Proof files grow ~13 % (e.g. 8.7 → 9.9 MB at 62k) only because the JSON `pretty_public_inputs` strings of the
     outputs are longer (nonzero values instead of "0.0") [inferred; raw halo2 proof bytes not re-measured separately].
  3. Error is slightly higher at K ≥ 4 (0.20 % vs 0.14 %): the averaged outputs are smaller (averaging more
     independent weights), so the fixed 2^-12 input quantisation is relatively larger.
