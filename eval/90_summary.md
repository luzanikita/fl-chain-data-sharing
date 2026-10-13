# 90 — ZKP scaling experiment: summary (2026-10-10)

Machine: Apple M4 Pro (8P+4E), 24 GB, macOS 26.6.2. ezkl **15.6.3 (= pinned)**, torch 2.2.2, Python 3.11.14 (`environment.json`).
Code under test: `fl/src/zkp.py` unmodified (`AggregationSetup(AggregateModel())`, `AggregationProver`, `AggregationVerifier`).
Inputs: synthetic `rng(0).normal(0,0.05,(K,d))`, 25,000 examples/client. Raw data: `results.csv`, `settings/*.json`, `runs/<cfg>/` (vk, proof, result.json, log). Notes: `10_env_setup.md`, `15_srs.md`, `20_smoke_test.md`, `25_logrows_finding.md`, `30_harness.md`, `40_results_small.md`.
All 20 configurations finished in ~5 min total; nothing hit the setup (30 min) or online (10 min) budgets; no OOM.

## Results (prove = median [min–max] of 3; verify median of 3, all verify=True)

| sweep | K | d | logrows | num_rows | t_setup s | t_witness s | prove s | verify ms | pk MB | vk kB | proof file kB (raw proof B) | peak RSS GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S0 | 2 | 5000 | 10 | 20053 | — | — | **setup error** | — | — | — | — | — |
| S0 | 2 | 10000 | 10 | 40061 | — | — | **setup error** | — | — | — | — | — |
| S1 | 2 | 100 | 10 | 450 | 0.15 | 0.003 | 0.46 [0.46–0.47] | 11 | 39.9 | 4.4 | 76 | 0.50 |
| S1 | 2 | 1000 | 10 | 4050 | 0.44 | 0.012 | 1.62 [1.58–1.67] | 36 | 76.7 | 13.4 | 345 (37,024) | 1.05 |
| S1 | 2 | 5000 | 13 | 20050 | 1.12 | 0.046 | 1.03 [1.02–1.10] | 26 | 126.9 | 58.2 | 751 (9,216) | 0.80 |
| S1 | 2 | 10000 | 14 | 40050 | 2.21 | 0.087 | 1.88 [1.84–1.91] | 48 | 264.4 | 113.7 | 1,448 | 1.22 |
| S1 | 2 | 20000 | 15 | 80050 | 4.00 | 0.175 | 2.87 [2.82–2.90] | 87 | 327.4 | 224.4 | 2,842 | 1.29 |
| S1 | 2 | 40000 | 16 | 160050 | 8.07 | 0.361 | 5.45 [5.39–5.86] | 172 | 654.8 | 445.6 | 5,642 | 2.57 |
| S1 | 2 | **62006** | 16 | 248074 | 11.8 | 0.545 | **7.48 [7.35–7.97]** | 269 | **843.6** | 594.0 | 8,736 (9,984) | 3.08 |
| S2 | 1 | 5000 | 13 | 10048 | 0.30 | 0.031 | 0.49 [0.48–0.53] | 24 | 92.8 | 39.0 | 736 | 0.69 |
| S2 | 2 | 5000 | 13 | 20050 | 1.12 | 0.047 | 1.05 [1.04–1.10] | 26 | 126.9 | 58.2 | 751 | 0.83 |
| S2 | 4 | 5000 | 13 | 35054 | 1.59 | 0.073 | 1.60 [1.58–1.66] | 31 | 195.1 | 96.8 | 782 | 1.15 |
| S2 | 8 | 5000 | 13 | 65060 | 2.05 | 0.131 | 2.28 [2.24–2.33] | 41 | 297.4 | 154.6 | 827 | 1.42 |
| S2 | 16 | 5000 | 13 | 125073 | 3.12 | 0.243 | 3.93 [3.93–4.09] | 86 | 570.2 | 308.7 | 947 | 2.88 |
| S3 | 2 | 10000 | 14 | 40050 | 2.18 | 0.088 | 1.84 [1.84–1.89] | 48 | 264.4 | 113.7 | 1,448 | 1.23 |
| S3 | 2 | 10000 | 16 | 40050 | 2.32 | 0.088 | 1.83 [1.82–1.84] | 49 | 277.0 | 148.7 | 1,417 | 1.11 |
| S3 | 2 | 10000 | 18 | 40050 | 4.42 | 0.087 | 4.51 [4.45–4.60] | 85 | 1,107.9 | 591.1 | 1,417 | 3.43 |
| S3+ | 2 | 62006 | 18 | 248074 | 12.5 | 0.534 | 8.60 [8.53–8.79] | 280 | 1,107.9 | 591.1 | 8,698 | 3.66 |
| S3+ | 2 | 62006 | **20** | 248074 | 22.8 | 0.600 | **21.3 [21.1–21.4]** | 450 | **4,431.5** | 2,360.6 | 8,698 (3,072) | 8.18 |
| smoke | 2 | 6 | 20 | 74 | ~ | | | | **4,431.5** | 2,360.6 | 18 | |

(S0 d=100/1000 at logrows 10 duplicate S1 rows. t_setup ≈ t_keygen: export+gen_settings+compile < 0.2 s. Plain NumPy average is 2–30 µs → prove overhead ~2×10^5–7×10^5.)
"S3+" = extra: paper config at logrows 18/20 (added because cheap and directly relevant to the 5 GB claim).

## Key findings / surprises
1. **logrows is never bumped by ezkl.** At logrows=10 it widens the circuit (d=1000: 4,050 rows > 1,024, still works) but **d ≥ ~1,018 fails at keygen** (`[halo2] k = 10 is too small for the given circuit` → `Failed to run setup: [halo2] General synthesis error`): all d public outputs share one instance column. The repo default logrows=10 can't run the paper's d=62,006; min valid is 16. Sweeps used logrows = max(10, ceil(log2(d+6))) (CSV `logrows_rule`).
2. **5 GB key reproduced only via logrows=20**: pk = 4,431,545,883 B at logrows 20 for *both* the 6-param toy (`simple_experiment`, logrows=20) and d=62,006. When the circuit fits in one column block, pk size is set by 2^logrows, not by d or K. At the minimal logrows=16 the paper config has pk 0.84 GB. The 5 GB number probably came from a logrows=20 setup.
3. **"Several minutes" not reproduced** on M4 Pro: prove 7.5 s (logrows 16) – 21 s (logrows 20); setup 12–23 s; peak RAM 3–8 GB. The original machine may be slower; unknown.
4. **Shape**: num_rows is exactly linear (≈4d for K=2; ≈(1.5K+1)·d for d=5k). pk, setup, and prove are step-like/piecewise: they jump with logrows (powers of 2) and grow ~linearly in-between. Notable: d=1000@lr10 (wide, 4 blocks) proves slower (1.62 s) than d=5000@lr13 (1.03 s). Over K at fixed logrows 13, pk/prove grow sub-linearly to ~linearly (K 1→16: rows ×12.4, pk ×6.1, prove ×8.0).
5. **Layout (S3, d=10k)**: logrows 14→16 costs nothing extra (pk 264→277 MB, prove 1.84→1.83 s); 18 → pk 4.2×, prove 2.4×. Over-sized logrows dominates cost.
6. **Correctness: every circuit output is exactly 0.** Reciprocal of Σn=50,000 at scale 7 rounds to 0 → rel. error 100% (max abs err = max |true avg| ≈ 0.05–0.2). Proofs verify, but prove a zero model. Toy case (Σn=15) has 5.5 % error. Timing is value-independent so cost numbers stand. (Fix suggested by `50_ezkl_source_research.md`: raise param_scale/input_scale — not done, zkp.py unchanged.)
7. **Proof file grows with d** (76 kB → 8.7 MB) but the raw halo2 proof is 3–37 kB; the file size is the d public outputs serialized in JSON (`instances` + `pretty_public_inputs`). Verify time grows with d (11 → 270 ms at logrows 16; 450 ms at 20) — still cheap for vehicles. vk grows with logrows (0.6 MB at 18, 2.4 MB at 20).

## Infrastructure note
EZKL's public SRS bucket now returns HTTP 403, so `get_srs` failed for uncached sizes. Rebuilt kzg10–19 by downsizing the cached official kzg20 with PSE halo2 (`tools/srs_downsize`); verified byte-identical to official kzg15, and ezkl's hash check accepts them. Used via `EZKL_REPO_PATH=eval/ezkl_repo` (see `15_srs.md`). Nothing above logrows 20 is possible.

## Not run / estimates
- 124k: not run (as instructed). Est. at min logrows 17: pk ~1.7 GB, prove ~15 s, RAM ~6 GB.
- Sequential aggregation estimate (decision file): pk ≈ pk(K=1,d=5k)=93 MB, prove ≈ K·0.49 s (vs 3.93 s for K=16 at once → 7.8 s sequential: no gain here at d=5k).
- No real-weights check, no calibration, no hashed inputs (E5 skipped).

## Open questions for the user
1. What logrows produced the original 5 GB key? (Data says 20, which matches `simple_experiment`.) Which logrows should the paper report — 16 (minimal) or 20?
2. The all-zero output: fix scales in zkp.py (separate change) and re-measure? With fixed scales `num_rows` reportedly unchanged, so cost numbers probably still hold.
3. The "several minutes" claim: re-word to measured numbers on this hardware, or re-measure on the original machine?
4. The claim "linear" should become "rows grow linearly; key size and prove time grow in power-of-two steps of logrows."
