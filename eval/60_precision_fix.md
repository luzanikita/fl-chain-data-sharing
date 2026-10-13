# 60 — Precision fix (benchmark only) and scale choice

**Problem.** With the repo's run args (input_scale = param_scale = 7, scale_rebase_multiplier = 1000) the division
`weighted_sum / Σn` becomes `weighted_sum · Recip(Σn)`, and Recip's output scale is max(input_scale, param_scale) = 7:
round(2^7 / 50,000) = 0, so every output is exactly 0 (`50_ezkl_source_research.md` §3).

**Fix (benchmark only; `fl/src/zkp.py` untouched).** `eval/bench_zkp_fixed.py` wraps `bench_zkp.py` unchanged. In the
worker process it swaps `zkp.AggregationSetup` (module attribute) for a subclass whose `_setup_run_args` calls `super()`
and then sets `input_scale`, `param_scale` and, if requested, `decomp_legs` from env vars. Everything else (visibility,
scale_rebase_multiplier = 1000, logrows rule, timing, budgets, cleanup) is the same harness code.
Raw data: `eval/results_precision.csv`, `eval/settings/P_*.json`, `eval/runs/P_*`.

## Why param_scale = 25
`param_scale` only matters here through the Recip scale (the model has no weights or constants). The reciprocal's
relative error is at most 0.5·Σn / 2^s. For Σn = 50,000:
s = 7 → 0 (100 %); 20 → round(20.97) = 21 (0.14 %); 22 → 84 (0.14 %); 25 → 671 (0.007 %).
At s = 25 the reciprocal error is about 20× below the input quantisation error even at input_scale 12, so a larger
param_scale gains nothing. A smaller one (20–22) leaves a 0.14 % bias that matters at input_scale 12. [run, gen_witness
only: in 12/p 20 and in 12/p 22 → rel. L2 0.196 %, in 12/p 25 → 0.141 %.] Output scale = 2·input + 25 (≤ 49 bits),
far below the field size. param_scale changes num_rows by 0.

## Mini-study: K = 2, d = 5000, logrows 13 (min valid rule), 3 proves each
| input / param / decomp_legs | status | num_rows | logrows | pk MB | setup s | prove med [min–max] s | verify ms | peak RSS GB | rel L2 err | max abs err | max rel err (|ref|>1e-3) | nonzero outputs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 7 / 7 / 2 (repo default, reference) | ok | 20,050 | 13 | 126.9 | 1.18 | 1.08 [1.05–1.08] | 24 | 0.91 | **100 %** | 0.141 | 100 % | 0 % |
| 7 / 25 / 2 | ok | 20,050 | 13 | 126.9 | 1.18 | 1.11 [1.10–1.13] | 27 | 0.92 | 4.43 % | 3.9e-3 | 281 % | 94.6 % |
| 10 / 25 / 2 | ok | 20,050 | 13 | 126.9 | 1.18 | 1.17 [1.15–1.19] | 26 | 0.89 | 0.564 % | 4.9e-4 | 45 % | 99.4 % |
| 12 / 25 / 2 | ok | 20,050 | 13 | 126.9 | 1.13 | 1.07 [1.06–1.08] | 26 | 0.89 | **0.141 %** | 1.2e-4 | 9.6 % | 99.9 % |
| 10 / 25 / 3 | ok | 20,060 | 13 | 126.9 | 1.12 | 1.10 [1.08–1.11] | 26 | 0.92 | 0.564 % | 4.9e-4 | 45 % | 99.4 % |
| **12 / 25 / 3 (chosen)** | ok | 20,060 | 13 | 126.9 | 1.13 | 1.11 [1.06–1.12] | 27 | 0.92 | **0.141 %** | 1.2e-4 | 9.6 % | 99.9 % |
| K=16: 10 / 25 / 2 | **witness fails** | 125,068 | 13 | 570.2 | 3.14 | — | — | 1.35 | — | — | — | — |
| K=16: 12 / 25 / 2 | **witness fails** | 125,072 | 13 | 570.2 | 3.16 | — | — | 1.36 | — | — | — | — |
| K=16: 10 / 25 / 3 | ok | 125,080 | 13 | 570.2 | 3.06 | 4.09 [4.02–4.15] | 86 | 2.93 | 0.577 % | 2.8e-4 | 17 % | 99.8 % |
| K=16: 12 / 25 / 3 | ok | 125,079 | 13 | 570.2 | 3.18 | 4.02 [3.99–4.07] | 88 | 2.84 | 0.198 % | 9.2e-5 | 4.7 % | 100 % |

(Max rel. error is dominated by the few outputs just above |ref| = 1e-3; rel. L2 is the meaningful number. "max rel err"
> 100 % at input 7 = small weights rounded to the wrong side.)

## Do scales change rows / logrows / lookups?
- **No for num_rows, logrows, pk:** at fixed decomp_legs, num_rows, total_assignments, pk bytes and the minimal valid
  logrows are identical for input scale 7, 10, 12 and param scale 7…25. `required_lookups = []` and
  `required_range_checks = [[-1,1],[0,16383]]` in all cases (Recip is not a lookup; `lookup_range` unused).
  Prove/setup times differ only by run-to-run noise (±5 %).
- **But the decomposition range limits Σn·2^input_scale.** Recip's optimality check decomposes values in base 16,384
  with `decomp_legs` legs. With the default 2 legs, gen_witness fails (`decomposition error: integer 456261632 is too
  large to be represented by base 16384 and n 2` → "General synthesis error") when Σn·2^input_scale ≳ 2.7·10^8:
  input 12 fails for K ≥ 4 (Σn ≥ 100k), input 10 for K = 16, input 13 already at K = 2. Keygen succeeds; the failure is
  at witness time, i.e. a deployment with more/larger clients would fail at its first round. **decomp_legs = 3** removes
  it (bound ≈ 4.4·10^12) for **+10 rows** (+0.05 %), no pk/logrows change.

## Choice
**input_scale = 12, param_scale = 25, decomp_legs = 3** for all re-runs (S0–S3 `_fix`).
Every input_scale ≥ 10 config meets < 1 % at the same cost (same logrows, same pk bytes, rows within 10, prove times
within noise), so "cheapest" doesn't separate them. The tie is broken by lower error (0.14 % vs 0.56 %) and by
needing one setting that works for the whole grid. decomp_legs = 3 is required anyway for K = 16 (at input 10 too),
so in = 10/legs 2 (nominally 10 rows cheaper) is not an option for one uniform setting. input_scale 13 (0.07 %) was
not pursued (outside the requested grid). An alternative that avoids decomp_legs is feeding n_i in thousands, but that
changes the inputs, not just the run args.
