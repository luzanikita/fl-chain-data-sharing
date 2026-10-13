# ZKP scaling experiment: report

Scope: measure how the cost of proving FedAvg with EZKL (`fl/src/zkp.py`) scales with model size *d* and number of
clients *K*, and check the quantitative statements in the paper's Limitations paragraph. All numbers come from
`eval/check_report_numbers.py` (reading `eval/results_fixed.csv`) or from the cited `eval/*.md` notes.
Measured sizes and memory are decimal GB (bytes / 10^9), as printed by the check script. The machine's RAM is quoted
by its nominal capacity, 24 GB (25,769,803,776 B = 24 GiB = 25.8 decimal GB).

## 1. Setup

- **Hardware/software** (`environment.json`): Apple M4 Pro (8 performance + 4 efficiency cores), 24 GB RAM (nominal),
  macOS 26.6.2; Python 3.11.14, EZKL 15.6.3 (matches the repo pin), torch 2.2.2. The original observation was made
  on a different, older machine; its hardware and settings were not available to this study.
- **Circuit**: the repo's `AggregateModel` via `AggregationSetup` / `AggregationProver` / `AggregationVerifier`,
  used unmodified (`fl/src/zkp.py` untouched). Repo run args: inputs private, params fixed, outputs public,
  scale_rebase_multiplier 1000.
- **Inputs**: synthetic client updates `rng(0).normal(0, 0.05, (K, d))`, 25,000 examples per client
  (`90_summary.md`).
- **Precision fix (benchmark only)**: with the repo's input_scale = param_scale = 7, the division by Σn is compiled
  to a reciprocal at scale 7, and round(2^7 / 50,000) = 0, so every circuit output is exactly 0 while proofs still
  verify (`50_ezkl_source_research.md` §3, `60_precision_fix.md`). All reported runs therefore use
  input_scale 12, param_scale 25, decomp_legs 3, injected by `bench_zkp_fixed.py` without editing `zkp.py`.
  Relative L2 error of the outputs is 0.14–0.20 % (`70_fixed_results.md`). The fix changes num_rows by +6…12, leaves
  pk/vk byte-identical, and changes prove/verify/setup time only within run-to-run noise (`70_fixed_results.md`),
  so cost numbers do not depend on it.
- **logrows rule**: EZKL's `gen_settings` never raises logrows, and the repo's default logrows 10 cannot run
  d ≥ ~1,018 (`25_logrows_finding.md`). Sweeps use the minimal valid value logrows = max(10, ceil(log2(d+6)));
  a second series fixes logrows = 20.
- **SRS**: EZKL's public SRS bucket returns HTTP 403. kzg10–19 were derived by downsizing the cached official kzg20
  with PSE `halo2_proofs` (`tools/srs_downsize`); the derived kzg15 is byte-identical to the official file and EZKL's
  hash check accepts them (`15_srs.md`). No SRS above logrows 20 was available.
- **Protocol**: one fresh worker process per configuration; prove and verify each run 3 times, median reported.
  Budgets: setup 30 min, online (witness + prove) 10 min. All 27 fixed-precision configurations finished with status
  ok and verify True on all repetitions; no timeouts or OOM (`70_fixed_results.md`, `80_extended_results.md`).

## 2. Verdicts

### C1 — circuit grows linearly with model parameters: **partly supported**
- Circuit rows: rows = 4.0000·d + 60.3 for K = 2 over d = 100…250,000 (log-log slope 0.985); residuals ≤ 1.7 rows
  (`80_extended_results.md` §1).
- At minimal logrows, from d = 100 (logrows 10) to d = 62,006 (logrows 16) to d = 250,000 (logrows 18):
  - proving key 0.04 GB, 0.84 GB, 3.37 GB;
  - prove time 0.5 s, 7.5 s, 29.3 s;
  - verify time 11 ms, 276 ms, 1,115 ms;
  - peak RSS 0.6 GB, 3.1 GB, 9.4 GB.
- Interpretation: row count is exactly linear in d. Proving-key size and prove time follow 2^logrows, so they grow
  in power-of-two steps. The inversion between d = 1,000 and d = 5,000 is real, not noise: d = 1,000 (logrows 10)
  takes 1.7 s to prove and 36 ms to verify, while d = 5,000 (logrows 13) takes 1.1 s and 27 ms. At logrows 10, EZKL
  fits 4,062 rows into 2^10 by adding column blocks (`25_logrows_finding.md`).

### C2 — circuit grows linearly with participating clients: **partly supported**
- d = 5,000, logrows 13, K = 1 → 32: rows ×24.4, proving key ×11.3, prove time ×14.5
  (0.50 s → 7.22 s; pk 92.8 MB → 1,047.6 MB; `80_extended_results.md` §3).
- Interpretation: rows are linear in K (from K = 2, +1.5·d rows per client), while key size and prove time grow
  sub-linearly at fixed logrows because extra rows are placed in additional columns.

### C3 — prover key files up to 5 GB: **reproduced only at logrows 20**
- At logrows 20 the proving key is 4.43 GB (4,431,545,883 B), byte-identical for every d from 100 to 62,006
  (`80_extended_results.md` §2), and also for the repo's 6-parameter toy example (`90_summary.md`). Peak RSS at
  logrows 20 is 9.1–11.2 GB.
- For the paper model (K = 2, d = 62,006) at the minimal valid logrows 16 the key is 0.84 GB.
- Interpretation: the ~5 GB figure is consistent with a logrows-20 setup; key size is governed by 2^logrows rather
  than by d.

### C4 — proof generation takes several minutes: **not reproduced on this machine**
- Paper model (K = 2, d = 62,006): 7.5 s at logrows 16; 20.8 s at logrows 20. Largest run (d = 250,000,
  logrows 18): 29.3 s. Setup (key generation): 11.8 s (logrows 16), 22.6 s (logrows 20) (`70_fixed_results.md`).
- Interpretation: proving took seconds, not minutes, on the Apple M4 Pro. The original measurement was made on a
  different, older machine, and this study cannot rule out that it took minutes there.

## 3. Proposed Limitations paragraph (ready to paste, 111 words)

> ZKP cost was measured on an Apple M4 Pro (24 GB) with EZKL 15.6.3. Circuit rows grow linearly with model
> parameters d and clients K, whereas proving-key size and proof time grow in power-of-two steps of 2^logrows rather
> than with d directly. For the evaluated configuration (K=2, d=62,006), the minimal valid logrows of 16 yields a
> 0.84 GB proving key and 7.5 s proof generation; logrows 20 yields 4.43 GB and 20.8 s. From K=1 to 32 clients
> (d=5,000), key size and proof time grew 11.3× and 14.5×. Verification took 276 ms (461 ms at logrows 20) on the
> same machine, with 3–37 kB halo2 proofs; vehicle-class hardware was not tested.

Notes for the authors:
- The JSON proof file is larger than the raw halo2 proof: 9.9 MB at d = 62,006, because it serialises the d public
  outputs (`70_fixed_results.md`, `90_summary.md`).
- The 3–37 kB raw proof sizes were measured in the original-precision run (`90_summary.md`).
- Proving takes 7.5–20.8 s, far below the benchmark's 10-min online budget. Whether that matches the IoV round
  length is for the paper to state.

## 4. Prototype issues found

- **Default logrows 10 cannot run d ≥ ~1,018.** All d public outputs share one instance column of 2^logrows rows.
  Keygen fails with `[halo2] k = 10 is too small for the given circuit`, followed by `Failed to run setup: [halo2]
  General synthesis error` (`25_logrows_finding.md`). The paper model needs logrows ≥ 16.
- **Default scales make every output 0.** With input/param scale 7, Recip(Σn = 50,000) rounds to 0. In the
  original-precision run (`results.csv`), all 20 successful configurations have relative L2 error 1.0, yet every
  proof verifies, so each proof certifies an all-zero model. Cost is unaffected (see §1).
- **decomp_legs 2 breaks witness generation after a naive scale fix.** With input_scale 12, Σn·2^input_scale exceeds
  the base-16,384 two-leg decomposition for K ≥ 4 (25,000 examples each). Keygen succeeds; the failure appears only
  at the first `gen_witness`/prove. decomp_legs 3 removes it for +10 rows (`60_precision_fix.md`).
- **SRS download returns HTTP 403.** `ezkl.get_srs` fails for uncached sizes; worked around by deriving SRS files
  locally (`15_srs.md`).

## 5. Sequential aggregation (estimate, not implemented)

- Estimate from K = 1 measurements (d = 5,000, logrows 13): K = 32 sequential proving ≈ 32 × 0.50 s = 15.9 s versus
  7.2 s measured for one monolithic K = 32 circuit; proving key 93 MB (constant in K) versus 1.05 GB monolithic.
  Sequential proving trades ~2.2× more total prover time for an 11.3× smaller, K-independent key
  (`80_extended_results.md` §3).
- Caveats (`50_ezkl_source_research.md` §5–6):
  - Steps must be chained, so step i+1 consumes step i's accumulator.
  - A public accumulator after step 1 exposes client 1's update.
  - EZKL 15.6.3 has one input visibility for all inputs, so it cannot make the accumulator public and the update
    private.
  - It has no folding/IVC, only KZG proof aggregation. Nova-style folding is future work.

## 6. Not measured

- Hashed inputs (in-circuit Poseidon commitments to client updates).
- `calibrate_settings` (EZKL's automatic scale/logrows calibration).
- Newer EZKL versions.
- Verification on vehicle-class hardware (all verify times are from the M4 Pro).
- logrows > 20 (no SRS available).
- d near 2^20: this would need logrows 20 with ~4.2 M rows. Peak memory is extrapolated to exceed the machine's
  physical RAM (`80_extended_results.md` §1; extrapolation, not run).
- Real model weights (synthetic inputs only; cost is value-independent).

## 7. Artifacts

| path | purpose |
|---|---|
| `eval/bench_zkp.py` | benchmark driver/worker around unmodified `fl/src/zkp.py` |
| `eval/bench_zkp_fixed.py` | same harness with input_scale 12 / param_scale 25 / decomp_legs 3 |
| `eval/results.csv` | original-precision runs (all outputs zero) |
| `eval/results_precision.csv` | precision mini-study (`60_precision_fix.md`) |
| `eval/results_fixed.csv` | fixed-precision runs used in this report |
| `eval/check_report_numbers.py` | prints every headline number in this report |
| `eval/plot_zkp.py`, `eval/test_plot_zkp.py` | figure generation and its tests |
| `eval/figs/zkp_scaling.pdf`, `.png` | six-panel scaling figure |
| `eval/results_table.tex` | LaTeX table (booktabs) of the minimal-logrows and logrows-20 series |
| `eval/environment.json` | hardware and software versions |
| `eval/settings/`, `eval/runs/` | EZKL settings; per-config logs, vk, proofs (runs/ is gitignored) |
| `eval/tools/srs_downsize` | Rust tool deriving kzg10–19 from kzg20 |
| `eval/15_srs.md` … `eval/90_summary.md` | working notes cited above |

**Figure caption draft (`figs/zkp_scaling.pdf`):**
*ZKP cost of the FedAvg aggregation circuit (EZKL 15.6.3, Apple M4 Pro, 24 GB; median of 3 runs, bars show min/max).
(a) Circuit rows vs number of parameters d (K = 2) with log-log fit (slope 0.99). (b) Proving-key size vs d at the
minimal valid logrows (○) and at logrows 20 (□). (c) Proof generation time vs d, same series; the d = 1,000 point
uses logrows 10 with extra columns and is slower than d = 5,000. (d) Verification time vs d (minimal logrows).
(e) Proof generation time vs number of clients K (d = 5,000): measured monolithic circuit and dashed sequential
estimate K·t(K=1). (f) Proving-key size vs K, with dashed line at pk(K=1) for the sequential scheme. Dotted vertical
lines mark the paper model (d = 62,006).*

**Table caption note (`results_table.tex`):** K = 2; err. = relative L2 error of circuit output vs NumPy FedAvg;
overhead = prove time / NumPy weighted-average time (baseline ≈ µs, so only the order of magnitude is meaningful).
Requires `\usepackage{booktabs}`.
