# 30 — Harness `eval/bench_zkp.py`

- `python bench_zkp.py run --sweep S1 --K 2 --d 100 1000 --logrows auto|<int>...` (driver); spawns `bench_zkp.py worker ...` per config, cwd `eval/`, env `EZKL_REPO_PATH=eval/ezkl_repo`.
- `--logrows auto` = `max(10, ceil(log2(d+6)))` (minimal valid logrows: all d public outputs live in one instance column, so 2^logrows ≥ d + blinding rows; per coordinator/source research `50_ezkl_source_research.md`). CSV column `logrows_rule` records this.
- Worker: synthetic `rng(0).normal(0,0.05,(K,d)).float32`, num_examples 25000/client; repo `AggregationSetup(AggregateModel())`, `AggregationProver`, `AggregationVerifier` unmodified.
  - Untimed probe gen_settings + `get_srs` (warm SRS) first.
  - `t_setup_s` = whole `AggregationSetup.setup` (export+gen_settings+compile+get_srs validation+keygen). Stage times via wrappers on `ezkl.gen_settings/compile_circuit/setup/get_srs` module attributes (zkp.py untouched): `t_gen_settings_s`, `t_compile_s`, `t_keygen_s` (= `ezkl.setup` alone), `t_export_s`.
  - `t_witness_s`, prove ×3, verify ×3 (`time.perf_counter`).
  - Output error: witness `pretty_elements.rescaled_outputs` (cross-checked with `ezkl.felt_to_float(outputs, model_output_scales[0])`) vs float64 NumPy weighted average. `out_max_rel_err` is elementwise over |ref|>1e-3; `out_rel_l2_err` = ||err||/||ref||.
  - `t_plain_med_s`: NumPy weighted average, median of 100.
- Driver: peak RSS via `os.wait4(pid)` rusage of that specific child (ru_maxrss, bytes on macOS; avoids RUSAGE_CHILDREN accumulating across configs) + psutil sampled RSS + system swap used. Kills: setup phase > 30 min → `timeout`; witness+first prove > 10 min → `exceeds_budget` (also worker self-skips reps if first prove finished past budget); RSS > 20 GB → `oom`. Free disk < 20 GB → abort. Deletes test.pk, witness, data json after recording sizes; keeps settings/vk/proof/onnx/compiled in `eval/runs/<cfg>/`, settings copy in `eval/settings/<cfg>.json`, worker log `eval/runs/<cfg>.log`. Resumable CSV `eval/results.csv`.

## Validation (VAL, K=2, d=100, logrows=10): ok
logrows_used 10, num_rows 450, assignments 901, num_inner_cols 2, setup 0.15 s (keygen 0.146 s), prove 0.47 s, verify 0.011 s, verify True,
pk 39.9 MB, vk 4.4 kB, proof 76 kB, peak RSS 0.53 GB. **All 100 circuit outputs are exactly 0** (rel err 1.0) — reciprocal of 50,000 at scale 7 rounds to 0.
