# 20 — Smoke test (toy `simple_experiment()`)

Run: `eval/scratch/smoke/smoke.py` → `asyncio.run(zkp.simple_experiment())` with `fl/src` on sys.path, cwd `eval/scratch/smoke`.
Toy input: 2 clients × 6 params, num_examples (10, 5), `logrows=20`.

## Result
- setup → witness → prove → verify: **OK, verify = True**, total wall time 32.4 s (SRS kzg20 was already cached in `~/.ezkl/srs`, from Mar 2025).
- Artifacts: **pk = 4,431,545,883 B (4.43 GB)**, vk = 2,360,583 B, proof = 18,201 B, compiled = 1,842 B.
- settings.json: `logrows=20`, `num_rows=74`, `total_assignments=149`, `num_inner_cols=2`, `input_scale=param_scale=7`, `model_output_scales=[21]`, `lookup_range=[-32768,32768]`, `required_range_checks=[[-1,1],[0,16383]]`, `decomp_base=16384`, `decomp_legs=2`, check_mode UNSAFE. Copy: `eval/scratch/smoke_settings_logrows20.json`.

**Observation (important for C3):** a 6-parameter circuit using only 74 rows already produces a 4.4 GB proving key at logrows=20. Key size is dominated by `2^logrows`, not by the circuit content. The paper's "5 GB" could plausibly come from a logrows=20 setup rather than from the 62k-param model.

## Accuracy of the toy output
Witness `pretty_elements.rescaled_outputs` = [3.867, 4.219, 4.570, 4.922, 5.273, 5.625].
True FedAvg = [3.667, 4.000, 4.333, 4.667, 5.000, 5.333] → abs error ≈ 0.20–0.29 (≈ 5.5 % relative).
Consistent with division by the total (15) being done as multiplication by a reciprocal quantised at scale 7: round(128/15)/128 = 9/128 = 0.0703 vs 0.0667 (+5.5 %). With num_examples total = 50,000 the reciprocal 1/50000·128 ≈ 0.0026 rounds to 0 — expect large error in the benchmark (to be measured).

## Async-ness (ezkl 15.6.3)
- `get_srs`, `gen_witness`: coroutines (calling `get_srs` outside an event loop → `RuntimeError: no running event loop`).
- `gen_settings`, `compile_circuit`, `setup`, `prove`, `verify`: synchronous, return bool.

Cached SRS at start: `~/.ezkl/srs/kzg15.srs`, `kzg20.srs` only — logrows=10/14/18 SRS will need download (timed separately/excluded).
