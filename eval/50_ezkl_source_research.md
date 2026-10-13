# 50 — EZKL 15.6.3 source research (for the ZKP scaling experiment)

Source: tag `v15.6.3` of github.com/zkonduit/ezkl (commit `8fde190e`).
Paths below are relative to a checkout of that tag unless a URL is given. halo2 fork pinned by `Cargo.lock:2375`:
`zkonduit/halo2@8b13a0d2a7a34d8daab010dadb2c47dfa47d37d0` (branch `ac/cache-lookup-commitments`).
tract pinned by `Cargo.toml:92`: `sonos/tract@40c64319291184814d9fea5fdf4fa16f5a4f7116`.

Labels: **[src]** = read in source; **[run]** = checked with a tiny run (gen_settings / gen_witness / `ezkl.mock`
only, K≤8, d≤2000, no keygen, no proving; scripts in scratchpad `gs/t*.py`); **[inferred]** = my reasoning, not verified.

---

## TL;DR

- **logrows is NOT bumped by `gen_settings`.** It stays 10. Extra rows are handled by adding more advice/fixed
  columns ("column duplication"). Only `calibrate_settings` changes logrows. **But** all public outputs sit in a
  single instance column of 2^logrows rows, so **d > 2^10 − 6 = 1018 fails at logrows=10** (mock panics
  `instance.len=2000, n=1024, cs.blinding_factors=5` at d=2000 [run]). So for d ≥ ~1k you *must* pass a larger logrows
  (≥ ceil(log2(d+6)); d=62,006 → logrows ≥ 16).
- **The circuit output is all zeros for num_examples = 25,000/client.** The division `weighted_sum / Σn` is turned
  into `weighted_sum * Recip(Σn)`, and the reciprocal is quantised at scale 7: round(128/50000) = 0. Witness
  `rescaled_outputs` are exactly 0.0 [run]. This is a correctness problem in the current setup, not just
  "precision". Fix: raise `param_scale` (Recip's output scale = max(input_scale, param_scale)) and ideally
  `input_scale`; e.g. input_scale=10/param_scale=25 → rel. L2 error 0.6 %, input 12/param 25 → 0.15 % [run],
  with identical `num_rows`.
- `scale_rebase_multiplier=1000` effectively disables rescaling, so there are no Div/lookup rescale ops and the output
  scale is 7+7+7 = 21 (or input+input+max(input,param) generally).

---

## 1. logrows in `gen_settings`; where rows are computed; size fields

**What gen_settings does** [src]: `ezkl.gen_settings` → `execute::gen_circuit_settings` (`src/execute.rs:820-829`)
which is just `GraphCircuit::from_run_args(&run_args, …)` + `settings.save()`. `from_run_args`
(`src/graph/mod.rs:1389-1395`) → `GraphCircuit::new` (`src/graph/mod.rs:752-790`) → `Model::gen_params`
(`src/graph/model.rs:524-594`). Nothing there touches `run_args.logrows`; it is copied verbatim into the settings
(`model.rs:568-569`). The only code that rewrites logrows is `calc_min_logrows` (`src/graph/mod.rs:1116-1211`,
sets `run_args.logrows` at line 1198), whose only caller is `calibrate` (`src/execute.rs:1162`). [run] confirmed:
K=2,d=2000, logrows=10 → settings `logrows=10`, `num_rows=8052`.

**How rows are counted** [src]: `gen_params` runs a "dummy layout" (`model.rs:1388-1475`) with a dummy region; the
region counter `next()` increments `linear_coord` per assigned cell and `row` every `num_inner_cols` cells
(`src/circuit/ops/region.rs:744-750`). Result: `num_rows = region.row()`, `total_assignments = region.linear_coord()`
(`model.rs:1465-1467`, stored at `model.rs:572-573`). So `num_rows ≈ total_assignments / num_inner_cols`.

**What happens when num_rows > 2^logrows** [src]: at circuit configuration `ModelVars::new`
(`src/graph/vars.rs:424-466`) creates 3 advice `VarTensor`s each sized for `total_assignments`;
`VarTensor::new_advice` (`src/tensor/var.rs:101-134`) computes
`modulo = (capacity + modulo)/(max_rows·num_inner_cols) + 1` blocks, each with `num_inner_cols` advice columns,
where `max_rows = 2^logrows − blinding_factors − 1` (`var.rs:49-52`). Constants likewise get
`num_constants/max_rows + 1` fixed columns (`var.rs:141-170`). I.e. **the circuit is widened, not lengthened**:
#advice columns ≈ 3 · num_inner_cols · ceil(total_assignments / (num_inner_cols · (2^logrows − 6))).
[run] K=8,d=120 → num_rows=1617 > 1024 at logrows 10, `ezkl.mock` = True.

**Hard limit: instance column** [src]: all public values go into *one* instance column
(`ValTensor::new_instance`, `src/tensor/val.rs:326-341`; shapes from `model.rs:1548-1557` — input shapes if
inputs public, output shapes if outputs public). halo2 rejects instances longer than `n − (blinding+1)`
(`halo2_proofs/src/plonk/prover.rs:126-127`, `Error::InstanceTooLarge`, zkonduit/halo2@8b13a0d), and keygen's
`copy`/`query_instance` return `NotEnoughRowsAvailable` for rows outside the usable range
(`halo2_proofs/src/plonk/keygen.rs:95-98, 147-156`). With `output_visibility="public"`, d outputs ⇒
**need 2^logrows ≥ d + 6**. [run] d=2000, logrows=10: `gen_settings` and `gen_witness` succeed, `mock` panics
`instance.len=2000, n=1024, cs.blinding_factors=5`. Setup would be expected to fail the same way [inferred from
keygen.rs:147-156]. Consequence for S0/S3 of the plan: "logrows=10 with d=10k" is not a valid config; minimum
logrows is 14 for d=10k, 16 for d=62,006.

Other columns that depend on logrows [src]: range-check tables (here `required_range_checks=[[-1,1],[0,16383]]`,
from the decomposition gadget) need `range_len / (2^logrows − reserved) + 1` table columns
(`src/circuit/table.rs:136-150`, configure at 166-186 and range-check variant ~378-400). At logrows=10 the
[0,16383] table needs ~17 columns; at logrows ≥ 15 one column [inferred arithmetic].

**settings.json fields describing size** (`src/graph/mod.rs:401-444`): `run_args.logrows`, `run_args.num_inner_cols`,
`num_rows`, `total_assignments`, `total_const_size`, `total_dynamic_col_size`, `num_dynamic_lookups`, `num_shuffles`,
`total_shuffle_col_size`, `max_dynamic_input_len`, `model_instance_shapes` (public instance count),
`required_lookups`, `required_range_checks`, `run_args.lookup_range`, `module_sizes`. The actual column counts are
not stored; they are logged at debug level as `CircuitSize` {num_instances, num_advice_columns, num_fixed,
num_selectors, logrows} (`src/graph/mod.rs:1483-1527`, logged at 1592-1602; enable with `RUST_LOG=debug`).

Measured [run] (K=2): d=100 → num_rows 450, total_assignments 901; d=2000 → 8052 / 16105. So
total_assignments ≈ 8·d + ~100 for K=2 (linear in d); K=8,d=120 → 3235. A plausible model is
total_assignments ≈ (2K+4)·d + c [inferred from 3 points; check against the sweep].

## 2. What determines proving-key size

halo2 `ProvingKey` (`halo2_proofs/src/plonk.rs:338-348`, written at 390-399; fork @8b13a0d) stores:
vk; `l0`, `l_last`, `l_active_row` (extended domain, 2^(k+e) each); for **each fixed column** (incl. lookup/range
table columns and compressed selectors) `fixed_values` (2^k) + `fixed_polys` (2^k) + `fixed_cosets` (2^(k+e));
and the permutation pk (`plonk/permutation.rs:130-134`) with, **for each equality-enabled column** (every advice
column — `var.rs:122-123` calls `enable_equality` — plus instance and constant columns), `permutations` (2^k),
`polys` (2^k), `cosets` (2^(k+e)). Advice columns themselves add no polynomials to the pk. 2^e is the extended-domain
factor: smallest power of two ≥ (max gate degree − 1) (`src/graph/mod.rs:1252-1264` mirrors this). EZKL writes keys
as `raw-bytes` (`src/lib.rs:185-191`), 32 B per field element.

So, approximately [inferred from the struct]:

    pk_bytes ≈ 32 · 2^k · (2 + 2^e) · (N_fixed + N_selector_fixed + N_perm_cols) + 32 · 3 · 2^(k+e) + vk

N_perm_cols ≈ 3·num_inner_cols·blocks + 1 instance + constant cols; N_fixed includes the range-check table columns.
Cross-check with smoke test (logrows 20, 74 rows): 4.43 GB / (32·2^20) ≈ 132 polynomial-lengths ⇒ ~13 columns if
2^e = 8 — plausible for a 1-block circuit [inferred].

**Shape in d** [inferred from §1 mechanics]:
- *Fixed logrows* (what the repo does): k constant, columns grow with ceil(total_assignments / 2^k) ⇒
  pk ≈ linear in d with tiny steps (one block = 3·num_inner_cols columns per ~2^k·num_inner_cols assignments).
  Prove time similarly ~linear in #columns.
- *Minimal/calibrated logrows* (k = ceil(log2(num_rows+reserved))): one block, k grows ⇒ **step-like, doubling at each
  power of two** of num_rows (and SRS size changes).
- Both are bounded below by the instance constraint 2^k ≥ d + 6.

## 3. Defaults, scale_rebase_multiplier, division, expected error

**Defaults** (`PyRunArgs::new()` = `RunArgs::default()`, `src/bindings/python.rs:204-208`;
defaults at `src/lib.rs:338-360`):
input_scale 7, param_scale 7, scale_rebase_multiplier 1, lookup_range (−32768, 32768), logrows 17,
num_inner_cols 2, tolerance 0 (CLI default `src/lib.rs:276`), input_visibility private, output_visibility public,
param_visibility private, decomp_base 16384, decomp_legs 2, check_mode UNSAFE, bounded_log_lookup false,
variables batch_size→1. The repo overrides param_visibility=fixed, scale_rebase_multiplier=1000, logrows=10.

**scale_rebase_multiplier** [src]: after each op, `RebaseScale::rebase` (`src/graph/node.rs:116-152`, invoked at
`node.rs:588-592`) inserts a rescale (`HybridOp::Div{denom = 2^(out_scale − global_scale·mult)}`) only if
`out_scale > global_scale · scale_rebase_multiplier`, where `global_scale = max(input_scale, param_scale)`
(`src/graph/vars.rs:257-259`). With 1000 the threshold is 7000, so **no rescaling ever happens**: scales just add
through multiplications. Here: w(7)·n(7) → 14, sum → 14, ·recip(7) → **output scale 21** (settings
`model_output_scales=[21]` [run]). With the default 1, the 14-scale product would be rescaled to 7 via a Div gadget
(integer denom → `layouts::div`, `src/circuit/ops/hybrid.rs:256-263`), costing extra constraints.

**Division by the private sum** [src]: tract declutters float `Div(a,b)` into `Mul(a, Recip(b))`
(tract `core/src/ops/math/mod.rs:411-456` @40c6431; EZKL calls `into_decluttered()` at `src/graph/model.rs:666-668`).
EZKL maps `Recip` to `HybridOp::Recip{input_scale, output_scale = max(scales.get_max(), in_scale)}`
(`src/graph/utilities.rs:805-813`); out scale = log2(output_scale) (`hybrid.rs:350-352`). The layout
(`src/circuit/ops/layouts.rs:176-252`) is **not a lookup table**: the prover supplies
`round(out_mult / (x/in_mult))` (`src/tensor/ops.rs:2260-2268`) and the circuit proves it is the integer minimiser of
|y·x − in_mult·out_mult| (convexity check `optimum_convex_function`, `layouts.rs:70-103`, using `l1_distance`/
`less_equal`, which use the base-16384, 2-leg decomposition + range checks `layouts.rs:5278-5377`), plus zero-handling
masks. Cost is O(1) per element (only one element here: Σn), so negligible vs the K·d part.

**Expected error** [src + run]:
- Recip at scale 7: y = round(128 / Σn). Σn = 50,000 → y = 0 ⇒ **output ≡ 0** (rel. error 100 %). [run] K=2,
  d=100 and d=2000: `max|out| = 0.0`; smoke test (Σn=15) gave +5.5 % because 9/128 ≠ 1/15. In general the relative
  error of the reciprocal is up to 0.5·Σn / 2^s_recip, so for Σn = 5·10^4 and < 0.1 % one needs s_recip ≥ 25.
- Input quantisation at scale 7: step 1/128 = 0.0078; for w ~ N(0, 0.05) the rounding error is ~U(±0.0039), RMS
  ≈ 0.0023 ≈ 4.5 % of σ, and ~6 % of weights round to exactly 0 [inferred arithmetic]. [run] with the recip fixed
  (input 7, param 25): rel. L2 error 4.4 %, max abs 0.0036 — matches.
- Fixes tested [run] (K=2, d=100, Σn=50,000; same num_rows=450): input 10/param 25 → rel L2 0.63 %, max abs 4e-4;
  input 12/param 25 → 0.15 %, 1.1e-4. Input 13/param 25 → `gen_witness` "General synthesis error" — likely the
  decomposition range (16384² ≈ 2.7·10^8) is exceeded by Σn·2^13 ≈ 4.1·10^8 [inferred]; raising `decomp_legs` would
  lift that at extra cost. Alternative: feed n_i in thousands (keeps the circuit, shrinks Σn).
- Output check is exact equality when tolerance=0 (`layouts.rs:5660-5670`).

## 4. calibrate_settings

`ezkl.calibrate_settings(data, model, settings, target="resources", lookup_safety_margin=2, scales=None,
scale_rebase_multiplier=[1,2,10], max_logrows=None)` (`src/bindings/python.rs:881-921`; defaults
`src/commands.rs:69,79,91`) → `execute::calibrate` (`src/execute.rs:967-1330`) [src]:
1. Grid over (input_scale, param_scale) ∈ `scales`² with input ≤ param (default `scales = 11..14` ⇒ {11,12,13}),
   × `scale_rebase_multiplier` ∈ {1,2,10} (`execute.rs:1004-1040`). It does **not** search num_inner_cols,
   decomp_base/legs or visibility.
2. For each, runs the real forward pass on the calibration data, records min/max lookup inputs and max range size,
   and calls `calc_min_logrows` (`mod.rs:1116-1211`): lower bound = max(instances, module constraints, dynamic
   lookups); upper bound = max(ceil(log2(num_rows+reserved)), table bits for the (safety-margined) lookup range,
   constants); then lowers logrows while the lookup tables would need > 12 columns or the extended domain is too big.
   It sets `lookup_range` to the observed range × margin and **logrows ≈ ceil(log2(num_rows + 8))** — i.e. the
   "one block, tall circuit" layout.
3. Selection (`execute.rs:1213-1275`): `resources` = min logrows, then max (input_scale, param_scale, multiplier);
   `accuracy` = max scales, then min logrows. `resources/col-overflow` additionally shrinks logrows down to the
   largest of lookup/range-check/instance/module bits (`execute.rs:1303-1323`) — that re-enables column duplication.
4. Prints a "Numerical Fidelity Report" comparing ONNX vs circuit outputs (`execute.rs:1277-1299`).

Effect on cost vs the repo default [inferred]: logrows becomes ~log2(num_rows) instead of 10 (once d > 1k the repo
setting is invalid anyway, see §1); pk/prove become power-of-two step functions; scales 11–13 with multiplier ≤ 10
raise accuracy but **would still give Recip output scale ≤ 13 ⇒ round(8192/50000) = 0**, so calibration with
default `scales` does not fix the zero output; and with multiplier 1/2/10 rescale Div ops appear. Pass explicit
`scales` (and/or set param_scale manually) if calibrating.

## 5. Per-input visibility

[src] `RunArgs.input_visibility` is one `Visibility` for all model inputs (`src/lib.rs:299-301`); instance shapes
add *all* input shapes when it is public (`model.rs:1548-1552`). Variants: Private, Public, Hashed{hash_is_public,
outlets}, KZGCommit ("polycommit"), Fixed (`src/graph/vars.rs:26-45`). The only per-input selector is
`"hashed/private/<i,j,…>"`, which Poseidon-hashes only the listed input outlets (`vars.rs:76-90, 230-235`); there is
no "input 0 public, input 1 private" mode in 15.6.3. Workarounds [inferred]: (a) make the accumulator an *output*
(outputs are public) of step i and pass it as a private input of step i+1, linking proofs by `polycommit`
visibility + `swap_proof_commitments` (`python.rs:638`) or by `aggregate(..., split_proofs=True)`, which asserts that
the output commitment of proof i equals the input commitment of proof i+1 (`src/pfsys/evm/aggregation_kzg.rs:141-170`)
— this keeps the accumulator hidden (commitment only), but with polycommit applying to *all* inputs; (b) put the
accumulator in the same input tensor and accept it is private; (c) public everything and leak.

## 6. Aggregation / folding

[src] 15.6.3 offers KZG proof *aggregation* via snark-verifier accumulation: `ezkl.setup_aggregate`,
`aggregate(aggregation_snarks, …, logrows=23, split_proofs=False)`, `mock_aggregate`, `verify_aggr`
(`src/bindings/python.rs:1021, 1240, 1349, 1414`; default `DEFAULT_AGGREGATED_LOGROWS = 23`, `src/commands.rs:71`). Inner proofs
must be made with `proof_type="for-aggr"` (Poseidon transcript, accumulator strategy; `src/pfsys/mod.rs:60-101`);
the aggregation circuit (`src/pfsys/evm/aggregation_kzg.rs:246-330`) verifies each inner proof in-circuit (non-native
BN254 EC arithmetic) and exposes 4·LIMBS accumulator limbs **plus all inner instances** as public inputs. Cost
[inferred, not measured]: needs a 2^23 SRS (~1 GB) and a correspondingly large pk (many GB) with minutes of proving
per aggregation, roughly growing with the number of inner proofs; it compresses verification but does not reduce
total prover work. There is **no folding/Nova/IVC** in 15.6.3 (no `nova`/`fold` matches in `src/`), so
sequential aggregation would be "K proofs + optional aggregate", with Nova-style folding as future work.

## 7. Reading outputs back as floats

[src] `gen_witness` writes `witness.json` with `pretty_elements.rescaled_outputs` (strings of floats) computed as
`dequantize(felt, output_scale) = int / 2^scale` (`src/graph/mod.rs:171-202`, `src/graph/utilities.rs:77-81`);
generated for every witness (`mod.rs:1372-1376`). The same rescaled values appear in the proof's
`pretty_public_inputs.rescaled_outputs` when outputs are public (`mod.rs:895-897`). Manually:
`ezkl.felt_to_float(felt_hex, scale)` (`src/bindings/python.rs:368-374`) on `witness["outputs"][0][i]` with
`scale = settings["model_output_scales"][0]` (21 here). Note: in the installed bindings `await ezkl.gen_witness(...)`
did not return a dict containing `pretty_elements` — read the JSON file instead [run].
