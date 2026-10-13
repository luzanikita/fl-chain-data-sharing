# 15 — SRS availability problem and fix

**Blocker found:** EZKL's public SRS host is gone. `ezkl.get_srs` for logrows=10 fails:
`RuntimeError: Failed to get srs: [io] input number is not less than field modulus`
(it downloaded the error page). `curl https://trusted-setup-halo2kzg.s3.eu-central-1.amazonaws.com/perpetual-powers-of-tau-raw-10`
→ HTTP 403 `<Code>AccessDenied</Code>` (same for -15, other URL forms). Only `~/.ezkl/srs/kzg15.srs` and `kzg20.srs` (Mar 2025) were cached.
`get_srs` validates existing files against a built-in SHA-256 table (a kzg20 file renamed kzg10 was rejected and deleted:
"SRS hash does not match the expected hash").

**Fix (no change to zkp.py, no insecure SRS):** `eval/tools/srs_downsize` — tiny Rust tool using PSE `halo2_proofs` v0.3.0
(`ParamsKZG::read_custom(RawBytes)` → `downsize(k)` → `write_custom(RawBytes)`), i.e. exactly how the official per-k files were derived.
**Verification:** kzg20 → downsize(15) is byte-identical to the official cached kzg15.srs
(sha256 `90807800a1c3b248a452e1732c45ee5099f38b737356f5542c0584ec9c3ebb45`).
Generated kzg10..kzg19 into `eval/ezkl_repo/srs/` (+ copies of official kzg15/kzg20); `ezkl.get_srs` accepts them
(hash check passes for 10/14/18). Workers run with `EZKL_REPO_PATH=eval/ezkl_repo` (EZKL env var), so `~/.ezkl` is untouched.
Limitation: no SRS above k=20 available → any config needing logrows>20 cannot run.

SRS loading is excluded from t_setup anyway: the worker warms with an untimed `get_srs` (t_srs_s, ~ms since files exist);
the `get_srs` inside `AggregationSetup.setup` then only re-validates (recorded as t_srs_in_setup_s, included in t_setup_s).
