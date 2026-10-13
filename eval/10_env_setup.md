# 10 — Environment setup

- venv: `uv venv --python 3.11 eval/.venv` → CPython 3.11.14 (uv 0.10.4).
- Installed: `uv pip install ezkl==15.6.3 torch==2.2.2 numpy==1.26.4 onnx==1.17.0 flwr==1.13.1 pandas matplotlib psutil`
  - All pinned versions installed cleanly from macOS arm64 wheels. **ezkl 15.6.3 = pinned version** (no deviation).
  - Unpinned: pandas 3.0.6, matplotlib 3.11.2, psutil 7.2.2 (analysis/monitoring only).
- Hardware: Apple M4 Pro (8 P + 4 E cores, 12 logical), 24 GB RAM, macOS 26.6.2, ~98 GB free disk at start (`df -g`).
- Full details: `eval/environment.json`.

## ezkl 15.6.3 API (signatures from `__text_signature__`)
- `gen_settings(model, output, py_run_args=None)`
- `compile_circuit(model, compiled_circuit, settings_path)`
- `get_srs(settings_path, logrows=None, srs_path=None, commitment=None)` — async (repo awaits it; confirmed in smoke test)
- `setup(model, vk_path, pk_path, srs_path=None, witness_path=None, disable_selector_compression=...)`
- `gen_witness(data, model, output, vk_path=None, srs_path=None)` — async
- `prove(witness, model, pk_path, proof_path=None, proof_type=..., srs_path=None)`
- `verify(proof_path, settings_path, vk_path, srs_path=None, reduced_srs=...)`
- `felt_to_float(felt, scale)`, `float_to_felt(input, scale)` available.
