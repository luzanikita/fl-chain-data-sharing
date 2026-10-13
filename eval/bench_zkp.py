#!/usr/bin/env python
"""ZKP scaling benchmark for fl/src/zkp.py (driver + worker).

Driver:  python bench_zkp.py run --sweep S1 --K 2 --d 100 1000 --logrows 10
Worker:  python bench_zkp.py worker --sweep S1 --K 2 --d 100 --logrows 10   (spawned by driver)

One fresh worker subprocess per config (sweep, K, d, logrows). Results appended to eval/results.csv
(resumable: configs already present are skipped). Repo classes from fl/src/zkp.py are used unmodified;
ezkl module functions are wrapped (monkeypatched at module attribute level, not in zkp.py) only to
time the individual setup stages and capture exception text.
"""
import argparse
import asyncio
import csv
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback

EVAL = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(EVAL)
RUNS = os.path.join(EVAL, "runs")
SETTINGS_DIR = os.path.join(EVAL, "settings")
CSV_PATH = os.path.join(EVAL, "results.csv")

SETUP_TIMEOUT_S = 30 * 60
ONLINE_BUDGET_S = 10 * 60  # witness + first prove
MIN_FREE_GB = 20
RSS_KILL_BYTES = 20 * 2**30
N_REPS = 3
NUM_EXAMPLES = 25000.0

FIELDS = [
    "sweep", "K", "d", "logrows_req", "logrows_rule", "status",
    "logrows_used", "num_rows", "total_assignments", "num_inner_cols", "total_const_size",
    "input_scale", "param_scale", "output_scale", "scale_rebase_multiplier", "lookup_range",
    "required_range_checks", "required_lookups", "decomp_base", "decomp_legs",
    "t_srs_s", "t_setup_s", "t_export_s", "t_gen_settings_s", "t_compile_s", "t_keygen_s",
    "t_witness_s", "t_prove_med_s", "t_prove_min_s", "t_prove_max_s", "n_prove",
    "t_verify_med_s", "t_verify_min_s", "t_verify_max_s", "n_verify", "verify_ok",
    "pk_bytes", "vk_bytes", "proof_bytes", "compiled_bytes", "onnx_bytes", "witness_bytes", "data_bytes",
    "peak_rss_bytes", "peak_rss_sampled_bytes", "swap_used_start_bytes", "swap_used_max_bytes",
    "t_plain_med_s", "overhead_prove_vs_plain",
    "out_max_abs_err", "out_mean_abs_err", "out_max_rel_err", "out_rel_l2_err", "true_max_abs",
    "wall_total_s", "error",
]


def cfg_name(sweep, K, d, logrows):
    return f"{sweep}_K{K}_d{d}_lr{logrows}"


# --------------------------------------------------------------------------------------------- worker
def worker(args):
    sys.path.insert(0, os.path.join(REPO, "fl", "src"))
    import numpy as np
    import torch
    import ezkl
    import zkp

    K, d, L = args.K, args.d, args.logrows
    name = cfg_name(args.sweep, K, d, L)
    rdir = os.path.join(RUNS, name)
    shutil.rmtree(rdir, ignore_errors=True)
    os.makedirs(rdir)
    res_path = os.path.join(rdir, "result.json")
    phase_path = os.path.join(rdir, "phase.json")
    res = {}

    def save():
        with open(res_path, "w") as f:
            json.dump(res, f, indent=1)

    def phase(p):
        with open(phase_path, "w") as f:
            json.dump({"phase": p, "t": time.time()}, f)

    # ---- timing wrappers around ezkl functions used by zkp.py (zkp.py itself untouched)
    stage_t = {}
    errors = []

    def wrap_sync(fn, key):
        def w(*a, **k):
            t = time.perf_counter()
            try:
                return fn(*a, **k)
            except Exception as e:
                errors.append(f"{key}: {e}")
                raise
            finally:
                stage_t[key] = time.perf_counter() - t
        return w

    def wrap_async(fn, key):
        async def w(*a, **k):
            t = time.perf_counter()
            try:
                return await fn(*a, **k)
            except Exception as e:
                errors.append(f"{key}: {e}")
                raise
            finally:
                stage_t[key] = time.perf_counter() - t
        return w

    orig_get_srs = ezkl.get_srs

    # ---- inputs
    rng = np.random.default_rng(0)
    W = rng.normal(0, 0.05, (K, d)).astype(np.float32)
    n = np.full(K, NUM_EXAMPLES, dtype=np.float32)
    inp = (torch.tensor(W, dtype=torch.float32), torch.tensor(n, dtype=torch.float32))

    # plain baseline
    ts = []
    for _ in range(100):
        t = time.perf_counter()
        ref = (W * n[:, None]).sum(0) / n.sum()
        ts.append(time.perf_counter() - t)
    res["t_plain_med_s"] = float(np.median(ts))
    ref64 = (W.astype(np.float64) * n.astype(np.float64)[:, None]).sum(0) / n.astype(np.float64).sum()

    # ---- pre-step (untimed): probe settings to learn logrows actually used, warm SRS
    phase("srs")
    pre = zkp.AggregationSetup(zkp.AggregateModel(), os.path.join(rdir, "probe"))
    pre._export_model(inp)
    ezkl.gen_settings(pre.model_path, pre.settings_path, py_run_args=pre._setup_run_args(L))
    t = time.perf_counter()
    async def _srs():
        return await orig_get_srs(pre.settings_path)
    asyncio.run(_srs())
    res["t_srs_s"] = time.perf_counter() - t
    shutil.rmtree(os.path.join(rdir, "probe"), ignore_errors=True)
    save()

    ezkl.gen_settings = wrap_sync(ezkl.gen_settings, "gen_settings")
    ezkl.compile_circuit = wrap_sync(ezkl.compile_circuit, "compile")
    ezkl.setup = wrap_sync(ezkl.setup, "keygen")
    ezkl.get_srs = wrap_async(orig_get_srs, "srs_in_setup")

    # ---- setup (timed, whole AggregationSetup.setup)
    phase("setup")
    setup = zkp.AggregationSetup(zkp.AggregateModel(), rdir)
    orig_export = setup._export_model

    def timed_export(x):
        t = time.perf_counter()
        orig_export(x)
        stage_t["export"] = time.perf_counter() - t
    setup._export_model = timed_export  # instance attribute only

    t = time.perf_counter()
    ok = asyncio.run(setup.setup(inp, logrows=L))
    res["t_setup_s"] = time.perf_counter() - t
    res["t_export_s"] = stage_t.get("export")
    res["t_gen_settings_s"] = stage_t.get("gen_settings")
    res["t_compile_s"] = stage_t.get("compile")
    res["t_keygen_s"] = stage_t.get("keygen")
    res["t_srs_in_setup_s"] = stage_t.get("srs_in_setup")

    def fsize(p):
        return os.path.getsize(p) if os.path.isfile(p) else None

    res["onnx_bytes"] = fsize(setup.model_path)
    res["compiled_bytes"] = fsize(setup.compiled_model_path)
    res["pk_bytes"] = fsize(setup.pk_path)
    res["vk_bytes"] = fsize(setup.vk_path)
    if os.path.isfile(setup.settings_path):
        with open(setup.settings_path) as f:
            res["settings"] = json.load(f)
    if not ok:
        res["status"] = "error: setup failed: " + " | ".join(errors) if errors else "error: setup returned False"
        save()
        return
    save()

    # ---- witness + prove
    prover = zkp.AggregationProver(setup.compiled_model_path, setup.pk_path)
    phase("online")
    t_online = time.perf_counter()
    t = time.perf_counter()
    data_path, wit_path = asyncio.run(prover.generate_witness(inp, os.path.join(rdir, "agg")))
    res["t_witness_s"] = time.perf_counter() - t
    res["witness_bytes"] = fsize(wit_path)
    res["data_bytes"] = fsize(data_path)

    # circuit output error
    with open(wit_path) as f:
        wit = json.load(f)
    out_scale = res.get("settings", {}).get("model_output_scales", [None])[0]
    try:
        out = np.array([float(x) for x in wit["pretty_elements"]["rescaled_outputs"][0]], dtype=np.float64)
        src = "pretty_elements.rescaled_outputs"
    except Exception:
        out = np.array([ezkl.felt_to_float(x, out_scale) for x in wit["outputs"][0]], dtype=np.float64)
        src = "felt_to_float"
    # cross-check first few with felt_to_float
    chk = [ezkl.felt_to_float(x, out_scale) for x in wit["outputs"][0][:5]]
    res["out_decode_src"] = src
    res["out_felt_check"] = [float(c) for c in chk]
    res["out_first5"] = out[:5].tolist()
    res["ref_first5"] = ref64[:5].tolist()
    err = out - ref64
    res["out_max_abs_err"] = float(np.max(np.abs(err)))
    res["out_mean_abs_err"] = float(np.mean(np.abs(err)))
    mask = np.abs(ref64) > 1e-3
    res["out_max_rel_err"] = float(np.max(np.abs(err[mask]) / np.abs(ref64[mask]))) if mask.any() else None
    res["out_rel_l2_err"] = float(np.linalg.norm(err) / np.linalg.norm(ref64))
    res["true_max_abs"] = float(np.max(np.abs(ref64)))
    res["out_nonzero_frac"] = float(np.mean(out != 0))
    save()

    with open(phase_path, "w") as f:
        json.dump({"phase": "online", "t": time.time() - (time.perf_counter() - t_online)}, f)

    proof_path = os.path.join(rdir, "aggregation_proof.pf")
    tp, tv, vres = [], [], []
    for i in range(N_REPS):
        t = time.perf_counter()
        okp = asyncio.run(prover.prove(wit_path, proof_path))
        tp.append(time.perf_counter() - t)
        if not okp:
            res["status"] = "error: prove returned False"
            break
        if i == 0 and (time.perf_counter() - t_online) > ONLINE_BUDGET_S:
            res["exceeds_budget_after_first"] = True
            break
        phase(f"prove{i+1}")
    res["t_prove_list"] = tp
    res["proof_bytes"] = fsize(proof_path)
    save()

    if os.path.isfile(proof_path):
        phase("verify")
        verifier = zkp.AggregationVerifier(setup.settings_path, setup.vk_path)
        nv = 1 if res.get("exceeds_budget_after_first") else N_REPS
        for _ in range(nv):
            t = time.perf_counter()
            vres.append(bool(verifier.verify(proof_path)))
            tv.append(time.perf_counter() - t)
    res["t_verify_list"] = tv
    res["verify_list"] = vres
    if "status" not in res:
        res["status"] = "exceeds_budget" if res.get("exceeds_budget_after_first") else "ok"
    phase("done")
    save()


# --------------------------------------------------------------------------------------------- driver
def done_configs():
    if not os.path.isfile(CSV_PATH):
        return set()
    with open(CSV_PATH) as f:
        return {(r["sweep"], int(r["K"]), int(r["d"]), int(r["logrows_req"])) for r in csv.DictReader(f)}


def append_row(row):
    new = not os.path.isfile(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def swap_used():
    try:
        import psutil
        return psutil.swap_memory().used
    except Exception:
        return None


def run_one(sweep, K, d, L, rule="fixed"):
    import psutil
    import statistics as st

    name = cfg_name(sweep, K, d, L)
    free_gb = shutil.disk_usage(EVAL).free / 2**30
    if free_gb < MIN_FREE_GB:
        print(f"ABORT: free disk {free_gb:.1f} GB < {MIN_FREE_GB} GB")
        sys.exit(2)
    rdir = os.path.join(RUNS, name)
    os.makedirs(RUNS, exist_ok=True)
    log = open(os.path.join(RUNS, name + ".log"), "w")
    cmd = [sys.executable, os.path.abspath(__file__), "worker", "--sweep", sweep,
           "--K", str(K), "--d", str(d), "--logrows", str(L)]
    print(f"[{time.strftime('%H:%M:%S')}] start {name} (free disk {free_gb:.0f} GB)", flush=True)
    t0 = time.time()
    env = dict(os.environ, EZKL_REPO_PATH=os.path.join(EVAL, "ezkl_repo"))  # local SRS (see 15_srs.md)
    p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=EVAL, env=env)
    ps = psutil.Process(p.pid)
    sw0 = swap_used()
    swmax = sw0 or 0
    rss_max = 0
    killed = None
    rusage = None
    status_code = None
    while True:
        pid, sc, ru = os.wait4(p.pid, os.WNOHANG)
        if pid == p.pid:
            rusage, status_code = ru, sc
            break
        try:
            rss_max = max(rss_max, ps.memory_info().rss)
        except Exception:
            pass
        s = swap_used()
        if s is not None:
            swmax = max(swmax, s)
        ph = None
        try:
            with open(os.path.join(rdir, "phase.json")) as f:
                ph = json.load(f)
        except Exception:
            pass
        if killed is None and ph:
            el = time.time() - ph["t"]
            if ph["phase"] == "setup" and el > SETUP_TIMEOUT_S:
                killed = "timeout"
            elif ph["phase"] == "online" and el > ONLINE_BUDGET_S:
                killed = "exceeds_budget"
            if rss_max > RSS_KILL_BYTES:
                killed = "oom"
            if killed:
                print(f"  killing worker: {killed} (phase {ph['phase']}, {el:.0f}s, rss {rss_max/2**30:.1f} GB)", flush=True)
                os.kill(p.pid, signal.SIGKILL)
        time.sleep(0.5)
    p.returncode = status_code
    log.close()

    res = {}
    try:
        with open(os.path.join(rdir, "result.json")) as f:
            res = json.load(f)
    except Exception:
        pass
    s = res.get("settings", {})
    ra = s.get("run_args", {})
    row = {"sweep": sweep, "K": K, "d": d, "logrows_req": L, "logrows_rule": rule,
           "logrows_used": ra.get("logrows"), "num_rows": s.get("num_rows"),
           "total_assignments": s.get("total_assignments"), "num_inner_cols": ra.get("num_inner_cols"),
           "total_const_size": s.get("total_const_size"),
           "input_scale": ra.get("input_scale"), "param_scale": ra.get("param_scale"),
           "output_scale": (s.get("model_output_scales") or [None])[0],
           "scale_rebase_multiplier": ra.get("scale_rebase_multiplier"),
           "lookup_range": json.dumps(ra.get("lookup_range")),
           "required_range_checks": json.dumps(s.get("required_range_checks")),
           "required_lookups": json.dumps(s.get("required_lookups")),
           "decomp_base": ra.get("decomp_base"), "decomp_legs": ra.get("decomp_legs"),
           "peak_rss_bytes": rusage.ru_maxrss if rusage else None,  # bytes on macOS
           "peak_rss_sampled_bytes": rss_max, "swap_used_start_bytes": sw0, "swap_used_max_bytes": swmax,
           "wall_total_s": round(time.time() - t0, 2)}
    for k in FIELDS:
        if k in res and k not in row:
            row[k] = res[k]
    tp, tv = res.get("t_prove_list", []), res.get("t_verify_list", [])
    if tp:
        row.update(t_prove_med_s=st.median(tp), t_prove_min_s=min(tp), t_prove_max_s=max(tp), n_prove=len(tp))
        if res.get("t_plain_med_s"):
            row["overhead_prove_vs_plain"] = st.median(tp) / res["t_plain_med_s"]
    if tv:
        row.update(t_verify_med_s=st.median(tv), t_verify_min_s=min(tv), t_verify_max_s=max(tv), n_verify=len(tv))
    if res.get("verify_list"):
        row["verify_ok"] = all(res["verify_list"])
    if killed:
        row["status"] = killed
    elif "status" in res:
        row["status"] = res["status"]
    else:
        row["status"] = f"error: worker exit status {status_code}, phase {res and 'partial'}; see runs/{name}.log"
        try:
            with open(os.path.join(RUNS, name + ".log")) as f:
                row["error"] = f.read()[-800:].replace("\n", " | ")
        except Exception:
            pass
    if res.get("status", "").startswith("error"):
        row["error"] = res["status"]

    # keep settings, vk, proof; record full result; delete pk / witness / data
    os.makedirs(SETTINGS_DIR, exist_ok=True)
    sp = os.path.join(rdir, "settings.json")
    if os.path.isfile(sp):
        shutil.copy(sp, os.path.join(SETTINGS_DIR, name + ".json"))
    for fn in ("test.pk", "agg_witness.json", "agg_data.json"):
        try:
            os.remove(os.path.join(rdir, fn))
        except FileNotFoundError:
            pass
    shutil.rmtree(os.path.join(rdir, "probe"), ignore_errors=True)
    append_row(row)
    print(f"  -> {row['status']}  logrows_used={row.get('logrows_used')} rows={row.get('num_rows')} "
          f"cols={row.get('num_inner_cols')} t_setup={row.get('t_setup_s')} t_prove={row.get('t_prove_med_s')} "
          f"pk={row.get('pk_bytes')} rss={row.get('peak_rss_bytes')} err={row.get('out_max_abs_err')}", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "worker"])
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--K", type=int, nargs="+", required=True)
    ap.add_argument("--d", type=int, nargs="+", required=True)
    ap.add_argument("--logrows", nargs="+", default=["10"],
                    help="ints, or 'auto' = max(10, ceil(log2(d+6))) (min valid: public outputs share one instance column)")
    a = ap.parse_args()
    if a.mode == "worker":
        a.K, a.d, a.logrows = a.K[0], a.d[0], int(a.logrows[0])
        worker(a)
        return
    done = done_configs()
    import math
    for Ls in a.logrows:
        for K in a.K:
            for d in a.d:
                if Ls == "auto":
                    L, rule = max(10, math.ceil(math.log2(d + 6))), "min_valid=max(10,ceil(log2(d+6)))"
                else:
                    L, rule = int(Ls), "fixed"
                if (a.sweep, K, d, L) in done:
                    print(f"skip {cfg_name(a.sweep, K, d, L)} (already in CSV)")
                    continue
                run_one(a.sweep, K, d, L, rule)


if __name__ == "__main__":
    main()
