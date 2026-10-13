#!/usr/bin/env python
"""Precision-fixed variant of bench_zkp.py (benchmark-only fix; fl/src/zkp.py untouched).

Reuses bench_zkp.py's driver/worker unchanged, except:
  * zkp.AggregationSetup is replaced (module attribute, in the worker process only) by a subclass whose
    _setup_run_args calls super() and then sets input_scale / param_scale (/ decomp_legs) from env vars
    BENCH_INPUT_SCALE, BENCH_PARAM_SCALE, BENCH_DECOMP_LEGS (unset = keep repo/ezkl default).
  * results go to --csv (default eval/results_fixed.csv); config names carry a scale tag.

Usage:
  python bench_zkp_fixed.py run --in 10 --ps 25 --legs 3 --csv results_fixed.csv --sweep S1_fix --K 2 --d 100 --logrows auto
"""
import os
import sys

import bench_zkp as B

B.FIELDS = B.FIELDS + ["req_input_scale", "req_param_scale", "req_decomp_legs", "out_nonzero_frac"]


def _tag():
    return "_in{}_p{}_l{}".format(os.environ.get("BENCH_INPUT_SCALE", "d"), os.environ.get("BENCH_PARAM_SCALE", "d"),
                                  os.environ.get("BENCH_DECOMP_LEGS", "d"))


_orig_cfg_name = B.cfg_name
B.cfg_name = lambda sweep, K, d, L: _orig_cfg_name(sweep, K, d, L) + _tag()
B.__file__ = os.path.abspath(__file__)  # driver spawns this script as the worker

_orig_append = B.append_row


def _append(row):
    for k, e in (("req_input_scale", "BENCH_INPUT_SCALE"), ("req_param_scale", "BENCH_PARAM_SCALE"),
                 ("req_decomp_legs", "BENCH_DECOMP_LEGS")):
        row[k] = os.environ.get(e, "default")
    try:
        import json
        with open(os.path.join(B.RUNS, B.cfg_name(row["sweep"], row["K"], row["d"], row["logrows_req"]),
                               "result.json")) as f:
            row["out_nonzero_frac"] = json.load(f).get("out_nonzero_frac")
    except Exception:
        pass
    _orig_append(row)


B.append_row = _append


def patch_zkp():
    sys.path.insert(0, os.path.join(B.REPO, "fl", "src"))
    import zkp

    base = zkp.AggregationSetup

    class FixedScaleSetup(base):
        def _setup_run_args(self, logrows: int = 10):
            ra = super()._setup_run_args(logrows)
            if os.environ.get("BENCH_INPUT_SCALE"):
                ra.input_scale = int(os.environ["BENCH_INPUT_SCALE"])
            if os.environ.get("BENCH_PARAM_SCALE"):
                ra.param_scale = int(os.environ["BENCH_PARAM_SCALE"])
            if os.environ.get("BENCH_DECOMP_LEGS"):
                ra.decomp_legs = int(os.environ["BENCH_DECOMP_LEGS"])
            return ra

    zkp.AggregationSetup = FixedScaleSetup


def main():
    argv = sys.argv[1:]

    def pop(flag, env=None):
        if flag in argv:
            i = argv.index(flag)
            v = argv[i + 1]
            del argv[i:i + 2]
            if env:
                os.environ[env] = v
            return v
        return None

    pop("--in", "BENCH_INPUT_SCALE")
    pop("--ps", "BENCH_PARAM_SCALE")
    pop("--legs", "BENCH_DECOMP_LEGS")
    csv_name = pop("--csv") or "results_fixed.csv"
    B.CSV_PATH = os.path.join(B.EVAL, csv_name)
    if argv and argv[0] == "worker":
        patch_zkp()
    sys.argv = [sys.argv[0]] + argv
    B.main()


if __name__ == "__main__":
    main()
