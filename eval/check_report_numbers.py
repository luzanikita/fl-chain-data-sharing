"""Print every headline number REPORT.md quotes, directly from results_fixed.csv."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_zkp import CSV, PAPER_D, loglog_fit, select_series  # noqa: E402

s = select_series(pd.read_csv(CSV))
mn, lr, cl = s["min"], s["lr20"], s["clients"]

a, b = np.polyfit(mn["d"], mn["num_rows"], 1)
print(f"rows = {a:.4f}*d + {b:.1f} (K=2, d {mn['d'].min()}..{mn['d'].max()})")
print(f"rows log-log slope: {loglog_fit(mn['d'], mn['num_rows'])[0]:.3f}")
for name, df in (("min", mn), ("lr20", lr)):
    for _, r in df.iterrows():
        print(f"{name:5} d={r['d']:>6} lr={r['logrows_used']} pk={r['pk_bytes']/1e9:.2f}GB "
              f"prove={r['t_prove_med_s']:.1f}s verify={r['t_verify_med_s']*1e3:.0f}ms "
              f"rss={r['peak_rss_bytes']/1e9:.1f}GB err={100*r['out_rel_l2_err']:.2f}%")
k1, k32 = cl.iloc[0], cl.iloc[-1]
print(f"K 1->{k32['K']}: rows x{k32['num_rows']/k1['num_rows']:.1f}, pk x{k32['pk_bytes']/k1['pk_bytes']:.1f}, "
      f"prove x{k32['t_prove_med_s']/k1['t_prove_med_s']:.1f}")
print(f"sequential est K={k32['K']}: prove {k32['K']*k1['t_prove_med_s']:.1f}s vs measured {k32['t_prove_med_s']:.1f}s; "
      f"pk {k1['pk_bytes']/1e6:.0f}MB vs {k32['pk_bytes']/1e9:.2f}GB")
paper = mn[mn["d"] == PAPER_D].iloc[0]
print(f"paper config min-logrows: lr={paper['logrows_used']}, pk={paper['pk_bytes']/1e9:.2f}GB, prove={paper['t_prove_med_s']:.1f}s")
