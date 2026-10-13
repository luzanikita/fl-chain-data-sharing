# 40 — S1 small-d results (K=2, logrows = max(10, ceil(log2(d+6))))

| d | logrows | num_rows | t_setup s | t_prove med s | pk MB | peak RSS GB | out max abs err |
|---|---|---|---|---|---|---|---|
| 100 | 10 | 450 | 0.15 | 0.46 | 39.9 | 0.54 | 0.076 |
| 1000 | 10 | 4050 | 0.44 | 1.62 | 76.7 | 1.13 | 0.132 |
| 5000 | 13 | 20050 | 1.12 | 1.03 | 126.9 | 0.85 | 0.141 |
| 10000 | 14 | 40050 | 2.21 | 1.88 | 264.4 | 1.31 | 0.133 |

All verify=True. All outputs exactly 0 (max abs err = max |true avg|).
Note d=1000 at logrows 10 (many columns) is slower to prove (1.62 s) than d=5000 at logrows 13 (1.03 s): wide-and-short layout is costly.

Extrapolation guard for 20k/40k/62006 (logrows 15/16/16): pk roughly doubles per logrows step → ~0.5–1.2 GB, setup ≲ 15 s, RAM ≲ 4 GB — far below the 25 min / 16 GB / 30 GB guard → run them.
