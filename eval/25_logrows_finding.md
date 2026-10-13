# 25 — logrows finding (S0: K=2, logrows=10 requested)

| d | settings logrows | num_rows | total_assignments | num_inner_cols | status | pk | prove (med) |
|---|---|---|---|---|---|---|---|
| 100 | 10 | 450 | 901 | 2 | ok, verify True | 39.9 MB | 0.48 s |
| 1000 | 10 | 4052 | 8105 | 2 | ok, verify True | 76.7 MB | 1.61 s |
| 5000 | 10 | 20053 | — | 2 | `error: setup failed: keygen: Failed to run setup: [halo2] General synthesis error` | — | — |
| 10000 | 10 | 40061 | — | 2 | same error | — | — |

Log line before the error (both failing configs): `[circuit] [halo2] k = 10 is too small for the given circuit. Try using a larger value of k`.

Conclusions:
- `gen_settings` **never bumps logrows** (settings.json logrows = 10 in all four). `num_inner_cols` stays 2 in settings;
  when `num_rows > 2^10` (d=1000: 4052 rows) EZKL widens the circuit by adding advice-column blocks internally
  (column counts are not written to settings.json; not captured — Python binding emits no debug log).
  Range-check table also needs 17 columns at logrows 10 ("Using 17 columns for range-check" in logs).
- d=1000 at logrows=10 works (4× more rows than 2^10 → more columns, pk 77 MB vs 40 MB).
- d ≥ ~1018 fails at keygen: all d public outputs go in a single instance column of 2^logrows rows (minus 6 reserved);
  consistent with the source research (`50_ezkl_source_research.md`). Fails fast (<1 s), during `ezkl.setup`
  (gen_settings/compile succeed). Hence **the paper configuration (K=2, d=62,006) cannot have run with logrows=10**;
  minimum is logrows 16 (2^16 = 65,536 ≥ 62,012).
- Consequence: S1/S2 use `logrows = max(10, ceil(log2(d+6)))` (CSV `logrows_rule`), S3 = K=2, d=10k, logrows ∈ {14,16,18}.
