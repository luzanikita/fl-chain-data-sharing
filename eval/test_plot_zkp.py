import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import plot_zkp  # noqa: E402

CSV = os.path.join(os.path.dirname(__file__), "results_fixed.csv")


def test_select_series_min():
    df = pd.read_csv(CSV)
    s = plot_zkp.select_series(df)
    assert list(s["min"]["d"]) == [100, 1000, 5000, 10000, 20000, 40000, 62006, 124012, 250000]
    assert (s["min"]["K"] == 2).all()


def test_select_series_lr20():
    df = pd.read_csv(CSV)
    s = plot_zkp.select_series(df)
    assert list(s["lr20"]["d"]) == [100, 1000, 5000, 10000, 20000, 40000, 62006]
    assert (s["lr20"]["logrows_used"] == 20).all()


def test_select_series_clients():
    df = pd.read_csv(CSV)
    s = plot_zkp.select_series(df)
    assert list(s["clients"]["K"]) == [1, 2, 4, 8, 16, 32]
    assert (s["clients"]["d"] == 5000).all()


def test_loglog_fit_recovers_power_law():
    x = np.array([10.0, 100.0, 1000.0])
    slope, coef = plot_zkp.loglog_fit(x, 3.0 * x**1.5)
    assert abs(slope - 1.5) < 1e-9
    assert abs(coef - 3.0) < 1e-9


def test_make_table_rows_and_format():
    df = pd.read_csv(CSV)
    tex = plot_zkp.make_table(plot_zkp.select_series(df))
    assert tex.startswith("\\begin{tabular}")
    assert "\\toprule" in tex and "\\bottomrule" in tex
    body = [line for line in tex.splitlines() if line.rstrip().endswith("\\\\") and line[:1].isdigit()]
    assert len(body) == 9 + 7  # min series + lr20 series
    assert any(line.startswith("62\\,006 & 16 &") for line in body)
    assert any(line.startswith("62\\,006 & 20 &") for line in body)
