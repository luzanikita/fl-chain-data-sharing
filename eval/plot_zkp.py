"""Combined figure and LaTeX table for the ZKP scaling experiment.

Reads eval/results_fixed.csv (fixed-precision runs) and writes
eval/figs/zkp_scaling.{pdf,png} and eval/results_table.tex.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "results_fixed.csv")
PAPER_D = 62006


def select_series(df: pd.DataFrame) -> dict:
    """Return {'min', 'lr20', 'clients'} DataFrames, sorted, status ok only."""
    ok = df[df["status"] == "ok"]
    min_ = ok[ok["sweep"].isin(["S1_fix", "S1min_fix"]) & (ok["K"] == 2)]
    lr20 = pd.concat([
        ok[ok["sweep"] == "S1lr20_fix"],
        ok[(ok["sweep"] == "S3_fix") & (ok["d"] == PAPER_D) & (ok["logrows_used"] == 20)],
    ])
    clients = ok[ok["sweep"] == "S2_fix"]
    return {
        "min": min_.sort_values("d").reset_index(drop=True),
        "lr20": lr20.sort_values("d").reset_index(drop=True),
        "clients": clients.sort_values("K").reset_index(drop=True),
    }


def loglog_fit(x, y):
    """Least-squares fit y = coef * x**slope in log space. Returns (slope, coef)."""
    slope, intercept = np.polyfit(np.log(np.asarray(x, float)), np.log(np.asarray(y, float)), 1)
    return float(slope), float(np.exp(intercept))


def _thousands(n: int) -> str:
    return f"{int(n):,}".replace(",", "\\,")


def make_table(series: dict) -> str:
    """booktabs tabular: one row per K=2 config, min-valid logrows block then logrows-20 block."""
    head = (
        "\\begin{tabular}{rrrrrrrrr}\n\\toprule\n"
        "$d$ & logrows & rows & PK (GB) & prove (s) & verify (s) & RAM (GB) & err. (\\%) & overhead ($\\times10^{5}$) \\\\\n"
        "\\midrule\n"
    )
    blocks = []
    for key in ("min", "lr20"):
        lines = []
        for _, r in series[key].iterrows():
            lines.append(
                f"{_thousands(r['d'])} & {int(r['logrows_used'])} & {_thousands(r['num_rows'])} & "
                f"{r['pk_bytes'] / 1e9:.2f} & {r['t_prove_med_s']:.1f} & {r['t_verify_med_s']:.2f} & "
                f"{r['peak_rss_bytes'] / 1e9:.1f} & {100 * r['out_rel_l2_err']:.2f} & "
                f"{r['overhead_prove_vs_plain'] / 1e5:.1f} \\\\"
            )
        blocks.append("\n".join(lines))
    return head + "\n\\midrule\n".join(blocks) + "\n\\bottomrule\n\\end{tabular}\n"


INK, GREY = "#222222", "#888888"
STYLE = {
    "min": dict(marker="o", color=INK, mfc="white", label="min. logrows"),
    "lr20": dict(marker="s", color=GREY, mfc="white", label="logrows 20"),
}


def _err(df, col):
    med = df[f"{col}_med_s"]
    return med, [med - df[f"{col}_min_s"], df[f"{col}_max_s"] - med]


def _points(ax, x, y, yerr=None, **kw):
    ax.errorbar(x, y, yerr=yerr, fmt=kw.pop("marker"), ms=3.5, lw=0, elinewidth=0.7,
                capsize=1.5, **kw)


def make_figure(series: dict):
    plt.rcParams.update({
        "font.family": "serif", "font.size": 7, "axes.labelsize": 7,
        "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.5,
    })
    fig, axs = plt.subplots(3, 2, figsize=(3.5, 4.1), constrained_layout=True)
    fig.set_constrained_layout_pads(h_pad=0.08, w_pad=0.04)
    mn, lr = series["min"], series["lr20"]

    # (a) circuit rows vs d, with log-log fit
    ax = axs[0, 0]
    _points(ax, mn["d"], mn["num_rows"], **STYLE["min"])
    slope, coef = loglog_fit(mn["d"], mn["num_rows"])
    xx = np.geomspace(mn["d"].min(), mn["d"].max(), 50)
    ax.plot(xx, coef * xx**slope, "--", color=GREY, lw=0.8)
    ax.text(0.97, 0.1, f"slope {slope:.2f}", transform=ax.transAxes, fontsize=6, color="#555", ha="right")
    ax.set_ylabel("Circuit rows")

    # (b) proving key, (c) prove time, (d) verify time - vs d
    for key, df in (("min", mn), ("lr20", lr)):
        _points(axs[0, 1], df["d"], df["pk_bytes"] / 1e9, **STYLE[key])
        y, e = _err(df, "t_prove")
        _points(axs[1, 0], df["d"], y, e, **STYLE[key])
    y, e = _err(mn, "t_verify")
    _points(axs[1, 1], mn["d"], y, e, **STYLE["min"])
    axs[0, 1].set_ylabel("Proving key (GB)")
    axs[1, 0].set_ylabel("Prove time (s)")
    axs[1, 1].set_ylabel("Verify time (s)")
    h, l = axs[0, 1].get_legend_handles_labels()
    axs[0, 0].legend(h, l, loc="upper left", frameon=False, handletextpad=0.2, borderaxespad=0.2)

    for ax in (axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]):
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.axvline(PAPER_D, color=GREY, ls=":", lw=0.7)
        ax.set_xlabel("Parameters $d$ ($K$=2)")

    # (e) prove time vs K, (f) proving key vs K, with sequential estimates
    cl = series["clients"]
    k1 = cl[cl["K"] == 1].iloc[0]
    y, e = _err(cl, "t_prove")
    _points(axs[2, 0], cl["K"], y, e, **{**STYLE["min"], "label": "measured"})
    axs[2, 0].plot(cl["K"], cl["K"] * k1["t_prove_med_s"], "--", color=GREY, lw=0.8, label="sequential (est.)")
    axs[2, 0].set_ylabel("Prove time (s)")
    axs[2, 0].legend(loc="upper left", frameon=False, handletextpad=0.3)
    _points(axs[2, 1], cl["K"], cl["pk_bytes"] / 1e9, **STYLE["min"])
    axs[2, 1].axhline(k1["pk_bytes"] / 1e9, color=GREY, ls="--", lw=0.8)
    axs[2, 1].set_ylabel("Proving key (GB)")
    for ax in (axs[2, 0], axs[2, 1]):
        ax.set_xscale("log", base=2)
        ax.set_xticks(list(cl["K"])); ax.set_xticklabels([str(k) for k in cl["K"]])
        ax.set_xlabel("Clients $K$ ($d$=5000)")
        ax.set_ylim(bottom=0)

    for ax, letter in zip(axs.flat, "abcdef"):
        ax.text(0.0, 1.03, f"({letter})", transform=ax.transAxes, fontsize=7, fontweight="bold", va="bottom", ha="left")
    return fig


def main():
    series = select_series(pd.read_csv(CSV))
    os.makedirs(os.path.join(HERE, "figs"), exist_ok=True)
    fig = make_figure(series)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, "figs", f"zkp_scaling.{ext}"), dpi=300)
    with open(os.path.join(HERE, "results_table.tex"), "w") as f:
        f.write(make_table(series))
    print("wrote figs/zkp_scaling.{pdf,png} and results_table.tex")


if __name__ == "__main__":
    main()
