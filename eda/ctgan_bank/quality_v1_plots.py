"""Insight plots for the CTGAN v1 quality checks (issues #49 and #58).

One figure per finding, each titled with the finding itself, sized 16:9 for slides. Run
``eda/ctgan_bank/quality_v1.py`` first: it writes the tables these plots read from
``artifacts/quality_v1/``. This script recomputes the rest from the data and the tools.

Colour follows the entity in every figure: real data blue, CTGAN v1 orange, SMOTENC aqua
(slots 1 to 3 of the reference palette, validated for colour-vision safety). Pass and fail use
the reserved status colours and always come with an icon and a word.

Run from the repo root:  uv run python eda/ctgan_bank/quality_v1_plots.py
"""

import json
import textwrap
from collections import Counter
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402

from agentic_gan.tools.quality_fidelity import (  # noqa: E402
    BANK_RULES,
    CRAMERS_V,
    MEAN_DISTANCE,
    SPEARMAN,
    WORST_DISTANCE,
    column_kinds,
    cramers_v,
    fidelity_checks,
    ks_distance,
    real_vs_real,
    spearman_matrix,
    suggest_thresholds,
    tv_distance,
)

# private helper: the same distance space the gate uses for the closest-record check
from agentic_gan.tools.quality_privacy import _preprocessor  # noqa: E402
from agentic_gan.utils.datasets import DATASETS  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "interim" / "bank_marketing"
ART = Path(__file__).resolve().parent / "artifacts" / "quality_v1"
OUT = ART / "plots"
KYLE = ROOT / "results" / "augmented_bank_marketing.csv"

# reference palette, light mode
REAL, V1, SMOTE = "#2a78d6", "#eb6834", "#1baf7a"
V1_TINT = "#f6bba4"  # v1 orange, lightened, for bars that are within the limit
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#ffffff"
GOOD, CRIT = "#0ca30c", "#d03b3b"  # status colours, always with an icon and a word

W, H, DPI = 12.0, 6.75, 160
LIMIT = 0.10  # suggested limit for one column's distance
BOTTOM, HEIGHT = 0.155, 0.665  # plot area, leaves two lines for the footnote

plt.rcParams.update(
    {
        "font.family": ["Arial", "DejaVu Sans"],
        "font.size": 12,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "legend.frameon": False,
    }
)

cfg = DATASETS["bank_marketing"]
TARGET = cfg.target


# --- helpers ----------------------------------------------------------------------------------


def new_fig(headline: str, sub: str | None = None, footer: str | None = None):
    fig = plt.figure(figsize=(W, H))
    fig.text(0.045, 0.955, headline, fontsize=19, fontweight="bold", color=INK, va="top")
    if sub:
        fig.text(0.045, 0.893, sub, fontsize=12.5, color=INK2, va="top")
    if footer:
        fig.text(
            0.045,
            0.014,
            textwrap.fill(footer, 172),
            fontsize=9.5,
            color=MUTED,
            va="bottom",
            linespacing=1.35,
        )
    return fig


def style(ax, grid: str | None = "x") -> None:
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", color=AXIS, labelcolor=INK2, length=3)


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=DPI)
    plt.close(fig)
    print("saved", name)


def pct(x: float, digits: int = 0) -> str:
    return f"{100 * x:.{digits}f}%"


def num(x: float, digits: int = 2) -> str:
    return f"{x:.{digits}f}".replace("-", "−")


def legend_entity(fig, items, x=0.985, y=0.965):
    fig.legend(
        handles=items,
        loc="upper right",
        bbox_to_anchor=(x, y),
        ncol=len(items),
        fontsize=11.5,
        labelcolor=INK2,
        handlelength=1.2,
        columnspacing=1.6,
    )


def square(color, label):
    return Patch(facecolor=color, label=label)


def dot(color, label):
    return Line2D([], [], marker="o", linestyle="", markersize=9, color=color, label=label)


def line(color, label):
    return Line2D([], [], color=color, lw=3, label=label)


def ecdf(ax, values, color):
    v = np.sort(np.asarray(values, dtype=float))
    ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=color, lw=2.3, zorder=3)


def load():
    train = pd.read_csv(DATA / "train_cleaned.csv")
    val = pd.read_csv(DATA / "validation_cleaned.csv")
    syn = {n: pd.read_csv(DATA / "synthetic" / f"bank_ctgan_fill{n}.csv") for n in (10, 25, 50)}
    return train, val, syn


# --- 1. the gate scorecard --------------------------------------------------------------------


def fig_scorecard(train, val, syn, pu):
    common = {"target": TARGET, "discrete": cfg.discrete, "rules": BANK_RULES}
    real = train[train[TARGET] == 1]
    baseline = real_vs_real(train, val, **common)
    limits = suggest_thresholds(baseline.values)
    v1 = fidelity_checks(real, syn[25], thresholds=limits, **common).values
    b = baseline.values
    p25 = pu.loc["fill25"]
    pr, f1 = pu["pr_auc_gain"], pu["best_f1_gain"]

    rows = [
        ("Fidelity", None),
        (
            "Mean column distance",
            f"{b[MEAN_DISTANCE]:.3f}",
            f"{v1[MEAN_DISTANCE]:.3f}",
            "≤ 0.05",
            "fail",
        ),
        (
            "Worst column distance (job)",
            f"{b[WORST_DISTANCE]:.3f}",
            f"{v1[WORST_DISTANCE]:.3f}",
            "≤ 0.10",
            "fail",
        ),
        (
            "Spearman difference, numeric pairs",
            f"{b[SPEARMAN]:.3f}",
            f"{v1[SPEARMAN]:.3f}",
            "≤ 0.045",
            "fail",
        ),
        (
            "Cramér's V difference, categorical pairs",
            f"{b[CRAMERS_V]:.3f}",
            f"{v1[CRAMERS_V]:.3f}",
            "≤ 0.065",
            "pass",
        ),
        ("Consistency rules (share of rows that break them)", None),
        ("pdays and poutcome rule", "0.06%", pct(v1["rule:pdays_poutcome"], 1), "≤ 0.5%", "fail"),
        ("Three other pdays and previous rules †", "0%", "0%", "≤ 0.5%", "pass"),
        ("Label balance, columns match †", "0", "0", "0.01", "pass"),
        ("Privacy", None),
        ("Exact copies, repeated rows †", "0", "0", "0", "pass"),
        (
            "Closest-record ratio (synthetic over real distance)",
            "1.00",
            f"{p25['dcr_ratio']:.2f}",
            "≥ 0.95",
            "pass",
        ),
        ("Utility: change when v1 rows are added to training", None),
        (
            f"PR-AUC at 25% fill (10%: {num(pr['fill10'], 3)}, 50%: {num(pr['fill50'], 3)})",
            "–",
            num(pr["fill25"], 3),
            "≥ 0",
            "fail",
        ),
        (
            f"Best-threshold F1 at 25% (10%: {num(f1['fill10'], 3)}, 50%: {num(f1['fill50'], 3)})",
            "–",
            num(f1["fill25"], 3),
            "≥ 0",
            "fail",
        ),
        ("Distinguishability (shown, not scored)", None),
        (
            "Random forest AUC, real vs synthetic",
            f"{p25['real_vs_real_auc']:.2f}",
            f"{p25['real_vs_synthetic_auc']:.2f}",
            "0.5 = same",
            "info",
        ),
    ]
    fig = new_fig(
        "CTGAN v1 fails the quality gate on fidelity and utility",
        "6 of 15 hard checks fail, and there is no sign of memorization. Fidelity limits are twice the real-vs-real value.",
        "† passes by construction: the sampler keeps only y = 1 rows, filters inconsistent rows and drops exact copies of real rows. "
        "Fidelity and privacy are the same at all three fill levels; the table shows the 25% file. Related checks are grouped, so 15 checks fit in 11 rows.",
    )
    ax = fig.add_axes([0.045, 0.095, 0.935, 0.755])
    ax.axis("off")
    n = len(rows) + 1.3
    ax.set_xlim(0, 100)
    ax.set_ylim(n, 0)
    xs = {"label": 1, "real": 62, "v1": 79, "limit": 90, "status": 92.5}
    for key, text in (("real", "Real data"), ("v1", "CTGAN v1"), ("limit", "Limit")):
        ax.text(
            xs[key], 0.5, text, ha="right", va="center", fontsize=11, color=INK2, fontweight="bold"
        )
    ax.text(
        xs["label"],
        0.5,
        "Check",
        ha="left",
        va="center",
        fontsize=11,
        color=INK2,
        fontweight="bold",
    )
    ax.text(
        xs["status"],
        0.5,
        "Result",
        ha="left",
        va="center",
        fontsize=11,
        color=INK2,
        fontweight="bold",
    )
    ax.plot([0, 100], [1, 1], color=AXIS, lw=1)
    y = 1
    for row in rows:
        if row[1] is None:
            y += 0.12
            ax.text(
                xs["label"],
                y + 0.5,
                row[0],
                ha="left",
                va="center",
                fontsize=11,
                color=INK2,
                fontweight="bold",
            )
            y += 1
            continue
        label, real_v, v1_v, limit, status = row
        yc = y + 0.5
        ax.text(xs["label"] + 1.5, yc, label, ha="left", va="center", fontsize=12, color=INK)
        ax.text(xs["real"], yc, real_v, ha="right", va="center", fontsize=12, color=INK2)
        ax.text(
            xs["v1"],
            yc,
            v1_v,
            ha="right",
            va="center",
            fontsize=12,
            color=INK,
            fontweight="bold" if status == "fail" else "normal",
        )
        ax.text(xs["limit"], yc, limit, ha="right", va="center", fontsize=12, color=INK2)
        icon, word, color, weight = {
            "pass": ("✓", "pass", GOOD, "normal"),
            "fail": ("✗", "fail", CRIT, "bold"),
            "info": ("–", "info", INK2, "normal"),
        }[status]
        ax.text(
            xs["status"],
            yc,
            icon,
            ha="left",
            va="center",
            fontsize=15,
            color=color,
            fontfamily="DejaVu Sans",
            fontweight="bold",
        )
        ax.text(
            xs["status"] + 2.3,
            yc,
            word,
            ha="left",
            va="center",
            fontsize=11.5,
            color=INK if status == "fail" else INK2,
            fontweight=weight,
        )
        ax.plot([0, 100], [y + 1, y + 1], color=GRID, lw=0.8)
        y += 1
    save(fig, "01_gate_scorecard.png")


# --- 2. distance per column -------------------------------------------------------------------


def fig_column_distances():
    cd = pd.read_csv(ART / "column_distances.csv", index_col=0).sort_values("fill50")
    top7 = cd["fill50"].nlargest(7).sum() / cd["fill50"].sum()
    job_share = cd.loc["job", "fill50"] / cd["fill50"].sum()
    fig = new_fig(
        "7 of 16 columns are far from real data, and job is the worst",
        f"Distance between v1 and real subscribers, per column (0 is identical). The seven columns over the limit carry {top7:.0%} of the total.",
        "Distance is the KS statistic for numeric columns and total variation distance for categorical ones. Real = 3,173 train subscribers; v1 = the 50% file (10,390 rows). "
        "The blue dot compares 1,058 real validation subscribers with the same train subscribers.",
    )
    ax = fig.add_axes([0.16, BOTTOM, 0.80, HEIGHT])
    y = np.arange(len(cd))
    over = cd["fill50"] > LIMIT
    ax.barh(y, cd["fill50"], height=0.58, color=[V1 if o else V1_TINT for o in over], zorder=3)
    ax.scatter(cd["real_vs_real"], y, s=70, color=REAL, edgecolor=SURFACE, linewidth=1.5, zorder=5)
    ax.axvline(LIMIT, color=INK2, lw=1.3, zorder=4)
    ax.text(LIMIT + 0.003, len(cd) - 0.35, "limit 0.10", color=INK2, fontsize=11.5, va="bottom")
    for yi, (v, o) in enumerate(zip(cd["fill50"], over, strict=True)):
        if o:
            ax.text(v + 0.005, yi, f"{v:.2f}", va="center", fontsize=11.5, color=INK2)
    ax.set_yticks(y, cd.index)
    ax.set_xlim(0, 0.31)
    ax.set_ylim(-0.7, len(cd) - 0.1)
    ax.set_xlabel("distance from real subscribers", fontsize=11.5)
    style(ax, "x")
    ax.tick_params(axis="y", length=0, labelsize=12)
    ax.text(
        0.305,
        1.6,
        f"job alone is {job_share:.0%} of the total.\nBars in light orange are\nwithin the limit.",
        ha="right",
        va="bottom",
        fontsize=11.5,
        color=INK2,
        linespacing=1.4,
    )
    legend_entity(fig, [square(V1, "CTGAN v1"), dot(REAL, "real vs real (noise floor)")])
    save(fig, "02_column_distances.png")


# --- 3. what is off in the categories ---------------------------------------------------------


def shares(real, synth, col):
    r = real[col].value_counts(normalize=True)
    s = synth[col].value_counts(normalize=True)
    labels = r.index.union(s.index)
    return pd.DataFrame(
        {"real": r.reindex(labels, fill_value=0), "v1": s.reindex(labels, fill_value=0)}
    )


def grouped_hbars(ax, df, title, annotate=True, xmax=None, step=None):
    yy = np.arange(len(df))[::-1]
    ax.barh(yy + 0.2, df["real"], height=0.38, color=REAL, zorder=3)
    ax.barh(yy - 0.2, df["v1"], height=0.38, color=V1, zorder=3)
    ax.set_yticks(yy, df.index, fontsize=11.5)
    ax.xaxis.set_major_formatter(lambda v, _: f"{100 * v:.0f}%")
    if step:
        ax.xaxis.set_major_locator(MultipleLocator(step))
    ax.set_title(title, loc="left", fontsize=12.5, fontweight="bold", color=INK, pad=8)
    style(ax, "x")
    ax.tick_params(axis="y", length=0)
    top = df.to_numpy().max()
    ax.set_xlim(0, xmax or top * 2.1)
    if annotate:
        k = (df["v1"] - df["real"]).abs().to_numpy().argmax()
        row = df.iloc[k]
        ax.text(
            max(row) + 0.02 * top,
            yy[k],
            f"{pct(row['real'])} → {pct(row['v1'])}",
            va="center",
            fontsize=10.5,
            color=INK2,
        )
    return yy


def fig_categories(train, syn):
    real = train[train[TARGET] == 1]
    s = syn[50]
    dist = pd.read_csv(ART / "column_distances.csv", index_col=0)["fill50"]
    fig = new_fig(
        "Technician is 42% of v1 subscribers and 15% of real ones",
        "Share of each category, real subscribers against v1: the biggest category gaps in the five categorical columns over the limit.",
        "Real = 3,173 train subscribers; v1 = the 50% file (10,390 rows). The label on each panel marks the category with the largest gap. "
        "The two numeric columns over the limit, day_of_month and pdays, are in the next figure.",
    )
    gs = fig.add_gridspec(
        2,
        4,
        width_ratios=[1.45, 1, 1, 1],
        left=0.115,
        right=0.985,
        top=0.80,
        bottom=0.155,
        wspace=0.62,
        hspace=0.62,
    )
    # job: the eight biggest categories, the rest folded
    jb = shares(real, s, "job")
    order = jb.max(axis=1).sort_values(ascending=False).index
    keep, rest = list(order[:8]), list(order[8:])
    jb = (
        pd.concat(
            [jb.loc[keep], pd.DataFrame([jb.loc[rest].sum()], index=[f"other ({len(rest)})"])]
        )
        if rest
        else jb.loc[keep]
    )
    ax = fig.add_subplot(gs[:, 0])
    yy = grouped_hbars(ax, jb, f"job (distance {dist['job']:.2f})", annotate=False, xmax=0.62)
    t = jb.index.get_loc("technician")
    ax.text(
        jb.loc["technician", "v1"] + 0.012,
        yy[t] - 0.2,
        f"{pct(jb.loc['technician', 'real'])} → {pct(jb.loc['technician', 'v1'])}",
        va="center",
        fontsize=12,
        color=INK,
        fontweight="bold",
    )
    # month: calendar order
    months = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
    mb = shares(real, s, "month").reindex(months).fillna(0)
    ax = fig.add_subplot(gs[0, 1:])
    x = np.arange(len(mb))
    ax.bar(x - 0.2, mb["real"], width=0.38, color=REAL, zorder=3)
    ax.bar(x + 0.2, mb["v1"], width=0.38, color=V1, zorder=3)
    ax.set_xticks(x, mb.index, fontsize=11)
    ax.yaxis.set_major_formatter(lambda v, _: f"{100 * v:.0f}%")
    ax.set_title(
        f"month (distance {dist['month']:.2f})",
        loc="left",
        fontsize=12.5,
        fontweight="bold",
        color=INK,
        pad=8,
    )
    style(ax, "y")
    for m in ("jul", "feb"):
        k = months.index(m)
        ax.text(
            k,
            max(mb.loc[m]) + 0.008,
            f"{pct(mb.loc[m, 'real'])} → {pct(mb.loc[m, 'v1'])}",
            ha="center",
            va="bottom",
            fontsize=10.5,
            color=INK2,
        )
    ax.set_ylim(0, mb.to_numpy().max() * 1.3)
    for slot, col in ((gs[1, 1], "marital"), (gs[1, 2], "poutcome"), (gs[1, 3], "housing")):
        df = shares(real, s, col)
        df = df.loc[df["real"].sort_values(ascending=False).index]
        grouped_hbars(fig.add_subplot(slot), df, f"{col} ({dist[col]:.2f})", step=0.5)
    legend_entity(fig, [square(REAL, "real subscribers"), square(V1, "CTGAN v1")])
    save(fig, "03_category_shifts.png")


# --- 4. numeric drift -------------------------------------------------------------------------


def fig_numeric(train, syn):
    real = train[train[TARGET] == 1]
    s = syn[50]
    kd = pd.read_csv(ART / "numeric_drift.csv").set_index("column")
    rc, sc = real.loc[real["pdays"] > 0, "pdays"], s.loc[s["pdays"] > 0, "pdays"]
    fig = new_fig(
        "v1 runs late: later in the month, longer between campaigns",
        "Cumulative share of subscribers up to each value. Where the orange line sits below the blue one, v1 has more large values.",
        "Real = 3,173 train subscribers; v1 = the 50% file (10,390 rows). KS is the largest gap between the two lines.",
    )
    gs = fig.add_gridspec(
        2, 2, left=0.065, right=0.985, top=0.80, bottom=0.155, wspace=0.16, hspace=0.55
    )
    panels = [
        (
            "day_of_month",
            real["day_of_month"],
            s["day_of_month"],
            (1, 31),
            f"day_of_month: median {s['day_of_month'].median():.0f} vs {real['day_of_month'].median():.0f}",
        ),
        (
            "pdays",
            rc,
            sc,
            (0, 600),
            f"pdays, contacted only: {pct((rc <= 120).mean())} within 120 days vs {pct((sc <= 120).mean())}",
        ),
        (
            "balance",
            real["balance"],
            s["balance"],
            (-800, 8000),
            f"balance: 95th percentile {s['balance'].quantile(0.95):,.0f} vs {real['balance'].quantile(0.95):,.0f}",
        ),
        (
            "age",
            real["age"],
            s["age"],
            (18, 95),
            f"age: 95th percentile {s['age'].quantile(0.95):.0f} vs {real['age'].quantile(0.95):.0f}",
        ),
    ]
    axes = []
    for slot, (col, r, v, xlim, title) in zip(
        (gs[0, 0], gs[0, 1], gs[1, 0], gs[1, 1]), panels, strict=True
    ):
        ax = fig.add_subplot(slot)
        axes.append(ax)
        ecdf(ax, r, REAL)
        ecdf(ax, v, V1)
        ax.set_xlim(*xlim)
        ax.set_ylim(0, 1.02)
        ax.yaxis.set_major_formatter(lambda x, _: f"{100 * x:.0f}%")
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=INK, pad=8)
        ax.text(
            0.97,
            0.06,
            f"KS {kd.loc[col, 'ks']:.2f}",
            transform=ax.transAxes,
            ha="right",
            fontsize=11.5,
            color=INK2,
        )
        style(ax, "y")
    axes[1].text(
        0.97,
        0.17,
        f"never contacted: {pct(1 - len(rc) / len(real))} real, {pct(1 - len(sc) / len(s))} v1",
        transform=axes[1].transAxes,
        ha="right",
        fontsize=10.5,
        color=INK2,
    )
    legend_entity(fig, [line(REAL, "real subscribers"), line(V1, "CTGAN v1")])
    save(fig, "04_numeric_drift.png")


# --- 5. associations that break ---------------------------------------------------------------


def fig_associations(train, syn):
    real = train[train[TARGET] == 1]
    s = syn[50]
    cols = [c for c in real if c not in {TARGET, "source_row", "is_synthetic"}]
    kinds = column_kinds(real, cols, cfg.discrete)
    cat = [c for c in cols if kinds[c] == "categorical"]
    numeric = [c for c in cols if kinds[c] == "numeric"]
    cv = pd.DataFrame(
        [
            (f"{a} × {b}", cramers_v(real[a], real[b]), cramers_v(s[a], s[b]))
            for a, b in combinations(cat, 2)
        ],
        columns=["pair", "real", "v1"],
    )
    sr, ss = spearman_matrix(real, numeric), spearman_matrix(s, numeric)
    sp = pd.DataFrame(
        [(f"{a} × {b}", sr.loc[a, b], ss.loc[a, b]) for a, b in combinations(numeric, 2)],
        columns=["pair", "real", "v1"],
    )
    pairs = pd.concat(
        [cv.assign(measure="cramers_v"), sp.assign(measure="spearman")], ignore_index=True
    )
    pairs["change"] = pairs["v1"] - pairs["real"]
    pairs.sort_values("change", key=lambda s: s.abs(), ascending=False).round(4).to_csv(
        ART / "association_pairs.csv", index=False
    )
    fig = new_fig(
        "v1 ties the calendar to the contact history far too tightly",
        "The pairs of columns whose link changes most between real subscribers and v1. A dot far from its partner is a relationship v1 got wrong.",
        "Cramér's V (0 to 1) between categorical columns, Spearman correlation (−1 to 1) between numeric ones. Values are labelled for the three biggest changes. "
        "Real = 3,173 train subscribers; v1 = the 50% file.",
    )
    for rect, df, title, xlim, k in (
        ([0.225, BOTTOM, 0.285, HEIGHT], cv, "Categorical pairs: Cramér's V", (-0.07, 0.5), 6),
        ([0.675, BOTTOM, 0.30, HEIGHT], sp, "Numeric pairs: Spearman correlation", (-0.3, 0.2), 5),
    ):
        ax = fig.add_axes(rect)
        d = (
            df.reindex((df["v1"] - df["real"]).abs().sort_values(ascending=False).index)
            .head(k)
            .iloc[::-1]
        )
        y = np.arange(len(d))
        ax.hlines(y, d["real"], d["v1"], color=AXIS, lw=3, zorder=2)
        ax.scatter(d["real"], y, s=95, color=REAL, edgecolor=SURFACE, linewidth=1.5, zorder=4)
        ax.scatter(d["v1"], y, s=95, color=V1, edgecolor=SURFACE, linewidth=1.5, zorder=4)
        ax.set_yticks(y, d["pair"], fontsize=11.5)
        ax.set_xlim(*xlim)
        ax.set_ylim(-0.6, len(d) - 0.4)
        ax.set_title(title, loc="left", fontsize=12.5, fontweight="bold", color=INK, pad=10)
        style(ax, "x")
        ax.tick_params(axis="y", length=0)
        if xlim[0] < -0.1:
            ax.axvline(0, color=AXIS, lw=1, zorder=1)
        for yi, (a, b) in enumerate(zip(d["real"], d["v1"], strict=True)):
            if yi >= len(d) - 3:
                up = b >= a
                ax.text(
                    b + (0.014 if up else -0.014),
                    yi,
                    num(b),
                    ha="left" if up else "right",
                    va="center",
                    fontsize=11,
                    color=INK,
                )
                ax.text(
                    a + (-0.014 if up else 0.014),
                    yi,
                    num(a),
                    ha="right" if up else "left",
                    va="center",
                    fontsize=11,
                    color=INK2,
                )
    legend_entity(fig, [dot(REAL, "real subscribers"), dot(V1, "CTGAN v1")])
    save(fig, "05_broken_associations.png")


# --- 6. does it matter for the prediction? ----------------------------------------------------


def fig_importance(train, val):
    cols = [c for c in train if c not in {TARGET, "source_row", "is_synthetic"}]
    kinds = column_kinds(train, cols, cfg.discrete)
    x, xv = train[cols].copy(), val[cols].copy()
    for c in cols:
        if kinds[c] == "categorical":
            cats = pd.Categorical(x[c]).categories
            x[c], xv[c] = (
                pd.Categorical(x[c], categories=cats),
                pd.Categorical(xv[c], categories=cats),
            )
    model = HistGradientBoostingClassifier(
        categorical_features="from_dtype", random_state=0, max_iter=200, learning_rate=0.08
    )
    model.fit(x, train[TARGET])
    imp = permutation_importance(
        model, xv, val[TARGET], scoring="average_precision", n_repeats=5, random_state=0
    )
    dist = pd.read_csv(ART / "column_distances.csv", index_col=0)["fill50"]
    df = pd.DataFrame(
        {"importance": imp.importances_mean, "distance": dist.reindex(cols).to_numpy()}, index=cols
    )
    df.round(4).to_csv(ART / "column_importance.csv", index_label="column")

    fig = new_fig(
        "v1 gets the columns that drive the prediction wrong: month, poutcome, day_of_month",
        "job is the biggest miss, but it barely matters for who subscribes. A model trained on v1 rows learns the wrong calendar and campaign history.",
        "Importance = drop in average precision when the column is shuffled (gradient boosting on real train rows, scored on real validation rows). Distance as in the column chart, "
        "50% file. The seven unlabelled dots are the other columns, all with importance under 0.01.",
    )
    ax = fig.add_axes([0.085, BOTTOM, 0.89, HEIGHT])
    ax.axvspan(LIMIT, 0.31, color="#faf1ec", zorder=0)
    ax.axvline(LIMIT, color=INK2, lw=1.3, zorder=1)
    ax.scatter(
        df["distance"],
        df["importance"],
        s=120,
        color=V1,
        edgecolor=SURFACE,
        linewidth=1.8,
        zorder=4,
    )
    offsets = {
        "month": (0.008, 0.004, "left"),
        "poutcome": (0.008, 0.004, "left"),
        "day_of_month": (0.008, 0.004, "left"),
        "contact": (0.008, 0.004, "left"),
        "job": (-0.008, 0.012, "right"),
        "pdays": (0.006, 0.012, "left"),
        "marital": (0.008, -0.004, "left"),
        "housing": (0.008, 0.013, "left"),
        "balance": (-0.004, 0.011, "right"),
    }
    for col, (dx, dy, ha) in offsets.items():
        ax.text(
            df.loc[col, "distance"] + dx,
            df.loc[col, "importance"] + dy,
            col,
            ha=ha,
            va="center",
            fontsize=12,
            color=INK,
            fontweight="bold" if col in {"month", "poutcome", "day_of_month", "job"} else "normal",
        )
    ax.text(LIMIT + 0.006, 0.168, "over the 0.10 limit", color=INK2, fontsize=11.5, va="center")
    ax.annotate(
        "biggest miss,\nbarely matters",
        xy=(df.loc["job", "distance"], df.loc["job", "importance"]),
        xytext=(0.255, 0.045),
        ha="center",
        fontsize=11.5,
        color=INK2,
        arrowprops={"arrowstyle": "-", "color": AXIS, "lw": 1.2},
    )
    ax.set_xlim(0, 0.31)
    ax.set_ylim(-0.012, 0.18)
    ax.set_xlabel("how far v1 is from real subscribers (distance)", fontsize=11.5)
    ax.set_ylabel("how much the prediction depends on the column", fontsize=11.5)
    style(ax, "y")
    save(fig, "06_fidelity_vs_importance.png")


# --- 7. can a classifier tell them apart ------------------------------------------------------


def fig_distinguishability():
    by = pd.read_csv(ART / "privacy_utility_by_seed.csv")
    fig = new_fig(
        "v1 rows are easy to spot: AUC 0.86, against 0.52 for real data",
        "Left: AUC of a random forest trained to separate real from synthetic rows. Right: job, day_of_month and month are in its top 5 in every run.",
        "5 seeds per file; the real-vs-real AUC compares 1,058 real validation subscribers with real train subscribers. "
        "Importance = permutation importance of the classifier, top 5 columns per run, 15 runs.",
    )
    ax = fig.add_axes([0.075, BOTTOM, 0.43, HEIGHT])
    groups = [("real vs real", by.loc[by["file"] == "fill10", "real_vs_real_auc"], REAL)] + [
        (f"v1 {n}%", by.loc[by["file"] == f"fill{n}", "real_vs_synthetic_auc"], V1)
        for n in (10, 25, 50)
    ]
    rng = np.random.default_rng(1)
    for i, (_, v, color) in enumerate(groups):
        ax.scatter(
            i + rng.uniform(-0.12, 0.12, len(v)),
            v,
            s=70,
            color=color,
            edgecolor=SURFACE,
            linewidth=1.5,
            zorder=4,
        )
        ax.hlines(v.mean(), i - 0.27, i + 0.27, color=INK, lw=2, zorder=5)
        ax.text(
            i + 0.31,
            v.mean(),
            f"{v.mean():.2f}",
            va="center",
            fontsize=12,
            color=INK,
            fontweight="bold",
        )
    ax.axhline(0.5, color=INK2, lw=1.3, zorder=1)
    ax.text(3.6, 0.485, "0.5 = cannot tell apart", ha="right", va="top", fontsize=11, color=INK2)
    ax.set_xticks(range(4), [g[0] for g in groups], fontsize=12)
    ax.set_xlim(-0.6, 3.75)
    ax.set_ylim(0.4, 0.95)
    ax.set_ylabel("AUC", fontsize=11.5)
    style(ax, "y")
    ax2 = fig.add_axes([0.66, BOTTOM, 0.31, HEIGHT])
    counts = Counter(c for item in by["revealing_columns"] for c in item.split(",") if c)
    top = pd.Series(counts).sort_values().tail(7)
    ax2.barh(np.arange(len(top)), top.to_numpy(), height=0.58, color=V1, zorder=3)
    for yi, v in enumerate(top.to_numpy()):
        ax2.text(v + 0.25, yi, f"{v} of 15", va="center", fontsize=11.5, color=INK2)
    ax2.set_yticks(np.arange(len(top)), top.index, fontsize=12)
    ax2.set_xticks([0, 5, 10, 15])
    ax2.set_xlim(0, 18)
    ax2.set_xlabel("runs where the column is in the top 5", fontsize=11.5)
    style(ax2, "x")
    ax2.tick_params(axis="y", length=0)
    save(fig, "07_distinguishability.png")


# --- 8. closest real record -------------------------------------------------------------------


def fig_closest(train, val, syn):
    cols = [c for c in train if c not in {TARGET, "source_row", "is_synthetic"}]
    tp, vp = train.loc[train[TARGET] == 1, cols], val.loc[val[TARGET] == 1, cols]
    transform = _preprocessor(tp).fit(tp)
    nn = NearestNeighbors(n_neighbors=1, algorithm="brute").fit(transform.transform(tp))
    d_val = nn.kneighbors(transform.transform(vp))[0].ravel()
    d_syn = nn.kneighbors(transform.transform(syn[50][cols]))[0].ravel()
    ratios = pd.read_csv(ART / "dcr_noise_floor_ratios.csv")["dcr_ratio"].to_numpy()
    below = json.loads((ART / "dcr_noise_floor.json").read_text())["share_below_1.00"]
    fig = new_fig(
        "No memorization: v1 rows sit farther from real rows than real rows do",
        "Left: distance from each row to its nearest real train subscriber. Right: why the starting limit of 1.0 is too strict, using real data only.",
        "Distances are in the space the gate uses (numbers scaled, categories one-hot). Ratio = median distance of the rows being tested over the median for real validation subscribers. "
        "Right: 100 random splits of the 4,231 real subscribers.",
    )
    ax = fig.add_axes([0.075, BOTTOM, 0.40, HEIGHT])
    ecdf(ax, d_val, REAL)
    ecdf(ax, d_syn, V1)
    for d, c, side in ((d_val, REAL, -1), (d_syn, V1, 1)):
        m = float(np.median(d))
        ax.plot([m, m], [0, 0.5], color=c, lw=1.2, zorder=2)
        ax.text(
            m + side * 0.05,
            0.5 - side * 0.06,
            f"median {m:.2f}",
            ha="right" if side < 0 else "left",
            va="center",
            fontsize=11.5,
            color=INK,
        )
    ax.set_xlim(0, float(np.percentile(np.r_[d_val, d_syn], 99.5)))
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_formatter(lambda x, _: f"{100 * x:.0f}%")
    ax.set_xlabel("distance to the nearest real train subscriber", fontsize=11.5)
    style(ax, "y")
    ax2 = fig.add_axes([0.58, BOTTOM, 0.39, HEIGHT])
    ax2.hist(ratios, bins=np.arange(0.95, 1.06, 0.005), color=REAL, zorder=3, rwidth=0.88)
    ax2.axvline(1.0, color=INK2, lw=1.3, zorder=4)
    ax2.axvline(0.95, color=INK2, lw=1.3, zorder=4)
    top = ax2.get_ylim()[1]
    ax2.text(
        1.0015,
        top * 0.97,
        f"starting limit 1.0:\n{pct(below)} of real splits\nfall below it",
        fontsize=11,
        color=INK,
        va="top",
        linespacing=1.3,
    )
    ax2.text(
        0.9485,
        top * 0.97,
        "suggested\nlimit 0.95",
        fontsize=11,
        color=INK,
        va="top",
        ha="right",
        linespacing=1.3,
    )
    ax2.set_xlim(0.90, 1.07)
    ax2.set_xlabel("ratio for real data (100 splits)", fontsize=11.5)
    ax2.set_ylabel("splits", fontsize=11.5)
    style(ax2, "y")
    ax2.annotate(
        f"v1: {np.median(d_syn) / np.median(d_val):.2f}  →",
        xy=(1.07, 0),
        xytext=(1.0695, top * 0.45),
        ha="right",
        fontsize=12,
        color=INK,
        fontweight="bold",
    )
    legend_entity(fig, [line(REAL, "real data"), line(V1, "CTGAN v1")])
    save(fig, "08_closest_record.png")


# --- 9. utility -------------------------------------------------------------------------------


def fig_utility():
    d = pd.read_csv(KYLE)
    sel = d[d["selected"]]
    g = sel.groupby(["arm", "fill_level"]).agg(
        pr=("test_pr_auc", "mean"),
        f1=("test_f1", "mean"),
        prec=("test_precision", "mean"),
        rec=("test_recall", "mean"),
    )
    base = g.loc[("real_only", 0.0)]
    levels = [0.0, 0.10, 0.25, 0.50]
    xs = [0, 10, 25, 50]

    def series(arm, col):
        return [base[col]] + [g.loc[(arm, lv), col] for lv in levels[1:]]

    fig = new_fig(
        "Adding v1 rows lowers PR-AUC at every fill level; the F1 gain is a threshold shift",
        "Left: test PR-AUC as synthetic rows are added. Right: the same v1 runs, PR-AUC against F1 at the fixed 0.5 threshold.",
        "Source: results/augmented_bank_marketing.csv (issue #35). Test set, mean of 5 seeds, best of nine models chosen on validation. "
        "Fill level = share of the class gap added as synthetic rows.",
    )
    ax = fig.add_axes([0.075, BOTTOM, 0.40, HEIGHT])
    for arm, color, label in (("ctgan_v1", V1, "CTGAN v1"), ("smotenc", SMOTE, "SMOTENC")):
        ys = series(arm, "pr")
        ax.plot(
            xs,
            ys,
            color=color,
            lw=2.6,
            marker="o",
            markersize=8,
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
            zorder=4,
        )
        ax.text(
            51.5,
            ys[-1],
            f"{label}  {ys[-1]:.3f}",
            va="center",
            fontsize=12,
            color=INK,
            fontweight="bold",
        )
    ax.axhline(base["pr"], color=REAL, lw=1.6, zorder=2)
    ax.text(
        0.8,
        base["pr"] + 0.003,
        f"real only  {base['pr']:.3f}",
        color=INK,
        fontsize=12,
        fontweight="bold",
        va="bottom",
    )
    ax.set_xticks(xs, ["0%", "10%", "25%", "50%"])
    ax.set_xlim(-2, 66)
    ax.set_ylim(0.38, 0.50)
    ax.set_xlabel("fill level", fontsize=11.5)
    ax.set_ylabel("test PR-AUC", fontsize=11.5)
    style(ax, "y")
    ax2 = fig.add_axes([0.58, BOTTOM, 0.39, HEIGHT])
    pr, f1 = series("ctgan_v1", "pr"), series("ctgan_v1", "f1")
    ax2.plot(
        xs,
        pr,
        color=V1,
        lw=2.6,
        marker="o",
        markersize=8,
        markeredgecolor=SURFACE,
        markeredgewidth=1.5,
        zorder=4,
    )
    ax2.plot(
        xs,
        f1,
        color=V1,
        lw=2.6,
        marker="s",
        markersize=8,
        markerfacecolor=SURFACE,
        markeredgewidth=2,
        zorder=4,
        linestyle=(0, (1, 1.6)),
    )
    ax2.text(
        51.5,
        pr[-1],
        "PR-AUC\n(threshold-free)",
        va="center",
        fontsize=11.5,
        color=INK,
        fontweight="bold",
        linespacing=1.2,
    )
    ax2.text(
        51.5,
        f1[-1],
        "F1 at 0.5\n(threshold-dependent)",
        va="center",
        fontsize=11.5,
        color=INK,
        fontweight="bold",
        linespacing=1.2,
    )
    rec1, p1 = g.loc[("ctgan_v1", 0.5), "rec"], g.loc[("ctgan_v1", 0.5), "prec"]
    ax2.text(
        1,
        0.30,
        f"recall {base['rec']:.3f} → {rec1:.3f}, precision {base['prec']:.3f} → {p1:.3f}\nmore positives predicted at 0.5, ranking gets worse",
        fontsize=11,
        color=INK2,
        va="bottom",
        linespacing=1.4,
    )
    ax2.set_xticks(xs, ["0%", "10%", "25%", "50%"])
    ax2.set_xlim(-2, 80)
    ax2.set_ylim(0.28, 0.54)
    ax2.set_xlabel("fill level", fontsize=11.5)
    style(ax2, "y")
    save(fig, "09_utility.png")


def extra_tables(train, syn):
    """Two checks behind the findings in the note: is v1 a class mix-up, and how diverse is it."""
    real, s = train[train[TARGET] == 1], syn[50]
    neg = train[train[TARGET] == 0]
    cols = [c for c in real if c not in {TARGET, "source_row", "is_synthetic"}]
    kinds = column_kinds(real, cols, cfg.discrete)

    def dist(a, b, c):
        return (ks_distance if kinds[c] == "numeric" else tv_distance)(a[c], b[c])

    mix = pd.DataFrame(
        {
            "v1_vs_subscribers": [dist(real, s, c) for c in cols],
            "v1_vs_non_subscribers": [dist(neg, s, c) for c in cols],
            "subscribers_vs_non_subscribers": [dist(real, neg, c) for c in cols],
        },
        index=cols,
    )
    mix.round(4).to_csv(ART / "class_mixup.csv", index_label="column")
    closer = mix.index[mix["v1_vs_non_subscribers"] < mix["v1_vs_subscribers"]].tolist()
    print(
        f"v1 is closer to non-subscribers than to subscribers in {len(closer)} of {len(cols)} columns: {closer}"
    )
    profile = ["job", "marital", "education", "housing", "loan", "contact", "month", "poutcome"]
    sample = s.sample(n=len(real), random_state=0)

    def spread(df):
        counts = df[profile].astype(str).agg("|".join, axis=1).value_counts()
        p = counts / counts.sum()
        out = {
            "unique_profiles": len(counts),
            "top10_share": round(float(p.head(10).sum()), 4),
            "entropy_bits": round(float(-(p * np.log2(p)).sum()), 3),
        }
        for c in ("job", "month", "education"):
            q = df[c].value_counts(normalize=True)
            out[f"{c}_entropy_bits"] = round(float(-(q * np.log2(q)).sum()), 3)
        return out

    table = pd.DataFrame(
        {"real subscribers": spread(real), "v1, same number of rows": spread(sample)}
    ).T
    table.to_csv(ART / "diversity.csv", index_label="set")
    print(table.to_string())


def main() -> None:
    train, val, syn = load()
    pu = pd.read_csv(ART / "privacy_utility.csv", index_col=0)
    fig_scorecard(train, val, syn, pu)
    fig_column_distances()
    fig_categories(train, syn)
    fig_numeric(train, syn)
    fig_associations(train, syn)
    fig_importance(train, val)
    fig_distinguishability()
    fig_closest(train, val, syn)
    fig_utility()
    extra_tables(train, syn)
    print(f"\nplots in {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
