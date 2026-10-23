"""Quality checks on the Bank CTGAN v1 files (issue #49).

Scores the three fill-level files with the fidelity tool (``agentic_gan.tools.quality_fidelity``)
against real train subscribers, under the starting thresholds (``QualityThresholds``) and under
thresholds suggested from the real-vs-real baseline. It also saves the detail behind the failing
checks: which categories overshoot, how the numeric columns drift, and which direction of each
rule breaks.

Then runs the privacy, distinguishability and utility checks (``agentic_gan.tools.quality_privacy``,
issue #47) on every file with five seeds, and estimates how much the closest-record ratio varies
for real data, so its limit can be set above the noise.

Results: docs/notes/ctgan_bank_v1_quality.md

Run from the repo root:  uv run python eda/ctgan_bank/quality_v1.py
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from agentic_gan.schemas import QualityThresholds
from agentic_gan.tools.quality_fidelity import (
    BANK_RULES,
    WORST_DISTANCE,
    fidelity_checks,
    ks_distance,
    real_vs_real,
    schema_thresholds,
    scorecard,
    suggest_thresholds,
)
from agentic_gan.tools.quality_privacy import closest_real_check, quality_privacy_checks
from agentic_gan.utils.datasets import DATASETS

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "interim" / "bank_marketing"
OUT = Path(__file__).resolve().parent / "artifacts" / "quality_v1"
LEVELS = (10, 25, 50)
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
SEEDS = (42, 43, 44, 45, 46)
DCR_SPLITS = 100

cfg = DATASETS["bank_marketing"]
common = {"target": cfg.target, "discrete": cfg.discrete, "rules": BANK_RULES}


def privacy_and_utility(train: pd.DataFrame, val: pd.DataFrame, synth: dict) -> pd.DataFrame:
    """The ``quality_privacy`` checks for every file and seed, one row each."""
    rows = []
    for label, frame in synth.items():
        for seed in SEEDS:
            checks, detail = quality_privacy_checks(train, val, frame, cfg.target, 1, seed=seed)
            value = {c.name: c.value for c in checks}
            rows.append(
                {
                    "file": label,
                    "seed": seed,
                    "exact_train_copies": value["exact_train_copies"],
                    "synthetic_duplicates": value["synthetic_duplicates"],
                    "dcr_ratio": value["dcr_ratio"],
                    "synthetic_median_distance": detail["synthetic_median"],
                    "validation_median_distance": detail["validation_median"],
                    "real_vs_synthetic_auc": value["real_vs_synthetic_auc"],
                    "real_vs_real_auc": detail["real_vs_real_auc"],
                    "revealing_columns": ",".join(detail["revealing_columns"]),
                    "real_only_pr_auc": detail["real_only_pr_auc"],
                    "augmented_pr_auc": detail["augmented_pr_auc"],
                    "pr_auc_gain": value["augmentation_pr_auc_gain"],
                    "real_only_best_f1": detail["real_only_best_f1"],
                    "augmented_best_f1": detail["augmented_best_f1"],
                    "best_f1_gain": value["augmentation_best_f1_gain"],
                }
            )
    return pd.DataFrame(rows)


def top_columns(lists: pd.Series, n: int = 5) -> str:
    """The columns that most often expose the synthetic rows, across seeds."""
    counts = Counter(c for item in lists for c in item.split(",") if c)
    return ",".join(c for c, _ in counts.most_common(n))


def summarize(by_seed: pd.DataFrame) -> pd.DataFrame:
    """Mean over seeds per file, plus the spread of the two AUCs."""
    grouped = by_seed.drop(columns=["seed", "revealing_columns"]).groupby("file", sort=False)
    out = grouped.mean()
    out["auc_sd"] = grouped["real_vs_synthetic_auc"].std()
    out["real_vs_real_auc_sd"] = grouped["real_vs_real_auc"].std()
    out["revealing_columns"] = by_seed.groupby("file", sort=False)["revealing_columns"].apply(
        top_columns
    )
    return out


def dcr_noise_floor(train: pd.DataFrame, val: pd.DataFrame) -> dict:
    """The closest-record ratio for real data, from random splits of the real subscribers.

    Each split takes three disjoint sets of real subscribers: a reference, a pseudo-synthetic set
    and a pseudo-validation set, with the sizes the real check uses. Real rows are as close to
    the reference as other real rows, so the ratio should centre on 1 and its spread is noise.
    """
    columns = [c for c in train if c not in {cfg.target, "source_row", "is_synthetic"}]
    pool = pd.concat([train[train[cfg.target] == 1], val[val[cfg.target] == 1]], ignore_index=True)
    n_val = int((val[cfg.target] == 1).sum())
    n_ref = len(pool) - 2 * n_val
    ratios = []
    for seed in range(DCR_SPLITS):
        order = np.random.default_rng(seed).permutation(len(pool))
        ref = pool.iloc[order[:n_ref]]
        pseudo = pool.iloc[order[n_ref : n_ref + n_val]]
        pseudo_val = pool.iloc[order[n_ref + n_val :]]
        check, _ = closest_real_check(
            ref, pseudo_val, pseudo, columns, cfg.target, 1, QualityThresholds()
        )
        ratios.append(check.value)
    r = np.array(ratios)
    return {
        "splits": DCR_SPLITS,
        "reference_rows": n_ref,
        "rows_per_set": n_val,
        "mean": round(float(r.mean()), 4),
        "sd": round(float(r.std(ddof=1)), 4),
        "p5": round(float(np.percentile(r, 5)), 4),
        "min": round(float(r.min()), 4),
        "max": round(float(r.max()), 4),
        "share_below_1.00": float((r < 1.0).mean()),
        "share_below_0.95": float((r < 0.95).mean()),
        "share_below_0.90": float((r < 0.90).mean()),
        "ratios": [round(float(x), 6) for x in r],
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(DATA / "train_cleaned.csv")
    val = pd.read_csv(DATA / "validation_cleaned.csv")
    real = train[train[cfg.target] == 1]  # real train subscribers: the reference
    synth = {
        f"fill{n}": pd.read_csv(DATA / "synthetic" / f"bank_ctgan_fill{n}.csv") for n in LEVELS
    }
    print(
        f"real train subscribers: {len(real):,}   real validation subscribers: "
        f"{(val[cfg.target] == 1).sum():,}   synthetic rows: "
        f"{ {k: len(v) for k, v in synth.items()} }"
    )

    # 1. scorecards under both threshold sets
    baseline = real_vs_real(train, val, **common)
    suggested = suggest_thresholds(baseline.values)
    starting = schema_thresholds(rules=BANK_RULES)
    for label, thresholds in (("suggested", suggested), ("schema", starting)):
        base = real_vs_real(train, val, thresholds=thresholds, **common)
        reports = {
            k: fidelity_checks(real, v, thresholds=thresholds, **common) for k, v in synth.items()
        }
        text = scorecard(base, reports, thresholds, per_column=True)
        (OUT / f"scorecard_{label}.txt").write_text(text + "\n")
        print(f"\n=== thresholds: {label}\n{text}")
    (OUT / "thresholds.json").write_text(
        json.dumps(
            {
                "starting (QualityThresholds)": starting,
                "suggested": suggested,
                "real_vs_real": baseline.values,
            },
            indent=2,
        )
        + "\n"
    )

    # 2. per-column distances, real-vs-real next to each file
    reports = {
        k: fidelity_checks(real, v, thresholds=suggested, **common) for k, v in synth.items()
    }
    columns = pd.DataFrame(
        {"kind": baseline.columns["kind"], "real_vs_real": baseline.columns["distance"]}
    )
    for key, report in reports.items():
        columns[key] = report.columns["distance"]
    columns.round(4).to_csv(OUT / "column_distances.csv")

    # 3. which categories overshoot or undershoot (fill50: the most rows, so the least noise)
    big = synth["fill50"]
    limit = suggested[WORST_DISTANCE]
    shifts = []
    for column in columns.index[columns["fill50"] > limit]:
        if columns.loc[column, "kind"] != "categorical":
            continue
        r = real[column].value_counts(normalize=True)
        s = big[column].value_counts(normalize=True)
        labels = r.index.union(s.index)
        t = pd.DataFrame(
            {
                "real_share": r.reindex(labels, fill_value=0),
                "synthetic_share": s.reindex(labels, fill_value=0),
            }
        )
        t["difference"] = t["synthetic_share"] - t["real_share"]
        t = t.reindex(t["difference"].abs().sort_values(ascending=False).index).head(4)
        t.insert(0, "category", t.index)
        t.insert(0, "column", column)
        shifts.append(t)
    pd.concat(shifts).round(4).to_csv(OUT / "category_shifts.csv", index=False)

    # 4. numeric drift: quantiles of real and synthetic (fill50)
    rows = []
    for column in columns.index[columns["kind"] == "numeric"]:
        rq = real[column].quantile(list(QUANTILES))
        sq = big[column].quantile(list(QUANTILES))
        row = {"column": column, "ks": round(ks_distance(real[column], big[column]), 4)}
        for q in QUANTILES:
            row[f"real_q{int(q * 100)}"] = rq[q]
            row[f"synthetic_q{int(q * 100)}"] = sq[q]
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / "numeric_drift.csv", index=False)

    # 5. which direction of the pdays / poutcome rule breaks
    def breaks(df: pd.DataFrame) -> dict:
        no_prev_but_outcome = (df["pdays"] == -1) & (df["poutcome"] != "unknown")
        prev_but_unknown = (df["pdays"] != -1) & (df["poutcome"] == "unknown")
        return {
            "rows": len(df),
            "pdays=-1 but poutcome known": int(no_prev_but_outcome.sum()),
            "pdays>-1 but poutcome unknown": int(prev_but_unknown.sum()),
            "share_breaking": round(float((no_prev_but_outcome | prev_but_unknown).mean()), 4),
        }

    rule_rows = {
        "real train subscribers": breaks(real),
        "real train, all rows": breaks(train),
        "real validation subscribers": breaks(val[val[cfg.target] == 1]),
    }
    rule_rows.update({f"synthetic {k}": breaks(v) for k, v in synth.items()})
    rule_table = pd.DataFrame(rule_rows).T
    rule_table.to_csv(OUT / "rule_breaks.csv", index_label="set")
    print("\n=== pdays / poutcome rule, by direction\n" + rule_table.to_string())
    print(
        "\n=== category shifts (fill50 vs real train subscribers)\n"
        + pd.read_csv(OUT / "category_shifts.csv").to_string(index=False)
    )

    # 6. privacy, distinguishability and utility (quality_privacy, issue #47)
    by_seed = privacy_and_utility(train, val, synth)
    by_seed.round(6).to_csv(OUT / "privacy_utility_by_seed.csv", index=False)
    summary = summarize(by_seed)
    summary.round(6).to_csv(OUT / "privacy_utility.csv", index_label="file")
    print(
        f"\n=== privacy, distinguishability and utility (mean over {len(SEEDS)} seeds)\n"
        + summary.drop(columns="revealing_columns").round(3).T.to_string()
        + "\nrevealing columns: "
        + "; ".join(f"{k}: {v}" for k, v in summary["revealing_columns"].items())
    )

    # 7. how much the closest-record ratio varies for real data
    floor = dcr_noise_floor(train, val)
    pd.Series(floor.pop("ratios"), name="dcr_ratio").to_csv(
        OUT / "dcr_noise_floor_ratios.csv", index_label="split"
    )
    (OUT / "dcr_noise_floor.json").write_text(json.dumps(floor, indent=2) + "\n")
    print(f"\n=== closest-record ratio, real data ({DCR_SPLITS} random splits)\n{floor}")
    print(f"\nsaved to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
