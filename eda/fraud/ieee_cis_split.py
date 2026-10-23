"""IEEE-CIS Fraud Detection: duplicates, the time split on TransactionDT and the training sample (issue #50).

Reads the two Kaggle tables from a local folder, joins them, and measures everything the settings
entry in ``src/agentic_gan/utils/datasets.json`` and ``docs/fraud_dataset_recommendation.md``
depend on:

* the join: transactions left-joined to identity on ``TransactionID``;
* duplicates: rows identical in every column except ``TransactionID`` (``TransactionDT`` included);
* the split: first 60% / next 20% / last 20% of rows by ``TransactionDT``, as two cut points;
* the sample: a stratified draw from the train part, for the generator and the arms;
* the discrete columns, and the columns that are mostly empty.

The data may not be redistributed (Kaggle rules, section 7B), so this prints and writes
aggregates only: counts, rates and cut points. No row, no ID and no value of a feature is saved.
The split is stored as a rule (cut points), not as a list of IDs.

Run from the repo root, with each person's own copy of the data in ``data/external``:

    uv run python -I eda/fraud/ieee_cis_split.py --write-splits splits
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR = ROOT / "data" / "external" / "ieee-fraud-detection"
TARGET, KEY, TIME = "isFraud", "TransactionID", "TransactionDT"
FRACTIONS = (0.6, 0.2, 0.2)
SAMPLE_ROWS, SEED = 100_000, 42
FILL_LEVELS = (0.10, 0.25, 0.50, 0.75, 1.00)

# the categorical columns named on the competition's data page
TRANSACTION_CATEGORICAL = [
    "ProductCD", "card1", "card2", "card3", "card4", "card5", "card6", "addr1", "addr2",
    "P_emaildomain", "R_emaildomain", *[f"M{i}" for i in range(1, 10)],
]  # fmt: skip
IDENTITY_CATEGORICAL = ["DeviceType", "DeviceInfo", *[f"id_{i}" for i in range(12, 39)]]
DISCRETE = TRANSACTION_CATEGORICAL + IDENTITY_CATEGORICAL


def row_hash(df: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """One 64-bit hash per row over the given columns (missing values hash the same each time)."""
    return pd.util.hash_pandas_object(df[columns], index=False).to_numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--write-splits", type=Path, help="folder for ieee_cis_time_split.csv")
    args = parser.parse_args()

    tx = pd.read_csv(args.data_dir / "train_transaction.csv")
    ident = pd.read_csv(args.data_dir / "train_identity.csv")
    assert tx[KEY].is_unique and ident[KEY].is_unique, "TransactionID must be unique in both tables"
    df = tx.merge(ident, on=KEY, how="left", validate="one_to_one")
    has_identity = df[KEY].isin(ident[KEY])
    del tx
    out = {
        "rows": len(df),
        "columns": df.shape[1],
        "rows_with_identity": int(has_identity.sum()),
        "share_with_identity": round(float(has_identity.mean()), 4),
        "fraud_rate_with_identity": round(float(df.loc[has_identity, TARGET].mean()), 4),
        "fraud_rate_without_identity": round(float(df.loc[~has_identity, TARGET].mean()), 4),
    }
    out["identity_columns"] = ident.shape[1] - 1
    out["transaction_columns"] = out["columns"] - out["identity_columns"]
    missing_discrete = [c for c in DISCRETE if c not in df.columns]
    assert not missing_discrete, f"discrete columns not in the data: {missing_discrete}"
    print(json.dumps(out, indent=2))

    # --- duplicates -------------------------------------------------------------------------
    features = [c for c in df.columns if c != KEY]
    no_time = [c for c in features if c != TIME]
    h_all, h_no_time = row_hash(df, features), row_hash(df, no_time)
    dup_all = pd.Series(h_all).duplicated(keep="first").to_numpy()
    dup_no_time = pd.Series(h_no_time).duplicated(keep="first").to_numpy()
    no_label = [c for c in no_time if c != TARGET]
    labels_per_group = df.assign(_h=row_hash(df, no_label)).groupby("_h")[TARGET].nunique()
    fraud = df[TARGET].to_numpy() == 1
    dups = {
        "exact_copies_extra_rows": int(dup_all.sum()),
        "exact_copies_extra_fraud_rows": int((dup_all & fraud).sum()),
        "groups_with_identical_features_but_both_labels": int((labels_per_group > 1).sum()),
        "extra_copies_when_time_ignored": int(dup_no_time.sum()),
        "of_which_at_a_different_time": int(dup_no_time.sum() - dup_all.sum()),
        "of_which_at_a_different_time_fraud": int(((dup_no_time & ~dup_all) & fraud).sum()),
    }
    print("\nduplicates", json.dumps(dups, indent=2))

    # --- time split on the data left after dropping exact copies -------------------------------
    keep = ~dup_all
    data = df.loc[keep].reset_index(drop=True)
    h_keep = h_no_time[keep]
    n = len(data)
    order = np.sort(data[TIME].to_numpy())
    t1, t2 = int(order[int(FRACTIONS[0] * n)]), int(order[int((FRACTIONS[0] + FRACTIONS[1]) * n)])
    part = np.where(data[TIME] < t1, "train", np.where(data[TIME] < t2, "validation", "test"))
    rows = []
    for name, rule in (
        ("train", f"{TIME} < {t1}"),
        ("validation", f"{t1} <= {TIME} < {t2}"),
        ("test", f"{TIME} >= {t2}"),
    ):
        sel = part == name
        sub = data[sel]
        rows.append(
            {
                "part": name,
                "rule": rule,
                "transaction_dt_from": int(sub[TIME].min()),
                "transaction_dt_to": int(sub[TIME].max()),
                "rows": int(sel.sum()),
                "fraud_rows": int(sub[TARGET].sum()),
                "fraud_rate": round(float(sub[TARGET].mean()), 6),
            }
        )
    table = pd.DataFrame(rows)
    print(
        f"\ncut points: {t1} (day {t1 / 86400:.1f}) and {t2} (day {t2 / 86400:.1f}), rows after dedupe: {n}"
    )

    # repeats at another time that fall in different parts (kept, as for ULB)
    train_hashes = set(h_keep[part == "train"].tolist())
    crossing = {}
    for name in ("validation", "test"):
        sel = (part == name) & np.isin(h_keep, list(train_hashes))
        crossing[name] = {"rows": int(sel.sum()), "fraud_rows": int(data.loc[sel, TARGET].sum())}
    print("rows with the same values as a train row at another time", crossing)

    # --- the sample drawn from the train part ---------------------------------------------------
    train_pos = np.flatnonzero(part == "train")
    sample_pos, _ = train_test_split(
        train_pos,
        train_size=SAMPLE_ROWS,
        stratify=data[TARGET].to_numpy()[train_pos],
        random_state=SEED,
    )
    sample = data.iloc[np.sort(sample_pos)]
    pos, neg = int(sample[TARGET].sum()), int(len(sample) - sample[TARGET].sum())
    gap = neg - pos
    fills = {str(level): round(level * gap) for level in FILL_LEVELS}
    table = pd.concat(
        [
            table,
            pd.DataFrame(
                [
                    {
                        "part": "train_sample",
                        "rule": f"stratified on {TARGET}, {SAMPLE_ROWS} rows of the train part, seed {SEED}",
                        "transaction_dt_from": int(sample[TIME].min()),
                        "transaction_dt_to": int(sample[TIME].max()),
                        "rows": len(sample),
                        "fraud_rows": pos,
                        "fraud_rate": round(pos / len(sample), 6),
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    print("\n" + table.to_string(index=False))
    print(
        f"\nsample class gap {gap}, fill rows {fills}, fill 1.0 is {gap / pos:.1f} times the real fraud rows"
    )

    # --- columns ---------------------------------------------------------------------------------
    cols = [c for c in data.columns if c not in {KEY, TARGET, TIME}]
    miss = sample[cols].isna().mean()
    for cap in (0.05, 0.10, 0.20):
        kept = miss[miss <= cap].index
        print(
            f"columns with at most {cap:.0%} missing in the sample: {len(kept)} "
            f"({sum(c in DISCRETE for c in kept)} of them discrete)"
        )
    card = pd.DataFrame(
        {
            "categories": sample[DISCRETE].nunique(),
            "missing": sample[DISCRETE].isna().mean().round(3),
        }
    ).sort_values("categories", ascending=False)
    print(f"\ndiscrete columns: {len(DISCRETE)}; over 50 categories in the sample:")
    print(card[card["categories"] > 50].to_string())
    print(f"discrete columns with at most 20% missing: {int((card['missing'] <= 0.2).sum())}")

    if args.write_splits:
        args.write_splits.mkdir(parents=True, exist_ok=True)
        path = args.write_splits / "ieee_cis_time_split.csv"
        table.to_csv(path, index=False)
        print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
