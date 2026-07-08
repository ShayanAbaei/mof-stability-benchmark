"""
audit_groups_before_ml.py

Purpose:
    Audit metal and topology group structure BEFORE any ML interpretation.

Inputs:
    raw_data/
        ASR_data_SI_20250204.csv
        FSR_data_SI_20250204.csv
        ION_data_SI_20250204.csv
        ASR_FSR_check.csv
        12089-recommended-screening-list.csv

Outputs:
    group_audit_outputs/
        01_group_summary_by_dataset_target.csv
        02_all_group_sizes_by_dataset_target.csv
        03_seed42_heldout_groups.csv
        04_seed42_train_test_row_counts.csv
        05_group_size_bins.csv
        06_group_audit_report.txt

What it reports:
    - number of distinct primary-metal groups
    - number of distinct topology groups
    - group size distribution
    - rare group counts
    - held-out groups under GroupShuffleSplit(seed=42, test_size=0.2)
    - test rows per held-out group
    - whether train/test groups overlap
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# =============================================================================
# CONFIG
# =============================================================================

ROOT = Path(__file__).parent.resolve()
RAW_DIR = ROOT / "raw_data"
OUT_DIR = ROOT / "group_audit_outputs"
OUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20

TARGETS = [
    "Solvent_stability",
    "Water_stability",
    "Thermal_stability (℃)",
]

DATASETS = {
    "ASR": "ASR_data_SI_20250204.csv",
    "FSR": "FSR_data_SI_20250204.csv",
    "ION": "ION_data_SI_20250204.csv",
}

GROUP_DEFINITIONS = {
    "metal": "primary_metal",
    "topology_AllNodes": "topology(AllNodes)",
    "topology_SingleNodes": "topology(SingleNodes)",
}

RARE_THRESHOLDS = [1, 2, 5, 10, 20]

# =============================================================================
# CLEANING HELPERS
# =============================================================================

def split_primary_metal(x):
    """
    Extract first alphabetically sorted metal from 'Metal Types'.
    This mirrors the earlier benchmark logic.
    """
    if pd.isna(x):
        return np.nan
    parts = sorted([p.strip() for p in str(x).split(",") if p.strip()])
    return parts[0] if parts else np.nan

def clean_table(df):
    df = df.copy()

    if "Metal Types" in df.columns and "primary_metal" not in df.columns:
        df["primary_metal"] = df["Metal Types"].apply(split_primary_metal)

    for t in TARGETS:
        if t in df.columns:
            df[t] = pd.to_numeric(df[t], errors="coerce")

    return df

def safe_group_series(df, group_col):
    return df[group_col].fillna("MISSING").astype(str)

def group_size_statistics(group_counts):
    values = group_counts.values
    out = {
        "n_groups": int(len(values)),
        "n_rows": int(values.sum()),
        "group_size_min": int(values.min()) if len(values) else 0,
        "group_size_q1": float(np.percentile(values, 25)) if len(values) else np.nan,
        "group_size_median": float(np.median(values)) if len(values) else np.nan,
        "group_size_mean": float(np.mean(values)) if len(values) else np.nan,
        "group_size_q3": float(np.percentile(values, 75)) if len(values) else np.nan,
        "group_size_max": int(values.max()) if len(values) else 0,
    }

    for th in RARE_THRESHOLDS:
        if th == 1:
            out["n_singleton_groups"] = int((values == 1).sum())
        else:
            out[f"n_groups_lt{th}"] = int((values < th).sum())

    return out

def size_bin(n):
    if n == 1:
        return "1"
    if n <= 2:
        return "2"
    if n <= 5:
        return "3-5"
    if n <= 10:
        return "6-10"
    if n <= 20:
        return "11-20"
    if n <= 50:
        return "21-50"
    if n <= 100:
        return "51-100"
    return ">100"

# =============================================================================
# MAIN AUDIT
# =============================================================================

summary_rows = []
all_group_size_rows = []
heldout_rows = []
split_count_rows = []
bin_rows = []

for dataset_name, filename in DATASETS.items():
    path = RAW_DIR / filename
    if not path.exists():
        print(f"Missing file, skipping: {path}")
        continue

    print(f"\nLoading {dataset_name}: {filename}")
    df0 = clean_table(pd.read_csv(path))
    print(f"  raw shape: {df0.shape}")

    available_targets = [t for t in TARGETS if t in df0.columns]

    # For ION, targets may exist but dataset is tiny; still audit.
    for target in available_targets:
        df = df0[df0[target].notna()].copy().reset_index(drop=True)
        if len(df) == 0:
            continue

        print(f"  Target={target}, n={len(df)}")

        for group_label, group_col in GROUP_DEFINITIONS.items():
            if group_col not in df.columns:
                continue

            groups = safe_group_series(df, group_col)
            counts = groups.value_counts(dropna=False)
            stats = group_size_statistics(counts)

            # Main summary
            summary_row = {
                "dataset": dataset_name,
                "target": target,
                "group_label": group_label,
                "group_col": group_col,
                **stats,
            }

            # GroupShuffleSplit seed-42 held-out groups
            if stats["n_groups"] >= 2:
                splitter = GroupShuffleSplit(
                    n_splits=1,
                    test_size=TEST_SIZE,
                    random_state=RANDOM_STATE,
                )
                train_idx, test_idx = next(splitter.split(df, groups=groups.values))

                train_groups = set(groups.iloc[train_idx])
                test_groups = set(groups.iloc[test_idx])
                overlap = train_groups.intersection(test_groups)

                train_rows = len(train_idx)
                test_rows = len(test_idx)
                train_group_count = len(train_groups)
                test_group_count = len(test_groups)

                summary_row.update({
                    "seed42_n_train_rows": int(train_rows),
                    "seed42_n_test_rows": int(test_rows),
                    "seed42_test_row_fraction": float(test_rows / len(df)),
                    "seed42_n_train_groups": int(train_group_count),
                    "seed42_n_test_groups": int(test_group_count),
                    "seed42_test_group_fraction": float(test_group_count / stats["n_groups"]),
                    "seed42_train_test_group_overlap": int(len(overlap)),
                    "seed42_heldout_groups": "; ".join(sorted(test_groups)),
                })

                split_count_rows.append({
                    "dataset": dataset_name,
                    "target": target,
                    "group_label": group_label,
                    "group_col": group_col,
                    "n_total_rows": int(len(df)),
                    "n_total_groups": int(stats["n_groups"]),
                    "n_train_rows": int(train_rows),
                    "n_test_rows": int(test_rows),
                    "test_row_fraction": float(test_rows / len(df)),
                    "n_train_groups": int(train_group_count),
                    "n_test_groups": int(test_group_count),
                    "test_group_fraction": float(test_group_count / stats["n_groups"]),
                    "train_test_group_overlap": int(len(overlap)),
                })

                heldout_counts = groups.iloc[test_idx].value_counts()
                for g, n in heldout_counts.items():
                    heldout_rows.append({
                        "dataset": dataset_name,
                        "target": target,
                        "group_label": group_label,
                        "group_col": group_col,
                        "heldout_group": g,
                        "n_test_rows_in_group": int(n),
                        "fraction_of_test_rows": float(n / test_rows),
                        "total_group_size_in_dataset": int(counts.loc[g]),
                    })
            else:
                summary_row.update({
                    "seed42_n_train_rows": np.nan,
                    "seed42_n_test_rows": np.nan,
                    "seed42_test_row_fraction": np.nan,
                    "seed42_n_train_groups": np.nan,
                    "seed42_n_test_groups": np.nan,
                    "seed42_test_group_fraction": np.nan,
                    "seed42_train_test_group_overlap": np.nan,
                    "seed42_heldout_groups": "",
                })

            summary_rows.append(summary_row)

            # All group sizes
            for group_name, n in counts.items():
                all_group_size_rows.append({
                    "dataset": dataset_name,
                    "target": target,
                    "group_label": group_label,
                    "group_col": group_col,
                    "group": group_name,
                    "n_rows": int(n),
                    "size_bin": size_bin(int(n)),
                })

            # Size bins
            bin_counts = pd.Series([size_bin(int(n)) for n in counts.values]).value_counts()
            for b, n_groups in bin_counts.items():
                bin_rows.append({
                    "dataset": dataset_name,
                    "target": target,
                    "group_label": group_label,
                    "group_col": group_col,
                    "size_bin": b,
                    "n_groups": int(n_groups),
                })

# =============================================================================
# SAVE OUTPUTS
# =============================================================================

summary_df = pd.DataFrame(summary_rows)
all_sizes_df = pd.DataFrame(all_group_size_rows)
heldout_df = pd.DataFrame(heldout_rows)
split_counts_df = pd.DataFrame(split_count_rows)
bins_df = pd.DataFrame(bin_rows)

summary_df.to_csv(OUT_DIR / "01_group_summary_by_dataset_target.csv", index=False)
all_sizes_df.to_csv(OUT_DIR / "02_all_group_sizes_by_dataset_target.csv", index=False)
heldout_df.to_csv(OUT_DIR / "03_seed42_heldout_groups.csv", index=False)
split_counts_df.to_csv(OUT_DIR / "04_seed42_train_test_row_counts.csv", index=False)
bins_df.to_csv(OUT_DIR / "05_group_size_bins.csv", index=False)

# =============================================================================
# TEXT REPORT
# =============================================================================

lines = []
lines.append("=" * 80)
lines.append("GROUP SPLIT AUDIT BEFORE ML")
lines.append("=" * 80)
lines.append("")
lines.append(f"Root folder: {ROOT}")
lines.append(f"Random state: {RANDOM_STATE}")
lines.append(f"GroupShuffleSplit test_size: {TEST_SIZE}")
lines.append("")
lines.append("Important interpretation:")
lines.append("- GroupShuffleSplit(test_size=0.2) selects 20% of UNIQUE GROUPS, not 20% of rows.")
lines.append("- Therefore test-row fractions can differ strongly from 0.2 when group sizes are imbalanced.")
lines.append("- This report quantifies that imbalance for primary-metal and topology grouping.")
lines.append("")

for dataset_name in summary_df["dataset"].unique():
    lines.append("")
    lines.append("-" * 80)
    lines.append(f"DATASET: {dataset_name}")
    lines.append("-" * 80)

    sdf = summary_df[summary_df["dataset"] == dataset_name]
    for _, row in sdf.iterrows():
        lines.append("")
        lines.append(f"Target: {row['target']}")
        lines.append(f"Grouping: {row['group_label']} ({row['group_col']})")
        lines.append(f"  rows with target: {row['n_rows']}")
        lines.append(f"  distinct groups: {row['n_groups']}")
        lines.append(
            f"  group size min/median/mean/max: "
            f"{row['group_size_min']} / {row['group_size_median']:.1f} / "
            f"{row['group_size_mean']:.1f} / {row['group_size_max']}"
        )
        lines.append(f"  singleton groups: {row.get('n_singleton_groups', 'NA')}")
        lines.append(f"  groups <5 rows: {row.get('n_groups_lt5', 'NA')}")
        lines.append(f"  groups <10 rows: {row.get('n_groups_lt10', 'NA')}")
        lines.append(
            f"  seed42 split rows train/test: "
            f"{row['seed42_n_train_rows']} / {row['seed42_n_test_rows']} "
            f"(test row fraction={row['seed42_test_row_fraction']:.3f})"
        )
        lines.append(
            f"  seed42 split groups train/test: "
            f"{row['seed42_n_train_groups']} / {row['seed42_n_test_groups']} "
            f"(test group fraction={row['seed42_test_group_fraction']:.3f})"
        )
        lines.append(f"  train/test group overlap: {row['seed42_train_test_group_overlap']}")
        lines.append(f"  held-out groups: {row['seed42_heldout_groups']}")

report = "\n".join(lines)
with open(OUT_DIR / "06_group_audit_report.txt", "w", encoding="utf-8") as f:
    f.write(report)

print("\nWrote outputs to:")
for fn in [
    "01_group_summary_by_dataset_target.csv",
    "02_all_group_sizes_by_dataset_target.csv",
    "03_seed42_heldout_groups.csv",
    "04_seed42_train_test_row_counts.csv",
    "05_group_size_bins.csv",
    "06_group_audit_report.txt",
]:
    print(" ", OUT_DIR / fn)

print("\nPreview:")
print(summary_df.head(20).to_string(index=False))