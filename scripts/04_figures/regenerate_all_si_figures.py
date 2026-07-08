"""
regenerate_all_si_figures.py

Builds ALL SI figures (S1-S9) from raw data + V14 results.
Clean publication-ready style. No inline titles. 600 dpi.

Inputs:
    raw_data/ASR_data_SI_20250204.csv
    raw_data/FSR_data_SI_20250204.csv
    raw_data/ION_data_SI_20250204.csv
    v14_results.csv
    v14_predictions.csv

Outputs (in figures/supporting_information/):
    figS1_missingness_ASR.pdf
    figS2_missingness_FSR.pdf
    figS3_missingness_ION.pdf
    figS4_descriptor_distributions.pdf
    figS5_metal_breakdown.pdf
    figS6_grouped_cv_details.pdf
    figS7_expanded_descriptor_sufficiency.pdf
    figS8_importance.pdf
    figS9_screening_subset.pdf
"""

from pathlib import Path
import warnings
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIG
# =============================================================================

ROOT = Path(__file__).parent.resolve()
RAW = ROOT / "raw_data"
FIG_DIR = ROOT / "figures" / "supporting_information"
FIG_DIR.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

REGIME_COLORS = {
    "A_metal_only": "#4C72B0",
    "B_metal_oms": "#55A868",
    "C_metal_oms_context": "#E07B39",
    "D_context_thermophysical": "#C44E52",
}
REGIME_LABEL_SHORT = {
    "A_metal_only": "A",
    "B_metal_oms": "B",
    "C_metal_oms_context": "C",
    "D_context_thermophysical": "D",
}
SPLIT_COLORS = {
    "random": "#4C72B0",
    "group_metal": "#E07B39",
    "group_topology": "#C44E52",
}
SPLIT_LABEL = {
    "random": "Random",
    "group_metal": "Metal-grouped",
    "group_topology": "Topology-grouped",
}


def clean_target_label(t):
    return t.replace("_stability", "").replace(" (℃)", "").strip()


def add_panel_label(ax, label, x=-0.15, y=1.05):
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="bottom", ha="left")


# =============================================================================
# LOAD DATA
# =============================================================================

print("Loading raw and V14 data...")
asr = pd.read_csv(RAW / "ASR_data_SI_20250204.csv")
fsr = pd.read_csv(RAW / "FSR_data_SI_20250204.csv")
ion = pd.read_csv(RAW / "ION_data_SI_20250204.csv")
res = pd.read_csv("v14_results.csv")
pred = pd.read_csv("v14_predictions.csv")

print(f"  ASR: {asr.shape}, FSR: {fsr.shape}, ION: {ion.shape}")
print(f"  v14_results: {res.shape}, v14_predictions: {pred.shape}")

# Helper: parse metals
def first_metal(x):
    if pd.isna(x): return np.nan
    parts = sorted([p.strip() for p in str(x).split(",") if p.strip()])
    return parts[0] if parts else np.nan

for df in [asr, fsr, ion]:
    if "Metal Types" in df.columns:
        df["primary_metal"] = df["Metal Types"].apply(first_metal)
        df["n_metals"] = df["Metal Types"].apply(
            lambda x: len([p for p in str(x).split(",") if p.strip()]) if pd.notna(x) else 0)
        df["is_mixed_metal"] = (df["n_metals"] > 1).astype(int)

summary = res.groupby(
    ["dataset", "target", "regime", "split_type", "model"]
).agg(
    Spearman_mean=("Spearman", "mean"),
    Spearman_std=("Spearman", "std"),
    R2_mean=("R2", "mean"),
    RMSE_mean=("RMSE", "mean"),
).reset_index()


# =============================================================================
# FIG S1, S2, S3: MISSINGNESS PER DATASET
# =============================================================================


def make_missingness_fig(df, dataset_name, filename):
    print(f"\nBuilding Fig {filename} (missingness)...")

    # Get missingness for ALL columns, then take top 20 by missingness
    miss = df.isna().mean().sort_values(ascending=True)
    miss = miss.tail(20)  # top 20 most missing (could include 0% if tied)

    # If there are fewer than 5 missing columns, just show the top 20 by name
    if (miss > 0).sum() < 5:
        # Fall back: show columns most likely to be metadata/auxiliary
        priority_cols = ["memo", "OMS Types", "secondary_metal", "Time",
                          "hall", "Year", "LFPD (Å)", "metal_list",
                          "NAV (A3)", "NAV_VF", "NPV (cm3/g)",
                          "std @ 300 K (J/g/K)", "std @ 350 K (J/g/K)",
                          "std @ 400 K (J/g/K)", "k_cp (J/g/K/K)", "cp0 (J/g/K)",
                          "dimension_by_topo", "number_spacegroup"]
        priority_present = [c for c in priority_cols if c in df.columns]
        miss = df[priority_present].isna().mean().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(7.5, max(3, len(miss) * 0.25)))
    y_pos = np.arange(len(miss))

    ax.barh(y_pos, miss.values * 100, color="#4C72B0",
            edgecolor="black", linewidth=0.4)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(miss.index, fontsize=8)
    ax.set_xlabel(f"Missingness in {dataset_name} (%)")
    ax.set_xlim(0, 100)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / filename, format="pdf")
    plt.close()
    print(f"  → {FIG_DIR / filename}")



def make_figS1():
    make_missingness_fig(asr, "ASR", "figS1_missingness_ASR.pdf")


def make_figS2():
    make_missingness_fig(fsr, "FSR", "figS2_missingness_FSR.pdf")


def make_figS3():
    make_missingness_fig(ion, "ION", "figS3_missingness_ION.pdf")


# =============================================================================
# FIG S4: DESCRIPTOR DISTRIBUTIONS
# =============================================================================

def make_figS4():
    print("\nBuilding Fig S4 (descriptor distributions)...")
    cols = ["Density (g/cm3)", "LCD (Å)", "PLD (Å)", "VF",
            "PV (cm3/g)", "average_atomic_mass", "natoms"]

    fig, axes = plt.subplots(2, 4, figsize=(11, 5.5))
    axes = axes.flatten()

    for i, col in enumerate(cols):
        ax = axes[i]
        vals = pd.to_numeric(asr[col], errors="coerce").dropna()
        ax.hist(vals, bins=40, color=REGIME_COLORS["D_context_thermophysical"],
                edgecolor="black", linewidth=0.4, alpha=0.85)
        # Clean axis labels
        label_clean = col.replace("(g/cm3)", "(g/cm³)").replace("(cm3/g)", "(cm³/g)")
        ax.set_xlabel(label_clean, fontsize=8.5)
        if i % 4 == 0:
            ax.set_ylabel("Count")
        ax.grid(axis="y", alpha=0.3)

    # Hide unused subplot
    axes[7].axis("off")
    axes[7].text(0.5, 0.5,
                  f"ASR descriptor\ndistributions\n\nn = {len(asr):,} frameworks",
                  transform=axes[7].transAxes,
                  fontsize=10, ha="center", va="center",
                  bbox=dict(boxstyle="round,pad=0.6",
                            facecolor="#f5f5f5", edgecolor="gray", linewidth=0.6))

    plt.tight_layout()
    plt.savefig(FIG_DIR / "figS4_descriptor_distributions.pdf", format="pdf")
    plt.close()
    print(f"  → {FIG_DIR / 'figS4_descriptor_distributions.pdf'}")


# =============================================================================
# FIG S5: METAL BREAKDOWN
# =============================================================================

def make_figS5():
    print("\nBuilding Fig S5 (metal breakdown)...")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))


    # Panel (a): Top 20 primary metals
    ax = axes[0]
    metal_counts = asr["primary_metal"].value_counts().head(20)
    y_pos = np.arange(len(metal_counts))
    ax.barh(y_pos, metal_counts.values, color="#4C72B0",
            edgecolor="black", linewidth=0.4)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(metal_counts.index, fontsize=8.5)
    ax.set_xlabel("Frequency in ASR")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    add_panel_label(ax, "(a)")

    # Panel (b): Single vs mixed
    ax = axes[1]
    single_count = (asr["is_mixed_metal"] == 0).sum()
    mixed_count = (asr["is_mixed_metal"] == 1).sum()
    counts = [single_count, mixed_count]
    labels = ["Single-metal", "Mixed-metal"]
    colors_b = [REGIME_COLORS["A_metal_only"], REGIME_COLORS["D_context_thermophysical"]]

    bars = ax.bar(labels, counts, color=colors_b,
                   edgecolor="black", linewidth=0.5, width=0.55)
    ax.set_ylabel("Number of frameworks (ASR)")
    ax.grid(axis="y", alpha=0.3)

    # Add count labels on top
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 10,
                f"{count}\n({100*count/sum(counts):.0f}%)",
                ha="center", va="bottom", fontsize=9)

    ax.set_ylim(0, max(counts) * 1.18)
    add_panel_label(ax, "(b)")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "figS5_metal_breakdown.pdf", format="pdf")
    plt.close()
    print(f"  → {FIG_DIR / 'figS5_metal_breakdown.pdf'}")


# =============================================================================
# FIG S6: GROUPED-BY-PRIMARY-METAL CV PERFORMANCE
# =============================================================================

def make_figS6():
    print("\nBuilding Fig S6 (grouped-by-metal CV)...")
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.0))

    targets = ["Solvent_stability", "Water_stability", "Thermal_stability (℃)"]
    target_labels = ["Solvent stability", "Water stability", "Thermal stability"]
    regimes = ["A_metal_only", "B_metal_oms", "C_metal_oms_context",
               "D_context_thermophysical"]
    models = ["Linear", "Ridge", "LASSO", "DecisionTree",
              "SVR_Linear", "RandomForest", "LightGBM"]
    model_labels = ["Lin", "Ridge", "LASSO", "DT", "SVR", "RF", "LGBM"]

    sub = summary[summary["split_type"] == "group_metal"]

    for i, (target, t_label) in enumerate(zip(targets, target_labels)):
        ax = axes[i]
        target_sub = sub[(sub["target"] == target) &
                          (sub["dataset"] == "ASR")]

        x = np.arange(len(models))
        width = 0.18

        for r_idx, regime in enumerate(regimes):
            vals = []
            errs = []
            for m in models:
                row = target_sub[(target_sub["regime"] == regime) &
                                  (target_sub["model"] == m)]
                vals.append(float(row["Spearman_mean"].iloc[0]) if len(row) else 0)
                errs.append(float(row["Spearman_std"].iloc[0])
                            if len(row) and pd.notna(row["Spearman_std"].iloc[0]) else 0)
            offset = (r_idx - 1.5) * width
            ax.bar(x + offset, vals, width,
                   yerr=errs, capsize=1.2,
                   color=REGIME_COLORS[regime],
                   edgecolor="black", linewidth=0.3,
                   label=f"Regime {REGIME_LABEL_SHORT[regime]}")

        ax.set_xticks(x)
        ax.set_xticklabels(model_labels, fontsize=7.5)
        ax.set_ylim(-0.3, 0.55)
        ax.axhline(0, color="black", linewidth=0.4)
        ax.set_ylabel("Spearman ± SD" if i == 0 else "")
        ax.set_title(f"ASR: {t_label}", fontsize=9.5)
        ax.grid(axis="y", alpha=0.3)
        if i == 1:
            ax.legend(loc="upper center", fontsize=7.5, ncol=4, frameon=False,
                      bbox_to_anchor=(0.5, -0.22))
        add_panel_label(ax, f"({chr(97+i)})")

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.20)
    plt.savefig(FIG_DIR / "figS6_grouped_cv_details.pdf", format="pdf")
    plt.close()
    print(f"  → {FIG_DIR / 'figS6_grouped_cv_details.pdf'}")


# =============================================================================
# FIG S7: EXPANDED DESCRIPTOR SUFFICIENCY (6 panels)
# =============================================================================

def make_figS7():
    print("\nBuilding Fig S7 (expanded descriptor sufficiency)...")
    fig, axes = plt.subplots(2, 3, figsize=(11, 7.0))

    regimes = ["A_metal_only", "B_metal_oms", "C_metal_oms_context",
               "D_context_thermophysical"]
    regime_x = np.arange(len(regimes))
    regime_labels = ["A", "B", "C", "D"]
    models = ["Linear", "Ridge", "LASSO", "DecisionTree",
              "SVR_Linear", "RandomForest", "LightGBM"]
    model_labels = ["Lin", "Ridge", "LASSO", "DT", "SVR", "RF", "LGBM"]
    model_colors = ["#8c8c8c", "#4C72B0", "#55A868", "#937860",
                    "#E15759", "#E07B39", "#C44E52"]

    # Panel (a): ASR solvent stability across models × regimes, random split
    ax = axes[0, 0]
    sub = summary[(summary["target"] == "Solvent_stability") &
                  (summary["dataset"] == "ASR") &
                  (summary["split_type"] == "random")]
    for m_idx, m in enumerate(models):
        vals = []
        for r in regimes:
            row = sub[(sub["regime"] == r) & (sub["model"] == m)]
            vals.append(float(row["Spearman_mean"].iloc[0]) if len(row) else np.nan)
        ax.plot(regime_x, vals, "-o", color=model_colors[m_idx],
                markersize=4, linewidth=1.2, label=model_labels[m_idx])
    ax.set_xticks(regime_x)
    ax.set_xticklabels(regime_labels)
    ax.set_ylim(0, 0.7)
    ax.set_xlabel("Descriptor regime")
    ax.set_ylabel("Spearman (random)")
    ax.set_title("ASR solvent stability", fontsize=9.5)
    ax.grid(axis="y", alpha=0.3)
    add_panel_label(ax, "(a)")

    # Panel (b): ASR water stability across models × regimes, random split
    ax = axes[0, 1]
    sub = summary[(summary["target"] == "Water_stability") &
                  (summary["dataset"] == "ASR") &
                  (summary["split_type"] == "random")]
    for m_idx, m in enumerate(models):
        vals = []
        for r in regimes:
            row = sub[(sub["regime"] == r) & (sub["model"] == m)]
            vals.append(float(row["Spearman_mean"].iloc[0]) if len(row) else np.nan)
        ax.plot(regime_x, vals, "-o", color=model_colors[m_idx],
                markersize=4, linewidth=1.2, label=model_labels[m_idx])
    ax.set_xticks(regime_x)
    ax.set_xticklabels(regime_labels)
    ax.set_ylim(0, 0.75)
    ax.set_xlabel("Descriptor regime")
    ax.set_ylabel("Spearman (random)")
    ax.set_title("ASR water stability", fontsize=9.5)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper center", fontsize=6.5, ncol=7, frameon=False,
              bbox_to_anchor=(0.5, -0.22))
    add_panel_label(ax, "(b)")

    # Panel (c): ROC-AUC across cutoffs as a function of regime, ASR water
    ax = axes[0, 2]
    cutoffs = [0.6, 0.7, 0.8]
    for r_idx, regime in enumerate(regimes):
        aucs = []
        for cut in cutoffs:
            pt = pred[(pred["dataset"] == "ASR") &
                      (pred["target"] == "Water_stability") &
                      (pred["regime"] == regime) &
                      (pred["split_type"] == "random") &
                      (pred["model"] == "LightGBM")].copy()
            if len(pt) > 0:
                y_class = (pt["y_true"] >= cut).astype(int)
                if y_class.nunique() > 1:
                    aucs.append(roc_auc_score(y_class, pt["y_pred"]))
                else:
                    aucs.append(0.5)
            else:
                aucs.append(0.5)
        ax.plot(cutoffs, aucs, "-o", color=REGIME_COLORS[regime],
                linewidth=1.5, markersize=5,
                label=f"Regime {REGIME_LABEL_SHORT[regime]}")
    ax.set_xticks(cutoffs)
    ax.set_xlabel("Stability cutoff")
    ax.set_ylabel("ROC-AUC (ASR water)")
    ax.set_ylim(0.5, 1.0)
    ax.set_title("Threshold sensitivity", fontsize=9.5)
    ax.legend(loc="lower right", fontsize=7, frameon=False)
    ax.grid(axis="y", alpha=0.3)
    add_panel_label(ax, "(c)")

    # Panel (d): Top-decile enrichment by regime — legend on LEFT as requested
    ax = axes[1, 0]
    for r_idx, regime in enumerate(regimes):
        enrichments = []
        for cut in cutoffs:
            pt = pred[(pred["dataset"] == "ASR") &
                      (pred["target"] == "Water_stability") &
                      (pred["regime"] == regime) &
                      (pred["split_type"] == "random") &
                      (pred["model"] == "LightGBM")].copy()
            if len(pt) > 0:
                y_class = (pt["y_true"] >= cut).astype(int)
                baseline = y_class.mean()
                if baseline > 0:
                    sorted_pt = pt.sort_values("y_pred", ascending=False)
                    top10 = sorted_pt.head(max(1, int(0.1 * len(sorted_pt))))
                    top10_class = (top10["y_true"] >= cut).astype(int)
                    enr = top10_class.mean() / baseline
                else:
                    enr = 0
                enrichments.append(enr)
            else:
                enrichments.append(0)
        ax.plot(cutoffs, enrichments, "-o", color=REGIME_COLORS[regime],
                linewidth=1.5, markersize=5,
                label=f"Regime {REGIME_LABEL_SHORT[regime]}")
    ax.set_xticks(cutoffs)
    ax.set_xlabel("Stability cutoff")
    ax.set_ylabel("Top-decile enrichment")
    ax.axhline(1, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title("Enrichment by regime", fontsize=9.5)
    ax.legend(loc="upper left", fontsize=7, frameon=False)
    ax.grid(axis="y", alpha=0.3)
    add_panel_label(ax, "(d)")

    # Panel (e): Random vs metal-grouped vs topology-grouped (ASR water, by regime)
    ax = axes[1, 1]
    best_d = summary.sort_values("Spearman_mean", ascending=False).groupby(
        ["dataset", "target", "regime", "split_type"]).head(1)
    asr_water = best_d[(best_d["dataset"] == "ASR") &
                       (best_d["target"] == "Water_stability")]
    x_pos = np.arange(len(regimes))
    width = 0.27

    for s_idx, split in enumerate(["random", "group_metal", "group_topology"]):
        vals = []
        errs = []
        for r in regimes:
            row = asr_water[(asr_water["regime"] == r) &
                            (asr_water["split_type"] == split)]
            vals.append(float(row["Spearman_mean"].iloc[0]) if len(row) else 0)
            errs.append(float(row["Spearman_std"].iloc[0])
                        if len(row) and pd.notna(row["Spearman_std"].iloc[0]) else 0)
        offset = (s_idx - 1) * width
        ax.bar(x_pos + offset, vals, width,
               yerr=errs, capsize=1.5,
               color=SPLIT_COLORS[split],
               edgecolor="black", linewidth=0.4,
               label=SPLIT_LABEL[split])
    ax.set_xticks(x_pos)
    ax.set_xticklabels(regime_labels)
    ax.set_xlabel("Descriptor regime")
    ax.set_ylabel("Best Spearman ± SD")
    ax.set_ylim(-0.1, 0.85)
    ax.axhline(0, color="black", linewidth=0.4)
    ax.set_title("ASR water: split discipline", fontsize=9.5)
    ax.legend(loc="upper left", fontsize=7, frameon=False)
    ax.grid(axis="y", alpha=0.3)
    add_panel_label(ax, "(e)")

    # Panel (f): Best Spearman by regime, all targets, random split, both datasets
    ax = axes[1, 2]
    best_random = summary[summary["split_type"] == "random"].sort_values(
        "Spearman_mean", ascending=False).groupby(
        ["dataset", "target", "regime"]).head(1)

    targets = ["Solvent_stability", "Water_stability", "Thermal_stability (℃)"]
    target_colors = ["#4C72B0", "#E07B39", "#C44E52"]
    linestyles = {"ASR": "-", "FSR": "--"}

    for ds in ["ASR", "FSR"]:
        for t_idx, t in enumerate(targets):
            vals = []
            for r in regimes:
                row = best_random[(best_random["dataset"] == ds) &
                                   (best_random["target"] == t) &
                                   (best_random["regime"] == r)]
                vals.append(float(row["Spearman_mean"].iloc[0]) if len(row) else np.nan)
            ax.plot(regime_x, vals, linestyles[ds] + "o",
                    color=target_colors[t_idx],
                    markersize=4, linewidth=1.2,
                    label=f"{ds} {clean_target_label(t)}")
    ax.set_xticks(regime_x)
    ax.set_xticklabels(regime_labels)
    ax.set_xlabel("Descriptor regime")
    ax.set_ylabel("Best Spearman (random)")
    ax.set_ylim(0, 0.75)
    ax.set_title("Descriptor ladder summary", fontsize=9.5)
    ax.legend(loc="lower right", fontsize=6.5, frameon=False, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    add_panel_label(ax, "(f)")

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.55, bottom=0.10)
    plt.savefig(FIG_DIR / "figS7_expanded_descriptor_sufficiency.pdf", format="pdf")
    plt.close()
    print(f"  → {FIG_DIR / 'figS7_expanded_descriptor_sufficiency.pdf'}")


# =============================================================================
# FIG S8: PERMUTATION IMPORTANCE (THERMAL STABILITY)
# =============================================================================

def make_figS8():
    print("\nBuilding Fig S8 (permutation importance)...")
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.inspection import permutation_importance
    import lightgbm as lgb

    asr_t = asr.copy()
    asr_t["Thermal_stability (℃)"] = pd.to_numeric(
        asr_t["Thermal_stability (℃)"], errors="coerce")
    asr_t = asr_t.dropna(subset=["Thermal_stability (℃)"])

    feat_cols = ["primary_metal", "is_mixed_metal", "n_metals", "Has OMS", "OMS Types",
                 "structure_dimension", "topology(SingleNodes)", "topology(AllNodes)",
                 "catenation",
                 "Density (g/cm3)", "LCD (Å)", "PLD (Å)", "VF", "PV (cm3/g)",
                 "average_atomic_mass",
                 "Heat_capacity@300K (J/g/K)", "Heat_capacity@350K (J/g/K)",
                 "Heat_capacity@400K (J/g/K)",
                 "k_cp (J/g/K/K)", "cp0 (J/g/K)", "natoms"]
    feat_cols = [c for c in feat_cols if c in asr_t.columns]

    for c in feat_cols:
        if c not in ["primary_metal", "Has OMS", "OMS Types",
                      "topology(SingleNodes)", "topology(AllNodes)"]:
            asr_t[c] = pd.to_numeric(asr_t[c], errors="coerce")

    X = asr_t[feat_cols]
    y = asr_t["Thermal_stability (℃)"].values

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    pre = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])
    X_proc = pre.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_proc, y, test_size=0.2, random_state=42)
    model = lgb.LGBMRegressor(n_estimators=200, num_leaves=31,
                               learning_rate=0.05, random_state=42, verbose=-1)
    model.fit(X_train, y_train)

    print("  Computing permutation importance...")
    result = permutation_importance(model, X_test, y_test,
                                     n_repeats=5, random_state=42, n_jobs=-1)

    feature_names = num_cols + [
        f"{c}={v}" for c, vals in zip(
            cat_cols,
            pre.named_transformers_["cat"]["onehot"].categories_
        ) for v in vals
    ]
    importances = pd.Series(result.importances_mean, index=feature_names)
    agg_imp = {}
    for c in num_cols:
        agg_imp[c] = importances[c] if c in importances.index else 0
    for c in cat_cols:
        mask = [n.startswith(f"{c}=") for n in importances.index]
        agg_imp[c] = importances[mask].sum()

    imp_s = pd.Series(agg_imp).sort_values().tail(15)

    short_names = {
        "primary_metal": "primary_metal",
        "average_atomic_mass": "avg atomic mass",
        "Heat_capacity@300K (J/g/K)": "Cp@300K",
        "Heat_capacity@350K (J/g/K)": "Cp@350K",
        "Heat_capacity@400K (J/g/K)": "Cp@400K",
        "k_cp (J/g/K/K)": "k_cp",
        "cp0 (J/g/K)": "cp0",
        "PLD (Å)": "PLD",
        "LCD (Å)": "LCD",
        "VF": "VF",
        "PV (cm3/g)": "PV",
        "Density (g/cm3)": "Density",
        "natoms": "natoms",
        "n_metals": "n_metals",
        "is_mixed_metal": "is_mixed_metal",
        "structure_dimension": "dim",
        "catenation": "catenation",
        "topology(SingleNodes)": "topo (single)",
        "topology(AllNodes)": "topo (all)",
        "Has OMS": "OMS",
        "OMS Types": "OMS types",
    }
    imp_s.index = [short_names.get(n, n) for n in imp_s.index]


    fig, ax = plt.subplots(figsize=(7, 5))
    y_pos = np.arange(len(imp_s))
    ax.barh(y_pos, imp_s.values, color="#4C72B0",
            edgecolor="black", linewidth=0.4)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(imp_s.index, fontsize=9)
    ax.set_xlabel("Mean permutation importance\n(ASR thermal stability, LightGBM, Regime D)")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "figS8_importance.pdf", format="pdf")
    plt.close()
    print(f"  → {FIG_DIR / 'figS8_importance.pdf'}")


# =============================================================================
# FIG S9: SCREENING SUBSET
# =============================================================================

def make_figS9():
    print("\nBuilding Fig S9 (screening subset)...")
    fig, ax = plt.subplots(figsize=(7, 5))

    cutoffs = [0.6, 0.7, 0.8]
    line_colors = {
        "Water_stability": "#4C72B0",
        "Solvent_stability": "#E07B39",
    }
    line_styles = {"ASR": "-", "FSR": "--"}
    markers = {"Water_stability": "o", "Solvent_stability": "s"}

    for ds in ["ASR", "FSR"]:
        for t in ["Water_stability", "Solvent_stability"]:
            aucs = []
            for cut in cutoffs:
                pt = pred[(pred["dataset"] == ds) &
                          (pred["target"] == t) &
                          (pred["regime"] == "D_context_thermophysical") &
                          (pred["split_type"] == "random") &
                          (pred["model"] == "LightGBM")].copy()
                if len(pt) > 0:
                    y_class = (pt["y_true"] >= cut).astype(int)
                    if y_class.nunique() > 1:
                        aucs.append(roc_auc_score(y_class, pt["y_pred"]))
                    else:
                        aucs.append(0.5)
                else:
                    aucs.append(0.5)
            label = f"{ds} {clean_target_label(t)}"
            ax.plot(cutoffs, aucs, line_styles[ds] + markers[t],
                    linewidth=1.5, markersize=8,
                    color=line_colors[t],
                    label=label)

    ax.set_xticks(cutoffs)
    ax.set_xlabel("Stability cutoff")
    ax.set_ylabel("ROC-AUC (Regime D, LightGBM)")
    ax.set_ylim(0.5, 1.0)
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "figS9_screening_subset.pdf", format="pdf")
    plt.close()
    print(f"  → {FIG_DIR / 'figS9_screening_subset.pdf'}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("REGENERATING ALL SI FIGURES (S1-S9) FOR V14 REVISION")
    print("=" * 70)

    make_figS1()
    make_figS2()
    make_figS3()
    make_figS4()
    make_figS5()
    make_figS6()
    make_figS7()
    make_figS8()
    make_figS9()

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"\nAll SI figures saved in: {FIG_DIR}")
    print("\nUpload all 9 to figures/supporting_information/ in Overleaf,")
    print("replacing the existing files.")