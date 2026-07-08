"""
regenerate_figures.py

Builds Figs 2, 3, 4, 5, 7 from V14 data for the JMCA revision.
No inline titles. Clean margins. 600 dpi PDF. Two-column-friendly aspect ratios.

Inputs (in local folder):
    raw_data/ASR_data_SI_20250204.csv
    raw_data/FSR_data_SI_20250204.csv
    v14_results.csv
    v14_predictions.csv

Outputs (in figures/):
    fig2_descriptor_sufficiency.pdf
    fig3_descriptor_interdependence.pdf
    fig4_chemistry_regime_map.pdf
    fig5_practical_screening.pdf
    fig7_importance_screening.pdf
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Rectangle
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score

# =============================================================================
# CONFIG: PUBLICATION-READY STYLING
# =============================================================================

ROOT = Path(__file__).parent.resolve()
RAW = ROOT / "raw_data"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

# Publication styling
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.titlesize": 11,
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

# Color palette
REGIME_COLORS = {
    "A_metal_only": "#4C72B0",
    "B_metal_oms": "#55A868",
    "C_metal_oms_context": "#E07B39",
    "D_context_thermophysical": "#C44E52",
}
REGIME_LABEL = {
    "A_metal_only": "A",
    "B_metal_oms": "B",
    "C_metal_oms_context": "C",
    "D_context_thermophysical": "D",
}
MODEL_COLORS = {
    "Linear": "#8c8c8c",
    "Ridge": "#4C72B0",
    "LASSO": "#55A868",
    "DecisionTree": "#937860",
    "SVR_Linear": "#937860",
    "RandomForest": "#E07B39",
    "LightGBM": "#C44E52",
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
DATASET_HATCH = {"ASR": "", "FSR": "//"}

# =============================================================================
# LOAD DATA
# =============================================================================

print("Loading V14 data...")
res = pd.read_csv("v14_results.csv")
pred = pd.read_csv("v14_predictions.csv")
asr = pd.read_csv(RAW / "ASR_data_SI_20250204.csv")
fsr = pd.read_csv(RAW / "FSR_data_SI_20250204.csv")
print(f"  results: {res.shape}")
print(f"  predictions: {pred.shape}")

# Summary per (dataset, target, regime, split_type, model)
summary = res.groupby(
    ["dataset", "target", "regime", "split_type", "model"]
).agg(
    Spearman_mean=("Spearman", "mean"),
    Spearman_std=("Spearman", "std"),
    R2_mean=("R2", "mean"),
    RMSE_mean=("RMSE", "mean"),
).reset_index()

# Helper: clean axis labels
def clean_target_label(t):
    return t.replace("_stability", "").replace(" (℃)", "").strip()

def add_panel_label(ax, label, x=-0.15, y=1.05):
    """Add bold panel label (a), (b) etc to top-left of axis."""
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="bottom", ha="left")


# =============================================================================
# FIG 2: DESCRIPTOR SUFFICIENCY (4 panels, 2x2)
# =============================================================================


def make_fig2():
    print("\nBuilding Fig 2 (descriptor sufficiency)...")
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 5.5))

    targets_order = ["Solvent_stability", "Water_stability", "Thermal_stability (℃)"]
    target_labels = ["Solvent", "Water", "Thermal"]
    regimes_order = ["A_metal_only", "B_metal_oms",
                     "C_metal_oms_context", "D_context_thermophysical"]
    regime_labels = ["A", "B", "C", "D"]

    # --- Panel (a): Regime ladder per (dataset, target) under random, best model ---
    ax = axes[0, 0]
    line_colors = {
        ("ASR", "Solvent_stability"): "#4C72B0",
        ("ASR", "Water_stability"): "#E07B39",
        ("ASR", "Thermal_stability (℃)"): "#C44E52",
        ("FSR", "Solvent_stability"): "#4C72B0",
        ("FSR", "Water_stability"): "#E07B39",
        ("FSR", "Thermal_stability (℃)"): "#C44E52",
    }
    line_styles = {"ASR": "-", "FSR": "--"}

    sub_rand = summary[summary["split_type"] == "random"]
    best_per_regime = (sub_rand.sort_values("Spearman_mean", ascending=False)
                       .groupby(["dataset", "target", "regime"]).head(1))

    for ds in ["ASR", "FSR"]:
        for t in targets_order:
            vals = []
            for r in regimes_order:
                row = best_per_regime[
                    (best_per_regime["dataset"] == ds) &
                    (best_per_regime["target"] == t) &
                    (best_per_regime["regime"] == r)
                ]
                vals.append(float(row["Spearman_mean"].iloc[0]) if len(row) else np.nan)

            t_clean = clean_target_label(t)
            ax.plot(range(len(regimes_order)), vals,
                    line_styles[ds] + "o",
                    color=line_colors[(ds, t)],
                    markersize=4, linewidth=1.3,
                    label=f"{ds} {t_clean}")

    ax.set_xticks(range(len(regimes_order)))
    ax.set_xticklabels(regime_labels)
    ax.set_xlabel("Descriptor regime")
    ax.set_ylabel("Best Spearman (random)")
    ax.set_ylim(0, 0.8)
    ax.legend(loc="upper left", fontsize=7, ncol=2, frameon=False)
    ax.grid(axis="y", alpha=0.3)
    add_panel_label(ax, "(a)")

    # --- Panel (b): D-regime split comparison, best model per (dataset, target, split) ---
    ax = axes[0, 1]
    sub_d = summary[summary["regime"] == "D_context_thermophysical"]
    best_split = (sub_d.sort_values("Spearman_mean", ascending=False)
                  .groupby(["dataset", "target", "split_type"]).head(1))

    splits = ["random", "group_metal", "group_topology"]
    n_targets = len(targets_order)
    group_positions = np.arange(n_targets)
    bar_width = 0.13
    bar_idx = 0

    for split_idx, split in enumerate(splits):
        for ds_idx, ds in enumerate(["ASR", "FSR"]):
            offset = (bar_idx - 2.5) * bar_width
            bar_idx += 1
            vals = []
            errs = []
            for t in targets_order:
                row = best_split[
                    (best_split["dataset"] == ds) &
                    (best_split["target"] == t) &
                    (best_split["split_type"] == split)
                ]
                vals.append(float(row["Spearman_mean"].iloc[0]) if len(row) else 0)
                errs.append(float(row["Spearman_std"].iloc[0]) if len(row) and pd.notna(row["Spearman_std"].iloc[0]) else 0)

            color = SPLIT_COLORS[split]
            alpha = 0.6 if ds == "FSR" else 1.0
            label = f"{SPLIT_LABEL[split]} ({ds})"
            ax.bar(group_positions + offset, vals, bar_width,
                   yerr=errs, capsize=1.5,
                   color=color, alpha=alpha,
                   edgecolor="black", linewidth=0.4,
                   label=label)

    ax.set_xticks(group_positions)
    ax.set_xticklabels(target_labels)
    ax.set_ylabel("Best Spearman ± SD (Regime D)")
    ax.set_ylim(-0.1, 0.85)
    ax.axhline(0, color="black", linewidth=0.4)
    ax.legend(loc="upper right", fontsize=6, ncol=2, frameon=False)
    ax.grid(axis="y", alpha=0.3)
    add_panel_label(ax, "(b)")

    # --- Panel (c): Model comparison at D, water stability, random split ---
    ax = axes[1, 0]
    water_d_random = summary[
        (summary["target"] == "Water_stability") &
        (summary["regime"] == "D_context_thermophysical") &
        (summary["split_type"] == "random")
    ]
    models_order = ["Linear", "Ridge", "LASSO", "DecisionTree",
                    "SVR_Linear", "RandomForest", "LightGBM"]
    model_labels = ["Lin", "Ridge", "LASSO", "DT", "SVR", "RF", "LGBM"]
    x = np.arange(len(models_order))
    width = 0.38

    for ds_idx, ds in enumerate(["ASR", "FSR"]):
        vals = []
        errs = []
        for m in models_order:
            row = water_d_random[
                (water_d_random["dataset"] == ds) &
                (water_d_random["model"] == m)
            ]
            vals.append(float(row["Spearman_mean"].iloc[0]) if len(row) else 0)
            errs.append(float(row["Spearman_std"].iloc[0]) if len(row) and pd.notna(row["Spearman_std"].iloc[0]) else 0)
        ax.bar(x + (ds_idx - 0.5) * width, vals, width,
               yerr=errs, capsize=2,
               color=["#4C72B0", "#E07B39"][ds_idx],
               edgecolor="black", linewidth=0.4,
               label=ds)

    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, rotation=0)
    ax.set_ylabel("Spearman (Water, D, random)")
    ax.set_ylim(0, 0.8)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.3)
    add_panel_label(ax, "(c)")

    # --- Panel (d): Random vs grouped Spearman for all 6 cells ---
    ax = axes[1, 1]
    random_vals = []
    metal_vals = []
    topo_vals = []
    labels = []
    for ds in ["ASR", "FSR"]:
        for t in targets_order:
            random_v = best_split[
                (best_split["dataset"] == ds) &
                (best_split["target"] == t) &
                (best_split["split_type"] == "random")
            ]["Spearman_mean"]
            metal_v = best_split[
                (best_split["dataset"] == ds) &
                (best_split["target"] == t) &
                (best_split["split_type"] == "group_metal")
            ]["Spearman_mean"]
            topo_v = best_split[
                (best_split["dataset"] == ds) &
                (best_split["target"] == t) &
                (best_split["split_type"] == "group_topology")
            ]["Spearman_mean"]
            random_vals.append(float(random_v.iloc[0]) if len(random_v) else np.nan)
            metal_vals.append(float(metal_v.iloc[0]) if len(metal_v) else np.nan)
            topo_vals.append(float(topo_v.iloc[0]) if len(topo_v) else np.nan)
            labels.append(f"{ds}\n{clean_target_label(t)[:7]}")

    x = np.arange(len(labels))
    ax.plot(x, random_vals, "o-", color="#4C72B0", linewidth=1.5,
            markersize=6, label="Random")
    ax.plot(x, metal_vals, "s-", color="#E07B39", linewidth=1.5,
            markersize=6, label="Metal-grouped")
    ax.plot(x, topo_vals, "^-", color="#C44E52", linewidth=1.5,
            markersize=6, label="Topology-grouped")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, fontsize=7.5)
    ax.set_ylabel("Best Spearman (Regime D)")
    ax.set_ylim(0, 0.8)
    ax.legend(loc="upper right", fontsize=7.5, frameon=False)
    ax.grid(axis="y", alpha=0.3)
    add_panel_label(ax, "(d)")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig2_descriptor_sufficiency.pdf", format="pdf")
    plt.close()
    print(f"  → {FIG_DIR / 'fig2_descriptor_sufficiency.pdf'}")


def make_fig3():
    print("\nBuilding Fig 3 (descriptor correlations + threshold robustness)...")
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.5))

    # Panel (a): Correlation matrix of continuous descriptors
    ax = axes[0]
    cont_cols = ["Density (g/cm3)", "LCD (Å)", "PLD (Å)", "VF", "PV (cm3/g)",
                 "average_atomic_mass", "natoms"]
    asr_cont = asr[cont_cols].apply(pd.to_numeric, errors="coerce").dropna()
    corr = asr_cont.corr()

    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    # Annotate cells
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            v = corr.values[i, j]
            color = "white" if abs(v) > 0.6 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=6.5, color=color)

    short_labels = ["Density", "LCD", "PLD", "VF", "PV", "<M>", "natoms"]
    ax.set_xticks(range(len(short_labels)))
    ax.set_xticklabels(short_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(short_labels)))
    ax.set_yticklabels(short_labels)
    cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label("Pearson r", fontsize=8)
    add_panel_label(ax, "(a)")

    # Panel (b): Threshold robustness — Spearman across regimes for water stability
    # Show how Spearman changes A→D and confirm robustness
    ax = axes[1]
    water_random = summary[
        (summary["target"] == "Water_stability") &
        (summary["split_type"] == "random")
    ]
    # Get best model per regime per dataset
    best_per_regime = (water_random.sort_values("Spearman_mean", ascending=False)
                      .groupby(["dataset", "regime"]).head(1))

    regimes_order = ["A_metal_only", "B_metal_oms",
                     "C_metal_oms_context", "D_context_thermophysical"]
    x = np.arange(len(regimes_order))

    for ds_idx, ds in enumerate(["ASR", "FSR"]):
        vals = []
        errs = []
        for r in regimes_order:
            row = best_per_regime[(best_per_regime["dataset"] == ds) &
                                  (best_per_regime["regime"] == r)]
            vals.append(row["Spearman_mean"].iloc[0] if len(row) else np.nan)
            errs.append(row["Spearman_std"].iloc[0] if len(row) else 0)
        ax.errorbar(x, vals, yerr=errs,
                    marker=["o", "s"][ds_idx], markersize=6,
                    linewidth=1.5, capsize=3,
                    color=["#4C72B0", "#E07B39"][ds_idx],
                    label=f"{ds} water stability")

    ax.set_xticks(x)
    ax.set_xticklabels(["A\n(metal)", "B\n(+OMS)", "C\n(+structure)", "D\n(+thermal)"])
    ax.set_xlabel("Descriptor regime")
    ax.set_ylabel("Spearman ± SD (random split)")
    ax.set_ylim(0.2, 0.8)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.3)
    add_panel_label(ax, "(b)")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig3_descriptor_interdependence.pdf", format="pdf")
    plt.close()
    print(f"  → {FIG_DIR / 'fig3_descriptor_interdependence.pdf'}")


# =============================================================================
# FIG 4: CHEMISTRY REGIME MAP (6 panels, 2x3)
# =============================================================================

def make_fig4():
    print("\nBuilding Fig 4 (chemistry regime map)...")
    fig, axes = plt.subplots(2, 3, figsize=(8.5, 5.5))

    # Use ASR water D random predictions
    p = pred[
        (pred["dataset"] == "ASR") &
        (pred["target"] == "Water_stability") &
        (pred["regime"] == "D_context_thermophysical") &
        (pred["split_type"] == "random") &
        (pred["model"] == "LightGBM")
    ].copy()
    p["abs_err"] = (p["y_true"] - p["y_pred"]).abs()

    # Panel (a): MAE by top metals
    ax = axes[0, 0]
    metal_mae = p.groupby("primary_metal")["abs_err"].agg(["mean", "count"])
    metal_mae = metal_mae[metal_mae["count"] >= 5].sort_values("mean")
    top15 = metal_mae.tail(15)
    y_pos = np.arange(len(top15))
    ax.barh(y_pos, top15["mean"], color="#4C72B0", edgecolor="black", linewidth=0.4)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top15.index, fontsize=7)
    ax.set_xlabel("Mean abs. error (water, D)")
    ax.invert_yaxis()
    add_panel_label(ax, "(a)")

    # Panel (b): Mixed-metal vs single-metal
    ax = axes[0, 1]
    mixed_data = [
        p[p["is_mixed_metal"] == 0]["abs_err"].values,
        p[p["is_mixed_metal"] == 1]["abs_err"].values,
    ]
    bp = ax.boxplot(mixed_data, labels=["Single", "Mixed"],
                    patch_artist=True, widths=0.5,
                    medianprops={"color": "black", "linewidth": 1.2},
                    flierprops={"marker": ".", "markersize": 3,
                                "markerfacecolor": "gray"})
    for patch, color in zip(bp["boxes"], ["#4C72B0", "#E07B39"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("Absolute error")
    ax.set_xlabel("Metal composition")
    add_panel_label(ax, "(b)")

    # Panel (c): OMS vs no-OMS
    ax = axes[0, 2]
    has_oms_str = p["Has OMS"].astype(str).str.strip().str.lower()
    oms_data = [
        p[has_oms_str == "no"]["abs_err"].values,
        p[has_oms_str == "yes"]["abs_err"].values,
    ]
    bp = ax.boxplot(oms_data, labels=["No OMS", "Has OMS"],
                    patch_artist=True, widths=0.5,
                    medianprops={"color": "black", "linewidth": 1.2},
                    flierprops={"marker": ".", "markersize": 3,
                                "markerfacecolor": "gray"})
    for patch, color in zip(bp["boxes"], ["#55A868", "#C44E52"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("Absolute error")
    add_panel_label(ax, "(c)")

    # Panel (d): Median true water stability by metal × OMS (heatmap)
    ax = axes[1, 0]
    asr_w = asr.copy()
    asr_w["Water_stability"] = pd.to_numeric(asr_w["Water_stability"], errors="coerce")
    asr_w = asr_w.dropna(subset=["Water_stability"])

    if "Metal Types" in asr_w.columns:
        def first_metal(x):
            if pd.isna(x): return np.nan
            parts = sorted([p.strip() for p in str(x).split(",") if p.strip()])
            return parts[0] if parts else np.nan
        asr_w["primary_metal"] = asr_w["Metal Types"].apply(first_metal)

    # Top 10 metals
    top_metals = asr_w["primary_metal"].value_counts().head(10).index.tolist()
    asr_w_top = asr_w[asr_w["primary_metal"].isin(top_metals)]
    oms_str = asr_w_top["Has OMS"].astype(str).str.strip().str.lower()
    asr_w_top = asr_w_top.assign(oms_clean=oms_str)
    asr_w_top = asr_w_top[asr_w_top["oms_clean"].isin(["yes", "no"])]

    pivot = asr_w_top.groupby(["primary_metal", "oms_clean"])["Water_stability"].median().unstack()
    pivot = pivot.reindex(top_metals)

    im = ax.imshow(pivot.values, cmap="RdYlBu", vmin=0, vmax=1, aspect="auto")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=7)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(["No OMS", "Has OMS"], fontsize=8)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=7, color="black")
    cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.03)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label("Median water stab.", fontsize=8)
    add_panel_label(ax, "(d)")

    # Panel (e): MAE by top topologies
    ax = axes[1, 1]
    topo_mae = p.groupby("topology")["abs_err"].agg(["mean", "count"])
    topo_mae = topo_mae[topo_mae["count"] >= 3].sort_values("mean")
    top15_topo = topo_mae.tail(15)
    y_pos = np.arange(len(top15_topo))
    ax.barh(y_pos, top15_topo["mean"], color="#E07B39",
            edgecolor="black", linewidth=0.4)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top15_topo.index, fontsize=7)
    ax.set_xlabel("Mean abs. error (water, D)")
    ax.invert_yaxis()
    add_panel_label(ax, "(e)")

    # Panel (f): Failure-enrichment by metal (worst 5% predictions)
    ax = axes[1, 2]
    threshold = p["abs_err"].quantile(0.95)
    p["is_failure"] = p["abs_err"] >= threshold
    failures = p[p["is_failure"]]
    fail_counts = failures["primary_metal"].value_counts().head(10)
    y_pos = np.arange(len(fail_counts))
    ax.barh(y_pos, fail_counts.values, color="#C44E52",
            edgecolor="black", linewidth=0.4)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(fail_counts.index, fontsize=7)
    ax.set_xlabel("Count among worst 5%")
    ax.invert_yaxis()
    add_panel_label(ax, "(f)")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig4_chemistry_regime_map.pdf", format="pdf")
    plt.close()
    print(f"  → {FIG_DIR / 'fig4_chemistry_regime_map.pdf'}")


# =============================================================================
# FIG 5: PRACTICAL SCREENING (6 panels, 2x3)
# =============================================================================

def make_fig5():
    print("\nBuilding Fig 5 (practical screening)...")
    fig, axes = plt.subplots(2, 3, figsize=(8.5, 5.5))

    # Convert water stability prediction to classification with cutoff 0.7
    p = pred[
        (pred["dataset"] == "ASR") &
        (pred["target"] == "Water_stability") &
        (pred["regime"] == "D_context_thermophysical") &
        (pred["split_type"] == "random") &
        (pred["model"] == "LightGBM")
    ].copy()

    cutoff = 0.7
    p["y_true_class"] = (p["y_true"] >= cutoff).astype(int)
    # Normalize y_pred to [0,1] as a pseudo-probability
    p["y_prob"] = np.clip(p["y_pred"], 0, 1)

    # Panel (a): Calibration curve
    ax = axes[0, 0]
    if p["y_true_class"].nunique() > 1:
        frac_pos, mean_pred = calibration_curve(p["y_true_class"], p["y_prob"],
                                                  n_bins=8, strategy="uniform")
        ax.plot(mean_pred, frac_pos, "o-", color="#4C72B0",
                linewidth=1.5, markersize=6, label="Model")
        ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="Perfect")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Fraction stable (observed)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    add_panel_label(ax, "(a)")

    # Panel (b): Precision at coverage
    ax = axes[0, 1]
    sorted_p = p.sort_values("y_prob", ascending=False)
    coverages = np.linspace(0.05, 1.0, 20)
    precisions = []
    for cov in coverages:
        k = max(1, int(cov * len(sorted_p)))
        precisions.append(sorted_p.head(k)["y_true_class"].mean())
    ax.plot(coverages, precisions, "-o", color="#E07B39",
            linewidth=1.5, markersize=4)
    ax.axhline(p["y_true_class"].mean(), linestyle="--", color="gray",
               linewidth=1, label=f"Baseline ({p['y_true_class'].mean():.2f})")
    ax.set_xlabel("Coverage (top fraction retained)")
    ax.set_ylabel("Precision at coverage")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower left", fontsize=8, frameon=False)
    add_panel_label(ax, "(b)")

    # Panel (c): Top-decile enrichment for water and solvent
    ax = axes[0, 2]
    enrichments = []
    targets_for_enr = ["Water_stability", "Solvent_stability"]
    for t in targets_for_enr:
        pt = pred[
            (pred["dataset"] == "ASR") &
            (pred["target"] == t) &
            (pred["regime"] == "D_context_thermophysical") &
            (pred["split_type"] == "random") &
            (pred["model"] == "LightGBM")
        ].copy()
        pt["y_true_class"] = (pt["y_true"] >= 0.7).astype(int)
        pt["y_prob"] = np.clip(pt["y_pred"], 0, 1)
        baseline = pt["y_true_class"].mean()
        sorted_pt = pt.sort_values("y_prob", ascending=False)
        top10 = sorted_pt.head(max(1, int(0.1 * len(sorted_pt))))
        enr = top10["y_true_class"].mean() / baseline if baseline > 0 else 0
        enrichments.append(enr)

    x = np.arange(len(targets_for_enr))
    ax.bar(x, enrichments, width=0.5,
           color=["#4C72B0", "#E07B39"],
           edgecolor="black", linewidth=0.4)
    ax.axhline(1, color="gray", linestyle="--", linewidth=1, label="No enrichment")
    ax.set_xticks(x)
    ax.set_xticklabels(["Water", "Solvent"])
    ax.set_ylabel("Top-decile enrichment")
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    add_panel_label(ax, "(c)")

    # Panel (d): ROC-AUC by cutoff for water and solvent
    ax = axes[1, 0]
    cutoffs = [0.6, 0.7, 0.8]
    for t_idx, t in enumerate(["Water_stability", "Solvent_stability"]):
        aucs = []
        for cut in cutoffs:
            pt = pred[
                (pred["dataset"] == "ASR") &
                (pred["target"] == t) &
                (pred["regime"] == "D_context_thermophysical") &
                (pred["split_type"] == "random") &
                (pred["model"] == "LightGBM")
            ].copy()
            y_class = (pt["y_true"] >= cut).astype(int)
            if y_class.nunique() > 1:
                auc = roc_auc_score(y_class, pt["y_pred"])
                aucs.append(auc)
            else:
                aucs.append(0.5)
        ax.plot(cutoffs, aucs, "-o", linewidth=1.5, markersize=5,
                color=["#4C72B0", "#E07B39"][t_idx],
                label=t.replace("_stability", ""))
    ax.set_xlabel("Stability cutoff")
    ax.set_ylabel("ROC-AUC (ASR Regime D)")
    ax.set_xticks(cutoffs)
    ax.set_ylim(0.5, 1.0)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.3)
    add_panel_label(ax, "(d)")

    # Panel (e): Workflow text box
    ax = axes[1, 1]
    ax.axis("off")
    workflow = (
        "Screening workflow\n\n"
        "1. Check descriptor completeness\n"
        "2. Verify in-domain chemistry\n"
        "    (metal & topology in training)\n"
        "3. Rank by predicted water stability\n"
        "4. Filter by solvent stability\n"
        "    (≥ 0.6 for humid CO$_2$)\n"
        "5. Flag confidence tier:\n"
        "    high / moderate / low\n"
        "6. Apply application-specific filter:\n"
        "    water harvesting (≥ 0.70)\n"
        "    humid CO$_2$ (water ≥ 0.70, solv. ≥ 0.60)\n"
        "    humid sep. (water ≥ 0.80, solv. ≥ 0.70)"
    )
    ax.text(0.05, 0.95, workflow, transform=ax.transAxes,
            fontsize=7.5, va="top", ha="left", family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f5f5f5",
                      edgecolor="black", linewidth=0.5))
    add_panel_label(ax, "(e)")

    # Panel (f): Predicted probability distribution
    ax = axes[1, 2]
    ax.hist(p["y_prob"].values, bins=30, color="#4C72B0",
            edgecolor="black", linewidth=0.4, alpha=0.7)
    ax.axvline(0.7, color="red", linestyle="--", linewidth=1.2,
               label="Cutoff 0.7")
    ax.set_xlabel("Predicted water stability")
    ax.set_ylabel("Count")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    add_panel_label(ax, "(f)")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig5_practical_screening.pdf", format="pdf")
    plt.close()
    print(f"  → {FIG_DIR / 'fig5_practical_screening.pdf'}")


# =============================================================================
# FIG 7: PERMUTATION IMPORTANCE + SCREENING SUBSET (2 panels, 1x2)
# =============================================================================

def make_fig7():
    print("\nBuilding Fig 7 (permutation importance + screening subset)...")
    from sklearn.inspection import permutation_importance
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.model_selection import train_test_split
    import lightgbm as lgb

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.5))

    # Panel (a): Permutation importance for ASR thermal stability D LightGBM
    ax = axes[0]

    # Train a LightGBM model on ASR thermal stability D and compute permutation importance
    print("  Training LightGBM for permutation importance...")
    asr_t = asr.copy()
    asr_t["Thermal_stability (℃)"] = pd.to_numeric(asr_t["Thermal_stability (℃)"],
                                                    errors="coerce")
    if "Metal Types" in asr_t.columns:
        def first_metal(x):
            if pd.isna(x): return np.nan
            parts = sorted([p.strip() for p in str(x).split(",") if p.strip()])
            return parts[0] if parts else np.nan
        asr_t["primary_metal"] = asr_t["Metal Types"].apply(first_metal)
        asr_t["n_metals"] = asr_t["Metal Types"].apply(
            lambda x: len([p for p in str(x).split(",") if p.strip()]) if pd.notna(x) else 0)
        asr_t["is_mixed_metal"] = (asr_t["n_metals"] > 1).astype(int)

    asr_t = asr_t.dropna(subset=["Thermal_stability (℃)"]).copy()

    feat_cols = ["primary_metal", "is_mixed_metal", "n_metals", "Has OMS", "OMS Types",
                 "structure_dimension", "topology(SingleNodes)", "topology(AllNodes)",
                 "catenation",
                 "Density (g/cm3)", "LCD (Å)", "PLD (Å)", "VF", "PV (cm3/g)",
                 "average_atomic_mass",
                 "Heat_capacity@300K (J/g/K)", "Heat_capacity@350K (J/g/K)",
                 "Heat_capacity@400K (J/g/K)",
                 "k_cp (J/g/K/K)", "cp0 (J/g/K)", "natoms"]
    feat_cols = [c for c in feat_cols if c in asr_t.columns]

    # Convert numeric columns
    for c in feat_cols:
        if c not in ["primary_metal", "Has OMS", "OMS Types",
                      "topology(SingleNodes)", "topology(AllNodes)"]:
            asr_t[c] = pd.to_numeric(asr_t[c], errors="coerce")

    X = asr_t[feat_cols]
    y = asr_t["Thermal_stability (℃)"].values

    # Simple preprocessing
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

    print("  Computing permutation importance (may take ~30s)...")
    result = permutation_importance(model, X_test, y_test,
                                     n_repeats=5, random_state=42, n_jobs=-1)

    # Map back to feature names
    feature_names = num_cols + [
        f"{c}={v}" for c, vals in zip(
            cat_cols,
            pre.named_transformers_["cat"]["onehot"].categories_
        ) for v in vals
    ]

    # Aggregate categorical importances back to parent
    importances = pd.Series(result.importances_mean, index=feature_names)
    agg_imp = {}
    for c in num_cols:
        agg_imp[c] = importances[c] if c in importances.index else 0
    for c in cat_cols:
        mask = [n.startswith(f"{c}=") for n in importances.index]
        agg_imp[c] = importances[mask].sum()

    imp_s = pd.Series(agg_imp).sort_values().tail(10)

    # Shorten names
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
        "is_mixed_metal": "is_mixed",
        "structure_dimension": "dim",
        "catenation": "catenation",
        "topology(SingleNodes)": "topo (single)",
        "topology(AllNodes)": "topo (all)",
        "Has OMS": "OMS",
        "OMS Types": "OMS types",
    }
    imp_s.index = [short_names.get(n, n) for n in imp_s.index]

    y_pos = np.arange(len(imp_s))
    ax.barh(y_pos, imp_s.values, color="#4C72B0",
            edgecolor="black", linewidth=0.4)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(imp_s.index, fontsize=8)
    ax.set_xlabel("Mean permutation importance\n(ASR thermal, LightGBM, D)")
    ax.grid(axis="x", alpha=0.3)
    add_panel_label(ax, "(a)")

    # Panel (b): Screening subset analysis — ROC-AUC at each cutoff for water vs solvent
    ax = axes[1]
    cutoffs = [0.6, 0.7, 0.8]
    for ds_idx, ds in enumerate(["ASR", "FSR"]):
        for t_idx, t in enumerate(["Water_stability", "Solvent_stability"]):
            aucs = []
            for cut in cutoffs:
                pt = pred[
                    (pred["dataset"] == ds) &
                    (pred["target"] == t) &
                    (pred["regime"] == "D_context_thermophysical") &
                    (pred["split_type"] == "random") &
                    (pred["model"] == "LightGBM")
                ].copy()
                y_class = (pt["y_true"] >= cut).astype(int)
                if y_class.nunique() > 1:
                    auc = roc_auc_score(y_class, pt["y_pred"])
                    aucs.append(auc)
                else:
                    aucs.append(0.5)
            linestyle = "-" if ds == "ASR" else "--"
            marker = "o" if t == "Water_stability" else "s"
            color = "#4C72B0" if t == "Water_stability" else "#E07B39"
            ax.plot(cutoffs, aucs, linestyle, marker=marker,
                    linewidth=1.5, markersize=6, color=color,
                    label=f"{ds} {clean_target_label(t)}")

    ax.set_xlabel("Stability cutoff")
    ax.set_ylabel("ROC-AUC")
    ax.set_xticks(cutoffs)
    ax.set_ylim(0.5, 1.0)
    ax.legend(loc="lower right", fontsize=7, frameon=False)
    ax.grid(axis="y", alpha=0.3)
    add_panel_label(ax, "(b)")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig7_importance_screening.pdf", format="pdf")
    plt.close()
    print(f"  → {FIG_DIR / 'fig7_importance_screening.pdf'}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("REGENERATING FIGURES FOR V14 REVISION")
    print("=" * 70)

    make_fig2()
    make_fig3()
    make_fig4()
    make_fig5()
    make_fig7()

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"\nFigures saved in: {FIG_DIR}")
    print("\nUpload these to figures/main_text/ in Overleaf:")
    print("  fig2_descriptor_sufficiency.pdf")
    print("  fig3_descriptor_interdependence.pdf")
    print("  fig4_chemistry_regime_map.pdf")
    print("  fig5_practical_screening.pdf")
    print("  fig7_importance_screening.pdf")
    print("\nKeep these existing files (no V14 update needed):")
    print("  fig1_dataset_architecture.pdf")
    print("  fig6_case_studies.png")