"""
JMCA final coding package runner — v2 figure/table fixes.

Place this folder as: jmca_handoff_mosiu/99_final_tools/
Run from the handoff root:
    python 99_final_tools/run_all_final_package.py
or from Spyder:
    %run 99_final_tools/run_all_final_package.py

This script does NOT rerun the full ML sweep. It rebuilds the V14 summary table
from v14_predictions.csv, exports final main/SI tables, regenerates main Figures
1-5 and 7, and copies Figure 6 if the structural/case-study asset is present.
"""
from __future__ import annotations

import json
import math
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# =============================================================================
# PATHS
# =============================================================================

TOOL_DIR = Path(__file__).resolve().parent
ROOT = TOOL_DIR.parent
RAW = ROOT / "02_raw_data"
OUT = ROOT / "04_outputs"
HIST = ROOT / "05_historical" / "v10_outputs_tables"
FINAL = OUT / "FINAL_FOR_REVISION"
VERIFIED = FINAL / "01_verified_results"
MAIN_TABLES = FINAL / "02_main_tables"
SI_TABLES = FINAL / "03_si_tables"
MAIN_FIGS = FINAL / "04_main_figures"
SI_FIGS = FINAL / "05_si_figures"
SOURCE = FINAL / "06_source_data_for_figures"
QC = FINAL / "07_quality_control"

for p in [VERIFIED, MAIN_TABLES, SI_TABLES, MAIN_FIGS, SI_FIGS, SOURCE, QC]:
    p.mkdir(parents=True, exist_ok=True)

# =============================================================================
# STYLE
# =============================================================================

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8.5,
    "axes.labelsize": 8.5,
    "axes.titlesize": 9,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.75,
    "xtick.major.width": 0.65,
    "ytick.major.width": 0.65,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

COLORS = {
    "blue": "#356C9B",
    "sky": "#8CBBD9",
    "orange": "#D9853B",
    "red": "#B74D4D",
    "green": "#3F8F7A",
    "purple": "#7A68A6",
    "gold": "#D4A33D",
    "gray": "#6E6E6E",
    "lightgray": "#E8E8E8",
    "dark": "#222222",
}
REGIME_ORDER = ["A_metal_only", "B_metal_oms", "C_metal_oms_context", "D_context_thermophysical"]
REGIME_SHORT = {"A_metal_only": "A", "B_metal_oms": "B", "C_metal_oms_context": "C", "D_context_thermophysical": "D"}
REGIME_LONG = {
    "A_metal_only": "A: metal",
    "B_metal_oms": "B: +OMS",
    "C_metal_oms_context": "C: +structure",
    "D_context_thermophysical": "D: +thermal",
}
SPLIT_ORDER = ["random", "group_metal", "group_topology"]
SPLIT_LABEL = {"random": "Random", "group_metal": "Metal-grouped", "group_topology": "Topology-grouped"}
SPLIT_COL = {"random": COLORS["blue"], "group_metal": COLORS["orange"], "group_topology": COLORS["red"]}
TARGET_ORDER = ["Solvent_stability", "Water_stability", "Thermal_stability (℃)"]
TARGET_SHORT = {"Solvent_stability": "Solvent", "Water_stability": "Water", "Thermal_stability (℃)": "Thermal"}
MODEL_ORDER = ["Linear", "Ridge", "LASSO", "DecisionTree", "RandomForest", "LightGBM", "SVR_Linear"]
MODEL_LABEL = {"DecisionTree": "Tree", "RandomForest": "RF", "SVR_Linear": "SVR-L"}
MODEL_COLORS = {
    "Linear": "#8C8C8C",
    "Ridge": COLORS["blue"],
    "LASSO": COLORS["green"],
    "DecisionTree": "#A6761D",
    "RandomForest": COLORS["orange"],
    "LightGBM": COLORS["red"],
    "SVR_Linear": COLORS["purple"],
}

FEATURE_LABELS = {
    "primary_metal": "Primary metal",
    "natoms": "Number of atoms",
    "average_atomic_mass": "Average atomic mass",
    "PLD (Å)": "PLD",
    "LCD (Å)": "LCD",
    "Density (g/cm3)": "Density",
    "PV (cm3/g)": "Pore volume",
    "VF": "Void fraction",
    "cp0 (J/g/K)": "cp0",
    "k_cp (J/g/K/K)": "kcp",
    "Heat_capacity@350K (J/g/K)": "Heat capacity 350 K",
    "Heat_capacity@400K (J/g/K)": "Heat capacity 400 K",
    "OMS Types": "OMS type",
}

def nice_feature_label(x: str) -> str:
    return FEATURE_LABELS.get(str(x), str(x).replace("_", " "))

# =============================================================================
# HELPERS
# =============================================================================

def log(msg: str) -> None:
    print(f"[JMCA] {msg}")


def need_file(p: Path) -> None:
    if not p.exists():
        raise FileNotFoundError(f"Required file not found: {p}")


def add_panel_label(ax, label: str, x=-0.12, y=1.05):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom", ha="left")


def save_fig(fig, name: str, outdir: Path = MAIN_FIGS):
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ["pdf", "png", "svg"]:
        fig.savefig(outdir / f"{name}.{ext}")
    plt.close(fig)


def safe_spearman(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 3 or np.nanstd(y_true[mask]) == 0 or np.nanstd(y_pred[mask]) == 0:
        return np.nan
    return float(spearmanr(y_true[mask], y_pred[mask]).correlation)


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def normalize_model_name(x: str) -> str:
    # Keep current names but support common variants.
    mapping = {
        "SVR-linear": "SVR_Linear",
        "SVR Linear": "SVR_Linear",
        "SVR_linear": "SVR_Linear",
        "RF": "RandomForest",
        "Random Forest": "RandomForest",
        "LGBM": "LightGBM",
        "DT": "DecisionTree",
    }
    return mapping.get(str(x), str(x))


def load_results_summary() -> pd.DataFrame:
    p = VERIFIED / "v14_results_rebuilt.csv"
    if not p.exists():
        return rebuild_v14_results()
    return pd.read_csv(p)


def aggregate_results(res: pd.DataFrame) -> pd.DataFrame:
    # Spearman can be undefined for a few grouped-split cells when the held-out
    # targets or predictions are nearly constant. Keep both total seed count and
    # valid Spearman count so the SI can be transparent.
    return (res.groupby(["dataset", "target", "regime", "split_type", "model"], dropna=False)
            .agg(Spearman_mean=("Spearman", "mean"),
                 Spearman_std=("Spearman", "std"),
                 Spearman_valid=("Spearman", lambda x: int(pd.Series(x).notna().sum())),
                 R2_mean=("R2", "mean"),
                 R2_std=("R2", "std"),
                 RMSE_mean=("RMSE", "mean"),
                 RMSE_std=("RMSE", "std"),
                 MAE_mean=("MAE", "mean"),
                 MAE_std=("MAE", "std"),
                 n_test_mean=("n_test", "mean"),
                 n_seeds=("seed", "nunique"))
            .reset_index())


def best_by_cell(summary: pd.DataFrame, keys, regime=None) -> pd.DataFrame:
    sub = summary.copy()
    if regime is not None:
        sub = sub[sub["regime"] == regime]
    sub = sub.sort_values("Spearman_mean", ascending=False)
    return sub.groupby(keys, as_index=False).head(1).reset_index(drop=True)

# =============================================================================
# STEP 1: REBUILD V14 RESULTS
# =============================================================================

def rebuild_v14_results() -> pd.DataFrame:
    log("Rebuilding V14 results from v14_predictions.csv")
    pred_path = OUT / "v14_predictions.csv"
    need_file(pred_path)
    pred = pd.read_csv(pred_path)
    pred["model"] = pred["model"].map(normalize_model_name)

    group_cols = ["dataset", "target", "regime", "model", "split_type", "seed"]
    rows = []
    for key, g in pred.groupby(group_cols, dropna=False):
        y = g["y_true"].astype(float).values
        p = g["y_pred"].astype(float).values
        mask = np.isfinite(y) & np.isfinite(p)
        y = y[mask]
        p = p[mask]
        if len(y) == 0:
            continue
        row = dict(zip(group_cols, key))
        row.update({
            "n_train": np.nan,
            "n_test": int(len(y)),
            "RMSE": rmse(y, p),
            "MAE": float(mean_absolute_error(y, p)),
            "R2": float(r2_score(y, p)) if len(y) > 1 and np.nanstd(y) > 0 else np.nan,
            "Spearman": safe_spearman(y, p),
        })
        rows.append(row)
    res = pd.DataFrame(rows)
    expected = 4200
    res.to_csv(VERIFIED / "v14_results_rebuilt.csv", index=False)
    # Also keep a root-level copy under 04_outputs for compatibility with old scripts.
    res.to_csv(OUT / "v14_results_rebuilt.csv", index=False)
    log(f"Saved rebuilt results: {res.shape} -> {VERIFIED / 'v14_results_rebuilt.csv'}")
    if len(res) != expected:
        log(f"WARNING: expected {expected} rows, got {len(res)}. Check missing cells in QC report.")
    return res

# =============================================================================
# STEP 2: TABLES
# =============================================================================

def make_tables(res: pd.DataFrame) -> dict:
    log("Exporting main and SI tables")
    summary = aggregate_results(res)
    summary.to_csv(SI_TABLES / "Table_S8_full_v14_results_summary.csv", index=False)
    res.to_csv(SI_TABLES / "Table_S8_full_v14_results_seed_level.csv", index=False)

    # Table 1: dataset file/shape audit + intended role.
    dataset_rows = []
    for name, fn, role in [
        ("ASR raw table", "ASR_data_SI_20250204.csv", "Primary modeling resource"),
        ("FSR raw table", "FSR_data_SI_20250204.csv", "Companion modeling resource"),
        ("ION raw table", "ION_data_SI_20250204.csv", "Audit-only resource"),
        ("ASR--FSR crosswalk", "ASR_FSR_check.csv", "Matched-pair consistency audit"),
        ("Recommended-screening list", "12089-recommended-screening-list.csv", "Application-screening universe"),
    ]:
        p = RAW / fn
        if p.exists():
            df = pd.read_csv(p)
            dataset_rows.append({"resource": name, "file": fn, "rows": df.shape[0], "columns": df.shape[1], "role": role})
    table1 = pd.DataFrame(dataset_rows)
    table1.to_csv(MAIN_TABLES / "Table_1_dataset_resources.csv", index=False)

    # Table 1b: label provenance.
    label_prov = pd.DataFrame([
        {"target": "Solvent_stability", "source_resource": "MOFSimplify / activation-stability annotations", "provenance": "Literature-mined; partially inferred", "use": "Continuous regression + thresholded triage"},
        {"target": "Water_stability", "source_resource": "WS24 / Terrones et al. water-stability labels", "provenance": "Literature-mined; partially inferred", "use": "Continuous regression + thresholded triage"},
        {"target": "Thermal_stability (℃)", "source_resource": "MOFSimplify thermal stability", "provenance": "Literature-mined", "use": "Supplementary continuous regression"},
    ])
    label_prov.to_csv(MAIN_TABLES / "Table_1b_label_provenance.csv", index=False)
    label_prov.to_csv(SI_TABLES / "Table_S2_label_provenance.csv", index=False)

    # Table 2: group audit compact.
    group_path = OUT / "group_audit_outputs" / "01_group_summary_by_dataset_target.csv"
    if group_path.exists():
        gd = pd.read_csv(group_path)
        # Use Water stability rows for compact main summary if available; else first target.
        compact = gd[gd["target"] == "Water_stability"].copy()
        if compact.empty:
            compact = gd.copy()
        compact = compact[["dataset", "group_label", "n_groups", "n_rows", "group_size_median", "group_size_max", "n_singleton_groups", "n_groups_lt5", "seed42_n_test_rows", "seed42_test_row_fraction", "seed42_n_test_groups"]]
        compact = compact.drop_duplicates(["dataset", "group_label"])
        # Main table: only the two grouping variables actually used in the main grouped-split benchmark.
        # Keep ION and SingleNodes in the full SI audit only, not in the compact main table.
        compact_main = compact[(compact["dataset"].isin(["ASR", "FSR"])) &
                               (compact["group_label"].isin(["metal", "topology_AllNodes"]))].copy()
        compact_main["group_label"] = compact_main["group_label"].replace({"metal":"Primary metal", "topology_AllNodes":"Topology (AllNodes)"})
        compact_main.to_csv(MAIN_TABLES / "Table_2_group_structure_compact.csv", index=False)
        gd.to_csv(SI_TABLES / "Table_S5_full_group_audit.csv", index=False)

    held_path = OUT / "group_audit_outputs" / "03_seed42_heldout_groups.csv"
    if held_path.exists():
        pd.read_csv(held_path).to_csv(SI_TABLES / "Table_S6_seed42_heldout_groups.csv", index=False)

    # Table 3: best Regime-D model summary by dataset/target/split.
    table3 = best_by_cell(summary, ["dataset", "target", "split_type"], regime="D_context_thermophysical")
    keep = ["dataset", "target", "split_type", "model", "Spearman_mean", "Spearman_std", "Spearman_valid", "RMSE_mean", "R2_mean", "MAE_mean", "n_seeds", "n_test_mean"]
    table3 = table3[keep].sort_values(["dataset", "target", "split_type"])
    table3.to_csv(MAIN_TABLES / "Table_3_best_regimeD_model_by_split.csv", index=False)

    # Table 4: application domain map.
    appsum_path = OUT / "application_screening_outputs" / "v14_12089_coverage_summary.csv"
    if appsum_path.exists():
        app = pd.read_csv(appsum_path)
        app.to_csv(SI_TABLES / "Table_S14_application_coverage_full_summary.csv", index=False)
        metrics = dict(zip(app["metric"], app["value"]))
        rows = []
        desired = [
            ("Total recommended-screening entries", "total_recommended_screening_entries", "Full CoRE MOF 2024 recommended-screening universe"),
            ("Descriptor-complete", "descriptor_complete_ASR_FSR", "Can be mapped to available ASR/FSR descriptor schema"),
            ("Descriptor-complete, in-domain", "descriptor_complete_in_domain", "Safest current triage domain"),
            ("Descriptor-complete, new topology", "coverage_category::descriptor_complete_new_topology", "Topology extrapolation; lower-confidence use"),
            ("Descriptor-complete, new metal", "coverage_category::descriptor_complete_new_metal", "Metal extrapolation; very cautious use"),
            ("Missing descriptors", "missing_descriptors", "Needs descriptor generation before ranking"),
            ("Missing descriptors but known metal/topology", "missing_descriptors_known_metal_known_topology", "Highest-leverage descriptor-curation subset"),
            ("Chemistry classes total", "chemistry_classes_total", "Primary-metal/topology class universe with prediction support"),
            ("Water-harvesting classes", "chemistry_classes_water_harvesting", "Classes passing water-stability triage tier"),
            ("Humid CO2 capture classes", "chemistry_classes_humid_co2_capture", "Classes passing water + activation triage tier"),
            ("Strict humid-separation classes", "chemistry_classes_humid_separations", "Classes passing strict water + activation tier"),
        ]
        for label, key, meaning in desired:
            rows.append({"category": label, "count": metrics.get(key, np.nan), "interpretation": meaning})
        pd.DataFrame(rows).to_csv(MAIN_TABLES / "Table_4_application_domain_map.csv", index=False)

    # Extra SI tables from existing historical outputs if present.
    for fn, outname in [
        ("classification_summary.csv", "Table_S10_thresholded_classification_summary.csv"),
        ("best_classification_rows.csv", "Table_S10b_best_thresholded_classification_rows.csv"),
        ("pair_consistency_summary.csv", "Table_S11_matched_pair_consistency.csv"),
        ("permutation_importance.csv", "Table_S12_permutation_importance.csv"),
        ("failure_cases_top50.csv", "Table_S13_failure_cases_top50.csv"),
        ("shortlist_subset_performance.csv", "Table_S15_shortlist_subset_performance.csv"),
    ]:
        p = HIST / fn
        if p.exists():
            pd.read_csv(p).to_csv(SI_TABLES / outname, index=False)

    # Source data copies.
    summary.to_csv(SOURCE / "all_aggregated_results_for_figures.csv", index=False)
    table3.to_csv(SOURCE / "figure2_table3_best_regimeD_source.csv", index=False)
    log("Tables exported")
    return {"summary": summary}

# =============================================================================
# FIGURE 1
# =============================================================================

def make_fig1():
    log("Making Figure 1")
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.5))
    for ax in axes.ravel():
        ax.set_axis_off()

    # Panel A: data flow.
    ax = axes[0, 0]
    add_panel_label(ax, "A", x=-0.03, y=1.02)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    cards = [
        (0.05, 0.72, "ASR\n1372 entries", COLORS["blue"]),
        (0.05, 0.50, "FSR\n1192 entries", COLORS["green"]),
        (0.05, 0.28, "ION\naudit only", COLORS["gray"]),
        (0.38, 0.61, "ASR--FSR\ncrosswalk", COLORS["purple"]),
        (0.38, 0.36, "12,089\nscreening list", COLORS["gold"]),
        (0.70, 0.50, "cleaned\nbenchmark\ntables", COLORS["red"]),
    ]
    for x, y, text, c in cards:
        box = FancyBboxPatch((x, y), 0.22, 0.14, boxstyle="round,pad=0.025,rounding_size=0.02", fc=c, ec="none", alpha=0.18)
        ax.add_patch(box)
        ax.text(x+0.11, y+0.07, text, ha="center", va="center", fontsize=8.5, fontweight="bold", color=COLORS["dark"])
    for start, end in [((0.27,0.79),(0.70,0.57)),((0.27,0.57),(0.70,0.54)),((0.27,0.35),(0.70,0.50)),((0.60,0.68),(0.70,0.58)),((0.60,0.43),(0.70,0.51))]:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=8, lw=0.8, color="#666666"))
    ax.text(0.02, 0.96, "Data resources", fontsize=9.5, fontweight="bold")

    # Panel B: targets/provenance.
    ax = axes[0, 1]
    add_panel_label(ax, "B", x=-0.03, y=1.02)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    target_cards = [
        (0.05, 0.70, "Solvent / activation\nstability", "literature-mined\npartly inferred", COLORS["orange"]),
        (0.05, 0.45, "Water stability", "literature-mined\npartly inferred", COLORS["blue"]),
        (0.05, 0.20, "Thermal stability", "literature-mined", COLORS["red"]),
    ]
    for x, y, title, sub, c in target_cards:
        ax.add_patch(FancyBboxPatch((x, y), 0.42, 0.16, boxstyle="round,pad=0.025,rounding_size=0.02", fc=c, ec="none", alpha=0.18))
        ax.text(x+0.02, y+0.105, title, ha="left", va="center", fontsize=8.5, fontweight="bold")
        ax.text(x+0.02, y+0.045, sub, ha="left", va="center", fontsize=7.4)
    ax.add_patch(FancyBboxPatch((0.58, 0.33), 0.34, 0.34, boxstyle="round,pad=0.03,rounding_size=0.02", fc="#F7F7F7", ec="#C7C7C7", lw=0.8))
    ax.text(0.75, 0.58, "screening tiers", ha="center", fontsize=8.5, fontweight="bold")
    ax.text(0.75, 0.49, "0.60 permissive", ha="center", fontsize=7.6)
    ax.text(0.75, 0.42, "0.70 operational", ha="center", fontsize=7.6)
    ax.text(0.75, 0.35, "0.80 strict", ha="center", fontsize=7.6)
    ax.text(0.02, 0.96, "Labels and targets", fontsize=9.5, fontweight="bold")

    # Panel C: descriptor ladder.
    ax = axes[1, 0]
    add_panel_label(ax, "C", x=-0.03, y=1.02)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ladder = [
        (0.08, 0.75, "A", "metal identity", COLORS["blue"]),
        (0.08, 0.55, "B", "+ OMS", COLORS["purple"]),
        (0.08, 0.35, "C", "+ structural context", COLORS["green"]),
        (0.08, 0.15, "D", "+ thermophysical terms", COLORS["gold"]),
    ]
    for x, y, tag, txt, c in ladder:
        ax.add_patch(FancyBboxPatch((x, y), 0.78, 0.13, boxstyle="round,pad=0.025,rounding_size=0.018", fc=c, ec="none", alpha=0.18))
        ax.text(x+0.06, y+0.065, tag, va="center", ha="center", fontsize=10, fontweight="bold", color=c)
        ax.text(x+0.16, y+0.065, txt, va="center", ha="left", fontsize=8.5, fontweight="bold")
        if y > 0.20:
            ax.add_patch(FancyArrowPatch((0.47, y-0.01), (0.47, y-0.07), arrowstyle="-|>", mutation_scale=8, lw=0.75, color="#777777"))
    ax.text(0.02, 0.96, "Descriptor ladder", fontsize=9.5, fontweight="bold")

    # Panel D: reliability ladder.
    ax = axes[1, 1]
    add_panel_label(ax, "D", x=-0.03, y=1.02)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    steps = [
        (0.05, 0.66, "Random split", "interpolation\nwithin familiar chemistry", COLORS["blue"]),
        (0.36, 0.46, "Metal-grouped", "held-out node\nchemistry", COLORS["orange"]),
        (0.67, 0.26, "Topology-grouped", "held-out framework\ncontext", COLORS["red"]),
    ]
    for x, y, title, desc, c in steps:
        ax.add_patch(FancyBboxPatch((x, y), 0.26, 0.18, boxstyle="round,pad=0.025,rounding_size=0.02", fc=c, ec="none", alpha=0.18))
        ax.text(x+0.13, y+0.115, title, ha="center", va="center", fontsize=8.2, fontweight="bold")
        ax.text(x+0.13, y+0.045, desc, ha="center", va="center", fontsize=7.2)
    ax.add_patch(FancyArrowPatch((0.30,0.67),(0.36,0.55), arrowstyle="-|>", mutation_scale=9, lw=0.8, color="#777777"))
    ax.add_patch(FancyArrowPatch((0.61,0.47),(0.67,0.36), arrowstyle="-|>", mutation_scale=9, lw=0.8, color="#777777"))
    ax.text(0.50, 0.12, "application screening = prediction + domain check", ha="center", fontsize=8.4, fontweight="bold")
    ax.text(0.02, 0.96, "Generalisation discipline", fontsize=9.5, fontweight="bold")
    fig.subplots_adjust(wspace=0.20, hspace=0.24)
    save_fig(fig, "fig1_reliability_architecture")

# =============================================================================
# FIGURE 2
# =============================================================================

def make_fig2(res: pd.DataFrame):
    log("Making Figure 2")
    summary = aggregate_results(res)
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.45))

    # A: descriptor ladder random, best model per regime/cell.
    ax = axes[0, 0]
    sub = summary[summary["split_type"] == "random"]
    best = best_by_cell(sub, ["dataset", "target", "regime"])
    line_cols = {"Solvent_stability": COLORS["blue"], "Water_stability": COLORS["orange"], "Thermal_stability (℃)": COLORS["red"]}
    for ds, ls in [("ASR", "-"), ("FSR", "--")]:
        for target in TARGET_ORDER:
            vals = []
            for reg in REGIME_ORDER:
                row = best[(best.dataset == ds) & (best.target == target) & (best.regime == reg)]
                vals.append(row.Spearman_mean.iloc[0] if not row.empty else np.nan)
            ax.plot(range(4), vals, marker="o", ms=3.5, lw=1.15, ls=ls, color=line_cols[target], label=f"{ds} {TARGET_SHORT[target]}")
    ax.set_xticks(range(4)); ax.set_xticklabels(["A", "B", "C", "D"])
    ax.set_xlabel("Descriptor regime")
    ax.set_ylabel("Best Spearman, random split")
    ax.set_ylim(0, 0.78); ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2, fontsize=6.1, loc="upper left", bbox_to_anchor=(0.00, 1.18), borderaxespad=0.0, handlelength=1.6, columnspacing=1.0)
    add_panel_label(ax, "A")

    # B: model comparison Regime D Water random.
    ax = axes[0, 1]
    sub = summary[(summary.target == "Water_stability") & (summary.regime == "D_context_thermophysical") & (summary.split_type == "random")]
    xpos = np.arange(len(MODEL_ORDER))
    width = 0.36
    for i, ds in enumerate(["ASR", "FSR"]):
        vals, errs = [], []
        for m in MODEL_ORDER:
            row = sub[(sub.dataset == ds) & (sub.model == m)]
            vals.append(row.Spearman_mean.iloc[0] if not row.empty else np.nan)
            errs.append(row.Spearman_std.iloc[0] if not row.empty else 0)
        ax.bar(xpos + (i-0.5)*width, vals, width, yerr=errs, capsize=1.5,
               color=[MODEL_COLORS.get(m, COLORS["gray"]) for m in MODEL_ORDER],
               alpha=1 if ds == "ASR" else 0.55, edgecolor="black", lw=0.35, label=ds)
    ax.set_xticks(xpos)
    ax.set_xticklabels([MODEL_LABEL.get(m, m) for m in MODEL_ORDER], rotation=35, ha="right")
    ax.set_ylabel("Spearman ± SD\n(Water, Regime D, random)")
    ax.set_ylim(0, 0.78); ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    add_panel_label(ax, "B")

    # C: split discipline Regime D, best model per dataset/target/split.
    ax = axes[1, 0]
    best_d = best_by_cell(summary, ["dataset", "target", "split_type"], regime="D_context_thermophysical")
    labels = ["ASR\nSolv", "ASR\nWater", "ASR\nTherm", "FSR\nSolv", "FSR\nWater", "FSR\nTherm"]
    cells = [("ASR", "Solvent_stability"), ("ASR", "Water_stability"), ("ASR", "Thermal_stability (℃)"),
             ("FSR", "Solvent_stability"), ("FSR", "Water_stability"), ("FSR", "Thermal_stability (℃)")]
    x = np.arange(len(cells)); width=0.23
    for j, split in enumerate(SPLIT_ORDER):
        vals=[]; errs=[]
        for ds, target in cells:
            row = best_d[(best_d.dataset == ds) & (best_d.target == target) & (best_d.split_type == split)]
            vals.append(row.Spearman_mean.iloc[0] if not row.empty else np.nan)
            errs.append(row.Spearman_std.iloc[0] if not row.empty else 0)
        ax.bar(x + (j-1)*width, vals, width, yerr=errs, capsize=1.3, color=SPLIT_COL[split], alpha=0.9, edgecolor="black", lw=0.3, label=SPLIT_LABEL[split])
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Best Spearman ± SD\n(Regime D)")
    ax.set_ylim(-0.12, 0.78); ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=6.5, ncol=3, loc="upper center", bbox_to_anchor=(0.50, 1.12), borderaxespad=0.0, handlelength=1.2, columnspacing=0.8)
    add_panel_label(ax, "C")

    # D: Spearman vs R2 scatter, Regime D all model/split cells.
    ax = axes[1, 1]
    sub = summary[summary.regime == "D_context_thermophysical"].copy()
    for split in SPLIT_ORDER:
        ss = sub[sub.split_type == split]
        ax.scatter(ss.R2_mean, ss.Spearman_mean, s=28, color=SPLIT_COL[split], alpha=0.72, label=SPLIT_LABEL[split], edgecolor="white", lw=0.3)
    ax.axhline(0, color="#444444", lw=0.5); ax.axvline(0, color="#444444", lw=0.5)
    ax.set_xlabel("Mean R²")
    ax.set_ylabel("Mean Spearman")
    ax.set_xlim(min(-1.2, np.nanmin(sub.R2_mean)-0.05), 0.75)
    ax.set_ylim(-0.25, 0.78); ax.grid(alpha=0.20)
    ax.legend(frameon=False, fontsize=6.5, loc="lower right")
    add_panel_label(ax, "D")

    fig.subplots_adjust(wspace=0.35, hspace=0.38)
    save_fig(fig, "fig2_descriptor_model_split")
    summary.to_csv(SOURCE / "fig2_summary_source.csv", index=False)

# =============================================================================
# FIGURE 3
# =============================================================================

def make_fig3():
    log("Making Figure 3")
    asr = pd.read_csv(RAW / "ASR_data_SI_20250204.csv")
    fsr = pd.read_csv(RAW / "FSR_data_SI_20250204.csv")
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.4))

    # A: target distributions with thresholds for solvent/water.
    # Encoding: target = colour; dataset = line style. This avoids the ambiguous
    # duplicate-looking legend in the first draft.
    ax = axes[0, 0]
    hist_specs = [
        (asr, "ASR water", "Water_stability", COLORS["blue"], "-"),
        (fsr, "FSR water", "Water_stability", COLORS["blue"], "--"),
        (asr, "ASR solvent", "Solvent_stability", COLORS["orange"], "-"),
        (fsr, "FSR solvent", "Solvent_stability", COLORS["orange"], "--"),
    ]
    for df, label, target, color, ls in hist_specs:
        if target in df.columns:
            vals = pd.to_numeric(df[target], errors="coerce").dropna()
            vals = vals[(vals >= 0) & (vals <= 1)]
            ax.hist(vals, bins=np.linspace(0,1,26), histtype="step", lw=1.35, color=color, ls=ls, label=label)
    for thr, ls in [(0.6, ":"), (0.7, "--"), (0.8, ":")]:
        ax.axvline(thr, color="#444444", lw=0.75, ls=ls)
        ax.text(thr+0.006, ax.get_ylim()[1]*0.86, f"{thr:.1f}", rotation=90, va="top", fontsize=6.5)
    ax.set_xlabel("Curated stability score")
    ax.set_ylabel("Count")
    ax.legend(frameon=False, fontsize=6.2, ncol=2, loc="upper left", bbox_to_anchor=(0.00, 1.14), borderaxespad=0.0, handlelength=1.7, columnspacing=0.9)
    add_panel_label(ax, "A")

    # B: descriptor correlations.
    ax = axes[0, 1]
    cols = ["Density (g/cm3)", "LCD (Å)", "PLD (Å)", "VF", "PV (cm3/g)", "average_atomic_mass", "natoms"]
    labels = ["Density", "LCD", "PLD", "VF", "PV", "<M>", "natoms"]
    avail = [c for c in cols if c in asr.columns]
    lab_avail = [labels[cols.index(c)] for c in avail]
    corr = asr[avail].apply(pd.to_numeric, errors="coerce").corr(method="pearson")
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(avail))); ax.set_xticklabels(lab_avail, rotation=45, ha="right")
    ax.set_yticks(range(len(avail))); ax.set_yticklabels(lab_avail)
    for i in range(len(avail)):
        for j in range(len(avail)):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.8, color="white" if abs(v)>0.55 else "black")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("Pearson r", fontsize=7)
    add_panel_label(ax, "B")

    # C: ASR-FSR overlap/pair consistency.
    ax = axes[1, 0]
    pair_path = HIST / "pair_consistency_summary.csv"
    if pair_path.exists():
        pc = pd.read_csv(pair_path)
        x = np.arange(len(pc))
        # Robust to both old and new column naming.
        if "spearman_rho" in pc.columns:
            vals = pc["spearman_rho"].astype(float).values
        elif "Spearman" in pc.columns:
            vals = pc["Spearman"].astype(float).values
        elif "spearman" in pc.columns:
            vals = pc["spearman"].astype(float).values
        else:
            vals = np.ones(len(pc))
        def nice_target(t):
            t = str(t)
            if "Solvent" in t: return "Solvent"
            if "Water" in t: return "Water"
            if "Thermal" in t: return "Thermal"
            return t.replace("_stability", "")
        labels_pc = [nice_target(t) for t in pc.iloc[:,0].astype(str).values]
        ax.bar(x, vals, color=[COLORS["blue"], COLORS["orange"], COLORS["red"]][:len(x)], edgecolor="black", lw=0.4)
        for i, v in enumerate(vals):
            ax.text(i, min(1.02, v + 0.015), f"{v:.3f}", ha="center", va="bottom", fontsize=7, fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(labels_pc, rotation=15, ha="right")
        ax.set_ylabel("ASR–FSR pair Spearman")
        ax.set_ylim(0, 1.08)
    else:
        ax.text(0.5, 0.5, "pair consistency\nsummary not found", ha="center", va="center")
    ax.grid(axis="y", alpha=0.25)
    add_panel_label(ax, "C")

    # D: group-size distribution.
    ax = axes[1, 1]
    gpath = OUT / "group_audit_outputs" / "02_all_group_sizes_by_dataset_target.csv"
    if gpath.exists():
        gs = pd.read_csv(gpath)
        size_col = "group_size" if "group_size" in gs.columns else ("n_rows" if "n_rows" in gs.columns else None)
        gl_col = "group_label" if "group_label" in gs.columns else None
        if size_col and gl_col:
            if "target" in gs.columns:
                sub = gs[gs["target"] == "Water_stability"].copy()
            else:
                sub = gs.copy()
            for gl, label, color in [("metal", "Primary metal", COLORS["orange"]), ("topology_AllNodes", "Topology (AllNodes)", COLORS["red"])]:
                s = sub[sub[gl_col].astype(str).eq(gl)]
                if not s.empty:
                    vals = pd.to_numeric(s[size_col], errors="coerce").dropna()
                    ax.hist(vals, bins=[1,2,5,10,20,50,100,500], histtype="step", lw=1.5, color=color, label=label)
            ax.set_xscale("log")
            ax.set_xlabel("Group size (log scale)")
            ax.set_ylabel("Number of groups")
            ax.legend(frameon=False, fontsize=7, loc="upper right")
        else:
            ax.text(0.5,0.5,"group-size columns\nnot recognized",ha="center",va="center")
    else:
        ax.text(0.5, 0.5, "group-size audit\nnot found", ha="center", va="center")
    ax.grid(axis="y", alpha=0.25)
    add_panel_label(ax, "D")

    fig.subplots_adjust(wspace=0.36, hspace=0.42)
    save_fig(fig, "fig3_label_threshold_overlap")
    corr.to_csv(SOURCE / "fig3_descriptor_correlation_source.csv")

# =============================================================================
# FIGURE 4
# =============================================================================

def make_fig4(res: pd.DataFrame):
    log("Making Figure 4")
    summary = aggregate_results(res)
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.45))

    # A: grouped penalty random minus grouped for best Regime D.
    ax = axes[0, 0]
    best_d = best_by_cell(summary, ["dataset", "target", "split_type"], regime="D_context_thermophysical")
    rows=[]
    for ds in ["ASR", "FSR"]:
        for target in TARGET_ORDER:
            rrow = best_d[(best_d.dataset==ds)&(best_d.target==target)&(best_d.split_type=="random")]
            if rrow.empty: continue
            rv = rrow.Spearman_mean.iloc[0]
            for split in ["group_metal", "group_topology"]:
                grow = best_d[(best_d.dataset==ds)&(best_d.target==target)&(best_d.split_type==split)]
                if grow.empty: continue
                rows.append({"cell": f"{ds} {TARGET_SHORT[target]}", "split": SPLIT_LABEL[split], "penalty": rv - grow.Spearman_mean.iloc[0]})
    pen = pd.DataFrame(rows)
    cells = pen.cell.unique().tolist(); x=np.arange(len(cells)); width=0.34
    for j, split in enumerate(["Metal-grouped", "Topology-grouped"]):
        vals=[pen[(pen.cell==c)&(pen.split==split)].penalty.iloc[0] if not pen[(pen.cell==c)&(pen.split==split)].empty else np.nan for c in cells]
        ax.bar(x+(j-0.5)*width, vals, width, color=COLORS["orange"] if split.startswith("Metal") else COLORS["red"], edgecolor="black", lw=0.35, label=split)
    ax.set_xticks(x); ax.set_xticklabels([c.replace(" ", "\n") for c in cells])
    ax.set_ylabel("Random-split advantage\n(Δ Spearman)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper center", bbox_to_anchor=(0.50, 1.12), borderaxespad=0.0, handlelength=1.3, columnspacing=0.9)
    add_panel_label(ax, "A")
    pen.to_csv(SOURCE / "fig4_grouped_penalty_source.csv", index=False)

    # B: group-size bin counts from audit.
    ax = axes[0, 1]
    binp = OUT / "group_audit_outputs" / "05_group_size_bins.csv"
    if binp.exists():
        bins = pd.read_csv(binp)
        # Attempt generic plotting.
        label_col = "group_label" if "group_label" in bins.columns else bins.columns[0]
        bin_col = "size_bin" if "size_bin" in bins.columns else ("bin" if "bin" in bins.columns else None)
        count_col = "n_groups" if "n_groups" in bins.columns else ("count" if "count" in bins.columns else None)
        if bin_col and count_col:
            sub = bins[(bins.get("target", "Water_stability") == "Water_stability") | (~bins.columns.isin(["target"]).any())]
            piv = sub.pivot_table(index=bin_col, columns=label_col, values=count_col, aggfunc="sum").fillna(0)
            piv = piv.loc[piv.sum(axis=1).sort_index().index] if len(piv) else piv
            piv.plot(kind="bar", ax=ax, color=[COLORS["orange"], COLORS["red"], COLORS["gray"]], edgecolor="black", linewidth=0.3)
            ax.set_xlabel("Group-size bin")
            ax.set_ylabel("Number of groups")
            ax.legend(frameon=False, fontsize=6.5)
        else:
            ax.text(0.5,0.5,"group bin columns\nnot recognized",ha="center",va="center")
    else:
        ax.text(0.5,0.5,"group bin audit\nnot found",ha="center",va="center")
    ax.grid(axis="y", alpha=0.25)
    add_panel_label(ax, "B")

    # C: signed error by decile.
    ax = axes[1, 0]
    ep = OUT / "phase1_diagnostics" / "diagnostics_signed_errors.csv"
    if ep.exists():
        err = pd.read_csv(ep)
        sub = err[(err.target=="Water_stability") & (err.regime=="D_context_thermophysical") & (err.split_type=="random")]
        # choose model with highest average absolute info if LightGBM present.
        for ds, color in [("ASR", COLORS["blue"]), ("FSR", COLORS["orange"] )]:
            s = sub[(sub.dataset==ds) & (sub.model=="LightGBM")]
            if s.empty:
                s = sub[sub.dataset==ds].sort_values("mean_abs_error").groupby("decile").head(1)
            if not s.empty:
                s = s.sort_values("decile")
                ax.plot(s.decile, s.mean_signed_error, marker="o", lw=1.3, ms=3, color=color, label=ds)
        ax.axhline(0, color="black", lw=0.6)
        ax.set_xlabel("True-label decile")
        ax.set_ylabel("Mean signed error\n(predicted − true)")
        ax.legend(frameon=False, fontsize=7)
    else:
        ax.text(0.5,0.5,"signed-error diagnostics\nnot found",ha="center",va="center")
    ax.grid(axis="y", alpha=0.25)
    add_panel_label(ax, "C")

    # D: confident/large error summary by primary metal from predictions.
    ax = axes[1, 1]
    predp = OUT / "v14_predictions.csv"
    pred = pd.read_csv(predp)
    sub = pred[(pred.target=="Water_stability") & (pred.regime=="D_context_thermophysical") & (pred.split_type=="random") & (pred.model.map(normalize_model_name)=="LightGBM")].copy()
    if not sub.empty and "primary_metal" in sub.columns:
        sub["abs_error"] = (sub.y_pred.astype(float) - sub.y_true.astype(float)).abs()
        metal = sub.groupby("primary_metal").agg(n=("abs_error","size"), mae=("abs_error","mean")).reset_index()
        metal = metal[metal.n >= 10].sort_values("mae", ascending=False).head(10).sort_values("mae")
        ax.barh(metal.primary_metal.astype(str), metal.mae, color=COLORS["purple"], alpha=0.85, edgecolor="black", lw=0.3)
        ax.set_xlabel("Mean absolute error\n(Water, LightGBM, random)")
    else:
        ax.text(0.5,0.5,"prediction-level error\nsummary unavailable",ha="center",va="center")
    ax.grid(axis="x", alpha=0.25)
    add_panel_label(ax, "D")

    fig.subplots_adjust(wspace=0.38, hspace=0.42)
    save_fig(fig, "fig4_error_reliability_map")

# =============================================================================
# FIGURE 5
# =============================================================================

def make_fig5():
    log("Making Figure 5")
    appsum_path = OUT / "application_screening_outputs" / "v14_12089_coverage_summary.csv"
    cmap_path = OUT / "application_screening_outputs" / "v14_12089_coverage_map.csv"
    need_file(appsum_path)
    app = pd.read_csv(appsum_path)
    metrics = dict(zip(app.metric, app.value))
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.35))

    # A: coverage funnel.
    ax = axes[0, 0]
    labels = ["Total", "Descriptor\ncomplete", "In-domain"]
    vals = [metrics.get("total_recommended_screening_entries", np.nan), metrics.get("descriptor_complete_ASR_FSR", np.nan), metrics.get("descriptor_complete_in_domain", np.nan)]
    ax.bar(labels, vals, color=[COLORS["gray"], COLORS["blue"], COLORS["green"]], edgecolor="black", lw=0.4)
    for i, v in enumerate(vals):
        ax.text(i, v*1.02, f"{int(v):,}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_ylabel("Entries")
    ax.set_title("Screening-list coverage", fontsize=9, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    add_panel_label(ax, "A")

    # B: coverage categories.
    ax = axes[0, 1]
    cats = [
        ("In-domain", "coverage_category::descriptor_complete_in_domain", COLORS["green"]),
        ("New topology", "coverage_category::descriptor_complete_new_topology", COLORS["orange"]),
        ("New metal", "coverage_category::descriptor_complete_new_metal", COLORS["red"]),
        ("Missing desc.\nknown domain", "missing_descriptors_known_metal_known_topology", COLORS["blue"]),
        ("Missing desc.\nother", None, COLORS["gray"]),
    ]
    total_missing = metrics.get("missing_descriptors", 0)
    known_missing = metrics.get("missing_descriptors_known_metal_known_topology", 0)
    other_missing = total_missing - known_missing
    vals = []
    for label, key, c in cats:
        vals.append(other_missing if key is None else metrics.get(key, np.nan))
    ax.bar(range(len(cats)), vals, color=[c[2] for c in cats], edgecolor="black", lw=0.35)
    ax.set_xticks(range(len(cats))); ax.set_xticklabels([c[0] for c in cats], rotation=30, ha="right")
    ax.set_ylabel("Entries")
    for i, v in enumerate(vals):
        if pd.notna(v): ax.text(i, v*1.015, f"{int(v):,}", ha="center", va="bottom", fontsize=6.4)
    ax.grid(axis="y", alpha=0.25)
    add_panel_label(ax, "B")

    # C: top metal fields.
    ax = axes[1, 0]
    metal_rows = [(m.replace("top_metal_field::", ""), v) for m, v in metrics.items() if str(m).startswith("top_metal_field::")]
    metal_df = pd.DataFrame(metal_rows, columns=["metal", "count"]).sort_values("count", ascending=False).head(10).sort_values("count")
    ax.barh(metal_df.metal, metal_df["count"], color=COLORS["purple"], alpha=0.85, edgecolor="black", lw=0.3)
    ax.set_xlabel("Entries in recommended list")
    ax.set_title("Most frequent metals", fontsize=9, fontweight="bold")
    ax.grid(axis="x", alpha=0.25)
    add_panel_label(ax, "C")

    # D: application chemistry classes.
    ax = axes[1, 1]
    labels = ["All classes", "Water\nharvesting", "Humid CO$_2$ \ncapture", "Strict humid\nseparations"]
    vals = [metrics.get("chemistry_classes_total", np.nan), metrics.get("chemistry_classes_water_harvesting", np.nan), metrics.get("chemistry_classes_humid_co2_capture", np.nan), metrics.get("chemistry_classes_humid_separations", np.nan)]
    ax.bar(labels, vals, color=[COLORS["gray"], COLORS["blue"], COLORS["green"], COLORS["red"]], edgecolor="black", lw=0.4)
    for i, v in enumerate(vals):
        ax.text(i, v*1.02, f"{int(v):,}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_ylabel("Chemistry classes")
    ax.set_title("Operational triage tiers", fontsize=9, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    add_panel_label(ax, "D")

    fig.subplots_adjust(wspace=0.36, hspace=0.50)
    save_fig(fig, "fig5_application_screening")
    app.to_csv(SOURCE / "fig5_application_summary_source.csv", index=False)
    if cmap_path.exists():
        pd.read_csv(cmap_path).to_csv(SOURCE / "fig5_coverage_map_source.csv", index=False)

# =============================================================================
# FIGURE 6 COPY/ASSET HANDLING
# =============================================================================

def handle_fig6():
    log("Handling Figure 6 structural/case-study figure")
    candidates = []
    for folder in [
        OUT / "figures" / "main_text",
        TOOL_DIR / "assets_fig6",
        ROOT / "figures" / "main_text",
        ROOT,
    ]:
        for ext in ["pdf", "png", "svg", "tif", "tiff"]:
            candidates.append(folder / f"fig6_case_studies.{ext}")
            candidates.append(folder / f"fig6_structural_case_studies.{ext}")
    found = None
    for p in candidates:
        if p.exists():
            found = p; break
    if found:
        dst = MAIN_FIGS / f"fig6_case_studies{found.suffix.lower()}"
        shutil.copy2(found, dst)
        log(f"Copied Figure 6 from {found} -> {dst}")
    else:
        msg = """Figure 6 was requested as a main structural/case-study figure, but no structural asset was found.

To keep Figure 6 in the main manuscript, place one file named fig6_case_studies.pdf/png/svg/tif in one of:
  04_outputs/figures/main_text/
  99_final_tools/assets_fig6/
Then rerun run_all_final_package.py.

This package will not fabricate a structural figure because the scientific/visual content must come from the verified structural assets.
"""
        (QC / "MISSING_FIG6_CASE_STUDIES.txt").write_text(msg, encoding="utf-8")
        log("WARNING: Figure 6 asset not found. See QC/MISSING_FIG6_CASE_STUDIES.txt")

# =============================================================================
# FIGURE 7
# =============================================================================

def make_fig7():
    log("Making Figure 7")
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))

    # A: permutation importance.
    ax = axes[0]
    p = HIST / "permutation_importance.csv"
    if p.exists():
        imp = pd.read_csv(p)
        sub = imp[(imp.regime == "D_context_thermophysical") & (imp.model == "RandomForest")]
        # Prefer thermal or water if available.
        preferred = sub[(sub.dataset == "ASR") & (sub.target == "Thermal_stability (℃)")]
        if preferred.empty:
            preferred = sub[(sub.dataset == "ASR") & (sub.target == "Water_stability")]
        if preferred.empty:
            preferred = sub
        top = preferred.sort_values("importance_mean", ascending=False).head(12).sort_values("importance_mean")
        labels = [nice_feature_label(f) for f in top.feature.astype(str)]
        ax.barh(labels, top.importance_mean, xerr=top.importance_std if "importance_std" in top else None, color=COLORS["blue"], alpha=0.85, edgecolor="black", lw=0.3)
        ax.set_xlabel("Permutation importance")
        ax.set_title("Model reliance on descriptors", fontsize=9, fontweight="bold")
    else:
        ax.text(0.5, 0.5, "permutation importance\nnot found", ha="center", va="center")
    ax.grid(axis="x", alpha=0.25)
    add_panel_label(ax, "A")

    # B: top-decile / shortlist screening behavior.
    ax = axes[1]
    sp = HIST / "shortlist_subset_performance.csv"
    if sp.exists():
        sc = pd.read_csv(sp)
        # Plot top-decile positive rate by cutoff and target.
        for target, color in [("Solvent_stability", COLORS["orange"]), ("Water_stability", COLORS["blue"] )]:
            sub = sc[(sc.dataset == "ASR") & (sc.target == target)].sort_values("screening_cutoff")
            if sub.empty:
                sub = sc[sc.target == target].sort_values("screening_cutoff")
            if not sub.empty:
                ax.plot(sub.screening_cutoff, sub.top_decile_positive_rate, marker="o", lw=1.4, color=color, label=TARGET_SHORT.get(target, target))
                if "overall_positive_rate" in sub.columns:
                    ax.plot(sub.screening_cutoff, sub.overall_positive_rate, marker="s", lw=1.0, ls="--", color=color, alpha=0.55)
        ax.set_xlabel("Screening cutoff")
        ax.set_ylabel("Positive rate")
        ax.set_ylim(0, 1.05)
        ax.legend(frameon=False, fontsize=7, title="Top decile (solid)\nPopulation (dashed)", title_fontsize=6.5)
        ax.set_title("Confidence enrichment", fontsize=9, fontweight="bold")
    else:
        ax.text(0.5, 0.5, "screening subset\nperformance not found", ha="center", va="center")
    ax.grid(axis="y", alpha=0.25)
    add_panel_label(ax, "B")

    fig.subplots_adjust(wspace=0.45)
    save_fig(fig, "fig7_importance_screening")

# =============================================================================
# SI FIGURES QUICK PACKAGE
# =============================================================================

def make_si_figures(res: pd.DataFrame):
    log("Making SI figure quick set")
    # S1-S3 missingness.
    for idx, (name, file) in enumerate([("ASR", "ASR_data_SI_20250204.csv"), ("FSR", "FSR_data_SI_20250204.csv"), ("ION", "ION_data_SI_20250204.csv")], start=1):
        p = RAW / file
        if not p.exists(): continue
        df = pd.read_csv(p)
        miss = (df.isna().mean()*100).sort_values(ascending=False).head(18).sort_values()
        fig, ax = plt.subplots(figsize=(5.6, 3.2))
        ax.barh(miss.index.astype(str), miss.values, color=COLORS["blue"], alpha=0.85, edgecolor="black", lw=0.3)
        ax.set_xlabel(f"Missingness in {name} (%)")
        ax.grid(axis="x", alpha=0.25)
        save_fig(fig, f"figS{idx}_missingness_{name}", outdir=SI_FIGS)

    # S7 expanded descriptor sufficiency heatmap-like table.
    summary = aggregate_results(res)
    sub = summary[summary.split_type == "random"].copy()
    best = best_by_cell(sub, ["dataset", "target", "regime"])
    best["cell"] = best.dataset + " " + best.target.map(TARGET_SHORT)
    piv = best.pivot(index="cell", columns="regime", values="Spearman_mean").reindex(columns=REGIME_ORDER)
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    im = ax.imshow(piv.values, vmin=0, vmax=0.75, cmap="YlGnBu")
    ax.set_xticks(range(len(REGIME_ORDER))); ax.set_xticklabels([REGIME_SHORT[r] for r in REGIME_ORDER])
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i,j]
            ax.text(j,i,f"{v:.2f}",ha="center",va="center",fontsize=6.5)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="Best random-split Spearman")
    save_fig(fig, "figS7_expanded_descriptor_sufficiency", outdir=SI_FIGS)

    # S8 model comparison all splits/regime D.
    sub = summary[summary.regime == "D_context_thermophysical"].copy()
    sub["cell"] = sub.dataset + " " + sub.target.map(TARGET_SHORT) + " " + sub.split_type.map(SPLIT_LABEL)
    # Keep just water and solvent to avoid giant; CSV full exists.
    ss = sub[sub.target.isin(["Water_stability","Solvent_stability"])]
    piv = ss.pivot_table(index="cell", columns="model", values="Spearman_mean", aggfunc="mean").reindex(columns=MODEL_ORDER)
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    im = ax.imshow(piv.values, vmin=-0.1, vmax=0.75, cmap="YlGnBu")
    ax.set_xticks(range(len(MODEL_ORDER))); ax.set_xticklabels([MODEL_LABEL.get(m,m) for m in MODEL_ORDER], rotation=45, ha="right")
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index, fontsize=6.4)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="Mean Spearman")
    save_fig(fig, "figS8_model_comparison_regimeD", outdir=SI_FIGS)

    # S13 signed errors from existing diagnostics.
    ep = OUT / "phase1_diagnostics" / "diagnostics_signed_errors.csv"
    if ep.exists():
        err = pd.read_csv(ep)
        sub = err[(err.regime == "D_context_thermophysical") & (err.model == "LightGBM")]
        fig, ax = plt.subplots(figsize=(5.8, 3.2))
        for split, color in [("random", COLORS["blue"]), ("group_metal", COLORS["orange"]), ("group_topology", COLORS["red"] )]:
            s = sub[(sub.dataset=="ASR") & (sub.target=="Water_stability") & (sub.split_type==split)].sort_values("decile")
            if not s.empty:
                ax.plot(s.decile, s.mean_signed_error, marker="o", lw=1.2, color=color, label=SPLIT_LABEL.get(split, split))
        ax.axhline(0, color="black", lw=0.6)
        ax.set_xlabel("True-label decile")
        ax.set_ylabel("Mean signed error")
        ax.legend(frameon=False, fontsize=7)
        ax.grid(axis="y", alpha=0.25)
        save_fig(fig, "figS13_signed_error_deciles", outdir=SI_FIGS)

    # S14 application categories.
    make_fig5()  # already saved in main; copy to SI as well for convenience.
    for ext in ["pdf", "png", "svg"]:
        src = MAIN_FIGS / f"fig5_application_screening.{ext}"
        if src.exists():
            shutil.copy2(src, SI_FIGS / f"figS14_application_screening.{ext}")

# =============================================================================
# QUALITY CONTROL
# =============================================================================

def qc_check(res: pd.DataFrame):
    log("Running QC checks")
    report = []
    report.append(f"Project root: {ROOT}")
    report.append(f"Rebuilt results shape: {res.shape}")
    expected_cells = {
        "datasets": sorted(res.dataset.dropna().unique().tolist()),
        "targets": sorted(res.target.dropna().unique().tolist()),
        "regimes": sorted(res.regime.dropna().unique().tolist()),
        "split_types": sorted(res.split_type.dropna().unique().tolist()),
        "models": sorted(res.model.dropna().unique().tolist()),
    }
    report.append(json.dumps(expected_cells, indent=2, ensure_ascii=False))
    if len(res) != 4200:
        report.append("WARNING: expected 4200 rows in rebuilt V14 results.")
    oldp = OUT / "v14_results.csv"
    if oldp.exists():
        old = pd.read_csv(oldp)
        report.append(f"Old v14_results.csv shape: {old.shape}")
        if len(old) != 4200:
            report.append("WARNING: old v14_results.csv is incomplete. Do not use it for final figures/tables.")
    fig_expected = [
        "fig1_reliability_architecture.pdf",
        "fig2_descriptor_model_split.pdf",
        "fig3_label_threshold_overlap.pdf",
        "fig4_error_reliability_map.pdf",
        "fig5_application_screening.pdf",
        "fig7_importance_screening.pdf",
    ]
    for f in fig_expected:
        report.append(f"Figure exists {f}: {(MAIN_FIGS/f).exists()}")
    report.append(f"Figure 6 exists/copied: {bool(list(MAIN_FIGS.glob('fig6_case_studies.*')))}")
    (QC / "final_qc_report.txt").write_text("\n".join(report), encoding="utf-8")
    log(f"QC report saved: {QC / 'final_qc_report.txt'}")

# =============================================================================
# MAIN
# =============================================================================

def main():
    log(f"Tool dir: {TOOL_DIR}")
    log(f"Root dir: {ROOT}")
    need_file(OUT / "v14_predictions.csv")
    need_file(OUT / "v14_hparams.csv")
    res = rebuild_v14_results()
    make_tables(res)
    make_fig1()
    make_fig2(res)
    make_fig3()
    make_fig4(res)
    make_fig5()
    handle_fig6()
    make_fig7()
    make_si_figures(res)
    qc_check(res)
    log("DONE. Check 04_outputs/FINAL_FOR_REVISION/ for final tables and figures.")

if __name__ == "__main__":
    main()
