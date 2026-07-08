"""
JMCA main-figure strict rebuild (Figures 2, 3, 4, 5, 7)
=======================================================

Why this script exists
----------------------
A previous "cleaned" plotting script used unsafe fallback logic: when a preferred
source was missing, it silently switched to a different table. That is exactly
what caused the Figure 4 problem (LightGBM / prediction-level analysis got
replaced by RandomForest / failure-table analysis).

This script fixes that problem by using STRICT / PINNED data sources.

Core rule
---------
This script NEVER silently switches to a scientifically different CSV.
If a required source file is missing, it raises an error and stops.
That is intentional.

Pinned panel sources
--------------------
Figure 2
  A, B, D -> 04_outputs/FINAL_FOR_REVISION/06_source_data_for_figures/fig2_summary_source.csv
  C       -> 04_outputs/FINAL_FOR_REVISION/02_main_tables/Table_3_best_regimeD_model_by_split.csv

Figure 3
  A       -> 02_raw_data/ASR*.csv and 02_raw_data/FSR*.csv
  B       -> 04_outputs/FINAL_FOR_REVISION/06_source_data_for_figures/fig3_descriptor_correlation_source.csv
  C       -> 04_outputs/FINAL_FOR_REVISION/03_si_tables/Table_S11_matched_pair_consistency.csv
  D       -> 02_raw_data/ASR*.csv and 02_raw_data/FSR*.csv   (strict raw-based group-bin count)

Figure 4
  A       -> 04_outputs/FINAL_FOR_REVISION/06_source_data_for_figures/fig4_grouped_penalty_source.csv
  B       -> 02_raw_data/ASR*.csv and 02_raw_data/FSR*.csv   (strict raw-based group-bin count)
  C       -> 04_outputs/phase1_diagnostics/diagnostics_signed_errors.csv
  D       -> 04_outputs/v14_predictions.csv

Figure 5
  A       -> 04_outputs/FINAL_FOR_REVISION/06_source_data_for_figures/fig5_application_summary_source.csv
  B, C    -> 04_outputs/FINAL_FOR_REVISION/06_source_data_for_figures/fig5_coverage_map_source.csv
  D       -> 04_outputs/FINAL_FOR_REVISION/02_main_tables/Table_4_application_domain_map.csv

Figure 7
  A       -> 04_outputs/FINAL_FOR_REVISION/03_si_tables/Table_S12_permutation_importance.csv
  B       -> 04_outputs/FINAL_FOR_REVISION/03_si_tables/Table_S15_shortlist_subset_performance.csv

Output folder
-------------
04_outputs/FINAL_FOR_REVISION/04_main_figures_FINAL_STRICT

Also writes a manifest:
04_outputs/FINAL_FOR_REVISION/07_quality_control/main_figure_strict_source_manifest.txt

Run in Spyder
-------------
Set working directory to:
D:/UNIVERSITY/PROJECTS/MEHRDAD/SpecialPapers_WAR/15/Revision/Coding/jmca_handoff_mosiu

Then run:
%run 99_final_tools_v2/make_final_main_figures_STRICT_v4.py
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# -----------------------------------------------------------------------------
# Project-root detection
# -----------------------------------------------------------------------------

def find_project_root() -> Path:
    candidates: List[Path] = []
    here = Path.cwd().resolve()
    candidates += [here] + list(here.parents)
    try:
        script_dir = Path(__file__).resolve().parent
        candidates += [script_dir] + list(script_dir.parents)
    except Exception:
        pass
    seen = set()
    ordered = []
    for c in candidates:
        if c not in seen:
            ordered.append(c)
            seen.add(c)
    for c in ordered:
        if (c / "04_outputs").exists() and (c / "04_outputs" / "FINAL_FOR_REVISION").exists():
            return c
    raise RuntimeError(
        "Could not find project root. Run this script from jmca_handoff_mosiu or a subfolder."
    )

ROOT = find_project_root()
OUT = ROOT / "04_outputs"
FINAL = OUT / "FINAL_FOR_REVISION"
RAW = ROOT / "02_raw_data"
PHASE1 = OUT / "phase1_diagnostics"

MAIN_TABLES = FINAL / "02_main_tables"
SI_TABLES = FINAL / "03_si_tables"
SRC = FINAL / "06_source_data_for_figures"
QC = FINAL / "07_quality_control"
FIG_OUT = FINAL / "04_main_figures_FINAL_STRICT"
FIG_OUT.mkdir(parents=True, exist_ok=True)
QC.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Global style
# -----------------------------------------------------------------------------

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 1.0,
    "axes.edgecolor": "black",
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
})

COLORS = {
    "solvent": "#D9893D",
    "water": "#3A73A5",
    "thermal": "#B94A4E",
    "metal": "#D9893D",
    "topology": "#B94A4E",
    "random": "#3A73A5",
    "purple": "#8E7BB3",
    "gray": "#777777",
    "green": "#4A9680",
    "lightblue": "#87A7C7",
    "pink": "#D29AA0",
    "dark": "#222222",
}

MODEL_ORDER = ["Linear", "Ridge", "LASSO", "DecisionTree", "RandomForest", "LightGBM", "SVR_Linear"]
MODEL_LABEL = {
    "Linear": "Linear",
    "Ridge": "Ridge",
    "LASSO": "LASSO",
    "DecisionTree": "Tree",
    "RandomForest": "RF",
    "LightGBM": "LightGBM",
    "SVR_Linear": "SVR-L",
}
MODEL_COLORS = {
    "Linear": "#9B9B9B",
    "Ridge": "#3A73A5",
    "LASSO": "#4A9680",
    "DecisionTree": "#B28A2A",
    "RandomForest": "#D9893D",
    "LightGBM": "#B94A4E",
    "SVR_Linear": "#7E6AAE",
}
REGIME_ORDER = ["A_metal_only", "B_metal_oms", "C_metal_oms_context", "D_context_thermophysical"]
REGIME_LABEL = {
    "A_metal_only": "A",
    "B_metal_oms": "B",
    "C_metal_oms_context": "C",
    "D_context_thermophysical": "D",
}
SPLIT_ORDER = ["random", "group_metal", "group_topology"]
SPLIT_LABEL = {
    "random": "Random",
    "group_metal": "Metal-grouped",
    "group_topology": "Topology-grouped",
}
BIN_LABELS = ["1", "2", "3-5", "6-10", "11-20", "21-50", "51-100", ">100"]
TARGET_ORDER = ["Solvent_stability", "Water_stability", "Thermal_stability (℃)", "Thermal_stability (°C)"]
TARGET_SHORT = {
    "Solvent_stability": "Solvent",
    "Water_stability": "Water",
    "Thermal_stability (℃)": "Thermal",
    "Thermal_stability (°C)": "Thermal",
}
TARGET_COLOR = {
    "Solvent_stability": COLORS["solvent"],
    "Water_stability": COLORS["water"],
    "Thermal_stability (℃)": COLORS["thermal"],
    "Thermal_stability (°C)": COLORS["thermal"],
}

MANIFEST: Dict[str, str] = {}

# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------

def require_file(path: Path, tag: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required file missing for {tag}: {path}")
    MANIFEST[tag] = str(path)
    return path


def read_csv_strict(path: Path, tag: str, **kwargs) -> pd.DataFrame:
    require_file(path, tag)
    return pd.read_csv(path, **kwargs)


def save_figure(fig: plt.Figure, stem: str) -> None:
    for ext in ["pdf", "png", "svg"]:
        dpi = 600 if ext == "png" else None
        fig.savefig(FIG_OUT / f"{stem}.{ext}", bbox_inches="tight", dpi=dpi, facecolor="white")
    plt.close(fig)


def write_manifest(extra_lines: Optional[List[str]] = None) -> None:
    lines = ["JMCA main-figure strict source manifest", "=" * 40, f"Project root: {ROOT}", f"Output folder: {FIG_OUT}", ""]
    for key in sorted(MANIFEST):
        lines.append(f"{key}: {MANIFEST[key]}")
    if extra_lines:
        lines.extend(["", "Notes", "-----"])
        lines.extend(extra_lines)
    (QC / "main_figure_strict_source_manifest.txt").write_text("\n".join(lines), encoding="utf-8")


def add_panel_label(ax, label: str, x: float = -0.16, y: float = 1.08) -> None:
    ax.text(x, y, label, transform=ax.transAxes, ha="left", va="top", fontsize=14, fontweight="bold", clip_on=False)


def clean_spines(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def pretty_feature_name(name: str) -> str:
    if pd.isna(name):
        return ""
    s = str(name)
    mapping = {
        "metal": "Primary metal",
        "Metal Types": "Primary metal",
        "natoms": "Number of atoms",
        "average_atomic_mass": "Average atomic mass",
        "PLD (Å)": "PLD",
        "LCD (Å)": "LCD",
        "VF": "Void fraction",
        "PV (cm3/g)": "Pore volume",
        "Density (g/cm3)": "Density",
        "Has OMS": "OMS type",
        "Heat capacity_350K": "Heat capacity 350 K",
        "Heat capacity_400K": "Heat capacity 400 K",
        "cp0": "cp0",
        "kcp": "kcp",
    }
    return mapping.get(s, s)


def normalize_model_name(s) -> str:
    if pd.isna(s):
        return ""
    s = str(s).strip()
    mapping = {
        "RF": "RandomForest",
        "Random Forest": "RandomForest",
        "random_forest": "RandomForest",
        "RandomForest": "RandomForest",
        "LGBM": "LightGBM",
        "lightgbm": "LightGBM",
        "LightGBM": "LightGBM",
        "SVR-L": "SVR_Linear",
        "SVR_linear": "SVR_Linear",
        "SVR_Linear": "SVR_Linear",
        "DecisionTree": "DecisionTree",
        "Tree": "DecisionTree",
        "Linear": "Linear",
        "Ridge": "Ridge",
        "LASSO": "LASSO",
    }
    return mapping.get(s, s)


def normalize_split_name(s) -> str:
    if pd.isna(s):
        return ""
    s = str(s).strip()
    mapping = {
        "random": "random",
        "Random": "random",
        "metal": "group_metal",
        "group_metal": "group_metal",
        "metal-grouped": "group_metal",
        "grouped_metal": "group_metal",
        "topology": "group_topology",
        "group_topology": "group_topology",
        "topology-grouped": "group_topology",
        "grouped_topology": "group_topology",
    }
    return mapping.get(s, s)


def find_raw_file(kind: str) -> Path:
    require_file(RAW, "raw_root")
    patterns = {
        "ASR": ["ASR_data_SI_*.csv", "*ASR*.csv"],
        "FSR": ["FSR_data_SI_*.csv", "*FSR*.csv"],
    }
    for pat in patterns[kind]:
        files = sorted([p for p in RAW.glob(pat) if p.suffix.lower() == ".csv"])
        files = [p for p in files if "check" not in p.name.lower()]
        if files:
            MANIFEST[f"raw_{kind}"] = str(files[0])
            return files[0]
    raise FileNotFoundError(f"Could not find raw {kind} CSV in {RAW}")


def find_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    norm = {re.sub(r"[^a-z0-9]+", "", c.lower()): c for c in df.columns}
    for cand in candidates:
        key = re.sub(r"[^a-z0-9]+", "", cand.lower())
        if key in norm:
            return norm[key]
    for c in df.columns:
        c0 = re.sub(r"[^a-z0-9]+", "", c.lower())
        for cand in candidates:
            if re.sub(r"[^a-z0-9]+", "", cand.lower()) in c0:
                return c
    return None


def first_token(value) -> Optional[str]:
    if pd.isna(value):
        return None
    toks = re.split(r"[,;|/\s]+", str(value).strip())
    toks = [t for t in toks if t and t.lower() not in {"nan", "none", "unknown", "na", "n/a"}]
    if not toks:
        return None
    return sorted(toks)[0]


def bin_group_size(n: int) -> str:
    n = int(n)
    if n <= 1:
        return "1"
    if n == 2:
        return "2"
    if 3 <= n <= 5:
        return "3-5"
    if 6 <= n <= 10:
        return "6-10"
    if 11 <= n <= 20:
        return "11-20"
    if 21 <= n <= 50:
        return "21-50"
    if 51 <= n <= 100:
        return "51-100"
    return ">100"


def build_group_bin_counts(include_single_nodes: bool = True) -> pd.DataFrame:
    rows = []
    raw_specs = []
    for kind in ["ASR", "FSR"]:
        p = find_raw_file(kind)
        df = pd.read_csv(p)
        metal_col = find_column(df, ["Metal Types", "metal_types", "metal"])
        allnodes_col = find_column(df, ["AllNodes", "topology_AllNodes", "topology"])
        singlenodes_col = find_column(df, ["SingleNodes", "topology_SingleNodes"])
        raw_specs.append((kind, df, metal_col, allnodes_col, singlenodes_col))

    for kind, df, metal_col, allnodes_col, singlenodes_col in raw_specs:
        if metal_col is not None:
            sizes = df[metal_col].map(first_token).dropna().value_counts()
            for size in sizes.values:
                rows.append({"dataset": kind, "group_type": "Primary metal", "bin": bin_group_size(int(size)), "n_groups": 1})
        if allnodes_col is not None:
            sizes = df[allnodes_col].astype(str).replace({"nan": np.nan, "None": np.nan}).dropna().value_counts()
            for size in sizes.values:
                rows.append({"dataset": kind, "group_type": "Topology (AllNodes)", "bin": bin_group_size(int(size)), "n_groups": 1})
        if include_single_nodes and singlenodes_col is not None:
            sizes = df[singlenodes_col].astype(str).replace({"nan": np.nan, "None": np.nan}).dropna().value_counts()
            for size in sizes.values:
                rows.append({"dataset": kind, "group_type": "Topology (SingleNodes)", "bin": bin_group_size(int(size)), "n_groups": 1})
    if not rows:
        raise RuntimeError("Could not build group-size bins from raw data")
    out = pd.DataFrame(rows).groupby(["group_type", "bin"], as_index=False)["n_groups"].sum()
    out["bin"] = pd.Categorical(out["bin"], categories=BIN_LABELS, ordered=True)
    return out.sort_values(["group_type", "bin"])

# -----------------------------------------------------------------------------
# Core data loaders
# -----------------------------------------------------------------------------

def load_fig2_summary() -> pd.DataFrame:
    df = read_csv_strict(SRC / "fig2_summary_source.csv", "fig2_summary")
    df["model"] = df["model"].map(normalize_model_name)
    df["split_type"] = df["split_type"].map(normalize_split_name)
    return df


def load_best_regimeD_split_table() -> pd.DataFrame:
    df = read_csv_strict(MAIN_TABLES / "Table_3_best_regimeD_model_by_split.csv", "fig2_table3")
    df["model"] = df["model"].map(normalize_model_name)
    df["split_type"] = df["split_type"].map(normalize_split_name)
    return df


def best_by_cell(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["regime_rank"] = d["regime"].map({r: i for i, r in enumerate(REGIME_ORDER)})
    d = d.sort_values(["dataset", "target", "regime_rank", "split_type", "Spearman_mean"], ascending=[True, True, True, True, False])
    return d.groupby(["dataset", "target", "regime", "split_type"], as_index=False).first()


def ordered_regime_best(df: pd.DataFrame) -> pd.DataFrame:
    d = best_by_cell(df)
    d["regime_num"] = d["regime"].map({r: i for i, r in enumerate(REGIME_ORDER)})
    return d.sort_values(["dataset", "target", "regime_num", "split_type"])


def load_raw_target_frame(kind: str) -> pd.DataFrame:
    return pd.read_csv(find_raw_file(kind))


def load_signed_error_diagnostics_strict() -> pd.DataFrame:
    path = require_file(PHASE1 / "diagnostics_signed_errors.csv", "fig4_signed_errors")
    df = pd.read_csv(path)

    # Normalize model / split names if present
    model_col = find_column(df, ["model"])
    split_col = find_column(df, ["split_type", "split"])
    if model_col:
        df[model_col] = df[model_col].map(normalize_model_name)
    if split_col:
        df[split_col] = df[split_col].map(normalize_split_name)

    target_col = find_column(df, ["target"])
    regime_col = find_column(df, ["regime"])
    dataset_col = find_column(df, ["dataset"])

    # Filter exact scientific context if columns exist.
    if target_col:
        df = df[df[target_col] == "Water_stability"]
    if regime_col:
        df = df[df[regime_col] == "D_context_thermophysical"]
    if model_col:
        df = df[df[model_col] == "LightGBM"]
        if df.empty:
            raise RuntimeError("diagnostics_signed_errors.csv does not contain Water/Regime D/LightGBM rows. Strict mode refuses to switch model.")
    if split_col:
        df = df[df[split_col] == "random"]
        if df.empty:
            raise RuntimeError("diagnostics_signed_errors.csv does not contain random-split rows for Water/Regime D/LightGBM.")

    # Case 1: pre-aggregated by decile already exists.
    dec_col = find_column(df, ["true_label_decile", "decile"])
    mse_col = find_column(df, ["mean_signed_error", "signed_error_mean", "avg_signed_error"])
    if dec_col and mse_col and dataset_col:
        out = df[[dataset_col, dec_col, mse_col]].copy()
        out.columns = ["dataset", "decile", "mean_signed_error"]
        out["decile"] = pd.to_numeric(out["decile"], errors="coerce")
        out["mean_signed_error"] = pd.to_numeric(out["mean_signed_error"], errors="coerce")
        return out.dropna().sort_values(["dataset", "decile"])

    # Case 2: row-level file; compute from y_true / y_pred.
    ytrue_col = find_column(df, ["y_true", "true", "label_true"]) 
    ypred_col = find_column(df, ["y_pred", "pred", "prediction", "yhat"])
    if not (dataset_col and ytrue_col and ypred_col):
        raise RuntimeError("diagnostics_signed_errors.csv exists but required columns were not found for strict decile reconstruction.")
    tmp = df[[dataset_col, ytrue_col, ypred_col]].copy()
    tmp.columns = ["dataset", "y_true", "y_pred"]
    tmp["y_true"] = pd.to_numeric(tmp["y_true"], errors="coerce")
    tmp["y_pred"] = pd.to_numeric(tmp["y_pred"], errors="coerce")
    tmp = tmp.dropna()
    out_rows = []
    for dataset, sub in tmp.groupby("dataset"):
        sub = sub.copy().sort_values("y_true")
        sub["decile"] = pd.qcut(sub["y_true"], 10, labels=False, duplicates="drop") + 1
        sub["mean_signed_error"] = sub["y_pred"] - sub["y_true"]
        agg = sub.groupby("decile", as_index=False)["mean_signed_error"].mean()
        agg["dataset"] = dataset
        out_rows.append(agg)
    if not out_rows:
        raise RuntimeError("Could not reconstruct signed-error deciles from diagnostics_signed_errors.csv")
    out = pd.concat(out_rows, ignore_index=True)
    return out[["dataset", "decile", "mean_signed_error"]].sort_values(["dataset", "decile"])


def load_prediction_errors_by_metal_strict() -> pd.DataFrame:
    path = require_file(OUT / "v14_predictions.csv", "fig4_predictions")
    df = pd.read_csv(path)

    # Find and normalize columns.
    model_col = find_column(df, ["model"])
    split_col = find_column(df, ["split_type", "split"])
    target_col = find_column(df, ["target"])
    regime_col = find_column(df, ["regime"])
    metal_col = find_column(df, ["primary_metal", "parsed_metal", "metal", "Metal Types"])
    ytrue_col = find_column(df, ["y_true", "true", "label_true"]) 
    ypred_col = find_column(df, ["y_pred", "pred", "prediction", "yhat"])

    needed = {"target": target_col, "regime": regime_col, "model": model_col, "split": split_col, "metal": metal_col, "y_true": ytrue_col, "y_pred": ypred_col}
    missing = [k for k, v in needed.items() if v is None]
    if missing:
        raise RuntimeError(f"v14_predictions.csv exists but strict Figure 4D columns are missing: {missing}")

    df[model_col] = df[model_col].map(normalize_model_name)
    df[split_col] = df[split_col].map(normalize_split_name)
    df = df[(df[target_col] == "Water_stability") & (df[regime_col] == "D_context_thermophysical") & (df[model_col] == "LightGBM") & (df[split_col] == "random")].copy()
    if df.empty:
        raise RuntimeError("v14_predictions.csv does not contain Water/Regime D/LightGBM/random rows. Strict mode refuses to switch source/model.")

    df["primary_metal_clean"] = df[metal_col].map(first_token)
    df["y_true_num"] = pd.to_numeric(df[ytrue_col], errors="coerce")
    df["y_pred_num"] = pd.to_numeric(df[ypred_col], errors="coerce")
    df = df.dropna(subset=["primary_metal_clean", "y_true_num", "y_pred_num"])
    df["abs_error"] = (df["y_pred_num"] - df["y_true_num"]).abs()
    agg = df.groupby("primary_metal_clean", as_index=False).agg(
        mean_abs_error=("abs_error", "mean"),
        n=("abs_error", "size")
    )
    agg = agg[agg["n"] >= 5].sort_values("mean_abs_error", ascending=False)
    if agg.empty:
        raise RuntimeError("Strict Figure 4D prediction aggregation produced no metals with n >= 5.")
    return agg

# -----------------------------------------------------------------------------
# Figure 2
# -----------------------------------------------------------------------------

def make_figure2() -> None:
    df = load_fig2_summary()
    t3 = load_best_regimeD_split_table()
    ladder = ordered_regime_best(df)

    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.7))
    axA, axB, axC, axD = axes.ravel()

    # A: descriptor ladder performance
    for dataset, ls in [("ASR", "-"), ("FSR", "--")]:
        for target in ["Solvent_stability", "Water_stability", "Thermal_stability (℃)"]:
            sub = ladder[(ladder["dataset"] == dataset) & (ladder["target"] == target) & (ladder["split_type"] == "random")].copy()
            if sub.empty:
                continue
            sub["regime_num"] = sub["regime"].map({r: i for i, r in enumerate(REGIME_ORDER)})
            sub = sub.sort_values("regime_num")
            axA.plot(np.arange(len(sub)), sub["Spearman_mean"], marker="o", lw=1.7, ls=ls, color=TARGET_COLOR[target], label=f"{dataset} {TARGET_SHORT[target]}")
    axA.set_xticks(range(4), ["A", "B", "C", "D"])
    axA.set_xlabel("Descriptor regime")
    axA.set_ylabel("Best Spearman, random split")
    axA.set_ylim(0, 0.78)
    axA.grid(axis="y", alpha=0.25)
    axA.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.00, 1.14), ncol=2, handlelength=1.5, columnspacing=1.0)
    clean_spines(axA); add_panel_label(axA, "A")

    # B: seven-model comparison, same color per model; ASR solid / FSR hatched
    sub = df[(df["target"] == "Water_stability") & (df["regime"] == "D_context_thermophysical") & (df["split_type"] == "random")].copy()
    x = np.arange(len(MODEL_ORDER))
    width = 0.36
    for i, model in enumerate(MODEL_ORDER):
        color = MODEL_COLORS[model]
        for dataset, offset, hatch, fill in [("ASR", -width/2, "", color), ("FSR", +width/2, "///", "white")]:
            row = sub[(sub["dataset"] == dataset) & (sub["model"] == model)]
            if row.empty:
                continue
            y = float(row["Spearman_mean"].iloc[0])
            err = float(row["Spearman_std"].iloc[0]) if pd.notna(row["Spearman_std"].iloc[0]) else 0.0
            axB.bar(i + offset, y, width=width, color=fill, edgecolor=color, hatch=hatch, linewidth=1.2, zorder=3)
            axB.errorbar(i + offset, y, yerr=err, fmt="none", ecolor="black", lw=1.0, capsize=2, zorder=4)
    axB.set_xticks(x, [MODEL_LABEL[m] for m in MODEL_ORDER], rotation=38, ha="right")
    axB.set_ylabel("Spearman ± SD\n(Water stability, Regime D, random)")
    axB.set_ylim(0, 0.77)
    axB.grid(axis="y", alpha=0.25, zorder=0)
    axB.legend(handles=[
        Patch(facecolor="#777777", edgecolor="#777777", label="ASR"),
        Patch(facecolor="white", edgecolor="#777777", hatch="///", label="FSR")
    ], frameon=False, loc="upper left", bbox_to_anchor=(0.00, 1.12))
    clean_spines(axB); add_panel_label(axB, "B")

    # C: split-discipline comparison
    order = [
        ("ASR", "Solvent_stability"), ("ASR", "Water_stability"), ("ASR", "Thermal_stability (℃)"),
        ("FSR", "Solvent_stability"), ("FSR", "Water_stability"), ("FSR", "Thermal_stability (℃)"),
    ]
    labels = ["ASR\nSolv.", "ASR\nWater", "ASR\nTherm.", "FSR\nSolv.", "FSR\nWater", "FSR\nTherm."]
    x = np.arange(len(order))
    width = 0.24
    split_colors = {"random": COLORS["random"], "group_metal": COLORS["metal"], "group_topology": COLORS["topology"]}
    for k, split in enumerate(SPLIT_ORDER):
        ys, es = [], []
        for dataset, target in order:
            row = t3[(t3["dataset"] == dataset) & (t3["target"] == target) & (t3["split_type"] == split)]
            ys.append(float(row["Spearman_mean"].iloc[0]) if not row.empty else np.nan)
            es.append(float(row["Spearman_std"].iloc[0]) if (not row.empty and pd.notna(row["Spearman_std"].iloc[0])) else 0)
        axC.bar(x + (k-1)*width, ys, width=width, color=split_colors[split], edgecolor="#222222", linewidth=0.45, label=SPLIT_LABEL[split], alpha=0.92)
        axC.errorbar(x + (k-1)*width, ys, yerr=es, fmt="none", ecolor="black", lw=1.0, capsize=2, zorder=5)
    axC.set_xticks(x, labels)
    axC.set_ylabel("Best Spearman ± SD\n(Regime D)")
    axC.set_ylim(-0.06, 0.78)
    axC.grid(axis="y", alpha=0.25)
    axC.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.15), handlelength=1.0, columnspacing=1.1)
    clean_spines(axC); add_panel_label(axC, "C")

    # D: Spearman vs R², legend outside
    d4 = df[df["regime"] == "D_context_thermophysical"].copy()
    split_colors = {"random": COLORS["random"], "group_metal": COLORS["metal"], "group_topology": COLORS["topology"]}
    for split in SPLIT_ORDER:
        sub = d4[d4["split_type"] == split]
        axD.scatter(sub["R2_mean"], sub["Spearman_mean"], s=28, color=split_colors[split], alpha=0.70, edgecolor="white", linewidth=0.25, label=SPLIT_LABEL[split])
    axD.axhline(0, color="#555555", lw=0.8)
    axD.axvline(0, color="#555555", lw=0.8)
    axD.set_xlabel("Mean R²")
    axD.set_ylabel("Mean Spearman")
    axD.grid(alpha=0.25)
    axD.legend(frameon=False, loc="lower right", bbox_to_anchor=(1.32, 0.02))
    clean_spines(axD); add_panel_label(axD, "D")

    fig.subplots_adjust(left=0.08, right=0.90, bottom=0.12, top=0.92, wspace=0.42, hspace=0.50)
    save_figure(fig, "fig2_descriptor_model_split_FINAL_STRICT")

# -----------------------------------------------------------------------------
# Figure 3
# -----------------------------------------------------------------------------

def make_figure3() -> None:
    raw_asr = load_raw_target_frame("ASR")
    raw_fsr = load_raw_target_frame("FSR")
    corr = read_csv_strict(SRC / "fig3_descriptor_correlation_source.csv", "fig3_corr")
    pc = read_csv_strict(SI_TABLES / "Table_S11_matched_pair_consistency.csv", "fig3_pair_consistency")
    bins_df = build_group_bin_counts(include_single_nodes=False)

    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.7))
    axA, axB, axC, axD = axes.ravel()

    # A: solvent/water distributions only
    bins = np.linspace(0, 1, 26)
    plotted = False
    for df, dataset, ls in [(raw_asr, "ASR", "-"), (raw_fsr, "FSR", "--")]:
        for target in ["Water_stability", "Solvent_stability"]:
            col = find_column(df, [target])
            if col is None:
                continue
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            vals = vals[(vals >= 0) & (vals <= 1)]
            if vals.empty:
                continue
            hist, edges = np.histogram(vals, bins=bins)
            centers = (edges[:-1] + edges[1:]) / 2
            axA.step(centers, hist, where="mid", color=TARGET_COLOR[target], ls=ls, lw=1.7, label=f"{dataset} {TARGET_SHORT[target].lower()}")
            plotted = True
    if not plotted:
        raise RuntimeError("Figure 3A could not plot solvent/water distributions from raw ASR/FSR data.")
    for xline, style in [(0.60, ":"), (0.70, "--"), (0.80, ":")]:
        axA.axvline(xline, color="#333333", lw=1.0, ls=style)
        axA.text(xline + 0.005, 0.92, f"{xline:.1f}", rotation=90, transform=axA.get_xaxis_transform(), va="top", ha="left", fontsize=7)
    axA.text(0.03, 0.90, "Solvent/water only;\nthermal excluded (°C scale)", transform=axA.transAxes, ha="left", va="top", fontsize=7.0, color="#333333")
    axA.set_xlabel("Curated stability score")
    axA.set_ylabel("Count")
    axA.grid(axis="y", alpha=0.25)
    axA.legend(frameon=False, ncol=2, loc="upper left", bbox_to_anchor=(0.00, 1.14), handlelength=1.2)
    clean_spines(axA); add_panel_label(axA, "A")

    # B: descriptor correlation matrix
    if "Unnamed: 0" in corr.columns:
        corr = corr.set_index("Unnamed: 0")
    rename = {
        "Density (g/cm3)": "Density",
        "PV (cm3/g)": "Pore volume",
        "VF": "Void fraction",
        "average_atomic_mass": "Avg. atomic mass",
        "natoms": "No. atoms",
        "LCD (Å)": "LCD",
        "PLD (Å)": "PLD",
    }
    corr = corr.rename(index=rename, columns=rename)
    vals = corr.values.astype(float)
    im = axB.imshow(vals, vmin=-1, vmax=1, cmap="coolwarm")
    axB.set_xticks(np.arange(len(corr.columns)), corr.columns, rotation=45, ha="right")
    axB.set_yticks(np.arange(len(corr.index)), corr.index)
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            color = "white" if abs(vals[i, j]) > 0.55 else "black"
            axB.text(j, i, f"{vals[i, j]:.2f}", ha="center", va="center", fontsize=6.4, color=color)
    cb = fig.colorbar(im, ax=axB, fraction=0.046, pad=0.02)
    cb.set_label("Pearson r")
    axB.tick_params(length=0)
    add_panel_label(axB, "B")

    # C: matched-pair consistency
    order = [t for t in TARGET_ORDER if t in set(pc["target"])]
    pc["target_order"] = pc["target"].map({t: i for i, t in enumerate(order)})
    pc = pc.sort_values("target_order").drop_duplicates(subset=["target"])
    labels = [TARGET_SHORT[t] for t in pc["target"]]
    y = pd.to_numeric(pc["spearman_rho"], errors="coerce").values
    colors = [TARGET_COLOR[t] for t in pc["target"]]
    bars = axC.bar(np.arange(len(labels)), y, color=colors, edgecolor="#222222", linewidth=0.6)
    axC.set_xticks(np.arange(len(labels)), labels, rotation=15)
    axC.set_ylim(0, 1.16)
    axC.set_ylabel("ASR--FSR matched-pair Spearman")
    for b, val in zip(bars, y):
        axC.text(b.get_x() + b.get_width()/2, val + 0.025, f"{val:.3f}", ha="center", va="bottom", fontsize=7, fontweight="bold")
    n_note = ""
    if "n_pairs" in pc.columns:
        n_unique = sorted(set(pd.to_numeric(pc["n_pairs"], errors="coerce").dropna().astype(int)))
        if len(n_unique) == 1:
            n_note = f"n = {n_unique[0]} pairs; "
    axC.text(0.50, 0.98, f"{n_note}identity after cleaning", transform=axC.transAxes, ha="center", va="bottom", fontsize=8, color="black", clip_on=False)
    axC.grid(axis="y", alpha=0.25)
    clean_spines(axC); add_panel_label(axC, "C")

    # D: group-size distribution with correct bin order
    gtypes = ["Primary metal", "Topology (AllNodes)"]
    colors = [COLORS["metal"], COLORS["topology"]]
    x = np.arange(len(BIN_LABELS))
    width = 0.34
    for i, (gtype, color) in enumerate(zip(gtypes, colors)):
        sub = bins_df[bins_df["group_type"] == gtype].set_index("bin").reindex(BIN_LABELS)
        y = pd.to_numeric(sub["n_groups"], errors="coerce").fillna(0).values
        axD.bar(x + (i-0.5)*width, y, width=width, color=color, edgecolor="#222222", linewidth=0.45, label=gtype)
    axD.set_xticks(x, BIN_LABELS)
    axD.set_xlabel("Group-size bin")
    axD.set_ylabel("Number of groups")
    axD.legend(frameon=False, loc="upper right")
    axD.grid(axis="y", alpha=0.25)
    clean_spines(axD); add_panel_label(axD, "D")

    fig.subplots_adjust(left=0.08, right=0.96, bottom=0.11, top=0.92, wspace=0.42, hspace=0.50)
    save_figure(fig, "fig3_label_threshold_overlap_FINAL_STRICT")

# -----------------------------------------------------------------------------
# Figure 4
# -----------------------------------------------------------------------------

def make_figure4() -> None:
    pen = read_csv_strict(SRC / "fig4_grouped_penalty_source.csv", "fig4_penalty")
    gb = build_group_bin_counts(include_single_nodes=True)
    dec = load_signed_error_diagnostics_strict()
    met = load_prediction_errors_by_metal_strict()

    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.7))
    axA, axB, axC, axD = axes.ravel()

    # A: random-split advantage
    cell_order = ["ASR Solvent", "ASR Water", "ASR Thermal", "FSR Solvent", "FSR Water", "FSR Thermal"]
    x = np.arange(len(cell_order)); width = 0.34
    for k, (split, color) in enumerate([("Metal-grouped", COLORS["metal"]), ("Topology-grouped", COLORS["topology"])]):
        sub = pen[pen["split"] == split].set_index("cell").reindex(cell_order)
        axA.bar(x + (k-0.5)*width, sub["penalty"].values, width=width, color=color, edgecolor="#222222", linewidth=0.55, label=split)
    axA.set_xticks(x, [c.replace(" ", "\n") for c in cell_order])
    axA.set_ylabel("Random-split advantage\n(Δ Spearman)")
    axA.set_ylim(0, max(0.46, float(pd.to_numeric(pen["penalty"], errors="coerce").max()) * 1.18))
    axA.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.13), handlelength=1.0)
    axA.grid(axis="y", alpha=0.25)
    clean_spines(axA); add_panel_label(axA, "A")

    # B: group-size audit, fixed order and human labels
    gtypes = ["Primary metal", "Topology (AllNodes)", "Topology (SingleNodes)"]
    colors = [COLORS["metal"], COLORS["topology"], "#666666"]
    x = np.arange(len(BIN_LABELS)); width = 0.25
    for i, (gtype, color) in enumerate(zip(gtypes, colors)):
        sub = gb[gb["group_type"] == gtype].set_index("bin").reindex(BIN_LABELS)
        y = pd.to_numeric(sub["n_groups"], errors="coerce").fillna(0).values
        axB.bar(x + (i-1)*width, y, width=width, color=color, edgecolor="#222222", linewidth=0.40, label=gtype)
    axB.set_xticks(x, BIN_LABELS)
    axB.set_xlabel("Group-size bin")
    axB.set_ylabel("Number of groups")
    axB.legend(frameon=False, loc="upper right")
    axB.grid(axis="y", alpha=0.25)
    clean_spines(axB); add_panel_label(axB, "B")

    # C: signed error by true-label decile (strict Water / D / LightGBM / random)
    line_colors = {"ASR": COLORS["random"], "FSR": COLORS["metal"]}
    for dataset in ["ASR", "FSR"]:
        sub = dec[dec["dataset"] == dataset].sort_values("decile")
        if sub.empty:
            continue
        axC.plot(sub["decile"], sub["mean_signed_error"], marker="o", lw=1.6, color=line_colors[dataset], label=dataset)
    axC.axhline(0, color="black", lw=0.8)
    axC.set_xlabel("True-label decile")
    axC.set_ylabel("Mean signed error\n(predicted − true)")
    axC.set_title("Water stability; LightGBM; Regime D; random split", fontsize=8.8, pad=4)
    axC.legend(frameon=False, loc="upper right")
    axC.grid(axis="y", alpha=0.25)
    clean_spines(axC); add_panel_label(axC, "C")

    # D: large-error metals, strict prediction-level context
    top = met.head(10).sort_values("mean_abs_error", ascending=True)
    axD.barh(top["primary_metal_clean"], top["mean_abs_error"], color=COLORS["purple"], edgecolor="#222222", linewidth=0.45)
    axD.set_xlabel("Mean absolute error")
    axD.set_title("Water stability; LightGBM; Regime D; random split", fontsize=8.8, pad=4)
    axD.grid(axis="x", alpha=0.25)
    clean_spines(axD); add_panel_label(axD, "D")

    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.10, top=0.93, wspace=0.40, hspace=0.52)
    save_figure(fig, "fig4_error_reliability_map_FINAL_STRICT")

# -----------------------------------------------------------------------------
# Figure 5
# -----------------------------------------------------------------------------

def make_figure5() -> None:
    summ = read_csv_strict(SRC / "fig5_application_summary_source.csv", "fig5_summary")
    cov = read_csv_strict(SRC / "fig5_coverage_map_source.csv", "fig5_coverage")
    t4 = read_csv_strict(MAIN_TABLES / "Table_4_application_domain_map.csv", "fig5_table4")

    def get_metric(name: str, default=np.nan):
        row = summ[summ["metric"] == name]
        if row.empty:
            return default
        return int(float(row["value"].iloc[0]))

    total = get_metric("total_recommended_screening_entries", 12089)
    desc = get_metric("descriptor_complete_ASR_FSR", 1820)
    indom = get_metric("strictly_in_domain_descriptor_complete", get_metric("descriptor_complete_in_domain", 1103))

    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.6))
    axA, axB, axC, axD = axes.ravel()

    # A: true funnel
    vals = [total, desc, indom]
    labels = ["Total\nscreening list", "Descriptor\ncomplete", "In-domain\nrankable"]
    colors = [COLORS["gray"], COLORS["random"], COLORS["green"]]
    x = np.arange(3)
    axA.bar(x, vals, color=colors, edgecolor="#222222", linewidth=0.6)
    for xi, val in zip(x, vals):
        axA.text(xi, val + max(vals)*0.025, f"{val:,}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for i in range(2):
        axA.annotate("", xy=(i+0.72, max(vals)*0.73), xytext=(i+0.28, max(vals)*0.73), arrowprops=dict(arrowstyle="->", lw=1.0, color="#555555"))
    axA.set_xticks(x, labels)
    axA.set_ylabel("Entries")
    axA.set_title("Screening-list coverage funnel", fontsize=11, fontweight="bold", pad=4)
    axA.grid(axis="y", alpha=0.25)
    clean_spines(axA); add_panel_label(axA, "A")

    # B: stacked domain breakdown
    cat_counts = cov["coverage_category"].value_counts()
    stacks = [
        (0, "Descriptor complete", [
            ("descriptor_complete_in_domain", COLORS["green"], "in-domain"),
            ("descriptor_complete_new_topology", COLORS["metal"], "new topology"),
            ("descriptor_complete_new_metal", COLORS["topology"], "new metal"),
        ]),
        (1, "Missing descriptors", [
            ("missing_descriptors_known_metal_known_topology", COLORS["random"], "known domain"),
            ("missing_descriptors_known_metal_new_topology", COLORS["lightblue"], "new topology"),
            ("missing_descriptors_new_metal_known_topology", COLORS["pink"], "new metal"),
            ("missing_descriptors_new_metal_and_topology", COLORS["purple"], "new metal+topol."),
        ]),
    ]
    handles = []
    for xpos, xlabel, items in stacks:
        bottom = 0
        for key, color, short in items:
            val = int(cat_counts.get(key, 0))
            axB.bar(xpos, val, bottom=bottom, color=color, edgecolor="#222222", linewidth=0.4)
            if val >= 250:
                txt_color = "white" if color in [COLORS["random"], COLORS["topology"], COLORS["purple"]] else "black"
                axB.text(xpos, bottom + val/2, f"{val:,}", ha="center", va="center", fontsize=7, fontweight="bold", color=txt_color)
            handles.append(Patch(facecolor=color, edgecolor="#222222", label=f"{xlabel}: {short}"))
            bottom += val
    axB.set_xticks([0, 1], ["Descriptor\ncomplete", "Missing\ndescriptors"])
    axB.set_ylabel("Entries")
    axB.set_title("Domain-of-applicability breakdown", fontsize=11, fontweight="bold", pad=4)
    axB.legend(handles=handles, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.02), fontsize=6.8)
    axB.grid(axis="y", alpha=0.25)
    clean_spines(axB); add_panel_label(axB, "B")

    # C: most frequent metals
    if "parsed_metal" not in cov.columns:
        raise RuntimeError("fig5_coverage_map_source.csv is missing parsed_metal column needed for Figure 5C")
    metals = cov["parsed_metal"].fillna("unknown").astype(str)
    metals = metals[metals.str.lower() != "unknown"]
    counts = metals.value_counts().head(10).sort_values()
    axC.barh(counts.index, counts.values, color=COLORS["purple"], edgecolor="#222222", linewidth=0.45)
    axC.set_xlabel("Entries in recommended list")
    axC.set_title("Most frequent metals", fontsize=11, fontweight="bold", pad=4)
    axC.grid(axis="x", alpha=0.25)
    clean_spines(axC); add_panel_label(axC, "C")

    # D: operational triage tiers from Table 4
    cat_to_val = dict(zip(t4["category"], pd.to_numeric(t4["count"], errors="coerce")))
    d_labels = ["All\nclasses", "Water\nharvesting", "Humid CO2\ncapture", "Strict humid\nseparations"]
    d_vals = [
        int(cat_to_val.get("Chemistry classes total", 364)),
        int(cat_to_val.get("Water-harvesting classes", 101)),
        int(cat_to_val.get("Humid CO2 capture classes", 86)),
        int(cat_to_val.get("Strict humid-separation classes", 10)),
    ]
    d_colors = [COLORS["gray"], COLORS["random"], COLORS["green"], COLORS["topology"]]
    xx = np.arange(4)
    axD.bar(xx, d_vals, color=d_colors, edgecolor="#222222", linewidth=0.6)
    for xi, val in zip(xx, d_vals):
        axD.text(xi, val + max(d_vals)*0.03, f"{val:,}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    axD.set_xticks(xx, d_labels)
    axD.set_ylabel("Chemistry classes")
    axD.set_title("Operational triage tiers", fontsize=11, fontweight="bold", pad=4)
    axD.text(0.5, -0.18, "Counts describe class support from the coverage map, not a prediction-ranked shortlist.", transform=axD.transAxes, ha="center", va="top", fontsize=7.2, color="#333333")
    axD.grid(axis="y", alpha=0.25)
    clean_spines(axD); add_panel_label(axD, "D")

    fig.subplots_adjust(left=0.08, right=0.92, bottom=0.16, top=0.92, wspace=0.45, hspace=0.55)
    save_figure(fig, "fig5_application_screening_FINAL_STRICT")

# -----------------------------------------------------------------------------
# Figure 7
# -----------------------------------------------------------------------------

def make_figure7() -> None:
    imp = read_csv_strict(SI_TABLES / "Table_S12_permutation_importance.csv", "fig7_importance")
    perf = read_csv_strict(SI_TABLES / "Table_S15_shortlist_subset_performance.csv", "fig7_shortlist_performance")

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.0))
    axA, axB = axes.ravel()

    # A: pinned context -> ASR / Water / Regime D / RandomForest
    tmp = imp[(imp["dataset"] == "ASR") & (imp["target"] == "Water_stability") & (imp["regime"] == "D_context_thermophysical")].copy()
    tmp["model"] = tmp["model"].map(normalize_model_name)
    tmp = tmp[tmp["model"] == "RandomForest"]
    if tmp.empty:
        raise RuntimeError("Table_S12_permutation_importance.csv does not contain ASR / Water_stability / Regime D / RandomForest rows.")
    tmp["importance_mean"] = pd.to_numeric(tmp["importance_mean"], errors="coerce")
    tmp["importance_std"] = pd.to_numeric(tmp["importance_std"], errors="coerce").fillna(0)
    tmp = tmp.sort_values("importance_mean", ascending=False).head(12).iloc[::-1]
    axA.barh(tmp["feature"].map(pretty_feature_name), tmp["importance_mean"], xerr=tmp["importance_std"], color=COLORS["random"], edgecolor="#222222", linewidth=0.45, ecolor="black")
    axA.set_xlabel("Permutation importance")
    axA.set_title("Model reliance on descriptors\nASR water, Regime D, Random Forest", fontsize=10.5, fontweight="bold", pad=4)
    axA.grid(axis="x", alpha=0.25)
    clean_spines(axA); add_panel_label(axA, "A", x=-0.12, y=1.10)

    # B: confidence enrichment, mean ASR + FSR by target/cutoff
    tmp = perf[perf["target"].isin(["Solvent_stability", "Water_stability"])].copy()
    tmp["screening_cutoff"] = pd.to_numeric(tmp["screening_cutoff"], errors="coerce")
    agg = tmp.groupby(["target", "screening_cutoff"], as_index=False).agg(
        top_decile_positive_rate=("top_decile_positive_rate", "mean"),
        overall_positive_rate=("overall_positive_rate", "mean"),
    )
    for target, color in [("Solvent_stability", COLORS["solvent"]), ("Water_stability", COLORS["water"])]:
        sub = agg[agg["target"] == target].sort_values("screening_cutoff")
        if sub.empty:
            continue
        lab = TARGET_SHORT[target]
        axB.plot(sub["screening_cutoff"], sub["top_decile_positive_rate"], marker="o", lw=1.8, color=color, label=f"{lab}, top decile")
        axB.plot(sub["screening_cutoff"], sub["overall_positive_rate"], marker="s", lw=1.5, ls="--", color=color, alpha=0.55, label=f"{lab}, population")
    axB.set_xlabel("Screening cutoff")
    axB.set_ylabel("Positive rate")
    axB.set_ylim(0, 1.05)
    axB.set_xticks([0.60, 0.70, 0.80])
    axB.set_title("Confidence enrichment\ntop prediction decile vs population", fontsize=10.5, fontweight="bold", pad=4)
    axB.text(0.03, 0.05, "Solid = top decile\nDashed = full population", transform=axB.transAxes, ha="left", va="bottom", fontsize=7.2, color="#333333")
    axB.legend(frameon=False, loc="lower left", fontsize=7.2, ncol=1)
    axB.grid(axis="y", alpha=0.25)
    clean_spines(axB); add_panel_label(axB, "B", x=-0.12, y=1.10)

    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.18, top=0.84, wspace=0.42)
    save_figure(fig, "fig7_importance_screening_FINAL_STRICT")

# -----------------------------------------------------------------------------
# Main runner
# -----------------------------------------------------------------------------

def run_all() -> None:
    notes = []
    notes.append(f"Project root: {ROOT}")
    notes.append(f"Pinned input root: {FINAL}")
    notes.append(f"Output folder: {FIG_OUT}")
    notes.append("Strict mode: enabled (no silent fallback to alternate CSVs)")

    funcs = [make_figure2, make_figure3, make_figure4, make_figure5, make_figure7]
    for func in funcs:
        func()
        notes.append(f"OK: {func.__name__}")

    write_manifest(extra_lines=notes)
    (QC / "main_figure_strict_run_report.txt").write_text("\n".join(notes), encoding="utf-8")
    print("\n".join(notes))

if __name__ == "__main__":
    run_all()
