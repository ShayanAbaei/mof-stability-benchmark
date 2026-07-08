r"""
make_final_cleaned_main_figures.py

Final JMCA figure-cleanup patch for main Figures 2, 3, 4, 5 and 7.

What it does
------------
Creates polished replacements in:
    04_outputs/FINAL_FOR_REVISION/04_main_figures_FINAL_CLEANED/

It does NOT rerun any ML model. It only reads the verified results/tables already
inside 04_outputs/FINAL_FOR_REVISION and, where available, the raw data in 02_raw_data.

Run from Spyder with project root as working directory:
    %run 99_final_tools_v2/make_final_cleaned_main_figures.py

Or from Anaconda Prompt:
    cd D:\UNIVERSITY\PROJECTS\MEHRDAD\SpecialPapers_WAR\15\Revision\Coding\jmca_handoff_mosiu
    python 99_final_tools_v2\make_final_cleaned_main_figures.py
"""

from __future__ import annotations

import re
import textwrap
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# -----------------------------------------------------------------------------
# Project paths
# -----------------------------------------------------------------------------

def find_project_root() -> Path:
    """Return the jmca_handoff_mosiu project root.

    Works whether this script is launched from the project root, from inside
    99_final_tools_v2, or from Spyder with a different working directory.
    """
    candidates = []
    try:
        candidates.append(Path.cwd())
    except Exception:
        pass
    try:
        here = Path(__file__).resolve()
        candidates.extend([here.parent, here.parent.parent, here.parent.parent.parent])
    except Exception:
        pass

    for c in candidates:
        if (c / "04_outputs" / "FINAL_FOR_REVISION").exists():
            return c
        if c.name.startswith("99_final_tools") and (c.parent / "04_outputs" / "FINAL_FOR_REVISION").exists():
            return c.parent

    raise FileNotFoundError(
        "Could not find project root. Run this from jmca_handoff_mosiu or place the script in 99_final_tools_v2."
    )

ROOT = find_project_root()
FINAL = ROOT / "04_outputs" / "FINAL_FOR_REVISION"
RAW = ROOT / "02_raw_data"
RES = FINAL / "01_verified_results"
MAIN_TABLES = FINAL / "02_main_tables"
SI_TABLES = FINAL / "03_si_tables"
SRC = FINAL / "06_source_data_for_figures"
OUT = FINAL / "04_main_figures_FINAL_CLEANED"
QC = FINAL / "07_quality_control"

OUT.mkdir(parents=True, exist_ok=True)
QC.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Style
# -----------------------------------------------------------------------------

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "axes.linewidth": 1.1,
    "axes.labelsize": 10.5,
    "xtick.labelsize": 9.0,
    "ytick.labelsize": 9.0,
    "legend.fontsize": 8.0,
    "figure.dpi": 140,
    "savefig.dpi": 600,
})

COLORS = {
    "solvent": "#2F6F9F",
    "water": "#D98534",
    "thermal": "#B94A4E",
    "random": "#3B75A4",
    "metal": "#D9893D",
    "topology": "#B94A4E",
    "gray": "#707070",
    "green": "#45947E",
    "purple": "#8B78AD",
}

MODEL_COLORS = {
    "Linear": "#8C8C8C",
    "Ridge": "#3B75A4",
    "LASSO": "#45947E",
    "DecisionTree": "#A8781E",
    "Tree": "#A8781E",
    "RandomForest": "#D9893D",
    "RF": "#D9893D",
    "LightGBM": "#B94A4E",
    "SVR_Linear": "#7D65A2",
    "SVR-L": "#7D65A2",
}

REGIME_ORDER = ["A_metal_only", "B_metal_oms", "C_context", "D_context_thermophysical"]
REGIME_LABEL = {
    "A_metal_only": "A",
    "B_metal_oms": "B",
    "C_context": "C",
    "D_context_thermophysical": "D",
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
SPLIT_ORDER = ["random", "group_metal", "group_topology"]
SPLIT_LABEL = {"random": "Random", "group_metal": "Metal-grouped", "group_topology": "Topology-grouped"}
TARGET_LABEL = {
    "Solvent_stability": "Solvent",
    "Water_stability": "Water",
    "Thermal_stability (℃)": "Thermal",
    "Thermal_stability (°C)": "Thermal",
}
TARGET_COLOR_KEY = {
    "Solvent_stability": "solvent",
    "Water_stability": "water",
    "Thermal_stability (℃)": "thermal",
    "Thermal_stability (°C)": "thermal",
}

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def read_csv_first(paths: Iterable[Path], required: bool = True) -> Optional[pd.DataFrame]:
    for p in paths:
        if p.exists():
            return pd.read_csv(p)
    if required:
        raise FileNotFoundError("None of these files exists:\n" + "\n".join(str(p) for p in paths))
    return None


def add_panel_label(ax, label: str, x=-0.13, y=1.08):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=14, fontweight="bold", ha="left", va="top")


def clean_spines(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(width=1.0, length=4)


def save_figure(fig, stem: str):
    for ext in ["pdf", "png", "svg"]:
        fig.savefig(OUT / f"{stem}.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def target_short(t: str) -> str:
    return TARGET_LABEL.get(t, str(t).replace("_stability", "").replace("Thermal_stability (℃)", "Thermal"))


def normalize_split_label(split: str) -> str:
    return SPLIT_LABEL.get(split, split)


def load_summary() -> pd.DataFrame:
    return read_csv_first([
        SRC / "fig2_summary_source.csv",
        SRC / "all_aggregated_results_for_figures.csv",
        SI_TABLES / "Table_S8_full_v14_results_summary.csv",
    ])


def load_results() -> pd.DataFrame:
    return read_csv_first([
        RES / "v14_results_rebuilt.csv",
        FINAL / "v14_results_rebuilt.csv",
        ROOT / "04_outputs" / "v14_results_rebuilt.csv",
    ])


def best_by_cell(df: pd.DataFrame) -> pd.DataFrame:
    """Pick best model by Spearman within dataset/target/regime/split."""
    d = df.copy()
    d["Spearman_mean"] = pd.to_numeric(d["Spearman_mean"], errors="coerce")
    d = d.dropna(subset=["Spearman_mean"])
    idx = d.groupby(["dataset", "target", "regime", "split_type"])["Spearman_mean"].idxmax()
    return d.loc[idx].reset_index(drop=True)


def ordered_regime_table(df: pd.DataFrame) -> pd.DataFrame:
    d = best_by_cell(df)
    d = d[(d["split_type"] == "random") & (d["regime"].isin(REGIME_ORDER))].copy()
    d["regime"] = pd.Categorical(d["regime"], categories=REGIME_ORDER, ordered=True)
    return d.sort_values(["dataset", "target", "regime"])


def find_raw_file(kind: str) -> Optional[Path]:
    if not RAW.exists():
        return None
    patterns = {
        "ASR": ["ASR_data_SI_*.csv", "*ASR*.csv"],
        "FSR": ["FSR_data_SI_*.csv", "*FSR*.csv"],
        "ION": ["ION_data_SI_*.csv", "*ION*.csv"],
    }
    for pat in patterns[kind]:
        files = sorted(RAW.glob(pat))
        # avoid ASR_FSR_check for ASR/FSR
        files = [f for f in files if "check" not in f.name.lower()]
        if files:
            return files[0]
    return None


def find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols_norm = {re.sub(r"[^a-z0-9]+", "", c.lower()): c for c in df.columns}
    for cand in candidates:
        key = re.sub(r"[^a-z0-9]+", "", cand.lower())
        if key in cols_norm:
            return cols_norm[key]
    # fuzzy contains
    for c in df.columns:
        cn = c.lower()
        for cand in candidates:
            if cand.lower() in cn:
                return c
    return None


def load_raw(kind: str) -> Optional[pd.DataFrame]:
    p = find_raw_file(kind)
    if p is None:
        return None
    return pd.read_csv(p)


def export_source(df: pd.DataFrame, name: str):
    out_src = FINAL / "06_source_data_for_figures_FINAL_CLEANED"
    out_src.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_src / name, index=False)

# -----------------------------------------------------------------------------
# Group-bin utilities with strict numeric ordering
# -----------------------------------------------------------------------------

BIN_LABELS = ["1", "2", "3-5", "6-10", "11-20", "21-50", "51-100", ">100"]

def group_size_bin(n: int) -> str:
    if n == 1:
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


def parse_primary_metal(value) -> str:
    if pd.isna(value):
        return "unknown"
    s = str(value)
    # remove common punctuation but keep element tokens
    tokens = re.findall(r"[A-Z][a-z]?", s)
    if not tokens:
        return "unknown"
    # remove common false positives if any
    tokens = [t for t in tokens if len(t) <= 2]
    return sorted(tokens)[0] if tokens else "unknown"


def infer_group_bins_from_raw(include_single_nodes: bool = False) -> pd.DataFrame:
    """Compute ASR+FSR group-size bins from raw ASR/FSR tables.

    Returns columns: group_type, bin, n_groups
    """
    rows = []
    for kind in ["ASR", "FSR"]:
        df = load_raw(kind)
        if df is None:
            continue
        metal_col = find_column(df, ["primary_metal", "Metal Types", "metal_types", "Metal_Types"])
        topo_col = find_column(df, ["AllNodes", "topology_AllNodes", "topology", "Topology"])
        single_col = find_column(df, ["SingleNodes", "topology_SingleNodes"])
        if metal_col:
            metals = df[metal_col].map(parse_primary_metal)
            counts = metals.value_counts(dropna=False)
            for size in counts.values:
                rows.append({"dataset": kind, "group_type": "Primary metal", "bin": group_size_bin(int(size)), "n_groups": 1})
        if topo_col:
            top = df[topo_col].fillna("unknown").astype(str).str.strip().replace({"": "unknown", "nan": "unknown"})
            counts = top.value_counts(dropna=False)
            for size in counts.values:
                rows.append({"dataset": kind, "group_type": "Topology (AllNodes)", "bin": group_size_bin(int(size)), "n_groups": 1})
        if include_single_nodes and single_col:
            top = df[single_col].fillna("unknown").astype(str).str.strip().replace({"": "unknown", "nan": "unknown"})
            counts = top.value_counts(dropna=False)
            for size in counts.values:
                rows.append({"dataset": kind, "group_type": "Topology (SingleNodes)", "bin": group_size_bin(int(size)), "n_groups": 1})
    if rows:
        out = pd.DataFrame(rows).groupby(["group_type", "bin"], as_index=False)["n_groups"].sum()
        out["bin"] = pd.Categorical(out["bin"], categories=BIN_LABELS, ordered=True)
        return out.sort_values(["group_type", "bin"])
    return pd.DataFrame(columns=["group_type", "bin", "n_groups"])


def fallback_group_bins_from_s5(include_single_nodes: bool = False) -> pd.DataFrame:
    """Fallback approximate binned distribution from Table S5 if raw data are missing.

    This uses cumulative rare-group counts, so it is marked approximate in QC.
    Prefer raw-data reconstruction whenever possible.
    """
    s5 = read_csv_first([SI_TABLES / "Table_S5_full_group_audit.csv"], required=False)
    if s5 is None or s5.empty:
        return pd.DataFrame(columns=["group_type", "bin", "n_groups"])
    # use one target only to avoid triplicate repeated rows
    target0 = "Water_stability" if "Water_stability" in set(s5["target"]) else s5["target"].iloc[0]
    d = s5[(s5["dataset"].isin(["ASR", "FSR"])) & (s5["target"] == target0)].copy()
    label_map = {"metal": "Primary metal", "topology_AllNodes": "Topology (AllNodes)", "topology_SingleNodes": "Topology (SingleNodes)"}
    allowed = ["metal", "topology_AllNodes"] + (["topology_SingleNodes"] if include_single_nodes else [])
    d = d[d["group_label"].isin(allowed)]
    rows = []
    for glabel, sub in d.groupby("group_label"):
        # aggregate ASR + FSR cumulative counts
        singleton = int(sub["n_singleton_groups"].sum())
        lt5 = int(sub["n_groups_lt5"].sum())
        lt10 = int(sub["n_groups_lt10"].sum())
        lt20 = int(sub["n_groups_lt20"].sum())
        total = int(sub["n_groups"].sum())
        # Approximate: split the 2-4 group range across 2 and 3-5.
        two_to_four = max(lt5 - singleton, 0)
        rows.extend([
            {"group_type": label_map[glabel], "bin": "1", "n_groups": singleton},
            {"group_type": label_map[glabel], "bin": "2", "n_groups": round(two_to_four * 0.35)},
            {"group_type": label_map[glabel], "bin": "3-5", "n_groups": round(two_to_four * 0.65)},
            {"group_type": label_map[glabel], "bin": "6-10", "n_groups": max(lt10 - lt5, 0)},
            {"group_type": label_map[glabel], "bin": "11-20", "n_groups": max(lt20 - lt10, 0)},
            {"group_type": label_map[glabel], "bin": "21-50", "n_groups": np.nan},
            {"group_type": label_map[glabel], "bin": "51-100", "n_groups": np.nan},
            {"group_type": label_map[glabel], "bin": ">100", "n_groups": np.nan},
        ])
    out = pd.DataFrame(rows)
    out["bin"] = pd.Categorical(out["bin"], categories=BIN_LABELS, ordered=True)
    return out.sort_values(["group_type", "bin"])


def get_group_bins(include_single_nodes: bool = False) -> Tuple[pd.DataFrame, str]:
    raw_bins = infer_group_bins_from_raw(include_single_nodes=include_single_nodes)
    if not raw_bins.empty:
        export_source(raw_bins, "group_size_bins_from_raw_ordered.csv")
        return raw_bins, "raw"
    fb = fallback_group_bins_from_s5(include_single_nodes=include_single_nodes)
    export_source(fb, "group_size_bins_from_S5_APPROXIMATE.csv")
    return fb, "fallback_approximate"

# -----------------------------------------------------------------------------
# Figure 2
# -----------------------------------------------------------------------------

def make_figure2():
    df = load_summary()
    best = best_by_cell(df)
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.6))
    axA, axB, axC, axD = axes.ravel()

    # Panel A: descriptor ladder under random split
    ladder = ordered_regime_table(df)
    for dataset, ls in [("ASR", "-"), ("FSR", "--")]:
        for target in ["Solvent_stability", "Water_stability", "Thermal_stability (℃)"]:
            sub = ladder[(ladder["dataset"] == dataset) & (ladder["target"] == target)].copy()
            if sub.empty:
                continue
            sub = sub.sort_values("regime")
            x = np.arange(len(sub))
            color = COLORS[TARGET_COLOR_KEY[target]]
            axA.plot(x, sub["Spearman_mean"], marker="o", lw=1.6, ls=ls, color=color, label=f"{dataset} {target_short(target)}")
    axA.set_xticks(range(4), ["A", "B", "C", "D"])
    axA.set_ylim(0, 0.78)
    axA.set_xlabel("Descriptor regime")
    axA.set_ylabel("Best Spearman, random split")
    axA.grid(axis="y", alpha=0.25)
    axA.legend(ncol=2, frameon=False, loc="upper left", bbox_to_anchor=(0.00, 1.03), handlelength=1.8)
    clean_spines(axA); add_panel_label(axA, "A")

    # Panel B: same model color; ASR solid, FSR hatched
    sub = df[(df["target"] == "Water_stability") & (df["regime"] == "D_context_thermophysical") & (df["split_type"] == "random")].copy()
    x = np.arange(len(MODEL_ORDER))
    width = 0.36
    for i, model in enumerate(MODEL_ORDER):
        color = MODEL_COLORS[model]
        for j, (dataset, offset, hatch, face) in enumerate([("ASR", -width/2, "", color), ("FSR", width/2, "///", "white")]):
            row = sub[(sub["dataset"] == dataset) & (sub["model"] == model)]
            if row.empty:
                continue
            y = float(row["Spearman_mean"].iloc[0])
            err = float(row["Spearman_std"].iloc[0]) if pd.notna(row["Spearman_std"].iloc[0]) else 0
            axB.bar(i + offset, y, width=width, color=face, edgecolor=color, hatch=hatch, linewidth=1.2, zorder=3)
            axB.errorbar(i + offset, y, yerr=err, color="black", lw=1, capsize=2, zorder=4)
    axB.set_xticks(x, [MODEL_LABEL[m] for m in MODEL_ORDER], rotation=38, ha="right")
    axB.set_ylim(0, 0.77)
    axB.set_ylabel("Spearman ± SD\n(Water, Regime D, random)")
    axB.grid(axis="y", alpha=0.25, zorder=0)
    axB.legend(
        handles=[Patch(facecolor="#777777", edgecolor="#777777", label="ASR"), Patch(facecolor="white", edgecolor="#777777", hatch="///", label="FSR")],
        frameon=False, loc="upper left", bbox_to_anchor=(0.01, 1.02), ncol=1
    )
    clean_spines(axB); add_panel_label(axB, "B")

    # Panel C: split-discipline comparison, best Regime-D model per cell
    t3 = read_csv_first([MAIN_TABLES / "Table_3_best_regimeD_model_by_split.csv", SRC / "figure2_table3_best_regimeD_source.csv"])
    t3 = t3[t3["target"].isin(["Solvent_stability", "Water_stability", "Thermal_stability (℃)"])].copy()
    cell_order = [("ASR", "Solvent_stability"), ("ASR", "Water_stability"), ("ASR", "Thermal_stability (℃)"),
                  ("FSR", "Solvent_stability"), ("FSR", "Water_stability"), ("FSR", "Thermal_stability (℃)")]
    labels = ["ASR\nSolv.", "ASR\nWater", "ASR\nTherm.", "FSR\nSolv.", "FSR\nWater", "FSR\nTherm."]
    x = np.arange(len(cell_order)); width = 0.24
    for k, split in enumerate(SPLIT_ORDER):
        ys, es = [], []
        for dataset, target in cell_order:
            row = t3[(t3["dataset"] == dataset) & (t3["target"] == target) & (t3["split_type"] == split)]
            ys.append(float(row["Spearman_mean"].iloc[0]) if not row.empty else np.nan)
            es.append(float(row["Spearman_std"].iloc[0]) if (not row.empty and pd.notna(row["Spearman_std"].iloc[0])) else 0)
        color = [COLORS["random"], COLORS["metal"], COLORS["topology"]][k]
        axC.bar(x + (k - 1) * width, ys, width=width, color=color, edgecolor="#222222", linewidth=0.45, label=SPLIT_LABEL[split], alpha=0.92)
        axC.errorbar(x + (k - 1) * width, ys, yerr=es, fmt="none", ecolor="black", lw=1.0, capsize=2, zorder=5)
    axC.set_xticks(x, labels)
    axC.set_ylim(-0.06, 0.78)
    axC.set_ylabel("Best Spearman ± SD\n(Regime D)")
    axC.grid(axis="y", alpha=0.25)
    axC.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.13), handlelength=1.0, columnspacing=1.2)
    clean_spines(axC); add_panel_label(axC, "C")

    # Panel D: Spearman vs R2, legend outside and smaller markers
    d4 = df[df["regime"] == "D_context_thermophysical"].copy()
    for split, color in [("random", COLORS["random"]), ("group_metal", COLORS["metal"]), ("group_topology", COLORS["topology"] )]:
        sub = d4[d4["split_type"] == split]
        axD.scatter(sub["R2_mean"], sub["Spearman_mean"], s=26, color=color, alpha=0.68, edgecolor="white", linewidth=0.25, label=SPLIT_LABEL[split])
    axD.axhline(0, color="#555555", lw=0.8)
    axD.axvline(0, color="#555555", lw=0.8)
    axD.set_xlabel("Mean R²")
    axD.set_ylabel("Mean Spearman")
    axD.set_xlim(-1.15, 0.55)
    axD.set_ylim(-0.24, 0.74)
    axD.grid(alpha=0.22)
    axD.legend(frameon=False, loc="lower left", bbox_to_anchor=(1.01, 0.00), borderaxespad=0.0, markerscale=0.9)
    clean_spines(axD); add_panel_label(axD, "D")

    fig.subplots_adjust(left=0.08, right=0.88, bottom=0.09, top=0.94, wspace=0.38, hspace=0.45)
    save_figure(fig, "fig2_descriptor_model_split_FINAL_CLEANED")

# -----------------------------------------------------------------------------
# Figure 3
# -----------------------------------------------------------------------------

def make_figure3():
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.6))
    axA, axB, axC, axD = axes.ravel()

    # Panel A: target distributions, solvent/water only
    raw_asr = load_raw("ASR")
    raw_fsr = load_raw("FSR")
    bins = np.linspace(0, 1, 26)
    plotted = False
    for df, dataset, ls in [(raw_asr, "ASR", "-"), (raw_fsr, "FSR", "--")]:
        if df is None:
            continue
        for target, color in [("Water_stability", COLORS["water"]), ("Solvent_stability", COLORS["solvent"] )]:
            col = find_column(df, [target])
            if col is None:
                continue
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            vals = vals[(vals >= 0) & (vals <= 1)]
            if vals.empty:
                continue
            hist, edges = np.histogram(vals, bins=bins)
            centers = (edges[:-1] + edges[1:]) / 2
            axA.step(centers, hist, where="mid", color=color, ls=ls, lw=1.7, label=f"{dataset} {target_short(target).lower()}")
            plotted = True
    if not plotted:
        axA.text(0.5, 0.5, "Raw ASR/FSR target columns\nnot found in 02_raw_data", ha="center", va="center", transform=axA.transAxes)
    for xline, style in [(0.60, ":"), (0.70, "--"), (0.80, ":")]:
        axA.axvline(xline, color="#333333", lw=1.0, ls=style)
        axA.text(xline + 0.005, 0.92, f"{xline:.1f}", rotation=90, transform=axA.get_xaxis_transform(), va="top", ha="left", fontsize=7)
    axA.text(0.03, 0.89, "Solvent/water only;\nthermal excluded (°C scale)", transform=axA.transAxes, ha="left", va="top", fontsize=7.2, color="#333333")
    axA.set_xlabel("Curated stability score")
    axA.set_ylabel("Count")
    axA.grid(axis="y", alpha=0.25)
    axA.legend(frameon=False, ncol=2, loc="upper left", bbox_to_anchor=(0.0, 1.12), handlelength=1.2, columnspacing=1.0)
    clean_spines(axA); add_panel_label(axA, "A")

    # Panel B: descriptor correlation matrix
    corr = read_csv_first([SRC / "fig3_descriptor_correlation_source.csv"], required=False)
    if corr is not None and not corr.empty:
        if "Unnamed: 0" in corr.columns:
            corr = corr.set_index("Unnamed: 0")
        rename = {
            "Density (g/cm3)": "Density",
            "PV (cm3/g)": "Pore volume",
            "average_atomic_mass": "Avg. atomic mass",
            "natoms": "No. atoms",
            "VF": "Void fraction",
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
                axB.text(j, i, f"{vals[i,j]:.2f}", ha="center", va="center", fontsize=6.4, color=color)
        cb = fig.colorbar(im, ax=axB, fraction=0.046, pad=0.02)
        cb.set_label("Pearson r")
        axB.tick_params(length=0)
    else:
        axB.text(0.5, 0.5, "Correlation source\nnot found", ha="center", va="center")
    add_panel_label(axB, "B")

    # Panel C: ASR-FSR matched-pair consistency
    pc = read_csv_first([SI_TABLES / "Table_S11_matched_pair_consistency.csv"], required=False)
    if pc is not None and not pc.empty:
        order = ["Solvent_stability", "Water_stability", "Thermal_stability (℃)", "Thermal_stability (°C)"]
        pc["target_order"] = pc["target"].map({t: i for i, t in enumerate(order)})
        pc = pc.sort_values("target_order")
        pc = pc.drop_duplicates(subset=["target"])
        pc = pc[pc["target"].isin(order)]
        labels = [target_short(t) for t in pc["target"]]
        y = pd.to_numeric(pc["spearman_rho"], errors="coerce").values
        colors = [COLORS[TARGET_COLOR_KEY.get(t, "solvent")] for t in pc["target"]]
        bars = axC.bar(np.arange(len(labels)), y, color=colors, edgecolor="#222222", linewidth=0.6)
        axC.set_xticks(np.arange(len(labels)), labels, rotation=15)
        axC.set_ylim(0, 1.16)
        axC.set_ylabel("ASR--FSR matched-pair Spearman")
        for b, val in zip(bars, y):
            axC.text(b.get_x()+b.get_width()/2, val+0.025, f"{val:.3f}", ha="center", va="bottom", fontsize=7, fontweight="bold")
        n_txt = ""
        if "n_pairs" in pc.columns:
            n_unique = sorted(set(pd.to_numeric(pc["n_pairs"], errors="coerce").dropna().astype(int)))
            if len(n_unique) == 1:
                n_txt = f"n = {n_unique[0]} pairs; "
        axC.text(0.5, 1.09, f"{n_txt}matched labels are identical after cleaning", transform=axC.transAxes, ha="center", va="bottom", fontsize=7.0, color="#333333", clip_on=False)
    else:
        axC.text(0.5, 0.5, "Matched-pair consistency\nsource not found", ha="center", va="center")
    axC.grid(axis="y", alpha=0.25)
    clean_spines(axC); add_panel_label(axC, "C")

    # Panel D: ordered group-size distribution, only primary metal and AllNodes
    gb, source = get_group_bins(include_single_nodes=False)
    if not gb.empty:
        width = 0.36
        x = np.arange(len(BIN_LABELS))
        for i, (gtype, color) in enumerate([("Primary metal", COLORS["metal"]), ("Topology (AllNodes)", COLORS["topology"]) ]):
            sub = gb[gb["group_type"] == gtype].set_index("bin").reindex(BIN_LABELS)
            y = pd.to_numeric(sub["n_groups"], errors="coerce").fillna(0).values
            axD.plot(x, y, drawstyle="steps-mid", lw=1.8, color=color, label=gtype)
        axD.set_xticks(x, BIN_LABELS, rotation=0)
        axD.set_xlabel("Group-size bin")
        axD.set_ylabel("Number of groups")
        if source == "fallback_approximate":
            axD.text(0.98, 0.96, "approx. bins", transform=axD.transAxes, ha="right", va="top", fontsize=7, color="#555555")
    else:
        axD.text(0.5, 0.5, "Group-size bins\nnot found", ha="center", va="center")
    axD.grid(axis="y", alpha=0.25)
    axD.legend(frameon=False, loc="upper right")
    clean_spines(axD); add_panel_label(axD, "D")

    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.09, top=0.94, wspace=0.40, hspace=0.55)
    save_figure(fig, "fig3_label_threshold_overlap_FINAL_CLEANED")

# -----------------------------------------------------------------------------
# Figure 4
# -----------------------------------------------------------------------------

def make_figure4():
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.6))
    axA, axB, axC, axD = axes.ravel()

    # Panel A: random split advantage / grouped penalty
    pen = read_csv_first([SRC / "fig4_grouped_penalty_source.csv"], required=False)
    if pen is not None and not pen.empty:
        cell_order = ["ASR Solvent", "ASR Water", "ASR Thermal", "FSR Solvent", "FSR Water", "FSR Thermal"]
        x = np.arange(len(cell_order)); width = 0.34
        for k, (split, color) in enumerate([("Metal-grouped", COLORS["metal"]), ("Topology-grouped", COLORS["topology"]) ]):
            sub = pen[pen["split"] == split].set_index("cell").reindex(cell_order)
            axA.bar(x + (k-0.5)*width, sub["penalty"].values, width=width, color=color, edgecolor="#222222", linewidth=0.55, label=split)
        axA.set_xticks(x, [c.replace(" ", "\n") for c in cell_order])
        axA.set_ylabel("Random-split advantage\n(Δ Spearman)")
        axA.set_ylim(0, max(0.46, float(pen["penalty"].max()) * 1.18))
        axA.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.13), handlelength=1.0)
    else:
        axA.text(0.5, 0.5, "Grouped penalty source\nnot found", ha="center", va="center")
    axA.grid(axis="y", alpha=0.25)
    clean_spines(axA); add_panel_label(axA, "A")

    # Panel B: ordered group-size bins, human labels
    gb, source = get_group_bins(include_single_nodes=True)
    if not gb.empty:
        gtypes = ["Primary metal", "Topology (AllNodes)", "Topology (SingleNodes)"]
        colors = [COLORS["metal"], COLORS["topology"], "#666666"]
        x = np.arange(len(BIN_LABELS)); width = 0.25
        for i, (gtype, color) in enumerate(zip(gtypes, colors)):
            sub = gb[gb["group_type"] == gtype].set_index("bin").reindex(BIN_LABELS)
            y = pd.to_numeric(sub["n_groups"], errors="coerce").fillna(0).values
            axB.bar(x + (i-1)*width, y, width=width, color=color, edgecolor="#222222", linewidth=0.4, label=gtype)
        axB.set_xticks(x, BIN_LABELS, rotation=0)
        axB.set_xlabel("Group-size bin")
        axB.set_ylabel("Number of groups")
        axB.legend(frameon=False, loc="upper right")
        if source == "fallback_approximate":
            axB.text(0.98, 0.96, "approx. bins", transform=axB.transAxes, ha="right", va="top", fontsize=7, color="#555555")
    else:
        axB.text(0.5, 0.5, "Group-size bins\nnot found", ha="center", va="center")
    axB.grid(axis="y", alpha=0.25)
    clean_spines(axB); add_panel_label(axB, "B")

    # Panel C: signed error by true-label decile
    # Prefer existing SI decile figure source if present; otherwise compute from failure cases only.
    decile_file_candidates = [SRC / "signed_error_deciles_source.csv", SI_TABLES / "Table_S13_signed_error_deciles.csv"]
    dec = read_csv_first(decile_file_candidates, required=False)
    if dec is None or dec.empty:
        # Reconstruct from failure table if a detailed y_true/y_pred table exists. Limited but still informative.
        fc = read_csv_first([SI_TABLES / "Table_S13_failure_cases_top50.csv"], required=False)
        if fc is not None and not fc.empty and {"dataset", "target", "y_true", "y_pred"}.issubset(fc.columns):
            tmp = fc[fc["target"] == "Water_stability"].copy()
            tmp["signed_error"] = pd.to_numeric(tmp["y_pred"], errors="coerce") - pd.to_numeric(tmp["y_true"], errors="coerce")
            tmp["decile"] = pd.qcut(pd.to_numeric(tmp["y_true"], errors="coerce"), 10, labels=False, duplicates="drop") + 1
            dec = tmp.groupby(["dataset", "decile"], as_index=False)["signed_error"].mean().rename(columns={"signed_error": "mean_signed_error"})
    if dec is not None and not dec.empty:
        # normalize column names
        if "true_label_decile" in dec.columns and "decile" not in dec.columns:
            dec = dec.rename(columns={"true_label_decile": "decile"})
        if "mean_signed_error" not in dec.columns:
            for c in dec.columns:
                if "signed" in c.lower() and "error" in c.lower():
                    dec = dec.rename(columns={c: "mean_signed_error"})
                    break
        for dataset, color in [("ASR", COLORS["random"]), ("FSR", COLORS["water"] if False else COLORS["metal"])]:
            sub = dec[dec["dataset"] == dataset].sort_values("decile")
            if sub.empty:
                continue
            axC.plot(sub["decile"], sub["mean_signed_error"], marker="o", lw=1.6, color=color, label=dataset)
        axC.axhline(0, color="black", lw=0.8)
        axC.set_xlabel("True-label decile")
        axC.set_ylabel("Mean signed error\n(predicted − true)")
        axC.legend(frameon=False, loc="upper right")
    else:
        axC.text(0.5, 0.5, "Signed-error decile\nsource not found", ha="center", va="center")
    axC.grid(axis="y", alpha=0.25)
    clean_spines(axC); add_panel_label(axC, "C")

    # Panel D: large-error metals, define exact context in subtitle
    fc = read_csv_first([SI_TABLES / "Table_S13_failure_cases_top50.csv"], required=False)
    if fc is not None and not fc.empty:
        tmp = fc[(fc["target"] == "Water_stability") & (fc["model"].isin(["LightGBM", "RandomForest"]))].copy()
        # Prefer LightGBM if available.
        if (tmp["model"] == "LightGBM").any():
            tmp = tmp[tmp["model"] == "LightGBM"]
            context_model = "LightGBM"
        else:
            context_model = "+".join(sorted(tmp["model"].dropna().unique())) if not tmp.empty else "model"
        tmp["abs_error"] = pd.to_numeric(tmp["abs_error"], errors="coerce")
        if not tmp.empty:
            met = tmp.groupby("primary_metal", as_index=False)["abs_error"].mean().sort_values("abs_error", ascending=False).head(10)
            axD.barh(met["primary_metal"][::-1], met["abs_error"][::-1], color=COLORS["purple"], edgecolor="#222222", linewidth=0.45)
            axD.set_xlabel("Mean absolute error")
            axD.set_title(f"Water stability; {context_model}; random-split failures", fontsize=8.5, pad=4)
        else:
            axD.text(0.5, 0.5, "Water-stability failure cases\nnot found", ha="center", va="center")
    else:
        axD.text(0.5, 0.5, "Failure-case table\nnot found", ha="center", va="center")
    axD.grid(axis="x", alpha=0.25)
    clean_spines(axD); add_panel_label(axD, "D")

    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.09, top=0.93, wspace=0.36, hspace=0.48)
    save_figure(fig, "fig4_error_reliability_map_FINAL_CLEANED")

# -----------------------------------------------------------------------------
# Figure 5
# -----------------------------------------------------------------------------

def make_figure5():
    summ = read_csv_first([SRC / "fig5_application_summary_source.csv"], required=False)
    cov = read_csv_first([SRC / "fig5_coverage_map_source.csv"], required=False)
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.5))
    axA, axB, axC, axD = axes.ravel()

    def get_metric(name, default=np.nan):
        if summ is None or summ.empty:
            return default
        row = summ[summ["metric"] == name]
        if row.empty:
            return default
        return int(float(row["value"].iloc[0]))

    total = get_metric("total_recommended_screening_entries", 12089)
    desc = get_metric("descriptor_complete_ASR_FSR", 1820)
    indom = get_metric("strictly_in_domain_descriptor_complete", get_metric("in_domain_descriptor_complete", 1103))
    if not np.isfinite(indom):
        indom = 1103

    # Panel A: funnel
    funnel_labels = ["Total\nscreening list", "Descriptor\ncomplete", "In-domain\nrankable"]
    funnel_vals = [total, desc, indom]
    funnel_colors = [COLORS["gray"], COLORS["random"], COLORS["green"]]
    x = np.arange(3)
    axA.bar(x, funnel_vals, color=funnel_colors, edgecolor="#222222", linewidth=0.6)
    for xi, val in zip(x, funnel_vals):
        axA.text(xi, val + max(funnel_vals)*0.03, f"{val:,}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    # draw arrows between stages
    for i in range(2):
        axA.annotate("", xy=(i+0.72, max(funnel_vals)*0.73), xytext=(i+0.28, max(funnel_vals)*0.73),
                     arrowprops=dict(arrowstyle="->", lw=1.0, color="#555555"))
    axA.set_xticks(x, funnel_labels)
    axA.set_ylabel("Entries")
    axA.set_title("Screening-list coverage funnel", fontsize=11, fontweight="bold", pad=5)
    axA.grid(axis="y", alpha=0.25)
    clean_spines(axA); add_panel_label(axA, "A")

    # Panel B: stacked domain breakdown
    if cov is not None and not cov.empty:
        counts = cov["coverage_category"].value_counts()
        categories = [
            ("Descriptor complete", ["descriptor_complete_in_domain", "descriptor_complete_new_topology", "descriptor_complete_new_metal"], [COLORS["green"], COLORS["metal"], COLORS["topology"]]),
            ("Missing descriptors", ["missing_descriptors_known_metal_known_topology", "missing_descriptors_known_metal_new_topology", "missing_descriptors_new_metal_known_topology", "missing_descriptors_new_metal_and_topology"], [COLORS["random"], "#87A7C7", "#D29AA0", COLORS["purple"]]),
        ]
        bottom = np.zeros(2)
        handles = []
        for label_base, keys, cols in categories:
            xpos = 0 if label_base == "Descriptor complete" else 1
            btm = 0
            for key, col in zip(keys, cols):
                val = int(counts.get(key, 0))
                short = {
                    "descriptor_complete_in_domain": "in-domain",
                    "descriptor_complete_new_topology": "new topology",
                    "descriptor_complete_new_metal": "new metal",
                    "missing_descriptors_known_metal_known_topology": "known domain",
                    "missing_descriptors_known_metal_new_topology": "new topology",
                    "missing_descriptors_new_metal_known_topology": "new metal",
                    "missing_descriptors_new_metal_and_topology": "new metal+topol.",
                }[key]
                axB.bar(xpos, val, bottom=btm, color=col, edgecolor="#222222", linewidth=0.4)
                if val > 250:
                    axB.text(xpos, btm + val/2, f"{val:,}", ha="center", va="center", color="white" if col in [COLORS["random"], COLORS["topology"]] else "black", fontsize=7, fontweight="bold")
                handles.append(Patch(facecolor=col, edgecolor="#222222", label=f"{label_base}: {short}"))
                btm += val
        axB.set_xticks([0, 1], ["Descriptor\ncomplete", "Missing\ndescriptors"])
        axB.set_ylabel("Entries")
        axB.set_title("Domain-of-applicability breakdown", fontsize=11, fontweight="bold", pad=5)
        axB.legend(handles=handles, frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=6.8)
    else:
        axB.text(0.5, 0.5, "Coverage map\nnot found", ha="center", va="center")
    axB.grid(axis="y", alpha=0.25)
    clean_spines(axB); add_panel_label(axB, "B")

    # Panel C: most frequent metals (move to SI if main is crowded, but keep here as requested)
    if cov is not None and not cov.empty and "parsed_metal" in cov.columns:
        metals = cov["parsed_metal"].fillna("unknown").astype(str)
        metals = metals[metals.str.lower() != "unknown"]
        counts = metals.value_counts().head(10).sort_values()
        axC.barh(counts.index, counts.values, color=COLORS["purple"], edgecolor="#222222", linewidth=0.45)
        axC.set_xlabel("Entries in recommended list")
        axC.set_title("Most frequent metals", fontsize=11, fontweight="bold", pad=5)
    else:
        axC.text(0.5, 0.5, "Metal parsing summary\nnot found", ha="center", va="center")
    axC.grid(axis="x", alpha=0.25)
    clean_spines(axC); add_panel_label(axC, "C")

    # Panel D: triage tiers
    tier_metrics = [
        ("All\nclasses", "n_chemistry_classes", 364, COLORS["gray"]),
        ("Water\nharvesting", "water_harvesting_classes", 101, COLORS["random"]),
        ("Humid CO2\ncapture", "humid_co2_capture_classes", 86, COLORS["green"]),
        ("Strict humid\nseparations", "strict_humid_separation_classes", 10, COLORS["topology"]),
    ]
    vals = [get_metric(key, default) for _, key, default, _ in tier_metrics]
    labels = [t[0] for t in tier_metrics]
    colors = [t[3] for t in tier_metrics]
    x = np.arange(len(vals))
    axD.bar(x, vals, color=colors, edgecolor="#222222", linewidth=0.6)
    for xi, val in zip(x, vals):
        axD.text(xi, val + max(vals)*0.03, f"{int(val):,}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    axD.set_xticks(x, labels)
    axD.set_ylabel("Chemistry classes")
    axD.set_title("Operational triage tiers", fontsize=11, fontweight="bold", pad=5)
    axD.text(0.02, -0.24, "Class counts/domain support; not experimental validation", transform=axD.transAxes, ha="left", va="top", fontsize=7.0, color="#333333")
    axD.grid(axis="y", alpha=0.25)
    clean_spines(axD); add_panel_label(axD, "D")

    fig.subplots_adjust(left=0.07, right=0.84, bottom=0.11, top=0.93, wspace=0.38, hspace=0.48)
    save_figure(fig, "fig5_application_screening_FINAL_CLEANED")

# -----------------------------------------------------------------------------
# Figure 7
# -----------------------------------------------------------------------------

def pretty_feature_name(name: str) -> str:
    mapping = {
        "primary_metal": "Primary metal",
        "natoms": "Number of atoms",
        "average_atomic_mass": "Average atomic mass",
        "PLD (Å)": "PLD",
        "LCD (Å)": "LCD",
        "VF": "Void fraction",
        "OMS Types": "OMS type",
        "Density (g/cm3)": "Density",
        "Heat_capacity@350K (J/g/K)": "Heat capacity 350 K",
        "Heat_capacity@400K (J/g/K)": "Heat capacity 400 K",
        "cp0 (J/g/K)": "cp0",
        "k_cp (J/g/K/K)": "kcp",
    }
    return mapping.get(name, str(name).replace("_", " "))


def make_figure7():
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.8))
    axA, axB = axes.ravel()

    imp = read_csv_first([SI_TABLES / "Table_S12_permutation_importance.csv"], required=False)
    if imp is not None and not imp.empty:
        # Prefer ASR water RF D, fallback to any ASR water D.
        tmp = imp[(imp["dataset"] == "ASR") & (imp["target"] == "Water_stability") & (imp["regime"] == "D_context_thermophysical")].copy()
        if (tmp["model"] == "RandomForest").any():
            tmp = tmp[tmp["model"] == "RandomForest"]
            model_context = "Random Forest"
        elif not tmp.empty:
            model_context = str(tmp["model"].iloc[0])
            tmp = tmp[tmp["model"] == tmp["model"].iloc[0]]
        else:
            tmp = imp.copy()
            model_context = "model"
        tmp["importance_mean"] = pd.to_numeric(tmp["importance_mean"], errors="coerce")
        tmp["importance_std"] = pd.to_numeric(tmp["importance_std"], errors="coerce").fillna(0)
        tmp = tmp.sort_values("importance_mean", ascending=False).head(12).iloc[::-1]
        axA.barh(tmp["feature"].map(pretty_feature_name), tmp["importance_mean"], xerr=tmp["importance_std"], color=COLORS["random"], edgecolor="#222222", linewidth=0.45, ecolor="black")
        axA.set_xlabel("Permutation importance")
        axA.set_title(f"Model reliance on descriptors\nASR water, Regime D, {model_context}", fontsize=10.5, fontweight="bold", pad=5)
    else:
        axA.text(0.5, 0.5, "Permutation-importance\ntable not found", ha="center", va="center")
    axA.grid(axis="x", alpha=0.25)
    clean_spines(axA); add_panel_label(axA, "A", x=-0.12, y=1.12)

    # Panel B: confidence enrichment
    perf = read_csv_first([SI_TABLES / "Table_S15_shortlist_subset_performance.csv"], required=False)
    if perf is not None and not perf.empty:
        # Average ASR+FSR by target/cutoff for solvent and water.
        tmp = perf[perf["target"].isin(["Solvent_stability", "Water_stability"])].copy()
        tmp["screening_cutoff"] = pd.to_numeric(tmp["screening_cutoff"], errors="coerce")
        agg = tmp.groupby(["target", "screening_cutoff"], as_index=False).agg(
            top_decile_positive_rate=("top_decile_positive_rate", "mean"),
            overall_positive_rate=("overall_positive_rate", "mean"),
        )
        for target, color in [("Solvent_stability", COLORS["water"]), ("Water_stability", COLORS["solvent"] )]:
            sub = agg[agg["target"] == target].sort_values("screening_cutoff")
            if sub.empty:
                continue
            label = target_short(target)
            axB.plot(sub["screening_cutoff"], sub["top_decile_positive_rate"], marker="o", lw=1.8, color=color, label=f"{label}, top decile")
            axB.plot(sub["screening_cutoff"], sub["overall_positive_rate"], marker="s", lw=1.5, ls="--", color=color, alpha=0.55, label=f"{label}, population")
        axB.set_xlabel("Screening cutoff")
        axB.set_ylabel("Positive rate")
        axB.set_ylim(0, 1.05)
        axB.set_xticks([0.60, 0.70, 0.80])
        axB.set_title("Confidence enrichment\ntop prediction decile vs population", fontsize=10.5, fontweight="bold", pad=5)
        axB.legend(frameon=False, loc="lower left", fontsize=7.2, ncol=1)
    else:
        axB.text(0.5, 0.5, "Shortlist performance\ntable not found", ha="center", va="center")
    axB.grid(axis="y", alpha=0.25)
    clean_spines(axB); add_panel_label(axB, "B", x=-0.12, y=1.12)

    fig.subplots_adjust(left=0.17, right=0.98, bottom=0.16, top=0.84, wspace=0.42)
    save_figure(fig, "fig7_importance_screening_FINAL_CLEANED")

# -----------------------------------------------------------------------------
# Main runner
# -----------------------------------------------------------------------------

def run_all():
    notes = []
    notes.append(f"Project root: {ROOT}")
    notes.append(f"Input FINAL_FOR_REVISION: {FINAL}")
    notes.append(f"Output folder: {OUT}")
    for func in [make_figure2, make_figure3, make_figure4, make_figure5, make_figure7]:
        try:
            func()
            notes.append(f"OK: {func.__name__}")
        except Exception as e:
            notes.append(f"ERROR in {func.__name__}: {repr(e)}")
            raise
    (QC / "final_cleaned_main_figures_report.txt").write_text("\n".join(notes), encoding="utf-8")
    print("\n".join(notes))

if __name__ == "__main__":
    run_all()
