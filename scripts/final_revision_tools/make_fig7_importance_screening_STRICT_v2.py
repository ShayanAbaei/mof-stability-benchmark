"""
Strict Figure 7 rebuild patch (v2)
----------------------------------

This script regenerates ONLY Figure 7 using pinned sources and an explicit target.
It exists because the old Figure 7 and the strict Figure 7 showed different
importance values. That difference was due to a target-context change:

- Old-looking Figure 7A values match ASR Thermal stability / Regime D / RandomForest.
- Strict Figure 7A uses ASR Water stability / Regime D / RandomForest.

Default below is WATER because Figure 7B discusses solvent/water screening
enrichment and the revised caption context previously asked for ASR water.
If you intentionally want the old-looking thermal importance panel, change:
    PANEL_A_TARGET = "Water_stability"
to:
    PANEL_A_TARGET = "Thermal_stability (℃)"

The script does not silently fall back to another target/model/source.
"""

from __future__ import annotations

from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
# -----------------------------
# Explicit scientific context
# -----------------------------
PANEL_A_DATASET = "ASR"
PANEL_A_TARGET = "Water_stability"          # change to "Thermal_stability (℃)" only if you want the old-looking thermal panel
PANEL_A_REGIME = "D_context_thermophysical"
PANEL_A_MODEL = "RandomForest"

# -----------------------------
# Locate project root
# -----------------------------
def find_project_root() -> Path:
    candidates = []
    here = Path.cwd().resolve()
    candidates += [here] + list(here.parents)
    try:
        sp = Path(__file__).resolve().parent
        candidates += [sp] + list(sp.parents)
    except Exception:
        pass
    seen = set()
    for c in candidates:
        if c in seen:
            continue
        seen.add(c)
        if (c / "04_outputs" / "FINAL_FOR_REVISION").exists():
            return c
    raise RuntimeError("Could not find project root containing 04_outputs/FINAL_FOR_REVISION")

ROOT = find_project_root()
FINAL = ROOT / "04_outputs" / "FINAL_FOR_REVISION"
SI_TABLES = FINAL / "03_si_tables"
QC = FINAL / "07_quality_control"
FIG_OUT = FINAL / "04_main_figures_FINAL_STRICT"
FIG_OUT.mkdir(parents=True, exist_ok=True)
QC.mkdir(parents=True, exist_ok=True)

IMP_PATH = SI_TABLES / "Table_S12_permutation_importance.csv"
PERF_PATH = SI_TABLES / "Table_S15_shortlist_subset_performance.csv"

# -----------------------------
# Style
# -----------------------------
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 1.0,
    "axes.edgecolor": "black",
})

COLORS = {
    "solvent": "#D9893D",
    "water": "#3A73A5",
    "random": "#3A73A5",
}
TARGET_SHORT = {
    "Solvent_stability": "Solvent",
    "Water_stability": "Water",
    "Thermal_stability (℃)": "Thermal",
    "Thermal_stability (°C)": "Thermal",
}

# -----------------------------
# Helpers
# -----------------------------
def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return path


def normalize_model(s) -> str:
    if pd.isna(s):
        return ""
    s = str(s).strip()
    return {
        "RF": "RandomForest",
        "Random Forest": "RandomForest",
        "random_forest": "RandomForest",
        "RandomForest": "RandomForest",
        "LightGBM": "LightGBM",
        "LGBM": "LightGBM",
        "lightgbm": "LightGBM",
    }.get(s, s)


def pretty_feature_name(name) -> str:
    if pd.isna(name):
        return ""
    s = str(name).strip()
    mapping = {
        "primary_metal": "Primary metal",
        "metal": "Primary metal",
        "Metal Types": "Primary metal",
        "natoms": "Number of atoms",
        "average_atomic_mass": "Average atomic mass",
        "PLD (Å)": "PLD",
        "LCD (Å)": "LCD",
        "VF": "Void fraction",
        "PV (A3)": "Pore volume",
        "PV (cm3/g)": "Pore volume",
        "Density (g/cm3)": "Density",
        "OMS Types": "OMS type",
        "Has OMS": "OMS type",
        "Heat_capacity@300K (J/g/K)": "Heat capacity 300 K",
        "Heat_capacity@350K (J/g/K)": "Heat capacity 350 K",
        "Heat_capacity@400K (J/g/K)": "Heat capacity 400 K",
        "cp0 (J/g/K)": "cp0",
        "k_cp (J/g/K/K)": "kcp",
        "topology(AllNodes)": "Topology (AllNodes)",
        "topology(SingleNodes)": "Topology (SingleNodes)",
    }
    if s in mapping:
        return mapping[s]
    # generic cleanup for unseen names
    s = re.sub(r"Heat_capacity@(\d+)K.*", r"Heat capacity \1 K", s)
    s = s.replace("_", " ")
    return s


def clean_spines(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_panel_label(ax, label: str, x=-0.11, y=1.10):
    ax.text(x, y, label, transform=ax.transAxes, ha="left", va="top", fontsize=14, fontweight="bold", clip_on=False)


def target_title(t: str) -> str:
    return TARGET_SHORT.get(t, t.replace("_", " "))

# -----------------------------
# Build Figure 7
# -----------------------------
def make_figure7():
    imp = pd.read_csv(require(IMP_PATH))
    perf = pd.read_csv(require(PERF_PATH))

    imp["model_norm"] = imp["model"].map(normalize_model)
    tmp = imp[
        (imp["dataset"] == PANEL_A_DATASET)
        & (imp["target"] == PANEL_A_TARGET)
        & (imp["regime"] == PANEL_A_REGIME)
        & (imp["model_norm"] == PANEL_A_MODEL)
    ].copy()
    if tmp.empty:
        available = imp[["dataset", "target", "regime", "model"]].drop_duplicates().to_string(index=False)
        raise RuntimeError(
            "No rows found for explicit Figure 7A context:\n"
            f"dataset={PANEL_A_DATASET}, target={PANEL_A_TARGET}, regime={PANEL_A_REGIME}, model={PANEL_A_MODEL}\n\n"
            f"Available contexts:\n{available}"
        )

    tmp["importance_mean"] = pd.to_numeric(tmp["importance_mean"], errors="coerce")
    tmp["importance_std"] = pd.to_numeric(tmp["importance_std"], errors="coerce").fillna(0)
    tmp = tmp.dropna(subset=["importance_mean"]).sort_values("importance_mean", ascending=False).head(12).iloc[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.0))
    axA, axB = axes.ravel()

    # Panel A
    axA.barh(
        tmp["feature"].map(pretty_feature_name),
        tmp["importance_mean"],
        xerr=tmp["importance_std"],
        color=COLORS["random"],
        edgecolor="#222222",
        linewidth=0.45,
        ecolor="black",
    )
    axA.set_xlabel("Permutation importance")
    axA.set_title(
        f"Model reliance on descriptors\n{PANEL_A_DATASET} {target_title(PANEL_A_TARGET).lower()}, Regime D, Random Forest",
        fontsize=10.5,
        fontweight="bold",
        pad=4,
    )
    axA.grid(axis="x", alpha=0.25)
    clean_spines(axA)
    add_panel_label(axA, "A")

    # Panel B: confidence enrichment, ASR+FSR averaged for solvent/water.
    d = perf[perf["target"].isin(["Solvent_stability", "Water_stability"])].copy()
    d["screening_cutoff"] = pd.to_numeric(d["screening_cutoff"], errors="coerce")
    agg = d.groupby(["target", "screening_cutoff"], as_index=False).agg(
        top_decile_positive_rate=("top_decile_positive_rate", "mean"),
        overall_positive_rate=("overall_positive_rate", "mean"),
    )

    for target, color in [("Solvent_stability", COLORS["solvent"]), ("Water_stability", COLORS["water"])]:
        sub = agg[agg["target"] == target].sort_values("screening_cutoff")
        if sub.empty:
            continue
        axB.plot(sub["screening_cutoff"], sub["top_decile_positive_rate"], marker="o", lw=1.8, color=color)
        axB.plot(sub["screening_cutoff"], sub["overall_positive_rate"], marker="s", lw=1.5, ls="--", color=color, alpha=0.55)

    axB.set_xlabel("Screening cutoff")
    axB.set_ylabel("Positive rate")
    axB.set_ylim(0, 1.05)
    axB.set_xticks([0.60, 0.70, 0.80])
    axB.set_title("Confidence enrichment\ntop prediction decile vs population", fontsize=10.5, fontweight="bold", pad=4)
    axB.grid(axis="y", alpha=0.25)
    clean_spines(axB)
    add_panel_label(axB, "B")

    # Compact, non-overlapping legend: color = target; line style = subset.
    handles = [
        Line2D([0], [0], color=COLORS["solvent"], lw=2, label="Solvent"),
        Line2D([0], [0], color=COLORS["water"], lw=2, label="Water"),
        Line2D([0], [0], color="#333333", lw=2, ls="-", label="top decile"),
        Line2D([0], [0], color="#333333", lw=2, ls="--", label="population"),
    ]
    axB.legend(handles=handles, frameon=False, loc="lower left", fontsize=7.3, ncol=2, columnspacing=0.9, handlelength=1.6)

    fig.subplots_adjust(left=0.17, right=0.98, bottom=0.18, top=0.84, wspace=0.42)

    stem = "fig7_importance_screening_FINAL_STRICT"
    for ext in ["pdf", "png", "svg"]:
        dpi = 600 if ext == "png" else None
        fig.savefig(FIG_OUT / f"{stem}.{ext}", bbox_inches="tight", dpi=dpi, facecolor="white")
    plt.close(fig)

    report = [
        "Figure 7 strict v2 source report",
        "=================================",
        f"Project root: {ROOT}",
        f"Panel A source: {IMP_PATH}",
        f"Panel A context: dataset={PANEL_A_DATASET}; target={PANEL_A_TARGET}; regime={PANEL_A_REGIME}; model={PANEL_A_MODEL}",
        f"Panel B source: {PERF_PATH}",
        "Panel B context: ASR+FSR averaged, solvent/water, top prediction decile vs full population",
        "",
        "Top Panel A features:",
    ]
    for _, r in tmp.iloc[::-1].iterrows():
        report.append(f"- {r['feature']}: {r['importance_mean']:.6f}")
    (QC / "fig7_strict_v2_source_report.txt").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report[:8]))


if __name__ == "__main__":
    make_figure7()
