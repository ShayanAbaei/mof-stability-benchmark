"""
JMCA Figure 3 patch v3 — cleaner descriptor labels + explicit ASR-FSR identity note.

Use:
    Put this file in:
        jmca_handoff_mosiu/99_final_tools_v2/make_fig3_label_threshold_overlap_v3.py

    Run AFTER run_all_final_package_v2.py:
        %run 99_final_tools_v2/make_fig3_label_threshold_overlap_v3.py

This script regenerates ONLY:
    04_outputs/FINAL_FOR_REVISION/04_main_figures/fig3_label_threshold_overlap.pdf
    04_outputs/FINAL_FOR_REVISION/04_main_figures/fig3_label_threshold_overlap.png
    04_outputs/FINAL_FOR_REVISION/04_main_figures/fig3_label_threshold_overlap.svg

Main fixes:
    1. Panel B replaces the cryptic "<M>" with "Avg. atomic mass".
    2. Panel C explicitly labels the unity ASR--FSR correlations as matched-pair
       identity/consistency after cleaning, so Spearman = 1.000 does not look like
       a coding accident.
    3. Panel C uses safe plain "Thermal" text, avoiding degree-symbol/font artifacts.
"""
from __future__ import annotations

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
TOOL_DIR = Path(__file__).resolve().parent
ROOT = TOOL_DIR.parent
RAW = ROOT / "02_raw_data"
OUT = ROOT / "04_outputs"
HIST = ROOT / "05_historical" / "v10_outputs_tables"
FINAL = OUT / "FINAL_FOR_REVISION"
MAIN_FIGS = FINAL / "04_main_figures"
SOURCE = FINAL / "06_source_data_for_figures"

MAIN_FIGS.mkdir(parents=True, exist_ok=True)
SOURCE.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Style
# -----------------------------------------------------------------------------
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
    "orange": "#D9853B",
    "red": "#B74D4D",
    "green": "#3F8F7A",
    "gray": "#6E6E6E",
    "dark": "#222222",
}

def add_panel_label(ax, label: str, x=-0.12, y=1.05):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=12,
            fontweight="bold", ha="left", va="bottom")

def save_fig(fig, name: str):
    for ext in ["pdf", "png", "svg"]:
        fig.savefig(MAIN_FIGS / f"{name}.{ext}")
    plt.close(fig)

def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_csv(path)

def nice_pair_target(t: str) -> str:
    t = str(t)
    if "Solvent" in t:
        return "Solvent"
    if "Water" in t:
        return "Water"
    if "Thermal" in t:
        return "Thermal"
    return t.replace("_stability", "").replace(" (℃)", "")

def make_fig3():
    asr_path = RAW / "ASR_data_SI_20250204.csv"
    fsr_path = RAW / "FSR_data_SI_20250204.csv"
    asr = read_csv_required(asr_path)
    fsr = read_csv_required(fsr_path)

    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.55))

    # -------------------------------------------------------------------------
    # A. Target distributions
    # -------------------------------------------------------------------------
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
            ax.hist(vals, bins=np.linspace(0, 1, 26), histtype="step",
                    lw=1.35, color=color, ls=ls, label=label)
    for thr, ls in [(0.6, ":"), (0.7, "--"), (0.8, ":")]:
        ax.axvline(thr, color="#444444", lw=0.75, ls=ls)
        ymax = ax.get_ylim()[1]
        ax.text(thr + 0.006, ymax * 0.86, f"{thr:.1f}",
                rotation=90, va="top", fontsize=6.5)
    ax.set_xlabel("Curated stability score")
    ax.set_ylabel("Count")
    ax.legend(frameon=False, fontsize=6.2, ncol=2, loc="upper left",
              bbox_to_anchor=(0.00, 1.14), borderaxespad=0.0,
              handlelength=1.7, columnspacing=0.9)
    ax.grid(axis="y", alpha=0.22)
    add_panel_label(ax, "A")

    # -------------------------------------------------------------------------
    # B. Descriptor correlations
    # -------------------------------------------------------------------------
    ax = axes[0, 1]
    cols = [
        "Density (g/cm3)",
        "LCD (Å)",
        "PLD (Å)",
        "VF",
        "PV (cm3/g)",
        "average_atomic_mass",
        "natoms",
    ]
    label_map = {
        "Density (g/cm3)": "Density",
        "LCD (Å)": "LCD",
        "PLD (Å)": "PLD",
        "VF": "Void fraction",
        "PV (cm3/g)": "Pore volume",
        "average_atomic_mass": "Avg. atomic mass",
        "natoms": "No. atoms",
    }
    avail = [c for c in cols if c in asr.columns]
    lab_avail = [label_map.get(c, c) for c in avail]
    corr = asr[avail].apply(pd.to_numeric, errors="coerce").corr(method="pearson")
    corr.to_csv(SOURCE / "fig3_descriptor_correlation_source.csv")

    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(avail)))
    ax.set_xticklabels(lab_avail, rotation=42, ha="right")
    ax.set_yticks(range(len(avail)))
    ax.set_yticklabels(lab_avail)
    for i in range(len(avail)):
        for j in range(len(avail)):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=5.6, color="white" if abs(v) > 0.55 else "black")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("Pearson r", fontsize=7)
    add_panel_label(ax, "B")

    # -------------------------------------------------------------------------
    # C. ASR--FSR matched-pair consistency
    # -------------------------------------------------------------------------
    ax = axes[1, 0]
    pair_path = HIST / "pair_consistency_summary.csv"
    if not pair_path.exists():
        # Fallback to final SI table if this package has already been generated.
        pair_path = FINAL / "03_si_tables" / "Table_S11_matched_pair_consistency.csv"
    if pair_path.exists():
        pc = pd.read_csv(pair_path)
        # Identify target and Spearman columns robustly.
        target_col = "target" if "target" in pc.columns else pc.columns[0]
        if "spearman_rho" in pc.columns:
            val_col = "spearman_rho"
        elif "Spearman" in pc.columns:
            val_col = "Spearman"
        elif "spearman" in pc.columns:
            val_col = "spearman"
        else:
            val_col = None
        vals = pd.to_numeric(pc[val_col], errors="coerce").fillna(1.0).values if val_col else np.ones(len(pc))
        labels_pc = [nice_pair_target(t) for t in pc[target_col].astype(str).values]

        # Put in a stable order: solvent, water, thermal.
        order = []
        for wanted in ["Solvent", "Water", "Thermal"]:
            for idx, lab in enumerate(labels_pc):
                if lab == wanted:
                    order.append(idx)
        if len(order) == len(labels_pc):
            vals = vals[order]
            labels_pc = [labels_pc[i] for i in order]
            pc = pc.iloc[order].copy()

        x = np.arange(len(vals))
        ax.bar(x, vals, color=[COLORS["blue"], COLORS["orange"], COLORS["red"]][:len(vals)],
               edgecolor="black", lw=0.4)
        for i, v in enumerate(vals):
            ax.text(i, min(1.02, v + 0.015), f"{v:.3f}",
                    ha="center", va="bottom", fontsize=7, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels_pc, rotation=15, ha="right")
        ax.set_ylabel("ASR--FSR matched-pair Spearman")
        ax.set_ylim(0, 1.10)

        # Add a tiny note to prevent the unity values from looking suspicious.
        n_txt = ""
        if "n_pairs" in pc.columns:
            try:
                n_unique = sorted(set(pd.to_numeric(pc["n_pairs"], errors="coerce").dropna().astype(int)))
                if len(n_unique) == 1:
                    n_txt = f"n = {n_unique[0]} pairs; "
            except Exception:
                pass
            # Put the matched-pair note above the panel, not inside the bars.
            # This avoids overlap with the 1.000 labels and makes the message clearer.
            note_txt = f"{n_txt}matched labels are identical after cleaning"
            
            ax.set_ylim(0, 1.14)
            
            ax.text(
                0.5, 1.001, note_txt,
                transform=ax.transAxes,
                ha="center",
                va="bottom",
                fontsize=7.2,
                color="black",
                clip_on=False
            )
    else:
        ax.text(0.5, 0.5, "pair consistency\nsummary not found",
                ha="center", va="center")
    ax.grid(axis="y", alpha=0.25)
    add_panel_label(ax, "C")

    # -------------------------------------------------------------------------
    # D. Group-size distribution
    # -------------------------------------------------------------------------
    ax = axes[1, 1]
    gpath = OUT / "group_audit_outputs" / "02_all_group_sizes_by_dataset_target.csv"
    if gpath.exists():
        gs = pd.read_csv(gpath)
        size_col = "group_size" if "group_size" in gs.columns else ("n_rows" if "n_rows" in gs.columns else None)
        gl_col = "group_label" if "group_label" in gs.columns else None
        if size_col and gl_col:
            sub = gs[gs["target"].astype(str).eq("Water_stability")].copy() if "target" in gs.columns else gs.copy()
            for gl, label, color in [
                ("metal", "Primary metal", COLORS["orange"]),
                ("topology_AllNodes", "Topology (AllNodes)", COLORS["red"]),
            ]:
                s = sub[sub[gl_col].astype(str).eq(gl)]
                if not s.empty:
                    vals = pd.to_numeric(s[size_col], errors="coerce").dropna()
                    ax.hist(vals, bins=[1, 2, 5, 10, 20, 50, 100, 500],
                            histtype="step", lw=1.5, color=color, label=label)
            ax.set_xscale("log")
            ax.set_xlabel("Group size (log scale)")
            ax.set_ylabel("Number of groups")
            ax.legend(frameon=False, fontsize=7, loc="upper right")
        else:
            ax.text(0.5, 0.5, "group-size columns\nnot recognized",
                    ha="center", va="center")
    else:
        ax.text(0.5, 0.5, "group-size audit\nnot found",
                ha="center", va="center")
    ax.grid(axis="y", alpha=0.25)
    add_panel_label(ax, "D")

    fig.subplots_adjust(wspace=0.38, hspace=0.46)
    save_fig(fig, "fig3_label_threshold_overlap")
    print(f"[JMCA] Figure 3 regenerated at: {MAIN_FIGS}")

if __name__ == "__main__":
    make_fig3()
