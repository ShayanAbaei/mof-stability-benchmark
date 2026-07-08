"""
make_figS14_application_expanded.py

Supplementary Figure S14 generator for the JMCA MOF stability revision.

Purpose
-------
This script gives TWO S14 options:

1) FAST OPTION
   Copies the main-text Figure 5 application-screening figure into the SI as
   figS14_fast_application_screening.*. This keeps the old/fast option available.

2) BETTER OPTION
   Generates a fuller SI-only application-screening figure:
   figS14_application_expanded_shortlist.*

The better S14 is designed to avoid repeating main Figure 5. It adds a fuller
application-screening diagnostic: coverage categories, a metal-by-domain heatmap,
top descriptor-complete chemistry classes, and operational triage class counts.

Where to put this file
----------------------
Place this file here:

    jmca_handoff_mosiu/99_final_tools_v2/make_figS14_application_expanded.py

How to run from Spyder
----------------------
First run the v2 package:

    %run 99_final_tools_v2/run_all_final_package_v2.py

Then run this file:

    %run 99_final_tools_v2/make_figS14_application_expanded.py

Outputs
-------
Writes SI figures to:

    04_outputs/FINAL_FOR_REVISION/05_si_figures/

Writes supporting SI tables/source data to:

    04_outputs/FINAL_FOR_REVISION/03_si_tables/
    04_outputs/FINAL_FOR_REVISION/06_source_data_for_figures/

Notes
-----
- This script does NOT rerun ML.
- It uses the already-generated application coverage map from the final package.
- If a true chemistry-class shortlist CSV is present in application_screening_outputs,
  the script will copy/export it. If not, it builds a support-based chemistry-class
  table from the 12,089-entry coverage map. That support table is not a predicted
  stability ranking; it is a coverage/domain-support diagnostic.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# =============================================================================
# PATHS
# =============================================================================

TOOL_DIR = Path(__file__).resolve().parent
# If placed inside 99_final_tools_v2, project root is the parent.
ROOT = TOOL_DIR.parent if TOOL_DIR.name.startswith("99_") else Path.cwd()
OUT = ROOT / "04_outputs"
FINAL = OUT / "FINAL_FOR_REVISION"
SI_FIGS = FINAL / "05_si_figures"
MAIN_FIGS = FINAL / "04_main_figures"
SI_TABLES = FINAL / "03_si_tables"
SOURCE = FINAL / "06_source_data_for_figures"
QC = FINAL / "07_quality_control"

for p in [SI_FIGS, SI_TABLES, SOURCE, QC]:
    p.mkdir(parents=True, exist_ok=True)

# =============================================================================
# STYLE
# =============================================================================

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8.0,
    "axes.labelsize": 8.0,
    "axes.titlesize": 8.8,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "legend.fontsize": 6.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.75,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

COL = {
    "gray": "#6E6E6E",
    "blue": "#356C9B",
    "orange": "#D9853B",
    "green": "#3F8F7A",
    "red": "#B74D4D",
    "purple": "#7A68A6",
    "gold": "#D4A33D",
    "lightgray": "#E8E8E8",
    "dark": "#222222",
}

CATEGORY_ORDER = [
    "descriptor_complete_in_domain",
    "descriptor_complete_new_topology",
    "descriptor_complete_new_metal",
    "missing_descriptors_known_metal_known_topology",
    "missing_descriptors_known_metal_new_topology",
    "missing_descriptors_new_metal_known_topology",
    "missing_descriptors_new_metal_and_topology",
]

CATEGORY_LABEL = {
    "descriptor_complete_in_domain": "Complete\nin-domain",
    "descriptor_complete_new_topology": "Complete\nnew topology",
    "descriptor_complete_new_metal": "Complete\nnew metal",
    "missing_descriptors_known_metal_known_topology": "Missing desc.\nknown domain",
    "missing_descriptors_known_metal_new_topology": "Missing desc.\nnew topology",
    "missing_descriptors_new_metal_known_topology": "Missing desc.\nnew metal",
    "missing_descriptors_new_metal_and_topology": "Missing desc.\nnew metal+topol.",
}

CATEGORY_COLOR = {
    "descriptor_complete_in_domain": COL["green"],
    "descriptor_complete_new_topology": COL["orange"],
    "descriptor_complete_new_metal": COL["red"],
    "missing_descriptors_known_metal_known_topology": COL["blue"],
    "missing_descriptors_known_metal_new_topology": "#8AA8C8",
    "missing_descriptors_new_metal_known_topology": "#C98E8E",
    "missing_descriptors_new_metal_and_topology": COL["purple"],
}

# =============================================================================
# HELPERS
# =============================================================================

def add_panel_label(ax, label: str, x=-0.10, y=1.04):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=11.5,
            fontweight="bold", va="bottom", ha="left")


def save_figure(fig, basename: str):
    for ext in ["pdf", "png", "svg"]:
        fig.savefig(SI_FIGS / f"{basename}.{ext}")
    plt.close(fig)


def read_csv_first(paths) -> Optional[pd.DataFrame]:
    for p in paths:
        p = Path(p)
        if p.exists():
            return pd.read_csv(p)
    return None


def find_shortlist_file() -> Optional[Path]:
    """Find a true chemistry-class shortlist file if Hosein's application script produced one."""
    candidates = []
    search_dirs = [
        OUT / "application_screening_outputs",
        FINAL / "03_si_tables",
        FINAL / "06_source_data_for_figures",
        OUT,
    ]
    for d in search_dirs:
        if d.exists():
            candidates.extend(list(d.glob("*chemistry*class*shortlist*.csv")))
            candidates.extend(list(d.glob("*shortlist*chemistry*class*.csv")))
            candidates.extend(list(d.glob("*descriptor_complete*shortlist*.csv")))
    if not candidates:
        return None
    # Prefer the most specific / largest file.
    candidates = sorted(candidates, key=lambda p: ("chemistry" not in p.name.lower(), -p.stat().st_size, p.name))
    return candidates[0]


def load_coverage_map() -> pd.DataFrame:
    paths = [
        SOURCE / "fig5_coverage_map_source.csv",
        OUT / "application_screening_outputs" / "v14_12089_coverage_map.csv",
        OUT / "application_screening_outputs" / "coverage_map.csv",
        SI_TABLES / "Table_S14_application_coverage_map.csv",
    ]
    df = read_csv_first(paths)
    if df is None:
        raise FileNotFoundError(
            "Could not find the application coverage map. Run run_all_final_package_v2.py first.\n"
            "Expected one of:\n" + "\n".join(str(p) for p in paths)
        )
    required = {"coverage_category", "parsed_metal", "parsed_topology", "descriptor_complete_ASR_FSR"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Coverage map is missing required columns: {missing}")
    return df


def load_summary_metrics() -> pd.DataFrame:
    paths = [
        SI_TABLES / "Table_S14_application_coverage_full_summary.csv",
        SOURCE / "fig5_application_summary_source.csv",
        OUT / "application_screening_outputs" / "v14_12089_coverage_summary.csv",
    ]
    df = read_csv_first(paths)
    if df is None:
        raise FileNotFoundError("Could not find application coverage summary. Run v2 package first.")
    return df


def metric_value(summary: pd.DataFrame, key: str, default=np.nan):
    if "metric" not in summary.columns or "value" not in summary.columns:
        return default
    s = summary.loc[summary["metric"].astype(str) == key, "value"]
    return s.iloc[0] if len(s) else default


def clean_topology(x) -> str:
    if pd.isna(x) or str(x).lower() in ["nan", "none", "", "null"]:
        return "unknown"
    return str(x)


def clean_metal(x) -> str:
    if pd.isna(x) or str(x).lower() in ["nan", "none", "", "null"]:
        return "unknown"
    return str(x)


def build_chemistry_class_support(cov: pd.DataFrame) -> pd.DataFrame:
    """Build a chemistry-class support table from the coverage map.

    This is a coverage/domain support table, not a predicted stability ranking.
    """
    work = cov.copy()
    work["parsed_metal"] = work["parsed_metal"].map(clean_metal)
    work["parsed_topology"] = work["parsed_topology"].map(clean_topology)
    work["chemistry_class"] = work["parsed_metal"] + "--" + work["parsed_topology"]

    g = (work.groupby(["chemistry_class", "parsed_metal", "parsed_topology", "coverage_category"], dropna=False)
            .size().reset_index(name="n_entries"))
    wide = g.pivot_table(index=["chemistry_class", "parsed_metal", "parsed_topology"],
                         columns="coverage_category", values="n_entries", fill_value=0, aggfunc="sum")
    wide = wide.reset_index()
    for cat in CATEGORY_ORDER:
        if cat not in wide.columns:
            wide[cat] = 0
    wide["total_entries"] = wide[CATEGORY_ORDER].sum(axis=1)
    wide["descriptor_complete_entries"] = (wide["descriptor_complete_in_domain"] +
                                            wide["descriptor_complete_new_topology"] +
                                            wide["descriptor_complete_new_metal"])
    wide["domain_supported_entries"] = wide["descriptor_complete_in_domain"]
    wide["descriptor_completion_fraction"] = np.where(
        wide["total_entries"] > 0, wide["descriptor_complete_entries"] / wide["total_entries"], np.nan
    )
    wide = wide.sort_values(["descriptor_complete_entries", "domain_supported_entries", "total_entries"], ascending=False)
    return wide


def copy_fast_option():
    """Keep the old/fast S14 option by copying main Figure 5 to SI."""
    copied = False
    for ext in ["pdf", "png", "svg"]:
        src = MAIN_FIGS / f"fig5_application_screening.{ext}"
        if src.exists():
            dst = SI_FIGS / f"figS14_fast_application_screening.{ext}"
            shutil.copy2(src, dst)
            copied = True
    return copied

# =============================================================================
# MAIN BETTER S14 FIGURE
# =============================================================================

def make_better_s14():
    cov = load_coverage_map()
    summary = load_summary_metrics()
    cov = cov.copy()
    cov["parsed_metal"] = cov["parsed_metal"].map(clean_metal)
    cov["parsed_topology"] = cov["parsed_topology"].map(clean_topology)

    # Export coverage-by-category table.
    cat_counts = (cov["coverage_category"].value_counts()
                  .reindex(CATEGORY_ORDER).dropna().astype(int).reset_index())
    cat_counts.columns = ["coverage_category", "n_entries"]
    cat_counts["label"] = cat_counts["coverage_category"].map(CATEGORY_LABEL)
    cat_counts.to_csv(SI_TABLES / "Table_S14b_coverage_category_counts.csv", index=False)
    cat_counts.to_csv(SOURCE / "figS14_coverage_category_counts_source.csv", index=False)

    # Top metals x categories heatmap.
    top_metals = cov["parsed_metal"].value_counts().head(14).index.tolist()
    metal_cat = (cov[cov["parsed_metal"].isin(top_metals)]
                 .pivot_table(index="parsed_metal", columns="coverage_category", values="core-mof-id",
                              aggfunc="count", fill_value=0)
                 .reindex(index=top_metals, columns=CATEGORY_ORDER, fill_value=0))
    metal_cat.to_csv(SI_TABLES / "Table_S14c_coverage_by_top_metal.csv")
    metal_cat.to_csv(SOURCE / "figS14_metal_category_heatmap_source.csv")

    # Chemistry class support. If a true predicted shortlist exists, also export it.
    shortlist_file = find_shortlist_file()
    true_shortlist = None
    if shortlist_file is not None:
        true_shortlist = pd.read_csv(shortlist_file)
        true_shortlist.to_csv(SI_TABLES / "Table_S15_full_chemistry_class_shortlist_detected.csv", index=False)

    class_support = build_chemistry_class_support(cov)
    class_support.to_csv(SI_TABLES / "Table_S15b_chemistry_class_domain_support_from_12089.csv", index=False)
    class_support.to_csv(SOURCE / "figS14_chemistry_class_support_source.csv", index=False)

    # Figure layout.
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.9))
    axA, axB, axC, axD = axes.ravel()

    # Panel A: coverage categories.
    labels = cat_counts["label"].tolist()
    vals = cat_counts["n_entries"].tolist()
    colors = [CATEGORY_COLOR.get(c, COL["gray"]) for c in cat_counts["coverage_category"]]
    x = np.arange(len(vals))
    axA.bar(x, vals, color=colors, edgecolor="black", linewidth=0.35)
    for xi, v in zip(x, vals):
        axA.text(xi, v + max(vals)*0.018, f"{int(v):,}", ha="center", va="bottom", fontsize=6.3, rotation=0)
    axA.set_xticks(x)
    axA.set_xticklabels(labels, rotation=38, ha="right", fontsize=6.2)
    axA.set_ylabel("Entries")
    axA.set_title("Coverage categories in the 12,089-entry list", fontweight="bold")
    axA.grid(axis="y", alpha=0.22)
    add_panel_label(axA, "A", x=-0.12, y=1.04)

    # Panel B: heatmap by top metal.
    heat = np.log10(metal_cat.values.astype(float) + 1.0)
    im = axB.imshow(heat, aspect="auto", cmap="YlGnBu", vmin=0)
    axB.set_yticks(np.arange(len(metal_cat.index)))
    axB.set_yticklabels(metal_cat.index)
    axB.set_xticks(np.arange(len(CATEGORY_ORDER)))
    axB.set_xticklabels([CATEGORY_LABEL[c] for c in CATEGORY_ORDER], rotation=42, ha="right", fontsize=5.7)
    axB.set_title("Top metals by coverage domain", fontweight="bold")
    for i in range(metal_cat.shape[0]):
        for j in range(metal_cat.shape[1]):
            val = int(metal_cat.values[i, j])
            if val > 0:
                axB.text(j, i, str(val), ha="center", va="center", fontsize=5.5,
                         color="white" if heat[i, j] > heat.max()*0.55 else "black")
    cbar = fig.colorbar(im, ax=axB, fraction=0.046, pad=0.02)
    cbar.set_label("log10(entries + 1)", labelpad=3)
    add_panel_label(axB, "B", x=-0.14, y=1.04)

    # Panel C: top descriptor-complete chemistry classes, stacked by domain category.
    top_classes = class_support[class_support["descriptor_complete_entries"] > 0].head(18).copy()
    if top_classes.empty:
        top_classes = class_support.head(18).copy()
    y = np.arange(len(top_classes))
    left = np.zeros(len(top_classes))
    stack_cats = ["descriptor_complete_in_domain", "descriptor_complete_new_topology", "descriptor_complete_new_metal"]
    for cat in stack_cats:
        vals = top_classes[cat].values.astype(float)
        axC.barh(y, vals, left=left, color=CATEGORY_COLOR[cat], edgecolor="black", linewidth=0.25,
                 label=CATEGORY_LABEL[cat].replace("\n", " "))
        left += vals
    axC.set_yticks(y)
    axC.set_yticklabels(top_classes["chemistry_class"].tolist(), fontsize=6.5)
    axC.invert_yaxis()
    axC.set_xlabel("Descriptor-complete entries")
    axC.set_title("Top supported metal--topology classes", fontweight="bold")
    axC.legend(frameon=False, ncol=1, loc="lower right", fontsize=6.0)
    axC.grid(axis="x", alpha=0.22)
    add_panel_label(axC, "C", x=-0.12, y=1.04)

    # Panel D: operational triage tiers and threshold enrichment.
    tiers = [
        ("All\nclasses", "chemistry_classes_total", COL["gray"]),
        ("Water\nharvesting", "chemistry_classes_water_harvesting", COL["blue"]),
        ("Humid CO$_2$ \ncapture", "chemistry_classes_humid_co2_capture", COL["green"]),
        ("Strict humid\nseparations", "chemistry_classes_humid_separations", COL["red"]),
    ]
    tier_vals = [metric_value(summary, k, np.nan) for _, k, _ in tiers]
    x = np.arange(len(tiers))
    axD.bar(x, tier_vals, color=[c for _, _, c in tiers], edgecolor="black", linewidth=0.35)
    for xi, v in zip(x, tier_vals):
        if pd.notna(v):
            axD.text(xi, v + max([vv for vv in tier_vals if pd.notna(vv)])*0.025, f"{int(v):,}",
                     ha="center", va="bottom", fontsize=7, fontweight="bold")
    axD.set_xticks(x)
    axD.set_xticklabels([lab for lab, _, _ in tiers], fontsize=6.6)
    axD.set_ylabel("Chemistry classes")
    axD.set_title("Operational triage tiers", fontweight="bold")
    axD.grid(axis="y", alpha=0.22)



    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.090, top=0.955, wspace=0.42, hspace=0.4)
    save_figure(fig, "figS14_application_expanded_shortlist")

    # Write a small QC/readme note.
    note_lines = [
        "Figure S14 application expanded generated.",
        f"Coverage map rows: {len(cov)}",
        f"True predicted chemistry-class shortlist CSV found: {shortlist_file if shortlist_file else 'NO'}",
        "If no predicted shortlist was found, Panel C is a domain-support plot from the 12,089-entry coverage map, not a stability-ranked shortlist.",
        "Use the full Table_S15b_chemistry_class_domain_support_from_12089.csv in the SI if the predicted class shortlist is absent.",
    ]
    (QC / "figS14_application_expanded_QC.txt").write_text("\n".join(note_lines), encoding="utf-8")


def make_s14_application_figures():
    copied = copy_fast_option()
    make_better_s14()
    print("[S14] Fast option copied:", copied)
    print("[S14] Better option written to:", SI_FIGS / "figS14_application_expanded_shortlist.pdf")
    print("[S14] Supporting tables written to:", SI_TABLES)


if __name__ == "__main__":
    make_s14_application_figures()
