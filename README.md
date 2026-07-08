# MOF stability benchmark

Code and documentation for a reproducible descriptor-sufficiency benchmark of metal--organic framework (MOF) water, solvent-removal/activation, and thermal stability.

This repository accompanies the manuscript:

**How Far Can Simple Descriptors Go? A Chemistry-Led Benchmark for MOF Water and Activation Stability**

The repository is intended to be **lightweight and reproducible**. It should contain the workflow scripts, figure/table generation code, environment files, and data-provenance documentation. It should **not** redistribute the external raw database files or large generated prediction/output files.

## What is included

Recommended repository contents:

```text
README.md
CITATION.cff
LICENSE
requirements.txt
.gitignore
data_README.md
docs/
  DATA_PROVENANCE_AND_CITATIONS.md
  REPRODUCTION_GUIDE.md
  REPOSITORY_UPLOAD_GUIDE.md
  archive_inventory_upload_recommendations.csv
scripts/
  01_audit_scripts/
  02_main_sweep/
  03_post_processing/
  04_figures/
  05_focused_studies/
  final_revision_tools/
```

## What is not included

Do not commit these to the GitHub main branch:

- raw external database CSV files;
- large prediction tables such as `v14_predictions.csv`;
- pickled model/result objects;
- deprecated historical output folders;
- manuscript-private revision notes;
- generated figures/tables unless you intentionally want a small `examples/` folder.

The external raw input files should be downloaded from the official data sources described in `data_README.md` and `docs/DATA_PROVENANCE_AND_CITATIONS.md`.

## Required local input files

Place the following files locally in `02_raw_data/` or `raw_data/` before running the full workflow:

```text
ASR_data_SI_20250204.csv
FSR_data_SI_20250204.csv
ION_data_SI_20250204.csv
ASR_FSR_check.csv
12089-recommended-screening-list.csv
```

These files are from the CoRE MOF 2024 Zenodo dataset release and should be cited separately.

## Minimal workflow

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Place the external raw CSVs in `02_raw_data/`.

3. Run the group audit:

```bash
python scripts/01_audit_scripts/audit_groups_before_ml.py
```

4. Run the main sweep if you need to reproduce the full benchmark:

```bash
python scripts/02_main_sweep/v14_workflow.py
```

5. Rebuild the aggregated result table from predictions:

```bash
python scripts/03_post_processing/rebuild_results_from_predictions.py
```

6. Regenerate figures/tables:

```bash
python scripts/final_revision_tools/run_all_final_package_v2.py
python scripts/final_revision_tools/make_final_main_figures_STRICT_v4.py
```

## Notes on strict figure generation

The strict figure script intentionally avoids silent fallback behavior. If a required source CSV is missing, it stops rather than switching to a different table. This protects the scientific identity of panels such as the Figure 4 error analysis.

## Citation

If you use this repository, cite the associated manuscript and the external data sources listed in `docs/DATA_PROVENANCE_AND_CITATIONS.md`.
