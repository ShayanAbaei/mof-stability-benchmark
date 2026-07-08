# Reproduction guide

## 1. Prepare external raw data

Download the required raw CSV files from the official CoRE MOF 2024 Zenodo release and place them in `02_raw_data/`.

Required files:

```text
ASR_data_SI_20250204.csv
FSR_data_SI_20250204.csv
ION_data_SI_20250204.csv
ASR_FSR_check.csv
12089-recommended-screening-list.csv
```

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

## 3. Group audit

```bash
python scripts/01_audit_scripts/audit_groups_before_ml.py
```

Expected outputs are written to `04_outputs/group_audit_outputs/`.

## 4. Main V14 sweep

```bash
python scripts/02_main_sweep/v14_workflow.py
```

This produces seed-level predictions and hyperparameters. It can be computationally expensive.

## 5. Rebuild result summary from predictions

```bash
python scripts/03_post_processing/rebuild_results_from_predictions.py
```

The expected complete result table has 4200 rows. Do not use an incomplete `v14_results.csv` if it contains only one dataset/target.

## 6. Figure and table generation

Recommended current figure workflow:

```bash
python scripts/final_revision_tools/run_all_final_package_v2.py
python scripts/final_revision_tools/make_final_main_figures_STRICT_v4.py
python scripts/final_revision_tools/make_figS14_application_expanded_v2.py
```

The strict main-figure script writes a source manifest and does not silently switch to alternate source tables.

## 7. Outputs

Generated outputs are written locally to `04_outputs/` and should not normally be committed to GitHub.
