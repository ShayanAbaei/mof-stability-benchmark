# Uploaded archive audit summary

The uploaded archive contains 225 files and about 358 MB after extraction.

## Main finding

The archive is useful as a local handoff and reproducibility working folder, but it should not be uploaded wholesale to GitHub.

## Safe to upload

- `03_code/` scripts
- `99_final_tools_v2/` figure/table generation scripts
- documentation adapted from README/HOW_TO_RUN
- citation/data provenance files
- small requirements/environment files

## Do not upload

- `02_raw_data/`: third-party raw dataset files; cite/download externally instead
- `04_outputs/v14_predictions.csv`: about 101 MB; too large for normal GitHub use
- `05_historical/`: deprecated and about 232 MB; contains large pkl/prediction files
- manuscript-private revision notes unless all authors approve
- generated output folders unless intentionally attached as a release/Zenodo artifact

## Important correction

The old `04_outputs/v14_results.csv` in earlier workflows was incomplete in prior audits. Use `v14_results_rebuilt.csv` as the verified aggregate result table when regenerating figures.

## Inventory

See `archive_inventory_upload_recommendations.csv` for file-level recommendations.
