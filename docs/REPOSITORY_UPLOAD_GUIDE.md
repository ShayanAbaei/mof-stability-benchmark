# Repository upload guide

## Strong recommendation

Keep the GitHub repository lightweight. Upload code and documentation. Do not upload raw external data, giant predictions, historical pkl files, or large generated output folders.

## Upload to GitHub main branch

Recommended:

```text
README.md
CITATION.cff
LICENSE
requirements.txt
.gitignore
data_README.md
references_database.bib
docs/DATA_PROVENANCE_AND_CITATIONS.md
docs/REPRODUCTION_GUIDE.md
docs/REPOSITORY_UPLOAD_GUIDE.md
docs/archive_inventory_upload_recommendations.csv
scripts/
```

## Do not upload to GitHub main branch

Do not upload:

```text
02_raw_data/
04_outputs/v14_predictions.csv
04_outputs/v14_hparams.csv
04_outputs/v14_results.csv
04_outputs/FINAL_FOR_REVISION/
05_historical/
*.pkl
*.joblib
*.7z
*.zip
*.pdf manuscript drafts unless intentionally public
internal revision notes with names/comments
```

## Optional, but better as Zenodo/GitHub release assets

If a journal/editor requires exact outputs, use Zenodo or GitHub Releases instead of the main branch for:

```text
v14_results_rebuilt.csv
small summary tables
final figures
source_data_for_figures/
```

Do not include `v14_predictions.csv` in the repo main branch because it is about 101 MB and exceeds/approaches GitHub file-size limits.

## Commands

```bash
git clone https://github.com/ShayanAbaei/mof-stability-benchmark.git
cd mof-stability-benchmark

# copy the contents of GITHUB_UPLOAD_RECOMMENDED into this folder

git status
git add README.md data_README.md CITATION.cff references_database.bib requirements.txt .gitignore docs scripts
git commit -m "Update reproducible JMCA stability benchmark workflow and data provenance"
git push origin main
```

## Before pushing

Run:

```bash
git status
find . -size +50M -type f
```

If any large file appears, remove it before commit.
