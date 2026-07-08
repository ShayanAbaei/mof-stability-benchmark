# Data provenance

This repository does not redistribute the raw external database CSV files. To reproduce the benchmark, download the required files from the official source and place them locally in `02_raw_data/` or `raw_data/`.

## Required files

```text
ASR_data_SI_20250204.csv
FSR_data_SI_20250204.csv
ION_data_SI_20250204.csv
ASR_FSR_check.csv
12089-recommended-screening-list.csv
```

## Primary source for the raw input files

The raw files used here are from the CoRE MOF 2024 Zenodo dataset release:

Zhao, G.; Brabson, L. M.; Chheda, S.; Huang, J.; Kim, H.; Liu, K.; et al. **Computation-Ready Experimental Metal-Organic Framework (CoRE MOF) 2024 Dataset**. Zenodo, 2025. DOI: `10.5281/zenodo.15055758`.

## Related database article

Zhao, G.; Brabson, L. M.; Chheda, S.; Huang, J.; Kim, H.; Liu, K.; et al. **CoRE MOF DB: A curated experimental metal-organic framework database with machine-learned properties for integrated material-process screening**. *Matter* **2025**, *8*, 102140. DOI: `10.1016/j.matt.2025.102140`.

## Stability-label sources to cite

Activation/solvent-removal and thermal-stability labels are connected to the MOFSimplify / stability-mining work:

Nandy, A.; Duan, C.; Kulik, H. J. **Using Machine Learning and Data Mining to Leverage Community Knowledge for the Engineering of Stable Metal--Organic Frameworks**. *J. Am. Chem. Soc.* **2021**, *143*, 17535--17547. DOI: `10.1021/jacs.1c07217`.

Nandy, A.; Terrones, G.; Arunachalam, N.; Duan, C.; Kastner, D. W.; Kulik, H. J. **MOFSimplify, machine learning models with extracted stability data of three thousand metal--organic frameworks**. *Scientific Data* **2022**, *9*, 74. DOI: `10.1038/s41597-022-01181-0`.

Water-stability labels are connected to the WS24 work:

Terrones, G. G.; Huang, S.-P.; Rivera, M. P.; Yue, S.; Hernandez, A.; Kulik, H. J. **Metal--Organic Framework Stability in Water and Harsh Environments from Data-Driven Models Trained on the Diverse WS24 Data Set**. *J. Am. Chem. Soc.* **2024**, *146*, 20333--20348. DOI: `10.1021/jacs.4c05879`.

## Redistribution note

Before uploading any third-party data to GitHub, check the license/terms of the original source. The safest default is to provide download instructions and citations rather than redistributing the raw CSV files.
