# Data provenance and citation list

## What this project uses

The benchmark uses three CoRE MOF SI structure sets and two supporting lookup/screening files:

- `ASR_data_SI_20250204.csv`
- `FSR_data_SI_20250204.csv`
- `ION_data_SI_20250204.csv`
- `ASR_FSR_check.csv`
- `12089-recommended-screening-list.csv`

These are treated as external data files and are not committed to the repository.

## Recommended ACS-style citations

1. Nandy, A.; Duan, C.; Kulik, H. J. Using Machine Learning and Data Mining to Leverage Community Knowledge for the Engineering of Stable Metal--Organic Frameworks. *J. Am. Chem. Soc.* **2021**, *143* (42), 17535--17547. https://doi.org/10.1021/jacs.1c07217

2. Nandy, A.; Terrones, G.; Arunachalam, N.; Duan, C.; Kastner, D. W.; Kulik, H. J. MOFSimplify, Machine Learning Models with Extracted Stability Data of Three Thousand Metal--Organic Frameworks. *Scientific Data* **2022**, *9*, 74. https://doi.org/10.1038/s41597-022-01181-0

3. Terrones, G. G.; Huang, S.-P.; Rivera, M. P.; Yue, S.; Hernandez, A.; Kulik, H. J. Metal--Organic Framework Stability in Water and Harsh Environments from Data-Driven Models Trained on the Diverse WS24 Data Set. *J. Am. Chem. Soc.* **2024**, *146* (29), 20333--20348. https://doi.org/10.1021/jacs.4c05879

4. Zhao, G.; Brabson, L. M.; Chheda, S.; Huang, J.; Kim, H.; Liu, K.; et al. Computation-Ready Experimental Metal-Organic Framework (CoRE MOF) 2024 Dataset. Zenodo, 2025. https://doi.org/10.5281/zenodo.15055758

5. Zhao, G.; Brabson, L. M.; Chheda, S.; Huang, J.; Kim, H.; Liu, K.; et al. CoRE MOF DB: A Curated Experimental Metal-Organic Framework Database with Machine-Learned Properties for Integrated Material-Process Screening. *Matter* **2025**, *8* (6), 102140. https://doi.org/10.1016/j.matt.2025.102140

## Which references to cite where

- Cite refs 1--2 for solvent-removal / activation stability and thermal stability mining/model background.
- Cite ref 3 for water-stability labels / WS24 stability modeling.
- Cite ref 4 for the exact CoRE MOF 2024 raw CSV dataset used in this repository.
- Cite ref 5 for the database article describing the updated CoRE MOF DB and its machine-learned properties.

## BibTeX keys

Use the keys in `references_database.bib`:

```text
Nandy2021StableMOFs
Nandy2022MOFSimplify
Terrones2024WS24
Zhao2025CoREMOF2024Zenodo
Zhao2025CoREMOFDBMatter
```
