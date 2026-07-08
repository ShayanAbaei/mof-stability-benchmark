"""
Rebuild v14_results.csv from v14_predictions.csv.

The predictions file has all per-row predictions across the full sweep.
The summary file got overwritten with only ASR Solvent.
We aggregate predictions back to per-cell metrics matching V14 schema.
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

print("Loading v14_predictions.csv...")
p = pd.read_csv('v14_predictions.csv')
print(f"  Loaded {len(p)} rows")
print(f"  Columns: {list(p.columns)}")

# Required columns
if 'seed' not in p.columns:
    print("  WARNING: 'seed' column not in predictions; using single-seed assumption")
    p['seed'] = 42

# Group keys
group_keys = ['dataset', 'target', 'regime', 'model', 'split_type', 'seed']
print(f"\nAggregating by: {group_keys}")

def metrics_from_group(g):
    yt = g['y_true'].values
    yp = g['y_pred'].values
    mask = ~(pd.isna(yt) | pd.isna(yp))
    yt = yt[mask]
    yp = yp[mask]
    if len(yt) < 3:
        return pd.Series({
            'n_train': np.nan,
            'n_test': len(yt),
            'RMSE': np.nan,
            'MAE': np.nan,
            'R2': np.nan,
            'Spearman': np.nan,
        })
    try:
        return pd.Series({
            'n_train': np.nan,  # not in predictions
            'n_test': len(yt),
            'RMSE': float(np.sqrt(mean_squared_error(yt, yp))),
            'MAE': float(mean_absolute_error(yt, yp)),
            'R2': float(r2_score(yt, yp)),
            'Spearman': float(spearmanr(yt, yp).statistic),
        })
    except Exception as e:
        return pd.Series({
            'n_train': np.nan,
            'n_test': len(yt),
            'RMSE': np.nan,
            'MAE': np.nan,
            'R2': np.nan,
            'Spearman': np.nan,
        })

print("Aggregating metrics per cell... (this takes ~30 seconds)")
results = p.groupby(group_keys, dropna=False).apply(metrics_from_group).reset_index()

print(f"\nReconstructed results shape: {results.shape}")
print(f"\nCells by dataset/target:")
print(results.groupby(['dataset', 'target']).size())
print(f"\nCells by regime:")
print(results['regime'].value_counts())
print(f"\nCells by split type:")
print(results['split_type'].value_counts())
print(f"\nModels present:")
print(results['model'].value_counts())

# Save with same column ordering as original v14_results.csv
out_cols = ['dataset', 'target', 'regime', 'model', 'split_type', 'seed',
            'n_train', 'n_test', 'RMSE', 'MAE', 'R2', 'Spearman']
results = results[out_cols]
results.to_csv('v14_results.csv', index=False)
print(f"\nSaved reconstructed v14_results.csv")
print(f"Original file overwritten. Backup it first if needed.")

# Quick verification of headline numbers
print("\n=== HEADLINE VERIFICATION ===")
d_random = results[(results['regime'] == 'D_context_thermophysical') &
                    (results['split_type'] == 'random')]
best_d = d_random.sort_values('Spearman', ascending=False).groupby(
    ['dataset', 'target']).head(1)
print(best_d[['dataset', 'target', 'model', 'Spearman', 'R2']].to_string(index=False))