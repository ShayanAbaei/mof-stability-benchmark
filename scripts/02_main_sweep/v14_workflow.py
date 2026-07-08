"""
v14_workflow.py — Multi-seed grouped splits + group-aware inner CV + data-sized hyperparameter ranges.

Models: Linear, Ridge, LASSO, DecisionTree, RandomForest, LightGBM, SVR_Linear
Targets: ASR/FSR x Solvent/Water/Thermal
Regimes: A, B, C, D
Splits:
  - random: 5 outer seeds, inner KFold
  - group_metal: 10 outer seeds, inner GroupKFold by primary_metal
  - group_topology: 10 outer seeds, inner GroupKFold by topology(AllNodes)
Tuning: Optuna 20 trials per model per cell, inner 3-fold CV
Hyperparameter ranges: tightened per best-practice for ~1000-row tabular data

Output: v14_results.csv, v14_predictions.csv, v14_hparams.csv
Expected runtime: ~6-8 hours on 12 cores.
"""

try:
    from sklearnex import patch_sklearn
    patch_sklearn()
    SKLEARNEX_ON = True
except ImportError:
    SKLEARNEX_ON = False

import os, math, json, time, warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from tqdm.auto import tqdm

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, GroupShuffleSplit, KFold, GroupKFold

import lightgbm as lgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIG
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.resolve()
RAW_DIR = PROJECT_ROOT / "raw_data"
RESULTS_FILE = PROJECT_ROOT / "v14_results.csv"
PREDS_FILE = PROJECT_ROOT / "v14_predictions.csv"
HPARAMS_FILE = PROJECT_ROOT / "v14_hparams.csv"

RANDOM_BASE_SEED = 42
N_INNER_FOLDS = 3
N_OPTUNA_TRIALS = 20
TEST_SIZE = 0.20

N_RANDOM_SEEDS = 5
N_GROUP_SEEDS = 10

TARGETS = ["Solvent_stability", "Water_stability", "Thermal_stability (℃)"]

CATEGORICAL_FEATURES = {
    "primary_metal", "OMS Types",
    "topology(SingleNodes)", "topology(AllNodes)",
    "Has OMS",
}

REGIMES = {
    "A_metal_only": ["primary_metal", "is_mixed_metal", "n_metals"],
    "B_metal_oms": ["primary_metal", "is_mixed_metal", "n_metals",
                    "Has OMS", "OMS Types"],
    "C_metal_oms_context": [
        "primary_metal", "is_mixed_metal", "n_metals", "Has OMS", "OMS Types",
        "structure_dimension", "topology(SingleNodes)", "topology(AllNodes)",
        "catenation",
        "Density (g/cm3)", "LCD (Å)", "PLD (Å)", "VF", "PV (cm3/g)",
    ],
    "D_context_thermophysical": [
        "primary_metal", "is_mixed_metal", "n_metals", "Has OMS", "OMS Types",
        "structure_dimension", "topology(SingleNodes)", "topology(AllNodes)",
        "catenation",
        "Density (g/cm3)", "LCD (Å)", "PLD (Å)", "VF", "PV (cm3/g)",
        "average_atomic_mass",
        "Heat_capacity@300K (J/g/K)", "Heat_capacity@350K (J/g/K)",
        "Heat_capacity@400K (J/g/K)",
        "k_cp (J/g/K/K)", "cp0 (J/g/K)", "natoms",
    ],
}

N_CORES = os.cpu_count() or 1

print("=" * 70)
print(f"V14 SWEEP — multi-seed grouped splits, group-aware inner CV")
print(f"  Random splits:    {N_RANDOM_SEEDS} seeds, inner KFold")
print(f"  Group splits:     {N_GROUP_SEEDS} seeds, inner GroupKFold by group_col")
print(f"  Optuna trials:    {N_OPTUNA_TRIALS} per model per cell per seed")
print(f"  Inner folds:      {N_INNER_FOLDS}")
print(f"  CPU cores: {N_CORES}, sklearnex: {SKLEARNEX_ON}")
print("=" * 70)

# ============================================================================
# DATA LOAD + CLEAN
# ============================================================================

def normalize_metal_types(x):
    if pd.isna(x): return np.nan
    parts = sorted([p.strip() for p in str(x).split(",") if str(p).strip()])
    return ",".join(parts) if parts else np.nan

def split_metal_types(x):
    if pd.isna(x):
        return pd.Series({"n_metals": np.nan, "is_mixed_metal": np.nan,
                          "primary_metal": np.nan, "secondary_metal": np.nan,
                          "metal_list": np.nan})
    parts = sorted([p.strip() for p in str(x).split(",") if str(p).strip()])
    return pd.Series({
        "n_metals": len(parts),
        "is_mixed_metal": int(len(parts) > 1),
        "primary_metal": parts[0] if parts else np.nan,
        "secondary_metal": parts[1] if len(parts) >= 2 else np.nan,
        "metal_list": ",".join(parts) if parts else np.nan,
    })

def yes_no_to_binary(s):
    ss = s.astype(str).str.strip().str.lower()
    out = pd.Series(np.nan, index=s.index, dtype="float")
    out[ss == "yes"] = 1.0
    out[ss == "no"] = 0.0
    return out

NUMERIC_COLUMNS = [
    "LCD (Å)", "PLD (Å)", "LFPD (Å)", "Density (g/cm3)",
    "ASA (A2)", "ASA (m2/cm3)", "ASA (m2/g)",
    "NASA (A2)", "NASA (m2/cm3)", "NASA (m2/g)",
    "PV (A3)", "VF", "PV (cm3/g)", "NAV (A3)", "NAV_VF", "NPV (cm3/g)",
    "structure_dimension", "catenation", "dimension_by_topo",
    "hall", "number_spacegroup", "average_atomic_mass",
    "Heat_capacity@300K (J/g/K)", "std @ 300 K (J/g/K)",
    "Heat_capacity@350K (J/g/K)", "std @ 350 K (J/g/K)",
    "Heat_capacity@400K (J/g/K)", "std @ 400 K (J/g/K)",
    "k_cp (J/g/K/K)", "cp0 (J/g/K)", "natoms", "Year", "Time",
    "Thermal_stability (℃)", "Solvent_stability", "Water_stability",
]

def clean_table(df):
    df = df.copy()
    if "Metal Types" in df.columns:
        df["metal_types_norm"] = df["Metal Types"].map(normalize_metal_types)
        df = pd.concat([df, df["Metal Types"].apply(split_metal_types)], axis=1)
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "Has OMS" in df.columns:
        df["Has OMS_binary"] = yes_no_to_binary(df["Has OMS"])
    return df

print("\nLoading and cleaning ASR + FSR...")
asr_clean = clean_table(pd.read_csv(RAW_DIR / "ASR_data_SI_20250204.csv"))
fsr_clean = clean_table(pd.read_csv(RAW_DIR / "FSR_data_SI_20250204.csv"))
print(f"  ASR: {asr_clean.shape}")
print(f"  FSR: {fsr_clean.shape}")

# ============================================================================
# PREPROCESSING — TWO PIPELINES
# ============================================================================

def split_cat_num(df, cols):
    cat = [c for c in cols if c in CATEGORICAL_FEATURES]
    num = [c for c in cols if c not in CATEGORICAL_FEATURES]
    return num, cat

def build_onehot_preprocessor(df, cols, scale_numeric=True):
    num, cat = split_cat_num(df, cols)
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    num_pipe = Pipeline(num_steps)
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        [("num", num_pipe, num), ("cat", cat_pipe, cat)],
        remainder="drop")

def prepare_native_cat(df_train, df_test, cols, scale_numeric=False):
    num, cat = split_cat_num(df_train, cols)
    X_train = df_train[cols].copy()
    X_test = df_test[cols].copy()
    for c in num:
        med = X_train[c].median()
        X_train[c] = X_train[c].fillna(med).astype(float)
        X_test[c] = X_test[c].fillna(med).astype(float)
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    if cat:
        X_train_cat = X_train[cat].astype(str).fillna("MISSING_CAT")
        X_test_cat = X_test[cat].astype(str).fillna("MISSING_CAT")
        enc.fit(X_train_cat)
        X_train[cat] = enc.transform(X_train_cat).astype(int)
        X_test[cat] = enc.transform(X_test_cat).astype(int)
    if scale_numeric and num:
        scaler = StandardScaler()
        scaler.fit(X_train[num])
        X_train[num] = scaler.transform(X_train[num])
        X_test[num] = scaler.transform(X_test[num])
    final_order = num + cat
    X_train = X_train[final_order].values
    X_test = X_test[final_order].values
    cat_indices = list(range(len(num), len(num) + len(cat)))
    return X_train, X_test, cat_indices

# ============================================================================
# METRICS
# ============================================================================

def regression_metrics(y_true, y_pred):
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "Spearman": float(spearmanr(y_true, y_pred).statistic)
                    if len(y_true) >= 3 else np.nan,
    }

# ============================================================================
# SPLITS — MULTI-SEED OUTER LOOPS
# ============================================================================

def get_random_split(df, seed):
    tr, te = train_test_split(np.arange(len(df)), test_size=TEST_SIZE,
                              random_state=seed, shuffle=True)
    return tr, te

def get_group_split(df, group_col, seed):
    if group_col not in df.columns:
        return None, None
    groups = df[group_col].fillna("MISSING").astype(str).values
    sp = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE,
                            random_state=seed)
    tr, te = next(sp.split(df, groups=groups))
    return tr, te

def get_outer_splits(df):
    """Return list of (split_type, seed, tr_idx, te_idx, inner_group_col_or_None)."""
    splits = []
    for s in range(N_RANDOM_SEEDS):
        seed = RANDOM_BASE_SEED + s
        tr, te = get_random_split(df, seed)
        splits.append(("random", seed, tr, te, None))
    for s in range(N_GROUP_SEEDS):
        seed = RANDOM_BASE_SEED + s
        tr, te = get_group_split(df, "primary_metal", seed)
        if tr is not None:
            splits.append(("group_metal", seed, tr, te, "primary_metal"))
    for s in range(N_GROUP_SEEDS):
        seed = RANDOM_BASE_SEED + s
        tr, te = get_group_split(df, "topology(AllNodes)", seed)
        if tr is not None:
            splits.append(("group_topology", seed, tr, te, "topology(AllNodes)"))
    return splits

# ============================================================================
# OPTUNA OBJECTIVES — WITH GROUP-AWARE INNER CV
# ============================================================================

def cv_rmse(model, X, y, inner_groups=None, n_folds=N_INNER_FOLDS, cat_idx=None):
    """Inner CV: GroupKFold if inner_groups provided, else KFold."""
    if inner_groups is not None:
        kf = GroupKFold(n_splits=n_folds)
        split_iter = kf.split(X, y, groups=inner_groups)
    else:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_BASE_SEED)
        split_iter = kf.split(X)
    rmses = []
    for tr, va in split_iter:
        try:
            if cat_idx is not None and isinstance(model, lgb.LGBMRegressor):
                model.fit(X[tr], y[tr], categorical_feature=cat_idx)
            else:
                model.fit(X[tr], y[tr])
            yp = model.predict(X[va])
            rmses.append(np.sqrt(mean_squared_error(y[va], yp)))
        except Exception:
            rmses.append(1e9)
    return float(np.mean(rmses))

def make_objective(model_name, X, y, inner_groups=None, cat_idx=None):
    def objective(trial):
        if model_name == "Ridge":
            alpha = trial.suggest_float("alpha", 1e-4, 1e3, log=True)
            m = Ridge(alpha=alpha, random_state=RANDOM_BASE_SEED)
        elif model_name == "LASSO":
            alpha = trial.suggest_float("alpha", 1e-4, 1e2, log=True)
            m = Lasso(alpha=alpha, random_state=RANDOM_BASE_SEED, max_iter=5000)
        elif model_name == "DecisionTree":
            params = {
                "max_depth": trial.suggest_int("max_depth", 3, 15),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            }
            m = DecisionTreeRegressor(random_state=RANDOM_BASE_SEED, **params)
        elif model_name == "RandomForest":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 300, step=50),
                "max_depth": trial.suggest_int("max_depth", 3, 15),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
                "max_features": trial.suggest_categorical("max_features",
                                                          ["sqrt", 0.5]),
            }
            m = RandomForestRegressor(random_state=RANDOM_BASE_SEED,
                                       n_jobs=N_CORES, **params)
        elif model_name == "LightGBM":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 250, step=50),
                "num_leaves": trial.suggest_int("num_leaves", 7, 31),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 30),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            }
            m = lgb.LGBMRegressor(random_state=RANDOM_BASE_SEED, n_jobs=N_CORES,
                                   verbose=-1, **params)
        elif model_name == "SVR_Linear":
            params = {
                "C": trial.suggest_float("C", 1e-3, 1e2, log=True),
                "epsilon": trial.suggest_float("epsilon", 0.01, 0.5),
            }
            m = SVR(kernel="linear", **params)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        return cv_rmse(m, X, y, inner_groups=inner_groups, cat_idx=cat_idx)
    return objective

def build_final(model_name, params):
    if model_name == "Linear":
        return LinearRegression()
    elif model_name == "Ridge":
        return Ridge(random_state=RANDOM_BASE_SEED, **params)
    elif model_name == "LASSO":
        return Lasso(random_state=RANDOM_BASE_SEED, max_iter=5000, **params)
    elif model_name == "DecisionTree":
        return DecisionTreeRegressor(random_state=RANDOM_BASE_SEED, **params)
    elif model_name == "RandomForest":
        return RandomForestRegressor(random_state=RANDOM_BASE_SEED,
                                      n_jobs=N_CORES, **params)
    elif model_name == "LightGBM":
        return lgb.LGBMRegressor(random_state=RANDOM_BASE_SEED, n_jobs=N_CORES,
                                  verbose=-1, **params)
    elif model_name == "SVR_Linear":
        return SVR(kernel="linear", **params)

# ============================================================================
# MAIN SWEEP
# ============================================================================

ONEHOT_MODELS = ["Linear", "Ridge", "LASSO", "DecisionTree", "SVR_Linear"]
NATIVE_MODELS = ["RandomForest", "LightGBM"]
ALL_MODELS = ONEHOT_MODELS + NATIVE_MODELS

datasets = {"ASR": asr_clean, "FSR": fsr_clean}
all_rows = []
all_preds = []
all_hps = []

# Enumerate work units
work_cells = []
for dataset_name, df in datasets.items():
    for target in TARGETS:
        if target not in df.columns:
            continue
        task_df = df[df[target].notna()].copy().reset_index(drop=True)
        if len(task_df) < 50:
            continue
        outer_splits = get_outer_splits(task_df)
        for regime, feature_cols in REGIMES.items():
            feat = [c for c in feature_cols if c in task_df.columns]
            for split_type, seed, tr_idx, te_idx, inner_group_col in outer_splits:
                work_cells.append({
                    "dataset_name": dataset_name,
                    "target": target,
                    "regime": regime,
                    "feat": feat,
                    "task_df": task_df,
                    "split_type": split_type,
                    "seed": seed,
                    "tr_idx": tr_idx,
                    "te_idx": te_idx,
                    "inner_group_col": inner_group_col,
                })

total_cells = len(work_cells)
print(f"\nTotal cells: {total_cells} (each runs {len(ALL_MODELS)} models)")
print(f"Total model fits: {total_cells * len(ALL_MODELS)}")

t_start = time.time()

pbar_cells = tqdm(work_cells, desc="Cells", unit="cell", position=0)
for cell_idx, cell in enumerate(pbar_cells):
    dataset_name = cell["dataset_name"]
    target = cell["target"]
    regime = cell["regime"]
    feat = cell["feat"]
    task_df = cell["task_df"]
    split_type = cell["split_type"]
    seed = cell["seed"]
    tr_idx = cell["tr_idx"]
    te_idx = cell["te_idx"]
    inner_group_col = cell["inner_group_col"]

    Xtr_df = task_df.iloc[tr_idx]
    Xte_df = task_df.iloc[te_idx]
    ytr = Xtr_df[target].astype(float).values
    yte = Xte_df[target].astype(float).values

    # Inner-CV groups (only for group splits)
    if inner_group_col is not None and inner_group_col in Xtr_df.columns:
        inner_groups = Xtr_df[inner_group_col].fillna("MISSING").astype(str).values
        # Sanity check: must have at least N_INNER_FOLDS distinct groups
        if len(set(inner_groups)) < N_INNER_FOLDS:
            inner_groups = None
    else:
        inner_groups = None

    cell_label = f"{dataset_name}/{target[:8]}/{regime[:3]}/{split_type[:10]}/s{seed}"
    pbar_cells.set_description(f"Cell {cell_idx+1}/{total_cells} {cell_label}")

    # One-hot pipeline
    try:
        pre_oh = build_onehot_preprocessor(Xtr_df, feat, scale_numeric=True)
        X_tr_oh = pre_oh.fit_transform(Xtr_df[feat])
        X_te_oh = pre_oh.transform(Xte_df[feat])
    except Exception as e:
        X_tr_oh = X_te_oh = None

    # Native pipeline
    try:
        X_tr_nat, X_te_nat, cat_idx_nat = prepare_native_cat(
            Xtr_df, Xte_df, feat, scale_numeric=False)
    except Exception as e:
        X_tr_nat = X_te_nat = cat_idx_nat = None

    pbar_models = tqdm(ALL_MODELS, desc="Models", unit="model",
                        position=1, leave=False)
    for mname in pbar_models:
        pbar_models.set_postfix_str(mname)
        try:
            if mname in ONEHOT_MODELS:
                if X_tr_oh is None:
                    continue
                X_tr_use = X_tr_oh
                X_te_use = X_te_oh
                cat_idx_use = None
            else:
                if X_tr_nat is None:
                    continue
                X_tr_use = X_tr_nat
                X_te_use = X_te_nat
                cat_idx_use = cat_idx_nat if mname == "LightGBM" else None

            if mname == "Linear":
                best_params = {}
                m = build_final(mname, {})
                m.fit(X_tr_use, ytr)
            else:
                study = optuna.create_study(
                    direction="minimize",
                    sampler=optuna.samplers.TPESampler(seed=RANDOM_BASE_SEED),
                )
                objective = make_objective(
                    mname, X_tr_use, ytr,
                    inner_groups=inner_groups,
                    cat_idx=cat_idx_use,
                )
                study.optimize(objective, n_trials=N_OPTUNA_TRIALS,
                                show_progress_bar=False)
                best_params = study.best_params
                m = build_final(mname, best_params)
                if mname == "LightGBM" and cat_idx_use:
                    m.fit(X_tr_use, ytr, categorical_feature=cat_idx_use)
                else:
                    m.fit(X_tr_use, ytr)

            ypred = m.predict(X_te_use)
            mets = regression_metrics(yte, ypred)

            all_rows.append({
                "dataset": dataset_name, "target": target,
                "regime": regime, "model": mname,
                "split_type": split_type, "seed": int(seed),
                "n_train": len(tr_idx), "n_test": len(te_idx),
                **mets,
            })
            all_preds.append(pd.DataFrame({
                "dataset": dataset_name, "target": target,
                "regime": regime, "model": mname,
                "split_type": split_type, "seed": int(seed),
                "y_true": yte, "y_pred": ypred,
                "primary_metal": Xte_df["primary_metal"].values,
                "topology": Xte_df.get("topology(AllNodes)",
                                        pd.Series([np.nan]*len(Xte_df))).values,
                "is_mixed_metal": Xte_df["is_mixed_metal"].values,
                "Has OMS": Xte_df.get("Has OMS",
                                       pd.Series([np.nan]*len(Xte_df))).values,
            }))
            all_hps.append({
                "dataset": dataset_name, "target": target,
                "regime": regime, "model": mname,
                "split_type": split_type, "seed": int(seed),
                "best_params": json.dumps(best_params),
            })
        except Exception as e:
            pass

    pbar_models.close()

    # Incremental save every 20 cells
    if (cell_idx + 1) % 20 == 0:
        pd.DataFrame(all_rows).to_csv(RESULTS_FILE, index=False)
        elapsed_min = (time.time() - t_start) / 60
        cells_remaining = total_cells - (cell_idx + 1)
        eta_min = (elapsed_min / (cell_idx + 1)) * cells_remaining
        pbar_cells.write(f"  [{cell_idx+1}/{total_cells}] elapsed={elapsed_min:.1f} min, ETA={eta_min:.0f} min")

pbar_cells.close()

# ============================================================================
# FINAL SAVE
# ============================================================================

print("\n" + "=" * 70)
print("Saving final results...")
print("=" * 70)

df_results = pd.DataFrame(all_rows)
df_preds = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
df_hps = pd.DataFrame(all_hps)

df_results.to_csv(RESULTS_FILE, index=False)
df_preds.to_csv(PREDS_FILE, index=False)
df_hps.to_csv(HPARAMS_FILE, index=False)

total_min = (time.time() - t_start) / 60
print(f"\nTotal runtime: {total_min:.1f} min")
print(f"Results: {len(df_results)} rows → {RESULTS_FILE}")
print(f"Predictions: {len(df_preds)} rows → {PREDS_FILE}")
print(f"Hyperparams: {len(df_hps)} rows → {HPARAMS_FILE}")

# ============================================================================
# QUICK SUMMARY — mean ± std across seeds
# ============================================================================

print("\n" + "=" * 70)
print("QUICK SUMMARY — Spearman mean ± std across seeds, per cell")
print("=" * 70)

if len(df_results) > 0:
    summary = (df_results
        .groupby(["dataset", "target", "regime", "split_type", "model"])
        .agg(spearman_mean=("Spearman", "mean"),
             spearman_std=("Spearman", "std"),
             spearman_n=("Spearman", "count"),
             r2_mean=("R2", "mean"))
        .reset_index())

    # Best model per cell by mean Spearman
    best = (summary
        .sort_values(["dataset", "target", "regime", "split_type", "spearman_mean"],
                     ascending=[True, True, True, True, False])
        .groupby(["dataset", "target", "regime", "split_type"]).head(1))

    print("\nBest model per (dataset, target, regime, split):")
    print(best[["dataset", "target", "regime", "split_type", "model",
                 "spearman_mean", "spearman_std", "spearman_n",
                 "r2_mean"]].to_string(index=False))

print("\nDone.")