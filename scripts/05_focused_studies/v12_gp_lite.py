"""
v12_gp_lite.py — Focused GP sanity test with structured chemistry-aware kernel.

Tests whether a chemistry-aware GP can beat simple baselines on the ASR water
stability task at Regime D. Single train/test split per condition, no outer repeats.
Expected runtime: 20-40 minutes total.

Used in the JMCA revision SI Section S8.3 as a focused uncertainty-quantification
analysis, explaining why GP was not retained in the main 7-model comparison panel.
"""

import os, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.linear_model import Ridge

import tensorflow as tf
import gpflow

warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")

# ============================================================================
# CONFIG
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.resolve()
RAW_DIR = PROJECT_ROOT / "raw_data"
OUT_FILE = PROJECT_ROOT / "v12_gp_lite_results.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_RESTARTS = 3
MAX_ITER = 200

CONTINUOUS_FEATURES_D = [
    "structure_dimension", "catenation",
    "Density (g/cm3)", "LCD (Å)", "PLD (Å)", "VF", "PV (cm3/g)",
    "average_atomic_mass",
    "Heat_capacity@300K (J/g/K)", "Heat_capacity@350K (J/g/K)",
    "Heat_capacity@400K (J/g/K)",
    "k_cp (J/g/K/K)", "cp0 (J/g/K)", "natoms",
    "n_metals",
]
CAT_METAL = "primary_metal"
CAT_TOPO = "topology(SingleNodes)"
BINARY_FEATURES = ["is_mixed_metal", "Has OMS_binary"]

# ============================================================================
# DATA LOAD (reuse cleaning logic, minimal)
# ============================================================================

def normalize_metal_types(x):
    if pd.isna(x): return np.nan
    parts = sorted([p.strip() for p in str(x).split(",") if str(p).strip()])
    return ",".join(parts) if parts else np.nan

def split_metal_types(x):
    if pd.isna(x):
        return pd.Series({"n_metals": np.nan, "is_mixed_metal": np.nan,
                          "primary_metal": np.nan})
    parts = sorted([p.strip() for p in str(x).split(",") if str(p).strip()])
    return pd.Series({
        "n_metals": len(parts),
        "is_mixed_metal": int(len(parts) > 1),
        "primary_metal": parts[0] if parts else np.nan,
    })

def yes_no_to_binary(s):
    ss = s.astype(str).str.strip().str.lower()
    out = pd.Series(np.nan, index=s.index, dtype="float")
    out[ss == "yes"] = 1.0
    out[ss == "no"] = 0.0
    return out

print("Loading ASR...")
asr = pd.read_csv(RAW_DIR / "ASR_data_SI_20250204.csv")
if "Metal Types" in asr.columns:
    metal_info = asr["Metal Types"].apply(split_metal_types)
    asr = pd.concat([asr, metal_info], axis=1)
if "Has OMS" in asr.columns:
    asr["Has OMS_binary"] = yes_no_to_binary(asr["Has OMS"])

# Coerce numerics
numeric_cols = CONTINUOUS_FEATURES_D + ["Water_stability", "n_metals"]
for c in numeric_cols:
    if c in asr.columns:
        asr[c] = pd.to_numeric(asr[c], errors="coerce")

# Keep only rows with water stability
task_df = asr[asr["Water_stability"].notna()].copy().reset_index(drop=True)
print(f"  ASR water-stability rows: {len(task_df)}")

# ============================================================================
# PREPROCESS
# ============================================================================

def prep_features(df_train, df_test):
    """Returns Xtr_cont, Xte_cont (numpy), metal_tr, metal_te (int codes),
    topo_tr, topo_te (int codes), n_metal_classes, n_topo_classes."""

    # Continuous + binary
    cont_cols = [c for c in CONTINUOUS_FEATURES_D + BINARY_FEATURES if c in df_train.columns]
    Xtr_cont_raw = df_train[cont_cols].copy()
    Xte_cont_raw = df_test[cont_cols].copy()

    imp = SimpleImputer(strategy="median")
    Xtr_cont_raw = pd.DataFrame(imp.fit_transform(Xtr_cont_raw), columns=cont_cols)
    Xte_cont_raw = pd.DataFrame(imp.transform(Xte_cont_raw), columns=cont_cols)

    scaler = StandardScaler()
    Xtr_cont = scaler.fit_transform(Xtr_cont_raw)
    Xte_cont = scaler.transform(Xte_cont_raw)

    # Metal codes
    metal_tr_raw = df_train[CAT_METAL].fillna("MISSING").astype(str).values.reshape(-1, 1)
    metal_te_raw = df_test[CAT_METAL].fillna("MISSING").astype(str).values.reshape(-1, 1)
    enc_metal = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    enc_metal.fit(metal_tr_raw)
    metal_tr = enc_metal.transform(metal_tr_raw).astype(int).flatten()
    metal_te = enc_metal.transform(metal_te_raw).astype(int).flatten()
    # Handle unknowns: replace -1 with 0 (most frequent)
    metal_te[metal_te == -1] = 0
    n_metal = len(enc_metal.categories_[0])

    # Topology codes
    topo_tr_raw = df_train[CAT_TOPO].fillna("MISSING").astype(str).values.reshape(-1, 1)
    topo_te_raw = df_test[CAT_TOPO].fillna("MISSING").astype(str).values.reshape(-1, 1)
    enc_topo = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    enc_topo.fit(topo_tr_raw)
    topo_tr = enc_topo.transform(topo_tr_raw).astype(int).flatten()
    topo_te = enc_topo.transform(topo_te_raw).astype(int).flatten()
    topo_te[topo_te == -1] = 0
    n_topo = len(enc_topo.categories_[0])

    return Xtr_cont, Xte_cont, metal_tr, metal_te, topo_tr, topo_te, n_metal, n_topo

# ============================================================================
# STRUCTURED KERNEL GP
# ============================================================================

def fit_gp_structured(Xtr_cont, metal_tr, topo_tr, ytr,
                       Xte_cont, metal_te, topo_te,
                       n_metal, n_topo, n_restarts=N_RESTARTS):
    """
    Build a GP with structured kernel:
      kernel = Matern52(continuous, ARD) * Coregion(metal) * Coregion(topology)
    Try multiple restarts and Tikhonov jitter ladder.
    Returns (mean_predictions, var_predictions, success_flag, final_loss)
    """
    n_cont = Xtr_cont.shape[1]

    # Build augmented input: [continuous | metal_idx | topo_idx]
    X_train = np.hstack([
        Xtr_cont.astype(np.float64),
        metal_tr.reshape(-1, 1).astype(np.float64),
        topo_tr.reshape(-1, 1).astype(np.float64),
    ])
    X_test = np.hstack([
        Xte_cont.astype(np.float64),
        metal_te.reshape(-1, 1).astype(np.float64),
        topo_te.reshape(-1, 1).astype(np.float64),
    ])
    y_train = ytr.reshape(-1, 1).astype(np.float64)

    # Standardize y for GP stability
    y_mean = float(y_train.mean())
    y_std = float(y_train.std()) if y_train.std() > 0 else 1.0
    y_train_std = (y_train - y_mean) / y_std

    # Structured kernel
    def build_kernel(init_ls):
        k_cont = gpflow.kernels.Matern52(
            lengthscales=init_ls,
            active_dims=list(range(n_cont)),
        )
        k_metal = gpflow.kernels.Coregion(
            output_dim=n_metal, rank=3,
            active_dims=[n_cont],
        )
        k_topo = gpflow.kernels.Coregion(
            output_dim=n_topo, rank=3,
            active_dims=[n_cont + 1],
        )
        return k_cont * k_metal * k_topo

    # Tikhonov jitter ladder
    jitter_ladder = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]

    best_loss = np.inf
    best_mean = None
    best_var = None
    success = False

    for restart in range(n_restarts):
        rng = np.random.RandomState(RANDOM_STATE + restart)
        init_ls = np.exp(rng.uniform(-1.0, 1.0, size=n_cont)).astype(np.float64)

        for jitter in jitter_ladder:
            try:
                kernel = build_kernel(init_ls)
                model = gpflow.models.GPR(
                    (X_train, y_train_std),
                    kernel=kernel,
                    noise_variance=jitter,
                )
                opt = gpflow.optimizers.Scipy()
                opt.minimize(
                    model.training_loss,
                    model.trainable_variables,
                    options={"maxiter": MAX_ITER},
                )
                loss = float(model.training_loss().numpy())
                if not np.isfinite(loss):
                    continue

                mean_std, var_std = model.predict_f(X_test)
                mean_std = mean_std.numpy().flatten()
                var_std = var_std.numpy().flatten()

                # Check for NaN in predictions
                if np.any(np.isnan(mean_std)) or np.any(np.isnan(var_std)):
                    continue

                # Unstandardize
                mean = mean_std * y_std + y_mean
                var = var_std * (y_std ** 2)

                if loss < best_loss:
                    best_loss = loss
                    best_mean = mean
                    best_var = var
                    success = True
                break  # success at this jitter, move to next restart
            except Exception as e:
                continue  # try next jitter

    return best_mean, best_var, success, best_loss

# ============================================================================
# RIDGE BASELINE (for comparison)
# ============================================================================

def fit_ridge_baseline(Xtr_cont, metal_tr, topo_tr, n_metal, n_topo, ytr,
                        Xte_cont, metal_te, topo_te):
    """Ridge baseline with one-hot encoded categoricals."""
    # One-hot metal
    metal_oh_tr = np.eye(n_metal)[metal_tr]
    metal_oh_te = np.eye(n_metal)[metal_te]
    topo_oh_tr = np.eye(n_topo)[topo_tr]
    topo_oh_te = np.eye(n_topo)[topo_te]

    X_tr = np.hstack([Xtr_cont, metal_oh_tr, topo_oh_tr])
    X_te = np.hstack([Xte_cont, metal_oh_te, topo_oh_te])

    ridge = Ridge(alpha=1.0, random_state=RANDOM_STATE)
    ridge.fit(X_tr, ytr)
    return ridge.predict(X_te)

# ============================================================================
# SPLITS
# ============================================================================

def get_splits(df):
    out = []

    # Random
    tr, te = train_test_split(np.arange(len(df)), test_size=TEST_SIZE,
                              random_state=RANDOM_STATE, shuffle=True)
    out.append(("random", tr, te))

    # Group by primary_metal
    groups = df["primary_metal"].fillna("MISSING").astype(str).values
    sp = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    tr, te = next(sp.split(df, groups=groups))
    out.append(("group_metal", tr, te))

    # Group by topology(AllNodes)
    if "topology(AllNodes)" in df.columns:
        groups = df["topology(AllNodes)"].fillna("MISSING").astype(str).values
        sp = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        tr, te = next(sp.split(df, groups=groups))
        out.append(("group_topology", tr, te))

    return out

# ============================================================================
# MAIN LOOP
# ============================================================================

print("\n" + "=" * 70)
print("V12 GP LITE — structured kernel GP on ASR water stability, Regime D")
print("=" * 70)

results = []
splits = get_splits(task_df)

for split_name, tr_idx, te_idx in splits:
    print(f"\n--- Split: {split_name}  (n_train={len(tr_idx)}, n_test={len(te_idx)}) ---")
    t0 = time.time()

    df_tr = task_df.iloc[tr_idx]
    df_te = task_df.iloc[te_idx]
    ytr = df_tr["Water_stability"].astype(float).values
    yte = df_te["Water_stability"].astype(float).values

    Xtr_cont, Xte_cont, metal_tr, metal_te, topo_tr, topo_te, n_metal, n_topo = prep_features(df_tr, df_te)
    print(f"  Continuous features: {Xtr_cont.shape[1]}")
    print(f"  Metal categories: {n_metal}, Topology(SingleNodes) categories: {n_topo}")

    # Ridge baseline
    print("  Fitting Ridge baseline...")
    t_ridge = time.time()
    ypred_ridge = fit_ridge_baseline(Xtr_cont, metal_tr, topo_tr, n_metal, n_topo, ytr,
                                       Xte_cont, metal_te, topo_te)
    ridge_spear = spearmanr(yte, ypred_ridge).statistic
    ridge_rmse = float(np.sqrt(np.mean((yte - ypred_ridge) ** 2)))
    print(f"    Ridge: Spearman={ridge_spear:.4f}, RMSE={ridge_rmse:.4f}  ({time.time()-t_ridge:.1f}s)")

    # GP structured
    print(f"  Fitting structured GP ({N_RESTARTS} restarts, Tikhonov ladder)...")
    t_gp = time.time()
    ypred_gp, yvar_gp, gp_success, gp_loss = fit_gp_structured(
        Xtr_cont, metal_tr, topo_tr, ytr,
        Xte_cont, metal_te, topo_te,
        n_metal, n_topo,
    )
    gp_time = time.time() - t_gp

    if gp_success and ypred_gp is not None:
        gp_spear = spearmanr(yte, ypred_gp).statistic
        gp_rmse = float(np.sqrt(np.mean((yte - ypred_gp) ** 2)))
        # Uncertainty calibration: correlation between predicted variance and absolute error
        abs_err = np.abs(yte - ypred_gp)
        cal_corr = spearmanr(np.sqrt(yvar_gp), abs_err).statistic
        print(f"    GP:    Spearman={gp_spear:.4f}, RMSE={gp_rmse:.4f}  ({gp_time:.1f}s)")
        print(f"           Variance-vs-error Spearman: {cal_corr:.4f}  (positive = well-calibrated)")
        print(f"           Marginal likelihood loss: {gp_loss:.2f}")
    else:
        gp_spear = np.nan
        gp_rmse = np.nan
        cal_corr = np.nan
        print(f"    GP:    ALL RESTARTS FAILED  ({gp_time:.1f}s)")

    results.append({
        "split": split_name,
        "n_train": len(tr_idx),
        "n_test": len(te_idx),
        "n_metal": n_metal,
        "n_topo": n_topo,
        "ridge_spearman": ridge_spear,
        "ridge_rmse": ridge_rmse,
        "gp_success": gp_success,
        "gp_spearman": gp_spear,
        "gp_rmse": gp_rmse,
        "gp_loss": gp_loss,
        "gp_var_vs_err_spearman": cal_corr,
        "gp_time_sec": gp_time,
    })

    print(f"  Split done in {(time.time()-t0):.1f}s total")

# ============================================================================
# SAVE + SUMMARIZE
# ============================================================================

df_results = pd.DataFrame(results)
df_results.to_csv(OUT_FILE, index=False)

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(df_results[["split", "ridge_spearman", "gp_spearman",
                   "gp_var_vs_err_spearman", "gp_time_sec"]].to_string(index=False))
print(f"\nResults saved to {OUT_FILE}")