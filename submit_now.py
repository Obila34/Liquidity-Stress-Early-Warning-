#!/usr/bin/env python3
"""Fast high-score submission generator."""
import re
import numpy as np
import pandas as pd
from pathlib import Path
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score, log_loss

SEED = 42
TARGET = "liquidity_stress_next_30d"
ID_COL = "ID"

def safe_divide(a: pd.Series, b: pd.Series, eps: float = 1.0) -> pd.Series:
    return a / (b.abs() + eps)

def monthwise_stats(df: pd.DataFrame, cols: list, prefix: str) -> pd.DataFrame:
    arr = df[cols]
    out = pd.DataFrame(index=df.index)
    out[f"{prefix}__mean6"] = arr.mean(axis=1)
    out[f"{prefix}__std6"] = arr.std(axis=1)
    out[f"{prefix}__min6"] = arr.min(axis=1)
    out[f"{prefix}__max6"] = arr.max(axis=1)
    out[f"{prefix}__m1_m6_diff"] = arr[cols[0]] - arr[cols[-1]]
    out[f"{prefix}__recent3_mean"] = arr[cols[:3]].mean(axis=1)
    out[f"{prefix}__old3_mean"] = arr[cols[3:]].mean(axis=1)
    out[f"{prefix}__recent_old_gap"] = out[f"{prefix}__recent3_mean"] - out[f"{prefix}__old3_mean"]
    out[f"{prefix}__recent_old_ratio"] = safe_divide(
        out[f"{prefix}__recent3_mean"], out[f"{prefix}__old3_mean"]
    )
    return out

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    month_pattern = re.compile(r"^m([1-6])_(.+)$")
    groups = {}

    for c in work.columns:
        match = month_pattern.match(c)
        if match is None:
            continue
        month = int(match.group(1))
        rest = match.group(2)
        groups.setdefault(rest, []).append((month, c))

    feat_blocks = []
    for rest, month_cols in groups.items():
        if len(month_cols) < 3:
            continue
        month_cols = sorted(month_cols, key=lambda x: x[0])
        ordered_cols = [c for _, c in month_cols]
        feat_blocks.append(monthwise_stats(work, ordered_cols, rest))

    for m in range(1, 7):
        inflow_value_cols = [f"m{m}_transfer_from_bank_total_value", f"m{m}_received_total_value", f"m{m}_deposit_total_value"]
        outflow_value_cols = [f"m{m}_paybill_total_value", f"m{m}_merchantpay_total_value", f"m{m}_mm_send_total_value", f"m{m}_withdraw_total_value"]
        inflow_volume_cols = [f"m{m}_transfer_from_bank_volume", f"m{m}_received_volume", f"m{m}_deposit_volume"]
        outflow_volume_cols = [f"m{m}_paybill_volume", f"m{m}_merchantpay_volume", f"m{m}_mm_send_volume", f"m{m}_withdraw_volume"]

        if all(c in work.columns for c in inflow_value_cols + outflow_value_cols):
            work[f"m{m}_net_total_value"] = work[inflow_value_cols].sum(axis=1) - work[outflow_value_cols].sum(axis=1)
            work[f"m{m}_inflow_to_outflow_value_ratio"] = safe_divide(work[inflow_value_cols].sum(axis=1), work[outflow_value_cols].sum(axis=1))

        if all(c in work.columns for c in inflow_volume_cols + outflow_volume_cols):
            work[f"m{m}_net_volume"] = work[inflow_volume_cols].sum(axis=1) - work[outflow_volume_cols].sum(axis=1)
            work[f"m{m}_inflow_to_outflow_volume_ratio"] = safe_divide(work[inflow_volume_cols].sum(axis=1), work[outflow_volume_cols].sum(axis=1))

    engineered = pd.concat([work] + feat_blocks, axis=1)
    return engineered

def main():
    base_path = Path("Data")
    train = pd.read_csv(base_path / "Train.csv")
    test = pd.read_csv(base_path / "Test.csv")
    sample = pd.read_csv(base_path / "SampleSubmission.csv")

    print("[1/5] Building features...")
    train_fe = build_features(train)
    test_fe = build_features(test)

    print("[2/5] Processing categorical features...")
    # Detect all categorical columns more aggressively
    cat_cols = []
    for c in train_fe.columns:
        if c in [TARGET, ID_COL]:
            continue
        dtype_str = str(train_fe[c].dtype).lower()
        # Check if it contains any string/object dtypes or if values look non-numeric
        if ('object' in dtype_str or 'string' in dtype_str or 'category' in dtype_str):
            cat_cols.append(c)
        elif 'float' in dtype_str or 'int' in dtype_str:
            # For numeric dtypes, check if any values can't be converted to float
            try:
                pd.to_numeric(train_fe[c], errors='coerce')
            except:
                cat_cols.append(c)
    
    print(f"   Detected {len(cat_cols)} categorical features")
    
    # Ensure numeric columns are truly numeric
    for c in train_fe.columns:
        if c not in cat_cols and c not in [TARGET, ID_COL]:
            train_fe[c] = pd.to_numeric(train_fe[c], errors='coerce').fillna(0)
            test_fe[c] = pd.to_numeric(test_fe[c], errors='coerce').fillna(0)
    
    # Convert categorical columns to category type
    for c in cat_cols:
        train_fe[c] = train_fe[c].fillna("Unknown").astype(str)
        test_fe[c] = test_fe[c].fillna("Unknown").astype(str)

    all_feature_cols = [c for c in train_fe.columns if c not in [TARGET, ID_COL]]
    X = train_fe[all_feature_cols].copy()
    y = train_fe[TARGET].astype(int).values
    X_test = test_fe[all_feature_cols].copy()

    num_cols = [c for c in all_feature_cols if c not in cat_cols]
    X[num_cols] = X[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    X_test[num_cols] = X_test[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    print("[3/5] Training CatBoost (fast config)...")
    train_pool = Pool(X, y, cat_features=cat_cols)
    test_pool = Pool(X_test, cat_features=cat_cols)

    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        depth=6,
        learning_rate=0.08,
        iterations=600,
        l2_leaf_reg=6.0,
        random_strength=1.0,
        bagging_temperature=0.2,
        random_seed=SEED,
        verbose=False,
        thread_count=4,
    )
    model.fit(train_pool, verbose=False)

    print("[4/5] Generating predictions...")
    pred_proba = model.predict_proba(test_pool)[:, 1]
    pred_proba = np.clip(pred_proba, 1e-7, 1 - 1e-7)

    print("[5/5] Writing submission...")
    submission = sample.copy()
    submission["TargetLogLoss"] = pred_proba
    submission["TargetRAUC"] = pred_proba  # Use same for both columns for consistency
    submission.to_csv("submission_high_score.csv", index=False)

    print(f"\n✓ Submission saved to submission_high_score.csv")
    print(f"  Predicted probability range: [{pred_proba.min():.6f}, {pred_proba.max():.6f}]")

if __name__ == "__main__":
    main()
