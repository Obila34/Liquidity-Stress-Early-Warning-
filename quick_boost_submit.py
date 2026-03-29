#!/usr/bin/env python3
import re
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

SEED = 42
TARGET = "liquidity_stress_next_30d"
ID_COL = "ID"


def safe_divide(a: pd.Series, b: pd.Series, eps: float = 1.0) -> pd.Series:
    return a / (b.abs() + eps)


def monthwise_stats(df: pd.DataFrame, cols: list[str], prefix: str) -> pd.DataFrame:
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
    out[f"{prefix}__recent_old_ratio"] = safe_divide(out[f"{prefix}__recent3_mean"], out[f"{prefix}__old3_mean"])
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    month_pattern = re.compile(r"^m([1-6])_(.+)$")
    groups: dict[str, list[tuple[int, str]]] = {}

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
        inflow_value_cols = [
            f"m{m}_transfer_from_bank_total_value",
            f"m{m}_received_total_value",
            f"m{m}_deposit_total_value",
        ]
        outflow_value_cols = [
            f"m{m}_paybill_total_value",
            f"m{m}_merchantpay_total_value",
            f"m{m}_mm_send_total_value",
            f"m{m}_withdraw_total_value",
        ]
        inflow_volume_cols = [
            f"m{m}_transfer_from_bank_volume",
            f"m{m}_received_volume",
            f"m{m}_deposit_volume",
        ]
        outflow_volume_cols = [
            f"m{m}_paybill_volume",
            f"m{m}_merchantpay_volume",
            f"m{m}_mm_send_volume",
            f"m{m}_withdraw_volume",
        ]

        if all(c in work.columns for c in inflow_value_cols + outflow_value_cols):
            work[f"m{m}_net_total_value"] = work[inflow_value_cols].sum(axis=1) - work[outflow_value_cols].sum(axis=1)
            work[f"m{m}_inflow_to_outflow_value_ratio"] = safe_divide(
                work[inflow_value_cols].sum(axis=1), work[outflow_value_cols].sum(axis=1)
            )

        if all(c in work.columns for c in inflow_volume_cols + outflow_volume_cols):
            work[f"m{m}_net_volume"] = work[inflow_volume_cols].sum(axis=1) - work[outflow_volume_cols].sum(axis=1)
            work[f"m{m}_inflow_to_outflow_volume_ratio"] = safe_divide(
                work[inflow_volume_cols].sum(axis=1), work[outflow_volume_cols].sum(axis=1)
            )

    return pd.concat([work] + feat_blocks, axis=1)


def optimize_rank_weights(y_val: np.ndarray, pred_matrix: np.ndarray) -> np.ndarray:
    best_auc = -1.0
    best_w = np.array([1 / pred_matrix.shape[1]] * pred_matrix.shape[1])

    # Small robust grid over 3-model simplex
    grid = np.linspace(0.1, 0.8, 8)
    for w1 in grid:
        for w2 in grid:
            w3 = 1.0 - w1 - w2
            if w3 < 0.05 or w3 > 0.85:
                continue
            w = np.array([w1, w2, w3])
            blended = np.zeros(pred_matrix.shape[0])
            for i in range(pred_matrix.shape[1]):
                blended += w[i] * (rankdata(pred_matrix[:, i]) / pred_matrix.shape[0])
            auc = roc_auc_score(y_val, blended)
            if auc > best_auc:
                best_auc = auc
                best_w = w
    return best_w


def main() -> None:
    base = Path("Data")
    train = pd.read_csv(base / "Train.csv")
    test = pd.read_csv(base / "Test.csv")
    sample = pd.read_csv(base / "SampleSubmission.csv")

    print("[1/7] Feature engineering...")
    train_fe = build_features(train)
    test_fe = build_features(test)

    base_cat_cols = [
        c
        for c in train.columns
        if c not in [TARGET, ID_COL]
        and any(k in str(train[c].dtype).lower() for k in ["object", "string", "category", "str"])
    ]

    feature_cols = [c for c in train_fe.columns if c not in [TARGET, ID_COL]]
    X = train_fe[feature_cols].copy()
    y = train_fe[TARGET].astype(int).values
    X_test = test_fe[feature_cols].copy()

    cat_cols = [c for c in base_cat_cols if c in X.columns]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    for c in cat_cols:
        X[c] = X[c].fillna("Unknown").astype(str)
        X_test[c] = X_test[c].fillna("Unknown").astype(str)

    X[num_cols] = X[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    X_test[num_cols] = X_test[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    print(f"[2/7] Features ready: {len(feature_cols)} total, {len(cat_cols)} categorical")

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    tr_idx, va_idx = next(splitter.split(X, y))
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    train_pool = Pool(X_tr, y_tr, cat_features=cat_cols)
    valid_pool = Pool(X_va, y_va, cat_features=cat_cols)
    test_pool = Pool(X_test, cat_features=cat_cols)

    configs = [
        {"name": "m1", "eval_metric": "AUC", "depth": 8, "learning_rate": 0.06, "iterations": 1100, "seed": SEED},
        {"name": "m2", "eval_metric": "Logloss", "depth": 7, "learning_rate": 0.07, "iterations": 1000, "seed": SEED + 1},
        {"name": "m3", "eval_metric": "AUC", "depth": 6, "learning_rate": 0.08, "iterations": 900, "seed": SEED + 2},
    ]

    val_preds = []
    test_preds = []
    best_iters = []

    print("[3/7] Training validation models...")
    for cfg in configs:
        model = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric=cfg["eval_metric"],
            depth=cfg["depth"],
            learning_rate=cfg["learning_rate"],
            iterations=cfg["iterations"],
            l2_leaf_reg=6.0,
            random_strength=1.0,
            bagging_temperature=0.25,
            random_seed=cfg["seed"],
            verbose=False,
            thread_count=4,
        )
        model.fit(train_pool, eval_set=valid_pool, use_best_model=True, early_stopping_rounds=120)
        p_val = model.predict_proba(valid_pool)[:, 1]
        p_test = model.predict_proba(test_pool)[:, 1]
        best_iter = int(model.get_best_iteration()) if model.get_best_iteration() > 0 else cfg["iterations"]
        best_iters.append(max(best_iter, 250))
        val_preds.append(p_val)
        test_preds.append(p_test)
        print(f"   {cfg['name']}: val_auc={roc_auc_score(y_va, p_val):.6f}, val_logloss={log_loss(y_va, np.clip(p_val,1e-7,1-1e-7)):.6f}, best_iter={best_iters[-1]}")

    val_mat = np.column_stack(val_preds)
    test_mat = np.column_stack(test_preds)

    print("[4/7] Optimizing blends...")
    calibrator = LogisticRegression(max_iter=2000)
    calibrator.fit(val_mat, y_va)
    val_logloss_blend = calibrator.predict_proba(val_mat)[:, 1]

    rank_w = optimize_rank_weights(y_va, val_mat)
    val_rauc_blend = np.zeros(len(y_va))
    for i in range(val_mat.shape[1]):
        val_rauc_blend += rank_w[i] * (rankdata(val_mat[:, i]) / len(y_va))

    print(f"   blended_logloss_val={log_loss(y_va, np.clip(val_logloss_blend,1e-7,1-1e-7)):.6f}")
    print(f"   blended_rauc_val_auc={roc_auc_score(y_va, val_rauc_blend):.6f}")
    print(f"   rank_weights={rank_w.round(4).tolist()}")

    print("[5/7] Skipping full retrain for speed; using tuned ensemble predictions...")
    full_test_mat = test_mat.copy()

    print("[6/7] Building submission candidates...")
    target_logloss = np.clip(calibrator.predict_proba(full_test_mat)[:, 1], 1e-7, 1 - 1e-7)
    target_rauc = np.zeros(full_test_mat.shape[0])
    for i in range(full_test_mat.shape[1]):
        target_rauc += rank_w[i] * (rankdata(full_test_mat[:, i]) / full_test_mat.shape[0])
    target_rauc = np.clip(target_rauc, 1e-7, 1 - 1e-7)

    submission = sample.copy()
    submission["TargetLogLoss"] = target_logloss
    submission["TargetRAUC"] = target_rauc
    submission.to_csv("submission_high_score.csv", index=False)

    # Backup candidate with slight perturbation blend for optional second try
    submission_alt = sample.copy()
    submission_alt["TargetLogLoss"] = np.clip(0.6 * target_logloss + 0.4 * np.mean(full_test_mat, axis=1), 1e-7, 1 - 1e-7)
    submission_alt["TargetRAUC"] = np.clip(0.5 * target_rauc + 0.5 * (rankdata(np.mean(full_test_mat, axis=1)) / full_test_mat.shape[0]), 1e-7, 1 - 1e-7)
    submission_alt.to_csv("submission_high_score_alt.csv", index=False)

    print("[7/7] Done")
    print("   wrote submission_high_score.csv")
    print("   wrote submission_high_score_alt.csv")


if __name__ == "__main__":
    main()
