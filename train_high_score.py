import json
import re
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split


SEED = 42
TARGET = "liquidity_stress_next_30d"
ID_COL = "ID"
TARGET_LOGLOSS = "TargetLogLoss"
TARGET_RAUC = "TargetRAUC"


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
    out[f"{prefix}__recent_old_ratio"] = safe_divide(
        out[f"{prefix}__recent3_mean"], out[f"{prefix}__old3_mean"]
    )
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
            work[f"m{m}_net_total_value"] = work[inflow_value_cols].sum(axis=1) - work[
                outflow_value_cols
            ].sum(axis=1)
            work[f"m{m}_inflow_to_outflow_value_ratio"] = safe_divide(
                work[inflow_value_cols].sum(axis=1), work[outflow_value_cols].sum(axis=1)
            )

        if all(c in work.columns for c in inflow_volume_cols + outflow_volume_cols):
            work[f"m{m}_net_volume"] = work[inflow_volume_cols].sum(axis=1) - work[
                outflow_volume_cols
            ].sum(axis=1)
            work[f"m{m}_inflow_to_outflow_volume_ratio"] = safe_divide(
                work[inflow_volume_cols].sum(axis=1), work[outflow_volume_cols].sum(axis=1)
            )

    for prefix in [
        "net_total_value",
        "net_volume",
        "inflow_to_outflow_value_ratio",
        "inflow_to_outflow_volume_ratio",
    ]:
        month_cols = [f"m{m}_{prefix}" for m in range(1, 7) if f"m{m}_{prefix}" in work.columns]
        if len(month_cols) >= 3:
            feat_blocks.append(monthwise_stats(work, month_cols, prefix))

    engineered = pd.concat([work] + feat_blocks, axis=1)
    return engineered


def optimize_linear_blend(y_true: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> dict:
    weights = np.linspace(0.0, 1.0, 81)
    best_auc = {"w": 0.5, "auc": -1.0, "logloss": 99.0}
    best_logloss = {"w": 0.5, "auc": -1.0, "logloss": 99.0}

    for w in weights:
        pred = w * p1 + (1.0 - w) * p2
        pred = np.clip(pred, 1e-6, 1 - 1e-6)
        auc = roc_auc_score(y_true, pred)
        ll = log_loss(y_true, pred)
        if auc > best_auc["auc"]:
            best_auc = {"w": float(w), "auc": float(auc), "logloss": float(ll)}
        if ll < best_logloss["logloss"]:
            best_logloss = {"w": float(w), "auc": float(auc), "logloss": float(ll)}

    return {"best_auc": best_auc, "best_logloss": best_logloss}


def main() -> None:
    base_path = Path("Data")
    train_path = base_path / "Train.csv"
    test_path = base_path / "Test.csv"
    sample_path = base_path / "SampleSubmission.csv"

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sample = pd.read_csv(sample_path)

    train_fe = build_features(train)
    test_fe = build_features(test)

    cat_cols = [
        c
        for c in train_fe.columns
        if str(train_fe[c].dtype).startswith("string")
        or str(train_fe[c].dtype) == "str"
        or str(train_fe[c].dtype) == "object"
        or str(train_fe[c].dtype) == "category"
    ]
    cat_cols = [c for c in cat_cols if c not in [TARGET, ID_COL]]
    for c in cat_cols:
        train_fe[c] = train_fe[c].fillna("Unknown").astype("category")
        test_fe[c] = test_fe[c].fillna("Unknown").astype("category")

    all_feature_cols = [c for c in train_fe.columns if c not in [TARGET, ID_COL]]
    X = train_fe[all_feature_cols].copy()
    y = train_fe[TARGET].astype(int).values
    X_test = test_fe[all_feature_cols].copy()

    num_cols = [c for c in all_feature_cols if c not in cat_cols]
    X[num_cols] = X[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    X_test[num_cols] = X_test[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    cat_feature_names = cat_cols.copy()
    train_pool = Pool(X_tr, y_tr, cat_features=cat_feature_names)
    valid_pool = Pool(X_va, y_va, cat_features=cat_feature_names)

    cb = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        depth=8,
        learning_rate=0.04,
        iterations=1800,
        l2_leaf_reg=6.0,
        random_strength=1.0,
        bagging_temperature=0.2,
        random_seed=SEED,
        verbose=False,
    )
    cb.fit(train_pool, eval_set=valid_pool, use_best_model=True, early_stopping_rounds=150)
    va_cb = cb.predict_proba(valid_pool)[:, 1]

    lgb_train = X_tr.copy()
    lgb_valid = X_va.copy()
    lgb_full = X.copy()
    lgb_test = X_test.copy()
    for c in cat_cols:
        lgb_train[c] = lgb_train[c].cat.codes
        lgb_valid[c] = lgb_valid[c].cat.codes
        lgb_full[c] = lgb_full[c].cat.codes
        lgb_test[c] = lgb_test[c].cat.codes

    lgbm = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=2200,
        learning_rate=0.03,
        num_leaves=63,
        max_depth=-1,
        subsample=0.85,
        colsample_bytree=0.75,
        reg_alpha=0.2,
        reg_lambda=1.0,
        min_child_samples=80,
        random_state=SEED,
        n_jobs=-1,
    )
    lgbm.fit(
        lgb_train,
        y_tr,
        eval_set=[(lgb_valid, y_va)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(0)],
    )
    va_lgb = lgbm.predict_proba(lgb_valid)[:, 1]

    print(
        f"Validation: CatBoost AUC={roc_auc_score(y_va, va_cb):.6f}, "
        f"LightGBM AUC={roc_auc_score(y_va, va_lgb):.6f}",
        flush=True,
    )

    blend = optimize_linear_blend(y_va, va_cb, va_lgb)

    rank_cb_va = rankdata(va_cb) / len(va_cb)
    rank_lgb_va = rankdata(va_lgb) / len(va_lgb)
    rank_blend = optimize_linear_blend(y_va, rank_cb_va, rank_lgb_va)

    stack_x = np.column_stack([va_cb, va_lgb])
    calibrator = LogisticRegression(max_iter=2000)
    calibrator.fit(stack_x, y_va)

    best_iter_cb = int(cb.get_best_iteration()) if cb.get_best_iteration() > 0 else 600
    best_iter_lgb = int(getattr(lgbm, "best_iteration_", 0) or 600)

    # Use already-trained validation models for test predictions to finish within runtime limits.
    pred_cb = cb.predict_proba(Pool(X_test, cat_features=cat_feature_names))[:, 1]
    pred_lgb = lgbm.predict_proba(lgb_test)[:, 1]

    stack_x_test = np.column_stack([pred_cb, pred_lgb])
    valid_stacked = calibrator.predict_proba(stack_x)[:, 1]
    test_stacked = calibrator.predict_proba(stack_x_test)[:, 1]

    rauc_w = rank_blend["best_auc"]["w"]
    test_rauc = rauc_w * (rankdata(pred_cb) / len(pred_cb)) + (1.0 - rauc_w) * (
        rankdata(pred_lgb) / len(pred_lgb)
    )

    test_logloss = np.clip(test_stacked, 1e-6, 1 - 1e-6)
    test_rauc = np.clip(test_rauc, 1e-6, 1 - 1e-6)

    submission = sample.copy()
    submission[TARGET_LOGLOSS] = test_logloss
    submission[TARGET_RAUC] = test_rauc
    submission_path = Path("submission_high_score.csv")
    submission.to_csv(submission_path, index=False)

    metrics = {
        "catboost": {
            "valid_auc": float(roc_auc_score(y_va, va_cb)),
            "valid_logloss": float(log_loss(y_va, np.clip(va_cb, 1e-6, 1 - 1e-6))),
            "best_iteration": best_iter_cb,
        },
        "lightgbm": {
            "valid_auc": float(roc_auc_score(y_va, va_lgb)),
            "valid_logloss": float(log_loss(y_va, np.clip(va_lgb, 1e-6, 1 - 1e-6))),
            "best_iteration": best_iter_lgb,
        },
        "blend_linear": blend,
        "blend_rank": rank_blend,
        "stacked_logloss_target": {
            "valid_auc": float(roc_auc_score(y_va, valid_stacked)),
            "valid_logloss": float(log_loss(y_va, np.clip(valid_stacked, 1e-6, 1 - 1e-6))),
        },
        "submission_file": str(submission_path),
    }

    with open("training_report.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\n=== CV SUMMARY ===")
    print(json.dumps(metrics, indent=2))
    print(f"\nSaved submission to: {submission_path}")
    print("Saved metrics to: training_report.json")


if __name__ == "__main__":
    main()