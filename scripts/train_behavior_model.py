"""Train per-video behavior predictors (multi-output regression).

For each target column we fit an independent LightGBM regressor and
write a model artifact + a JSON metrics report.

Targets:
  like_rate, comment_rate, follow_rate, collect_rate, forward_rate,
  hate_rate, effective_view_rate, mean_watch_time

Loss / metric choices:
  - Rates (0..1): MSE training, report MAE / Spearman / weighted AUC
    (AUC is computed by binarising the rate at its median).
  - mean_watch_time: log1p transform + MSE; report MAE / RMSE / Spearman.

Usage:
  python scripts/train_behavior_model.py \
      --in-dir data_processed/behavior \
      --out-dir data_processed/behavior/models \
      --test-frac 0.1 --seed 20260519
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr


RATE_TARGETS = [
    "like_rate", "comment_rate", "follow_rate", "collect_rate",
    "forward_rate", "hate_rate", "effective_view_rate",
]
REGRESS_TARGETS = ["mean_watch_time"]
ALL_TARGETS = RATE_TARGETS + REGRESS_TARGETS


def fit_one(X_tr, y_tr, X_va, y_va, w_tr, w_va, transform: bool):
    """Train one LightGBM model. Returns (model, predictions on val)."""
    try:
        import lightgbm as lgb
    except Exception as exc:
        raise RuntimeError("lightgbm is required: pip install lightgbm") from exc

    y_tr_t = np.log1p(y_tr) if transform else y_tr
    y_va_t = np.log1p(y_va) if transform else y_va

    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        reg_lambda=0.1,
        verbose=-1,
    )
    model.fit(
        X_tr, y_tr_t, sample_weight=w_tr,
        eval_set=[(X_va, y_va_t)], eval_sample_weight=[w_va],
        callbacks=[lgb.early_stopping(50, verbose=False),
                   lgb.log_evaluation(period=0)],
    )
    pred_t = model.predict(X_va)
    pred = np.expm1(pred_t) if transform else pred_t
    return model, pred


def evaluate(y_true, y_pred, kind: str) -> dict:
    out = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }
    rho, _ = spearmanr(y_true, y_pred)
    out["spearman"] = float(rho) if not np.isnan(rho) else None
    if kind == "rate":
        # Binarise at median to get a coarse AUC; useful for skewed rates.
        thresh = float(np.median(y_true))
        labels = (y_true > thresh).astype(int)
        if labels.min() != labels.max():
            try:
                out["auc_above_median"] = float(roc_auc_score(labels, y_pred))
            except ValueError:
                out["auc_above_median"] = None
        else:
            out["auc_above_median"] = None
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260519)
    args = parser.parse_args()

    X = np.load(args.in_dir / "X.npy")
    Y = np.load(args.in_dir / "Y.npy")
    W = np.load(args.in_dir / "weights.npy")
    meta = pd.read_parquet(args.in_dir / "meta.parquet")
    print(f"X: {X.shape}, Y: {Y.shape}")

    # train / val / test split (stratification not meaningful on rates → random)
    X_tv, X_te, Y_tv, Y_te, W_tv, W_te, idx_tv, idx_te = train_test_split(
        X, Y, W, np.arange(len(X)), test_size=args.test_frac, random_state=args.seed
    )
    val_size = args.val_frac / (1 - args.test_frac)
    X_tr, X_va, Y_tr, Y_va, W_tr, W_va = train_test_split(
        X_tv, Y_tv, W_tv, test_size=val_size, random_state=args.seed
    )
    print(f"train: {len(X_tr)}, val: {len(X_va)}, test: {len(X_te)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {}
    test_preds = np.zeros_like(Y_te)
    for j, name in enumerate(ALL_TARGETS):
        transform = name == "mean_watch_time"
        kind = "regress" if transform else "rate"
        print(f"\n=== training {name} ({kind}) ===")
        model, _ = fit_one(
            X_tr, Y_tr[:, j], X_va, Y_va[:, j],
            W_tr, W_va, transform=transform,
        )
        # eval on holdout test
        pred_t = model.predict(X_te)
        pred = np.expm1(pred_t) if transform else pred_t
        test_preds[:, j] = pred
        m = evaluate(Y_te[:, j], pred, kind=kind)
        metrics[name] = m
        print({k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()})

        # save model
        booster_path = args.out_dir / f"{name}.txt"
        model.booster_.save_model(str(booster_path))

    # write metrics report
    report = {
        "n_train": int(len(X_tr)),
        "n_val": int(len(X_va)),
        "n_test": int(len(X_te)),
        "feature_dim": int(X.shape[1]),
        "targets": ALL_TARGETS,
        "metrics": metrics,
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(report, indent=2))

    # save test predictions for inspection
    test_meta = meta.iloc[idx_te].reset_index(drop=True)
    for j, name in enumerate(ALL_TARGETS):
        test_meta[f"pred_{name}"] = test_preds[:, j]
    test_meta.to_parquet(args.out_dir / "test_predictions.parquet", index=False)

    print("\n=== done ===")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
