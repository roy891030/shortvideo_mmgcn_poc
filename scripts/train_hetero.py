"""Train the Pipeline C heterogeneous GNN for video performance prediction.

Loss (train mask only, exposure-weighted)
-----------------------------------------
  7 rate metrics : weighted MSE
  mean_watch_time: weighted Huber (log-space, δ=1.0)

Evaluation (test set, per-target)
----------------------------------
  MAE / RMSE / Spearman / nDCG@10 / AUC@median

Output
------
  data_processed/behavior/models_gnn/
    best_model.pt         — best validation checkpoint
    metrics_gnn.json      — test-set metrics

Usage
-----
  python scripts/train_hetero.py
  python scripts/train_hetero.py --model hgt --epochs 300 --seed 0
  python scripts/train_hetero.py --d 256 --layers 3
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent))
from hetero_gnn import build_model

GRAPH_PATH   = Path("data_processed/behavior/hetero_graph.pt")
LGBM_METRICS = Path("data_processed/behavior/models/metrics.json")
TARGETS = [
    "like_rate", "comment_rate", "follow_rate", "collect_rate",
    "forward_rate", "hate_rate", "effective_view_rate", "mean_watch_time",
]


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def weighted_loss(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Exposure-weighted multi-task loss.

    Rates (col 0-6): weighted MSE in original space.
    watch_time (col 7): weighted Huber in log space (target already log1p-transformed).
    """
    w = weight / (weight.sum() + 1e-8)  # normalise so total weight = 1
    w = w.unsqueeze(1)                   # [N, 1]

    rate_loss = (w * (pred[:, :7] - target[:, :7]) ** 2).sum()
    wt_loss   = (w * F.huber_loss(pred[:, 7:8], target[:, 7:8], delta=1.0, reduction="none")).sum()
    return rate_loss + wt_loss


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def compute_metrics(pred_np: np.ndarray, true_np: np.ndarray) -> dict[str, dict[str, float]]:
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score, ndcg_score

    metrics: dict[str, dict[str, float]] = {}
    N = len(true_np)

    for i, name in enumerate(TARGETS):
        y_true = true_np[:, i]
        y_pred = pred_np[:, i]
        # Undo log1p for watch_time
        if name == "mean_watch_time":
            y_true = np.expm1(y_true)
            y_pred = np.expm1(np.clip(y_pred, -30, 30))

        mae  = float(np.mean(np.abs(y_pred - y_true)))
        rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))

        sp, _ = spearmanr(y_true, y_pred)
        sp = float(sp) if not np.isnan(sp) else 0.0

        # nDCG@10: use true values as relevance, predict-based ranking
        # Shift to non-negative for ndcg_score
        shifted = y_true - y_true.min()
        try:
            ndcg = float(ndcg_score(shifted.reshape(1, -1), y_pred.reshape(1, -1), k=10))
        except Exception:
            ndcg = float("nan")

        # AUC@median: classify above/below median
        median = np.median(y_true)
        binary = (y_true >= median).astype(int)
        try:
            if binary.sum() > 0 and binary.sum() < N:
                auc = float(roc_auc_score(binary, y_pred))
            else:
                auc = float("nan")
        except Exception:
            auc = float("nan")

        metrics[name] = {"MAE": mae, "RMSE": rmse, "Spearman": sp, "nDCG@10": ndcg, "AUC@median": auc}

    return metrics


def print_metrics(metrics: dict[str, dict[str, float]], label: str = "GNN") -> None:
    header = f"\n{'Target':<22} {'MAE':>8} {'RMSE':>8} {'Spearman':>10} {'nDCG@10':>9} {'AUC@med':>9}"
    print(f"\n{'─'*70}")
    print(f"  {label} — test-set metrics")
    print(header)
    print("─" * 70)
    for name, m in metrics.items():
        print(f"  {name:<20} {m['MAE']:>8.4f} {m['RMSE']:>8.4f} "
              f"{m['Spearman']:>10.4f} {m['nDCG@10']:>9.4f} {m['AUC@median']:>9.4f}")
    print("─" * 70)


def compare_with_lgbm(gnn_metrics: dict, lgbm_path: Path) -> None:
    if not lgbm_path.exists():
        print(f"\n[compare] LightGBM metrics not found at {lgbm_path}; skipping comparison.")
        return

    with open(lgbm_path) as f:
        lgbm_raw = json.load(f)
    # Support both flat {"like_rate": {...}} and nested {"metrics": {"like_rate": {...}}}
    lgbm_per_target = lgbm_raw.get("metrics", lgbm_raw)

    print(f"\n{'═'*90}")
    print("  Pipeline C (HeteroGNN) vs Pipeline B (LightGBM) — Spearman & MAE (test set)")
    print(f"  {'Target':<22} {'GNN Spearman':>14} {'LGBM Spearman':>15} {'GNN MAE':>10} {'LGBM MAE':>10}")
    print("─" * 75)
    for name in TARGETS:
        gnn_sp  = gnn_metrics.get(name, {}).get("Spearman", float("nan"))
        gnn_mae = gnn_metrics.get(name, {}).get("MAE", float("nan"))
        lgbm_m  = lgbm_per_target.get(name, {})
        lgbm_sp = lgbm_m.get("spearman", float("nan"))
        lgbm_mae = lgbm_m.get("mae", float("nan"))
        print(f"  {name:<22} {gnn_sp:>14.4f} {lgbm_sp:>15.4f} {gnn_mae:>10.4f} {lgbm_mae:>10.4f}")
    print(f"{'═'*90}\n")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load graph
    graph_path = args.graph
    print(f"Loading graph from {graph_path}...")
    data = torch.load(graph_path, weights_only=False)
    data = data.to(device)

    # log1p-transform watch_time target (col 7)
    y_raw = data["video"].y.clone()
    data["video"].y[:, 7] = torch.log1p(data["video"].y[:, 7].clamp(min=0))

    train_mask = data["video"].train_mask
    val_mask   = data["video"].val_mask
    test_mask  = data["video"].test_mask
    weight     = data["video"].weight

    model = build_model(data, args).to(device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "best_model.pt"

    # Warm-up forward to initialise LazyLinear parameters
    model.eval()
    with torch.no_grad():
        _ = model(data.x_dict, data.edge_index_dict)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=20
    )

    best_val_loss = float("inf")
    patience_counter = 0
    t0 = time.time()

    print(f"\nTraining {args.model.upper()} — {args.epochs} epochs, patience={args.patience}")
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(1, args.epochs + 1):
        # --- train ---
        model.train()
        optimizer.zero_grad()
        out = model(data.x_dict, data.edge_index_dict)           # [N_v, 8]
        loss = weighted_loss(out[train_mask], data["video"].y[train_mask], weight[train_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # --- val ---
        model.eval()
        with torch.no_grad():
            val_out  = model(data.x_dict, data.edge_index_dict)
            val_loss = weighted_loss(
                val_out[val_mask], data["video"].y[val_mask], weight[val_mask]
            ).item()

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_counter += 1

        if epoch % 20 == 0 or epoch == 1:
            elapsed = time.time() - t0
            print(f"  epoch {epoch:4d} | train_loss={loss.item():.4f} "
                  f"val_loss={val_loss:.4f} | best={best_val_loss:.4f} "
                  f"| patience={patience_counter}/{args.patience} | {elapsed:.0f}s")

        if patience_counter >= args.patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    # --- test ---
    print(f"\nLoading best checkpoint from {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()
    with torch.no_grad():
        final_out = model(data.x_dict, data.edge_index_dict)

    pred_np = final_out[test_mask].cpu().numpy()
    true_np = data["video"].y[test_mask].cpu().numpy()   # log-space watch_time for col 7

    metrics = compute_metrics(pred_np, true_np)
    print_metrics(metrics, label=f"HeteroGNN ({args.model.upper()})")

    metrics_path = out_dir / "metrics_gnn.json"
    with open(metrics_path, "w") as f:
        json.dump({"model": args.model, "metrics": metrics}, f, indent=2)
    print(f"Metrics saved → {metrics_path}")

    compare_with_lgbm(
        {k: v for k, v in metrics.items()},
        LGBM_METRICS,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Pipeline C HeteroGNN")
    parser.add_argument("--graph",   type=Path, default=GRAPH_PATH)
    parser.add_argument("--model",   type=str, default="sage", choices=["sage", "hgt"])
    parser.add_argument("--d",       type=int, default=128,   help="Hidden dim")
    parser.add_argument("--layers",  type=int, default=2,     help="GNN layers")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr",      type=float, default=1e-3)
    parser.add_argument("--epochs",  type=int, default=300)
    parser.add_argument("--patience",type=int, default=50)
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--out-dir", type=str, default="data_processed/behavior/models_gnn")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
