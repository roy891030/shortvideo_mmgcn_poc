"""Align per-video visual + text features with per-video behavior labels.

Two feature sources are supported:

(1) Pre-extracted .npy folder (tiny dataset format):
    --visual-dir  data_raw/shortvideo_tiny/video_feature_total
    Each file is named "{pid}.npy" (shape: (D,) or (1, D)).

(2) Stacked matrix (recommendation format):
    --visual-matrix data_raw/video_rec_dataset/image_feat.npy
    --pid-list      data_raw/video_rec_dataset/pids.txt
    The matrix row i corresponds to pids.txt line i.

Text features can also be supplied either as a folder of .txt files
(title_en/asr_en, GloVe encoded on the fly) or as a pre-computed matrix.

Output:
  data_processed/behavior/X.npy    (N, D_visual + D_text)
  data_processed/behavior/Y.npy    (N, T)
  data_processed/behavior/meta.parquet  pid + n_impressions + target names

Usage example (tiny dataset path):
  python scripts/align_features_labels.py \
      --labels data_processed/behavior/video_behavior_labels.parquet \
      --visual-dir data_raw/shortvideo_tiny/video_feature_total \
      --text-dir data_raw/shortvideo_tiny/title_en \
      --out-dir data_processed/behavior
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


TARGETS = [
    "like_rate",
    "comment_rate",
    "follow_rate",
    "collect_rate",
    "forward_rate",
    "hate_rate",
    "effective_view_rate",
    "mean_watch_time",
]


def load_visual_dir(visual_dir: Path, pids: Iterable[int]) -> tuple[np.ndarray, list[int]]:
    feats, keep = [], []
    missing = 0
    for pid in pids:
        f = visual_dir / f"{pid}.npy"
        if not f.exists():
            missing += 1
            continue
        v = np.load(f).astype(np.float32).reshape(-1)
        feats.append(v)
        keep.append(int(pid))
    print(f"visual: kept {len(keep)} / {len(keep) + missing} (missing {missing})")
    if not feats:
        raise RuntimeError("no visual features matched any pid")
    return np.stack(feats), keep


def load_visual_matrix(matrix_path: Path, pid_list_path: Path,
                       pids: Iterable[int]) -> tuple[np.ndarray, list[int]]:
    mat = np.load(matrix_path, mmap_mode="r")
    pid_list = [int(x) for x in pid_list_path.read_text().splitlines() if x.strip()]
    pid_to_row = {p: i for i, p in enumerate(pid_list)}
    rows, keep = [], []
    for pid in pids:
        if pid in pid_to_row:
            rows.append(pid_to_row[pid])
            keep.append(int(pid))
    feats = mat[rows].astype(np.float32)
    print(f"visual matrix: kept {len(keep)} pids")
    return feats, keep


def load_text_dir_glove(text_dir: Path, pids: Iterable[int], glove_dim: int) -> np.ndarray:
    """Encode each video's title/ASR text via mean-pooled GloVe."""
    try:
        from torchtext.vocab import GloVe
        from torchtext.data.utils import get_tokenizer
    except Exception as exc:
        raise RuntimeError(
            "torchtext is required for --text-dir mode; "
            "install torchtext or use --text-matrix instead"
        ) from exc

    glove = GloVe(name="6B", dim=glove_dim)
    tokenizer = get_tokenizer("basic_english")
    feats = []
    zero_count = 0
    for pid in pids:
        f = text_dir / f"{pid}.txt"
        if not f.exists():
            feats.append(np.zeros(glove_dim, dtype=np.float32))
            zero_count += 1
            continue
        text = f.read_text(encoding="utf-8", errors="ignore").strip()
        tokens = tokenizer(text)
        vecs = [glove[t].numpy() for t in tokens if t in glove.stoi]
        if vecs:
            feats.append(np.mean(np.stack(vecs), axis=0).astype(np.float32))
        else:
            feats.append(np.zeros(glove_dim, dtype=np.float32))
            zero_count += 1
    print(f"text: {zero_count} videos had zero embedding (will impute)")
    return np.stack(feats)


def load_text_matrix(matrix_path: Path, pid_list_path: Path,
                     pids: Iterable[int]) -> np.ndarray:
    mat = np.load(matrix_path, mmap_mode="r")
    pid_list = [int(x) for x in pid_list_path.read_text().splitlines() if x.strip()]
    pid_to_row = {p: i for i, p in enumerate(pid_list)}
    rows = [pid_to_row[p] for p in pids if p in pid_to_row]
    return mat[rows].astype(np.float32)


def impute_zero_rows(feats: np.ndarray) -> np.ndarray:
    zero_mask = ~np.any(feats != 0, axis=1)
    if zero_mask.any():
        nonzero_mean = feats[~zero_mask].mean(axis=0)
        feats[zero_mask] = nonzero_mean
        print(f"imputed {int(zero_mask.sum())} all-zero text rows with global mean")
    return feats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--visual-dir", type=Path, default=None)
    parser.add_argument("--visual-matrix", type=Path, default=None)
    parser.add_argument("--visual-pid-list", type=Path, default=None)
    parser.add_argument("--text-dir", type=Path, default=None)
    parser.add_argument("--text-matrix", type=Path, default=None)
    parser.add_argument("--text-pid-list", type=Path, default=None)
    parser.add_argument("--glove-dim", type=int, default=50)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    labels = pd.read_parquet(args.labels)
    pids = labels["pid"].astype(int).tolist()
    print(f"loaded {len(pids)} candidate pids from labels")

    if args.visual_dir is not None:
        visual, kept_pids = load_visual_dir(args.visual_dir, pids)
    elif args.visual_matrix is not None:
        if args.visual_pid_list is None:
            raise ValueError("--visual-matrix needs --visual-pid-list")
        visual, kept_pids = load_visual_matrix(args.visual_matrix, args.visual_pid_list, pids)
    else:
        raise ValueError("provide either --visual-dir or --visual-matrix")

    # subset labels to kept_pids order to keep alignment
    labels = labels.set_index("pid").loc[kept_pids].reset_index()

    if args.text_dir is not None:
        text = load_text_dir_glove(args.text_dir, kept_pids, args.glove_dim)
    elif args.text_matrix is not None:
        if args.text_pid_list is None:
            raise ValueError("--text-matrix needs --text-pid-list")
        text = load_text_matrix(args.text_matrix, args.text_pid_list, kept_pids)
    else:
        # no text source: use zero vector (model will downweight)
        text = np.zeros((len(kept_pids), args.glove_dim), dtype=np.float32)

    text = impute_zero_rows(text)

    X = np.concatenate([visual, text], axis=1).astype(np.float32)
    Y = labels[TARGETS].astype(np.float32).values
    weights = labels["n_impressions"].astype(np.float32).values

    np.save(args.out_dir / "X.npy", X)
    np.save(args.out_dir / "Y.npy", Y)
    np.save(args.out_dir / "weights.npy", weights)
    meta = labels[["pid", "n_impressions", *TARGETS]].copy()
    meta.to_parquet(args.out_dir / "meta.parquet", index=False)

    print(f"\nX: {X.shape}, Y: {Y.shape}, weights: {weights.shape}")
    print(f"visual dim: {visual.shape[1]}, text dim: {text.shape[1]}")
    print(f"wrote artifacts to {args.out_dir}")


if __name__ == "__main__":
    main()
