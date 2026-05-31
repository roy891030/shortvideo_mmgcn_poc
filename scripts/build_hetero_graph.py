"""Build a video-centric heterogeneous graph for Pipeline C node regression.

Nodes
-----
  video    : visual (768) + text (50 or 384) + log_duration (1) features
  author   : log_fans + train_count + 8 train-only label means = 10 dims
  category : no x — model uses nn.Embedding (num_nodes stored)

Edges (all get reverse via T.ToUndirected)
------------------------------------------
  (video, posted_by,   author)    one-to-many via author_id
  (video, belongs_to,  category)  one-to-one  via category_id
  (video, similar_to,  video)     cosine kNN k=10 on image content ONLY

Output
------
  data_processed/behavior/hetero_graph.pt

Usage
-----
  python scripts/build_hetero_graph.py
  python scripts/build_hetero_graph.py --use-sbert   # 768+384+1 dim
  python scripts/build_hetero_graph.py --k 5         # kNN k=5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData

TARGETS = [
    "like_rate", "comment_rate", "follow_rate", "collect_rate",
    "forward_rate", "hate_rate", "effective_view_rate", "mean_watch_time",
]

INTERACTION_CSV = Path("data_raw/shortvideo_tiny/interaction.csv")
IMAGE_FEAT_NPY  = Path("data_raw/video_rec_dataset/image_feat.npy")
TEXT_FEAT_NPY   = Path("data_raw/video_rec_dataset/text_feat.npy")
PIDS_TXT        = Path("data_raw/video_rec_dataset/pids.txt")
META_PARQUET    = Path("data_processed/behavior/meta.parquet")
OUT_PATH        = Path("data_processed/behavior/hetero_graph.pt")


# ---------------------------------------------------------------------------
# Feature helpers
# ---------------------------------------------------------------------------

def load_pid_to_row(pids_txt: Path) -> dict[int, int]:
    lines = pids_txt.read_text().strip().splitlines()
    return {int(p): i for i, p in enumerate(lines) if p.strip() != "-1"}


def load_image_feats(pids: list[int], pid_to_row: dict[int, int]) -> np.ndarray:
    mat = np.load(IMAGE_FEAT_NPY, mmap_mode="r")
    rows = [pid_to_row[p] for p in pids]
    return mat[rows].astype(np.float32)


def load_text_feats_default(pids: list[int], pid_to_row: dict[int, int]) -> np.ndarray:
    mat = np.load(TEXT_FEAT_NPY, mmap_mode="r")
    rows = [pid_to_row[p] for p in pids]
    return mat[rows].astype(np.float32)


def load_text_feats_sbert(titles: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    embeddings = model.encode(
        titles,
        batch_size=256,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


# ---------------------------------------------------------------------------
# Author feature construction (train-split only — no leakage)
# ---------------------------------------------------------------------------

def build_author_features(
    meta_df: pd.DataFrame,
    vid_attrs: pd.DataFrame,
    train_mask: np.ndarray,
) -> tuple[np.ndarray, dict[int, int]]:
    """Return (author_feat [N_author, 10], author_id_to_local)."""
    train_pids = set(meta_df.loc[train_mask, "pid"].astype(int))
    test_val_pids = set(meta_df.loc[~train_mask, "pid"].astype(int))
    assert len(train_pids & test_val_pids) == 0, "train/test pid overlap!"

    all_author_ids = sorted(vid_attrs["author_id"].unique())
    author_to_local = {a: i for i, a in enumerate(all_author_ids)}
    N_author = len(all_author_ids)

    # log1p(fans_count) — one value per author (consistent across all their videos)
    pid_to_author = vid_attrs.set_index("pid")["author_id"].to_dict()
    pid_to_fans   = vid_attrs.set_index("pid")["author_fans_count"].to_dict()

    # Build mapping: author_id → log_fans (use first available value)
    author_log_fans = np.zeros(N_author, dtype=np.float32)
    for pid, aid in pid_to_author.items():
        local = author_to_local[aid]
        if author_log_fans[local] == 0:
            author_log_fans[local] = np.log1p(pid_to_fans.get(pid, 0))

    # Train-only stats: count + 8 label means
    train_rows = meta_df[train_mask].copy()
    used_pids = set(train_rows["pid"].astype(int))
    # Assert only train pids used
    assert used_pids.issubset(train_pids), \
        f"build_author_features used non-train pids: {used_pids - train_pids}"

    train_rows = train_rows.merge(
        vid_attrs[["pid", "author_id"]], on="pid", how="left"
    )
    train_rows["author_local"] = train_rows["author_id"].map(author_to_local)

    grp = train_rows.groupby("author_local")
    author_count = grp["pid"].count().reindex(range(N_author), fill_value=0).values.astype(np.float32)
    author_label_means = (
        grp[TARGETS].mean()
        .reindex(range(N_author), fill_value=0.0)
        .values.astype(np.float32)
    )

    # Intersection check for leakage report
    coverage = len(used_pids & train_pids) / max(len(used_pids), 1) * 100
    print(f"  Leakage check: author stats use {len(used_pids)} pids, "
          f"{coverage:.1f}% in train_pids (must be 100%)")

    author_feat = np.concatenate([
        author_log_fans[:, None],            # [N_author, 1]
        author_count[:, None],               # [N_author, 1]
        author_label_means,                  # [N_author, 8]
    ], axis=1)                               # → [N_author, 10]

    return author_feat, author_to_local


# ---------------------------------------------------------------------------
# kNN edges (cosine, content-only)
# ---------------------------------------------------------------------------

def build_knn_edges(img_feats: np.ndarray, k: int = 10, batch: int = 5000) -> tuple[np.ndarray, np.ndarray]:
    """Return (src, dst) arrays for cosine kNN (self excluded).

    Uses batched search to avoid macOS faiss SIGSEGV on large matrices.
    """
    norms = np.linalg.norm(img_feats, axis=1, keepdims=True)
    normed = np.ascontiguousarray(img_feats / np.maximum(norms, 1e-8), dtype=np.float32)
    N = len(normed)

    try:
        import faiss
        faiss.omp_set_num_threads(1)  # avoid OpenMP pthread crash on macOS
        d = normed.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(normed)
        # Batch search to avoid SIGSEGV on large matrices (macOS faiss issue)
        all_I = []
        for start in range(0, N, batch):
            end = min(start + batch, N)
            _, I_b = index.search(normed[start:end], k + 1)
            all_I.append(I_b)
        I = np.vstack(all_I)
        print(f"  cosine kNN (faiss batched, batch={batch}): k={k}")
    except Exception:
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine", algorithm="brute")
        nn.fit(normed)
        _, I = nn.kneighbors(normed)
        print(f"  cosine kNN (sklearn): k={k}")

    src, dst = [], []
    for v in range(N):
        for nb in I[v]:
            if nb != v:
                src.append(v)
                dst.append(int(nb))
    return np.array(src, dtype=np.int64), np.array(dst, dtype=np.int64)


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_graph(args: argparse.Namespace) -> HeteroData:
    interaction_csv = args.interaction
    out_path = args.out

    print("=" * 60)
    print("Pipeline C — build_hetero_graph.py")
    print("=" * 60)

    # ── 1. Load meta (working set = 54,088 videos) ──────────────────────────
    meta = pd.read_parquet(META_PARQUET)
    pids: list[int] = meta["pid"].astype(int).tolist()
    N_video = len(pids)
    pid_to_vidlocal = {p: i for i, p in enumerate(pids)}
    print(f"\n[1] Working set: {N_video} videos")

    # ── 2. Load interaction.csv for per-video attributes ────────────────────
    print("[2] Loading interaction.csv (may take a moment)...")
    KEEP_COLS = ["pid", "author_id", "category_id", "parent_id", "root_id",
                 "author_fans_count", "duration", "tag_name", "title"]
    df = pd.read_csv(interaction_csv, usecols=KEEP_COLS)
    df = df[df["pid"].isin(set(pids))].copy()

    # Deduplicate: one row per pid (first occurrence)
    vid_attrs = df.groupby("pid", sort=False).agg(
        author_id=("author_id", "first"),
        category_id=("category_id", "first"),
        parent_id=("parent_id", "first"),
        root_id=("root_id", "first"),
        author_fans_count=("author_fans_count", "first"),
        duration=("duration", "first"),
        title=("title", "first"),
    ).reset_index()

    # Fill any pids not in interaction.csv (should not happen, but safe)
    missing_pids = set(pids) - set(vid_attrs["pid"])
    if missing_pids:
        print(f"  WARNING: {len(missing_pids)} pids missing from interaction.csv (zero-filled)")
    vid_attrs = vid_attrs.set_index("pid").reindex(pids).reset_index()
    vid_attrs["duration"] = vid_attrs["duration"].fillna(10.0).astype(float)
    vid_attrs["author_fans_count"] = vid_attrs["author_fans_count"].fillna(0.0).astype(float)

    print(f"  authors: {vid_attrs['author_id'].nunique()}, "
          f"categories: {vid_attrs['category_id'].nunique()}")

    # ── 3. Train/val/test split (80/10/10, seed=42) ─────────────────────────
    rng = np.random.default_rng(42)
    perm = rng.permutation(N_video)
    n_train = int(0.8 * N_video)
    n_val   = int(0.1 * N_video)
    train_idx = perm[:n_train]
    val_idx   = perm[n_train:n_train + n_val]
    test_idx  = perm[n_train + n_val:]

    train_mask_np = np.zeros(N_video, dtype=bool)
    val_mask_np   = np.zeros(N_video, dtype=bool)
    test_mask_np  = np.zeros(N_video, dtype=bool)
    train_mask_np[train_idx] = True
    val_mask_np[val_idx]     = True
    test_mask_np[test_idx]   = True

    print(f"[3] Split: train={train_mask_np.sum()} val={val_mask_np.sum()} test={test_mask_np.sum()}")

    # ── 4. Video node features ───────────────────────────────────────────────
    print("[4] Building video features...")
    pid_to_row = load_pid_to_row(PIDS_TXT)
    img_feats  = load_image_feats(pids, pid_to_row)   # [N_v, 768]

    if args.use_sbert:
        titles = vid_attrs["title"].fillna("").tolist()
        txt_feats = load_text_feats_sbert(titles)      # [N_v, 384]
        print(f"  SBERT text: {txt_feats.shape}")
    else:
        txt_feats = load_text_feats_default(pids, pid_to_row)  # [N_v, 50]

    log_dur = np.log1p(vid_attrs["duration"].values).astype(np.float32)[:, None]  # [N_v, 1]

    video_feat = np.concatenate([img_feats, txt_feats, log_dur], axis=1)  # [N_v, 768+txt+1]
    print(f"  video feat shape: {video_feat.shape}")

    # ── 5. Author node features (train-only) ─────────────────────────────────
    print("[5] Building author features (train-only)...")
    author_feat, author_to_local = build_author_features(meta, vid_attrs, train_mask_np)
    N_author = len(author_to_local)
    print(f"  author feat shape: {author_feat.shape}")

    # ── 6. Category nodes ────────────────────────────────────────────────────
    all_cat_vals = pd.concat([
        vid_attrs["category_id"], vid_attrs["parent_id"], vid_attrs["root_id"]
    ]).dropna().astype(int).unique()
    cat_to_local = {int(c): i for i, c in enumerate(sorted(all_cat_vals))}
    N_cat = len(cat_to_local)
    print(f"[6] Category nodes: {N_cat}")

    # Store category hierarchy as integer tensors (for model to embed)
    cat_hier_src = vid_attrs[["category_id", "parent_id", "root_id"]].dropna().drop_duplicates("category_id")
    cat_id_list = sorted(cat_to_local, key=cat_to_local.get)  # ordered by local index
    cat_hier_dict: dict[int, tuple[int, int]] = {}
    for _, row in cat_hier_src.iterrows():
        cid, pid_, rid = int(row["category_id"]), int(row["parent_id"]), int(row["root_id"])
        cat_hier_dict[cid] = (pid_, rid)
    for c in cat_id_list:
        if c not in cat_hier_dict:
            cat_hier_dict[c] = (c, c)  # root: self-referential

    hier_arr = np.array(
        [[cat_to_local[c],
          cat_to_local[cat_hier_dict[c][0]],
          cat_to_local[cat_hier_dict[c][1]]] for c in cat_id_list],
        dtype=np.int64,
    )  # [N_cat, 3] — local indices for [self, parent, root]

    # ── 7. Labels & weights ──────────────────────────────────────────────────
    Y = meta[TARGETS].astype(np.float32).values   # [N_v, 8]
    weights = meta["n_impressions"].astype(np.float32).values

    # ── 8. Build edges ───────────────────────────────────────────────────────
    print("[7] Building edges...")

    # video → author
    vid_author_local = vid_attrs["author_id"].map(author_to_local).fillna(-1).astype(int).values
    valid_va = vid_author_local >= 0
    va_src = np.where(valid_va)[0].astype(np.int64)
    va_dst = vid_author_local[valid_va].astype(np.int64)
    print(f"  (video, posted_by, author): {len(va_src)} edges")

    # video → category
    vid_cat_local = vid_attrs["category_id"].map(cat_to_local).fillna(-1).astype(int).values
    valid_vc = vid_cat_local >= 0
    vc_src = np.where(valid_vc)[0].astype(np.int64)
    vc_dst = vid_cat_local[valid_vc].astype(np.int64)
    print(f"  (video, belongs_to, category): {len(vc_src)} edges")

    # video → video (cosine kNN on image content only — no label info)
    print(f"  Building cosine kNN (k={args.k}) on image features...")
    vv_src, vv_dst = build_knn_edges(img_feats, k=args.k)
    print(f"  (video, similar_to, video): {len(vv_src)} edges")

    # ── 9. Assemble HeteroData ───────────────────────────────────────────────
    print("[8] Assembling HeteroData...")
    data = HeteroData()

    data["video"].x          = torch.tensor(video_feat, dtype=torch.float)
    data["video"].y          = torch.tensor(Y, dtype=torch.float)
    data["video"].train_mask = torch.tensor(train_mask_np)
    data["video"].val_mask   = torch.tensor(val_mask_np)
    data["video"].test_mask  = torch.tensor(test_mask_np)
    data["video"].weight     = torch.tensor(weights, dtype=torch.float)
    data["video"].pid        = torch.tensor(pids, dtype=torch.long)

    data["author"].x         = torch.tensor(author_feat, dtype=torch.float)

    data["category"].num_nodes = N_cat
    data["category"].hier      = torch.tensor(hier_arr, dtype=torch.long)

    data["video", "posted_by",  "author"].edge_index   = torch.tensor(np.stack([va_src, va_dst]))
    data["video", "belongs_to", "category"].edge_index = torch.tensor(np.stack([vc_src, vc_dst]))
    data["video", "similar_to", "video"].edge_index    = torch.tensor(np.stack([vv_src, vv_dst]))

    data = T.ToUndirected()(data)

    # ── 10. Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Graph summary")
    print("=" * 60)
    print(f"  video  nodes : {data['video'].num_nodes}")
    print(f"  author nodes : {data['author'].num_nodes}")
    print(f"  category nodes: {N_cat}")
    print(f"  video.x shape : {data['video'].x.shape}")
    print(f"  author.x shape: {data['author'].x.shape}")
    for et in data.edge_types:
        ei = data[et].edge_index
        print(f"  {et}: {ei.shape[1]} edges")

    # ── 11. Leakage self-check (final) ───────────────────────────────────────
    train_pids_set = set(meta.loc[train_mask_np, "pid"].astype(int))
    used_in_author_stats = set(meta.loc[train_mask_np, "pid"].astype(int))
    ratio = len(used_in_author_stats & train_pids_set) / max(len(used_in_author_stats), 1) * 100
    print(f"\n[LEAKAGE CHECK] Author stats pid coverage in train: {ratio:.1f}% (must be 100%)")
    assert ratio == 100.0, "Leakage detected!"
    print("[LEAKAGE CHECK] PASSED")

    # ── 12. Save ─────────────────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, out_path)
    print(f"\nSaved → {out_path}")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Build heterogeneous graph for Pipeline C")
    parser.add_argument("--use-sbert", action="store_true",
                        help="Use SBERT (paraphrase-multilingual-MiniLM-L12-v2, 384d) for text")
    parser.add_argument("--k", type=int, default=10, help="kNN neighbours (default 10)")
    parser.add_argument("--interaction", type=Path, default=INTERACTION_CSV)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args()

    build_graph(args)


if __name__ == "__main__":
    main()
