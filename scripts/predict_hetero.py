"""Pipeline C inference — predict 8 performance metrics for a video.

Two modes
---------
by-pid   : graph is already built; look up the video node by pid and run forward.
from-raw : cold-start; extract features from cover image or video file,
           attach a new node to the graph, and predict.

Usage
-----
  # In-graph video
  python scripts/predict_hetero.py by-pid --pid 84199269992

  # Cold-start: cover image only
  python scripts/predict_hetero.py from-raw --cover /path/to/cover.jpg

  # Cold-start: video file + known metadata
  python scripts/predict_hetero.py from-raw --video /path/to/clip.mp4 \
      --author-id 193159093 --category-id 36 --duration 91.9

Notes
-----
* Dimension assert: the script checks that extracted feature dim matches
  training graph video.x dim to catch any 768-vs-3072 mismatch early.
* watch_time output is expm1-restored (seconds).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from hetero_gnn import build_model

GRAPH_PATH  = Path("data_processed/behavior/hetero_graph.pt")
MODEL_PATH  = Path("data_processed/behavior/models_gnn/best_model.pt")
PIDS_TXT    = Path("data_raw/video_rec_dataset/pids.txt")
IMAGE_FEAT  = Path("data_raw/video_rec_dataset/image_feat.npy")
TEXT_FEAT   = Path("data_raw/video_rec_dataset/text_feat.npy")

TARGETS = [
    "like_rate", "comment_rate", "follow_rate", "collect_rate",
    "forward_rate", "hate_rate", "effective_view_rate", "mean_watch_time",
]

NUM_VIT_FRAMES = 4    # frames to sample for video → ViT embedding


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

def extract_vit_cover(image_path: str) -> np.ndarray:
    """Extract 768-dim ViT-B/16 feature from a single cover image."""
    import cv2
    import torchvision.transforms as TVT
    from torchvision.models import vit_b_16, ViT_B_16_Weights

    weights = ViT_B_16_Weights.IMAGENET1K_V1
    vit = vit_b_16(weights=weights)
    vit.heads = torch.nn.Identity()  # strip classifier → 768-dim
    vit.eval()

    preprocess = TVT.Compose([
        TVT.ToPILImage(),
        weights.transforms(),
    ])

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = preprocess(img_rgb).unsqueeze(0)   # [1, 3, 224, 224]
    with torch.no_grad():
        feat = vit(tensor).squeeze(0).numpy()   # [768]
    return feat.astype(np.float32)


def extract_vit_video(video_path: str, n_frames: int = NUM_VIT_FRAMES) -> np.ndarray:
    """Extract 768-dim ViT feature from a video by averaging n_frames frames."""
    import cv2
    import torchvision.transforms as TVT
    from torchvision.models import vit_b_16, ViT_B_16_Weights

    weights = ViT_B_16_Weights.IMAGENET1K_V1
    vit = vit_b_16(weights=weights)
    vit.heads = torch.nn.Identity()
    vit.eval()

    preprocess = TVT.Compose([
        TVT.ToPILImage(),
        weights.transforms(),
    ])

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise ValueError(f"Video has no frames: {video_path}")

    frame_indices = np.linspace(0, total - 1, n_frames, dtype=int)
    frame_feats = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        t = preprocess(rgb).unsqueeze(0)
        with torch.no_grad():
            f = vit(t).squeeze(0).numpy()
        frame_feats.append(f)
    cap.release()

    if not frame_feats:
        raise RuntimeError("Failed to extract any frame from the video.")
    return np.mean(frame_feats, axis=0).astype(np.float32)   # [768]


def extract_text_feature(title: str, expected_text_dim: int = 50) -> np.ndarray:
    """Encode text matching the exact dimension used when building the graph.

    The training graph stores text features at a fixed dimension (50 or 384).
    This function MUST produce the same dimension, otherwise the assert in
    predict_from_raw will fail.

    - expected_text_dim == 384: use SBERT (graph was built with --use-sbert)
    - expected_text_dim == 50 : return zero vector (the pre-extracted 50-dim
      features in text_feat.npy were computed offline; we cannot reproduce them
      from title text alone, so we use zeros as a placeholder)
    """
    if expected_text_dim == 384:
        try:
            from sentence_transformers import SentenceTransformer
            sbert = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
            return sbert.encode([title], convert_to_numpy=True)[0].astype(np.float32)
        except Exception as e:
            print(f"  [WARN] SBERT failed ({e}); falling back to zeros")
            return np.zeros(384, dtype=np.float32)
    else:
        # Graph was built with the pre-extracted 50-dim text_feat.npy.
        # We cannot reproduce those exact features from title text alone,
        # so we use a zero vector (same as videos that had no text in training).
        if title:
            print(f"  [NOTE] Graph uses 50-dim text features (not SBERT). "
                  "Text '{title[:30]}...' replaced with zero vector. "
                  "Rebuild graph with --use-sbert for better text signal.")
        return np.zeros(50, dtype=np.float32)


# ---------------------------------------------------------------------------
# Mode: by-pid
# ---------------------------------------------------------------------------

def predict_by_pid(args: argparse.Namespace) -> None:
    device = torch.device("cpu")
    data   = torch.load(args.graph, weights_only=False, map_location=device)

    # Find local video index for the requested pid
    pids = data["video"].pid.numpy()
    matches = np.where(pids == int(args.pid))[0]
    if len(matches) == 0:
        print(f"ERROR: pid {args.pid} not found in graph (has {len(pids)} videos).")
        sys.exit(1)
    vid_local = int(matches[0])

    gnn = build_model(data, args)
    state = torch.load(args.model_path, map_location=device, weights_only=True)
    gnn.load_state_dict(state)
    gnn.eval()

    with torch.no_grad():
        out = gnn(data.x_dict, data.edge_index_dict)

    pred = out[vid_local].numpy()
    _print_predictions(pred, pid=args.pid)


# ---------------------------------------------------------------------------
# Mode: from-raw
# ---------------------------------------------------------------------------

def predict_from_raw(args: argparse.Namespace) -> None:
    device = torch.device("cpu")
    data   = torch.load(args.graph, weights_only=False, map_location=device)
    gnn    = build_model(data, args)
    state  = torch.load(args.model_path, map_location=device, weights_only=True)
    gnn.load_state_dict(state)
    gnn.eval()

    expected_feat_dim = data["video"].x.shape[1]

    # ── 1. Extract visual feature ───────────────────────────────────────────
    if args.cover:
        print(f"Extracting ViT cover feature from {args.cover}...")
        vis = extract_vit_cover(args.cover)
    elif args.video:
        print(f"Extracting ViT video feature from {args.video} ({NUM_VIT_FRAMES} frames)...")
        vis = extract_vit_video(args.video)
    else:
        raise ValueError("Provide --cover or --video for from-raw mode.")
    print(f"  visual dim: {vis.shape[0]}")

    # ── 2. Text feature (dimension MUST match how the graph was built) ─────────
    # Infer expected text dim from graph: total - visual(768) - duration(1)
    expected_text_dim = expected_feat_dim - 768 - 1
    title = args.title or ""
    txt = extract_text_feature(title, expected_text_dim=expected_text_dim)
    print(f"  text dim: {txt.shape[0]}  (graph expects {expected_text_dim})")

    # ── 3. Duration ─────────────────────────────────────────────────────────
    duration = float(args.duration) if args.duration else 60.0
    log_dur  = np.array([np.log1p(duration)], dtype=np.float32)

    # ── 4. Concat & dimension check ─────────────────────────────────────────
    new_feat = np.concatenate([vis, txt, log_dur])  # should match training
    if new_feat.shape[0] != expected_feat_dim:
        raise AssertionError(
            f"Feature dim mismatch: got {new_feat.shape[0]}, "
            f"expected {expected_feat_dim} (training graph). "
            "Ensure you use the same text feature type (SBERT vs default) "
            "as when the graph was built."
        )
    print(f"  combined feat dim: {new_feat.shape[0]} ✓ (matches training)")

    # ── 5. Add new node to a copy of the graph ──────────────────────────────
    import copy
    from torch_geometric.data import HeteroData
    new_data = copy.copy(data)

    N_old = data["video"].x.shape[0]
    new_x  = torch.cat([data["video"].x,
                         torch.tensor(new_feat, dtype=torch.float).unsqueeze(0)], dim=0)
    new_data["video"].x = new_x
    new_node_idx = N_old

    # Dummy y & masks for the new node (not used in inference)
    new_data["video"].y = torch.cat([data["video"].y,
                                     torch.zeros(1, 8)], dim=0)

    # ── 6. Connect new node to author/category if known ─────────────────────
    extra_edges: list[tuple[str, str, str, torch.Tensor]] = []

    if args.author_id is not None:
        # Check if author is in graph
        author_pids = data["video"].pid.numpy()
        # Find any video by this author to get author local idx
        # Use interaction lookup via edge_index
        va_edge = data["video", "posted_by", "author"].edge_index  # [2, E]
        # Build reverse: author_local → first vid_local
        author_nodes_in_graph = va_edge[1].unique()
        # We need to know which author_id = args.author_id corresponds to which local idx
        # This requires the author mapping stored at build time (not stored in graph)
        # Fallback: use kNN only
        print("  [NOTE] author_id lookup requires build-time mapping; using kNN only for author.")

    if args.category_id is not None:
        # Find category local idx by searching the category hier
        hier = data["category"].hier      # [N_cat, 3]: [local, parent_local, root_local]
        # hier[:, 0] = cat_local = 0..N_cat-1 (always equal to row index)
        # We need: which row corresponds to original category_id?
        # Not stored at build time — use fallback
        print("  [NOTE] category_id re-mapping not stored in graph; using kNN only for category.")

    # ── 7. Connect via cosine kNN on content embedding ───────────────────────
    import faiss
    img_feats = data["video"].x[:, :768].numpy()  # content-only for kNN
    new_vis_norm = vis / (np.linalg.norm(vis) + 1e-8)
    db_norms = img_feats / (np.linalg.norm(img_feats, axis=1, keepdims=True) + 1e-8)

    k = 10
    scores = db_norms @ new_vis_norm          # [N_old]
    top_k  = np.argsort(-scores)[:k].astype(np.int64)

    # Add edges: new_node → top_k  and  top_k → new_node
    new_src = np.full(k, new_node_idx, dtype=np.int64)
    new_dst = top_k
    knn_fwd = torch.tensor(np.stack([new_src, new_dst]))
    knn_rev = torch.tensor(np.stack([new_dst, new_src]))

    old_sim_ei = data["video", "similar_to", "video"].edge_index
    new_sim_ei = torch.cat([old_sim_ei, knn_fwd, knn_rev], dim=1)
    new_data["video", "similar_to", "video"].edge_index = new_sim_ei

    # Reverse edges for other edge types already added by ToUndirected at build time
    # — no change needed for author/category since we skip them in fallback mode

    # ── 8. Inference ─────────────────────────────────────────────────────────
    with torch.no_grad():
        out = gnn(new_data.x_dict, new_data.edge_index_dict)

    pred = out[new_node_idx].numpy()
    _print_predictions(pred, pid=None)


# ---------------------------------------------------------------------------
# Output formatter
# ---------------------------------------------------------------------------

def _print_predictions(pred: np.ndarray, pid: int | None) -> None:
    label = f"pid={pid}" if pid else "new video (cold-start)"
    print(f"\n{'─'*50}")
    print(f"  Predicted performance metrics — {label}")
    print("─" * 50)
    for i, name in enumerate(TARGETS):
        val = float(pred[i])
        if name == "mean_watch_time":
            val = float(np.expm1(max(val, -30)))   # restore from log space
            print(f"  {name:<25} {val:>8.2f} s")
        else:
            print(f"  {name:<25} {val:>8.4f}")
    print("─" * 50)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline C inference")
    sub = parser.add_subparsers(dest="mode", required=True)

    # by-pid
    p_pid = sub.add_parser("by-pid", help="Predict for an in-graph video by pid")
    p_pid.add_argument("--pid",   type=int, required=True)
    p_pid.add_argument("--graph", type=Path, default=GRAPH_PATH)
    p_pid.add_argument("--model-path", type=Path, default=MODEL_PATH)
    p_pid.add_argument("--model", type=str, default="sage")
    p_pid.add_argument("--d",     type=int, default=128)
    p_pid.add_argument("--layers",type=int, default=2)

    # from-raw
    p_raw = sub.add_parser("from-raw", help="Cold-start: extract from file")
    p_raw.add_argument("--cover",      type=str, default=None, help="Path to cover image")
    p_raw.add_argument("--video",      type=str, default=None, help="Path to video file")
    p_raw.add_argument("--title",      type=str, default="",   help="Video title text")
    p_raw.add_argument("--author-id",  type=int, default=None)
    p_raw.add_argument("--category-id",type=int, default=None)
    p_raw.add_argument("--duration",   type=float, default=None, help="Duration in seconds")
    p_raw.add_argument("--graph",      type=Path, default=GRAPH_PATH)
    p_raw.add_argument("--model-path", type=Path, default=MODEL_PATH)
    p_raw.add_argument("--model",      type=str, default="sage")
    p_raw.add_argument("--d",          type=int, default=128)
    p_raw.add_argument("--layers",     type=int, default=2)

    args = parser.parse_args()

    if args.mode == "by-pid":
        predict_by_pid(args)
    else:
        predict_from_raw(args)


if __name__ == "__main__":
    main()
