"""Predict per-video user behavior from video content features.

Two modes:

(A) By pid (existing video in the dataset):
    python scripts/predict_behavior.py by-pid \
        --pid 12345 \
        --in-dir data_processed/behavior \
        --models-dir data_processed/behavior/models

(B) From a new .mp4 + title/asr text:
    python scripts/predict_behavior.py from-raw \
        --video data/new_clip.mp4 \
        --title "Underwater photography demo" \
        --asr-text "today we go diving ..." \
        --models-dir data_processed/behavior/models \
        --glove-dim 50

Outputs a JSON with predictions per target.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np


TARGETS = [
    "like_rate", "comment_rate", "follow_rate", "collect_rate",
    "forward_rate", "hate_rate", "effective_view_rate", "mean_watch_time",
]
TRANSFORMED = {"mean_watch_time"}


def load_models(models_dir: Path):
    try:
        import lightgbm as lgb
    except Exception as exc:
        raise RuntimeError("lightgbm required to load models") from exc
    return {
        name: lgb.Booster(model_file=str(models_dir / f"{name}.txt"))
        for name in TARGETS
    }


def predict(models, x: np.ndarray) -> dict:
    out = {}
    for name in TARGETS:
        pred = float(models[name].predict(x)[0])
        if name in TRANSFORMED:
            pred = float(np.expm1(pred))
        out[name] = pred
    return out


# -------------------------- mode A -----------------------------------------

def cmd_by_pid(args):
    import pandas as pd
    meta = pd.read_parquet(args.in_dir / "meta.parquet")
    X = np.load(args.in_dir / "X.npy")
    if args.pid not in set(meta["pid"]):
        print(f"pid {args.pid} not found in meta", file=sys.stderr)
        sys.exit(1)
    row_idx = int(np.where(meta["pid"].values == args.pid)[0][0])
    x = X[row_idx:row_idx + 1]
    models = load_models(args.models_dir)
    pred = predict(models, x)
    print(json.dumps({"pid": int(args.pid), "predictions": pred}, indent=2))


# -------------------------- mode B -----------------------------------------

def extract_visual_vit(video_path: Path) -> np.ndarray:
    """Replicate ShortVideo_dataset/video_feature_process.py for a single video.

    Returns a 1D float32 vector. By default uses 4 uniformly-sampled frames
    through ViT-B/16 and concatenates the per-frame outputs (matching the
    upstream script). If you trained on the 768-d processed features instead,
    swap this for `np.load(...)` of the pre-extracted matrix row.
    """
    try:
        import cv2
        import torch
        from torchvision import models, transforms
    except Exception as exc:
        raise RuntimeError(
            "opencv-python and torchvision required for video feature extraction"
        ) from exc

    device = torch.device("cuda" if torch.cuda.is_available()
                          else ("mps" if torch.backends.mps.is_available() else "cpu"))
    net = models.vit_b_16(weights="DEFAULT")
    net.heads = torch.nn.Identity()
    net.eval().to(device)

    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    num_frames = 4
    frames = []
    indices = np.linspace(0, max(total - 1, 0), num_frames, dtype=int)
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    to_tensor = transforms.ToTensor()
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            frames.append(torch.zeros(3, 224, 224))
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224))
        frames.append(norm(to_tensor(frame)))
    cap.release()
    batch = torch.stack(frames).to(device)
    with torch.no_grad():
        feats = net(batch)  # (4, 768)
    return feats.flatten().cpu().numpy().astype(np.float32)


def extract_text_glove(texts: Iterable[str], glove_dim: int) -> np.ndarray:
    from torchtext.vocab import GloVe
    from torchtext.data.utils import get_tokenizer

    glove = GloVe(name="6B", dim=glove_dim)
    tok = get_tokenizer("basic_english")
    joined = " ".join(t for t in texts if t).strip()
    vecs = [glove[w].numpy() for w in tok(joined) if w in glove.stoi]
    if not vecs:
        return np.zeros(glove_dim, dtype=np.float32)
    return np.mean(np.stack(vecs), axis=0).astype(np.float32)


def cmd_from_raw(args):
    visual = extract_visual_vit(args.video)
    text = extract_text_glove([args.title or "", args.asr_text or ""],
                              args.glove_dim)
    x = np.concatenate([visual, text])[None, :]
    print(f"visual dim: {visual.shape[0]}, text dim: {text.shape[0]}, total: {x.shape[1]}")

    models = load_models(args.models_dir)
    expected_dim = models[TARGETS[0]].num_feature()
    if x.shape[1] != expected_dim:
        print(
            f"WARNING: feature dim {x.shape[1]} != model expected {expected_dim}. "
            "Make sure the visual extractor matches the one used during training.",
            file=sys.stderr,
        )
    pred = predict(models, x)
    print(json.dumps({
        "video": str(args.video),
        "title": args.title,
        "predictions": pred,
    }, indent=2))


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)

    p_pid = sub.add_parser("by-pid")
    p_pid.add_argument("--pid", type=int, required=True)
    p_pid.add_argument("--in-dir", type=Path, required=True)
    p_pid.add_argument("--models-dir", type=Path, required=True)
    p_pid.set_defaults(fn=cmd_by_pid)

    p_raw = sub.add_parser("from-raw")
    p_raw.add_argument("--video", type=Path, required=True)
    p_raw.add_argument("--title", type=str, default="")
    p_raw.add_argument("--asr-text", type=str, default="")
    p_raw.add_argument("--glove-dim", type=int, default=50)
    p_raw.add_argument("--models-dir", type=Path, required=True)
    p_raw.set_defaults(fn=cmd_from_raw)

    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
