from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data_raw" / "video_rec_dataset"
PROCESSED_DIR = ROOT / "data_processed" / "video"
MMREC_DATA_DIR = ROOT / "MMRec" / "data" / "video"
MMREC_VIDEO_YAML = ROOT / "MMRec" / "src" / "configs" / "dataset" / "video.yaml"


def choose_users(df: pd.DataFrame, target_interactions: int, seed: int) -> list[int]:
    split_counts = df.groupby(["userID", "x_label"]).size().unstack(fill_value=0)
    eligible = split_counts[
        (split_counts.get(0, 0) > 0)
        & (split_counts.get(1, 0) > 0)
        & (split_counts.get(2, 0) > 0)
    ].index.to_numpy()
    rng = np.random.default_rng(seed)
    rng.shuffle(eligible)

    selected: list[int] = []
    total = 0
    user_sizes = df.groupby("userID").size()
    for user_id in eligible:
        selected.append(int(user_id))
        total += int(user_sizes.loc[user_id])
        if total >= target_interactions:
            break

    if not selected:
        raise RuntimeError("No users with train/valid/test interactions were found.")
    return selected


def write_video_yaml() -> None:
    MMREC_VIDEO_YAML.write_text(
        """# ShortVideo processed recommendation smoke-test subset.
USER_ID_FIELD: userID
ITEM_ID_FIELD: itemID
TIME_FIELD: timestamp
inter_splitting_label: x_label
filter_out_cod_start_users: True

inter_file_name: video.inter
vision_feature_file: image_feat.npy
text_feature_file: text_feat.npy

field_separator: "\\t"

use_gpu: False
epochs: 1
stopping_step: 5
train_batch_size: 2048
eval_batch_size: 256
embedding_size: 64
topk: [10, 20]
metrics: ["Recall", "NDCG"]
valid_metric: NDCG@10
eval_step: 1
save_recommended_topk: False
clip_grad_norm:
eval_type: full
use_neighborhood_loss: False
alpha1: 0.0
alpha2: 0.0
beta: 1
""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-interactions", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20260519)
    args = parser.parse_args()

    inter_path = RAW_DIR / "video.inter"
    image_path = RAW_DIR / "image_feat.npy"
    text_path = RAW_DIR / "text_feat.npy"
    for path in [inter_path, image_path, text_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    df = pd.read_csv(inter_path, sep="\t")
    users = choose_users(df, args.target_interactions, args.seed)
    subset = df[df["userID"].isin(users)].copy()

    user_map = {old: new for new, old in enumerate(sorted(subset["userID"].unique()))}
    item_ids = np.array(sorted(subset["itemID"].unique()), dtype=np.int64)
    item_map = {int(old): new for new, old in enumerate(item_ids)}

    subset["userID"] = subset["userID"].map(user_map).astype(np.int64)
    subset["itemID"] = subset["itemID"].map(item_map).astype(np.int64)
    subset = subset.sort_values(["userID", "x_label", "timestamp", "itemID"])

    image_feat = np.load(image_path, mmap_mode="r")[item_ids].astype(np.float32)
    text_feat = np.load(text_path, mmap_mode="r")[item_ids].astype(np.float32)

    for directory in [PROCESSED_DIR, MMREC_DATA_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
        subset.to_csv(directory / "video.inter", sep="\t", index=False)
        np.save(directory / "image_feat.npy", image_feat)
        np.save(directory / "text_feat.npy", text_feat)

    write_video_yaml()
    shutil.copy2(MMREC_VIDEO_YAML, ROOT / "data_processed" / "video.yaml")

    split_counts = subset["x_label"].value_counts().sort_index().to_dict()
    density = len(subset) / (subset["userID"].nunique() * subset["itemID"].nunique())
    print(f"source interactions: {len(df)}")
    print(f"selected users: {subset['userID'].nunique()}")
    print(f"selected items: {subset['itemID'].nunique()}")
    print(f"selected interactions: {len(subset)}")
    print(f"split counts: {split_counts}")
    print(f"image feature shape: {image_feat.shape}")
    print(f"text feature shape: {text_feat.shape}")
    print(f"sparsity: {1.0 - density:.8f}")
    print(f"wrote: {PROCESSED_DIR}")
    print(f"wrote: {MMREC_DATA_DIR}")
    print(f"wrote: {MMREC_VIDEO_YAML}")


if __name__ == "__main__":
    main()
