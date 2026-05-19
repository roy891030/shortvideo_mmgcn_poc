from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "MMRec" / "data" / "video",
        help="Directory containing video.inter and feature .npy files.",
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    inter_path = data_dir / "video.inter"
    image_path = data_dir / "image_feat.npy"
    text_path = data_dir / "text_feat.npy"

    if not inter_path.exists():
        raise FileNotFoundError(inter_path)

    df = pd.read_csv(inter_path, sep="\t")
    num_users = int(df["userID"].nunique())
    num_items = int(df["itemID"].nunique())
    num_interactions = int(len(df))
    density = num_interactions / (num_users * num_items)
    split_counts = df["x_label"].value_counts().sort_index().to_dict()

    print(f"data_dir: {data_dir}")
    print(f"num_users: {num_users}")
    print(f"num_items: {num_items}")
    print(f"num_interactions: {num_interactions}")
    print(f"sparsity: {1.0 - density:.8f}")
    print(f"train_count: {int(split_counts.get(0, 0))}")
    print(f"valid_count: {int(split_counts.get(1, 0))}")
    print(f"test_count: {int(split_counts.get(2, 0))}")
    print(f"user_id_range: {int(df['userID'].min())}..{int(df['userID'].max())}")
    print(f"item_id_range: {int(df['itemID'].min())}..{int(df['itemID'].max())}")

    if image_path.exists():
        image_feat = np.load(image_path, mmap_mode="r")
        print(f"visual feature shape: {image_feat.shape}")
        print(f"visual feature dtype: {image_feat.dtype}")
    else:
        print("visual feature shape: missing")

    if text_path.exists():
        text_feat = np.load(text_path, mmap_mode="r")
        print(f"text feature shape: {text_feat.shape}")
        print(f"text feature dtype: {text_feat.dtype}")
    else:
        print("text feature shape: missing")


if __name__ == "__main__":
    main()
