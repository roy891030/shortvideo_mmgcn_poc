# ShortVideo + MMRec/MMGCN PoC Reproduction

日期：2026-05-19

## 目標狀態

已完成一條最小可行 ShortVideo recommendation pipeline：

- 使用 `enoche/MMRec` 作為主框架
- 使用 `tsinghua-fib-lab/ShortVideo_dataset` 官方 README 指定的 processed recommendation dataset
- 未下載 raw videos，未重新抽影片特徵
- 已建立可跑的 `video` dataset
- 已成功輸出 dataset statistics
- 已成功跑通 LightGCN / VBPR / MMGCN 1 epoch
- 已成功跑通 MMGCN 5 epochs
- evaluation 已輸出 `Recall@10` / `Recall@20` / `NDCG@10` / `NDCG@20`

## Repo 與目錄

```bash
cd /Users/luoyi/Desktop
mkdir -p shortvideo_mmgcn_poc/{notes,logs,scripts,data_raw,data_processed}
git clone https://github.com/enoche/MMRec.git shortvideo_mmgcn_poc/MMRec
git clone https://github.com/tsinghua-fib-lab/ShortVideo_dataset.git shortvideo_mmgcn_poc/ShortVideo_dataset
```

舊的 `/Users/luoyi/Desktop/GNN_final_project/MM19-MMGCN` clone 已在確認 commit 後刪除。

## 環境

本機：

- macOS arm64
- Python 3.10.20
- torch 2.12.0
- torch-geometric 2.7.0
- CUDA：False
- MPS：True
- 本次 smoke test 使用 CPU，避免 MMRec 原始 CUDA 假設與 PyG/MPS 相容性風險

建立環境：

```bash
conda create -p /Users/luoyi/Desktop/shortvideo_mmgcn_poc/.python310 python=3.10 -y
/Users/luoyi/Desktop/shortvideo_mmgcn_poc/.python310/bin/python -m venv /Users/luoyi/Desktop/shortvideo_mmgcn_poc/.venv
/Users/luoyi/Desktop/shortvideo_mmgcn_poc/.venv/bin/python -m pip install --upgrade pip setuptools wheel
/Users/luoyi/Desktop/shortvideo_mmgcn_poc/.venv/bin/pip install torch torchvision torchaudio torch_geometric numpy pandas scipy pyyaml lmdb tqdm matplotlib
```

完整 freeze：`requirements.lock`

## 資料來源

ShortVideo_dataset 官方 README 的 recommendation benchmark 指定 processed dataset：

```text
https://www.dropbox.com/scl/fo/ha0e0wolgqgg5qskr52l1/AHZUySejwWJzfyJy8WGo2k4?rlkey=xwpqosx7b906yb7g8nwy53oqh&st=slry1d1l&dl=0
```

自動下載命令：

```bash
curl -L --fail --retry 3 \
  'https://www.dropbox.com/scl/fo/ha0e0wolgqgg5qskr52l1/AHZUySejwWJzfyJy8WGo2k4?rlkey=xwpqosx7b906yb7g8nwy53oqh&st=slry1d1l&dl=1' \
  -o /Users/luoyi/Desktop/shortvideo_mmgcn_poc/data_raw/video-rec-dataset.zip
```

下載內容：

- `video.inter`
- `image_feat.npy`
- `text_feat.npy`

完整 processed dataset 檢查結果：

- interactions：294355
- users：9609
- used items：88098
- feature rows：153561
- image feature shape：`(153561, 768)`
- text feature shape：`(153561, 50)`

## Smoke-Test Dataset

為了在 5 小時內完成可驗證雛形，沒有直接對完整 153k item feature matrix 跑 benchmark，而是用官方 processed recommendation dataset 建立 deterministic subset。

產生命令：

```bash
/Users/luoyi/Desktop/shortvideo_mmgcn_poc/.venv/bin/python scripts/prepare_video_yaml.py
```

輸出：

- `data_processed/video/video.inter`
- `data_processed/video/image_feat.npy`
- `data_processed/video/text_feat.npy`
- `MMRec/data/video/video.inter`
- `MMRec/data/video/image_feat.npy`
- `MMRec/data/video/text_feat.npy`
- `MMRec/src/configs/dataset/video.yaml`

subset statistics：

- users：69
- items：2822
- interactions：3033
- train：2452
- valid：276
- test：305
- sparsity：0.98442363
- visual feature shape：`(2822, 768)`
- text feature shape：`(2822, 50)`

## MMRec 設定

建立了 `MMRec/src/configs/dataset/video.yaml`：

- `dataset: video`
- `inter_file_name: video.inter`
- `vision_feature_file: image_feat.npy`
- `text_feature_file: text_feat.npy`
- `train_batch_size: 2048`
- `embedding_size: 64`
- `topk: [10, 20]`
- `epochs: 1`
- `stopping_step: 5`
- `eval_step: 1`
- `metrics: ["Recall", "NDCG"]`
- `valid_metric: NDCG@10`

最小相容性修補：

- `MMRec/src/main.py`：讓 `--epochs 1` / `--epochs 5` 這類 CLI override 生效
- `MMRec/src/utils/metrics.py`：將 NumPy 2 不支援的 `np.float` 改成 `float`
- `MMRec/src/models/lightgcn.py`：將 SciPy 新版不支援的 DOK `_update` 改成 COO adjacency 建構
- `MMRec/src/models/mmgcn.py`：修正 PyG 2.7 `MessagePassing.message()` signature
- `LightGCN.yaml` / `VBPR.yaml` / `MMGCN.yaml`：改成 smoke-test 單一參數組合，避免跑完整 hyperparameter grid

MMRec 本機 commit：

```text
863e3e6 Add ShortVideo MMGCN smoke pipeline
```

## 執行指令與 Logs

環境檢查：

```bash
/Users/luoyi/Desktop/shortvideo_mmgcn_poc/.venv/bin/python scripts/check_env.py \
  2>&1 | tee logs/check_env.log
```

資料檢查：

```bash
/Users/luoyi/Desktop/shortvideo_mmgcn_poc/.venv/bin/python scripts/inspect_shortvideo_data.py \
  2>&1 | tee logs/inspect_data.log
```

LightGCN：

```bash
cd /Users/luoyi/Desktop/shortvideo_mmgcn_poc/MMRec/src
/Users/luoyi/Desktop/shortvideo_mmgcn_poc/.venv/bin/python main.py --model LightGCN --dataset video --epochs 1 \
  2>&1 | tee /Users/luoyi/Desktop/shortvideo_mmgcn_poc/logs/lightgcn_smoke.log
```

VBPR：

```bash
cd /Users/luoyi/Desktop/shortvideo_mmgcn_poc/MMRec/src
/Users/luoyi/Desktop/shortvideo_mmgcn_poc/.venv/bin/python main.py --model VBPR --dataset video --epochs 1 \
  2>&1 | tee /Users/luoyi/Desktop/shortvideo_mmgcn_poc/logs/vbpr_smoke.log
```

MMGCN 1 epoch：

```bash
cd /Users/luoyi/Desktop/shortvideo_mmgcn_poc/MMRec/src
/Users/luoyi/Desktop/shortvideo_mmgcn_poc/.venv/bin/python main.py --model MMGCN --dataset video --epochs 1 \
  2>&1 | tee /Users/luoyi/Desktop/shortvideo_mmgcn_poc/logs/mmgcn_smoke.log
```

MMGCN 5 epochs：

```bash
cd /Users/luoyi/Desktop/shortvideo_mmgcn_poc/MMRec/src
/Users/luoyi/Desktop/shortvideo_mmgcn_poc/.venv/bin/python main.py --model MMGCN --dataset video --epochs 5 \
  2>&1 | tee /Users/luoyi/Desktop/shortvideo_mmgcn_poc/logs/mmgcn_5epoch.log
```

## 成功結果

LightGCN 1 epoch：

| split | Recall@10 | Recall@20 | NDCG@10 | NDCG@20 |
|---|---:|---:|---:|---:|
| valid | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| test | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

VBPR 1 epoch：

| split | Recall@10 | Recall@20 | NDCG@10 | NDCG@20 |
|---|---:|---:|---:|---:|
| valid | 0.0000 | 0.0036 | 0.0000 | 0.0016 |
| test | 0.0036 | 0.0036 | 0.0037 | 0.0036 |

MMGCN 1 epoch：

| split | Recall@10 | Recall@20 | NDCG@10 | NDCG@20 |
|---|---:|---:|---:|---:|
| valid | 0.0000 | 0.0048 | 0.0000 | 0.0018 |
| test | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

MMGCN 5 epochs best-valid checkpoint：

| split | Recall@10 | Recall@20 | NDCG@10 | NDCG@20 |
|---|---:|---:|---:|---:|
| valid | 0.0000 | 0.0048 | 0.0000 | 0.0018 |
| test | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

MMGCN 5 epochs final epoch test output：

| split | Recall@10 | Recall@20 | NDCG@10 | NDCG@20 |
|---|---:|---:|---:|---:|
| test epoch 4 | 0.0012 | 0.0012 | 0.0010 | 0.0009 |

## 目前已成功 / 未成功

已成功：

- data loading 成功
- dataset statistics 成功輸出
- visual/text features 成功載入
- LightGCN 1 epoch 成功
- VBPR 1 epoch 成功
- MMGCN 1 epoch 成功
- MMGCN 5 epochs 成功
- Recall/NDCG @ 10/20 成功輸出

未做：

- 未下載 3.2 TB raw videos
- 未抽取新的 video/text features
- 未跑完整 benchmark
- 未對完整 153561 item processed feature matrix 跑全量訓練

## 下一步

1. 擴大 subset，例如 `--target-interactions 30000`
2. 在 GPU/RunPod 上跑完整 processed recommendation dataset
3. 比較 LightGCN / VBPR / MMGCN 的穩定指標，多跑幾個 seed
4. 若要正式報告，加入 full dataset 的時間/顯存估算與 ablation 設計
