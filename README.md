# ShortVideo MMGCN PoC

這個專案用公開的 ShortVideo 資料集，探索短影音推薦與影片表現預測。

核心問題：一支影片剛上架、還沒有足夠互動紀錄時，能不能只靠內容特徵、作者資訊、分類與相似影片關係，預測它在平台上的表現。

## 專案重點

本專案分成三條 pipeline：

| Pipeline | 任務 | 方法 | 主要用途 |
|---|---|---|---|
| A | 使用者影片推薦 | LightGCN / VBPR / MMGCN | 預測某個使用者可能喜歡哪些影片 |
| B | 影片表現預測 baseline | LightGBM | 只用影片內容特徵做非圖模型對照 |
| C | 冷啟動影片表現預測 | HeteroGNN / GraphSAGE | 用影片、作者、分類、相似影片組成異質圖做節點回歸 |

目前主軸是 Pipeline C。

## Pipeline C 做什麼

Pipeline C 不放 user 節點，避免模型依賴使用者互動歷史。它把資料建成影片中心的異質圖：

```text
video --posted_by--> author
video --belongs_to--> category
video --similar_to--> video
```

輸入包含：

- 影片視覺特徵：ViT 768 維
- 影片文字特徵：預抽取 50 維，或 SBERT 384 維
- 影片長度
- 作者粉絲數與 train-only 統計
- 分類與內容相似影片鄰居

輸出 8 個影片層級指標：

- `like_rate`
- `comment_rate`
- `follow_rate`
- `collect_rate`
- `forward_rate`
- `hate_rate`
- `effective_view_rate`
- `mean_watch_time`

## 目前結果

工作集規模：54,088 部影片，train / val / test = 80 / 10 / 10。

最重要結果是 `mean_watch_time` 的排序能力：

| 方法 | `mean_watch_time` Spearman |
|---|---:|
| Pipeline B LightGBM | 0.047 |
| Pipeline C HeteroGNN SAGE | 0.251 |

解讀：

- GNN 對觀看時長排序明顯優於只看單支影片特徵的 LightGBM。
- 多數 rate 類指標仍接近隨機，主因是按讚、留言、追蹤等事件非常稀疏。
- `mean_watch_time` 的 MAE 仍偏大，因此目前更適合用來排序，不適合直接當精準秒數預測。

詳細實驗與推導見 [docs/final_project_report.tex](docs/final_project_report.tex) 或 [docs/final_project_report_notion.md](docs/final_project_report_notion.md)。

## 專案結構

```text
.
├── scripts/
│   ├── build_behavior_labels.py      # 從 interaction.csv 聚合影片層級標籤
│   ├── align_features_labels.py      # 對齊影片特徵與標籤
│   ├── train_behavior_model.py       # Pipeline B: LightGBM baseline
│   ├── build_hetero_graph.py         # Pipeline C: 建異質圖
│   ├── train_hetero.py               # Pipeline C: 訓練 HeteroGNN
│   └── predict_hetero.py             # Pipeline C: by-pid / cold-start 推論
├── data_raw/
│   ├── shortvideo_tiny/              # tiny dataset 與 interaction.csv
│   └── video_rec_dataset/            # image_feat.npy, text_feat.npy, pids.txt, video.inter
├── data_processed/
│   └── behavior/                     # labels, meta, models, metrics
├── docs/                             # 報告、簡報與參考資料
├── MMRec/                            # Pipeline A 使用的推薦系統框架
└── requirements.lock
```

注意：大型 raw data 與部分模型產物被 `.gitignore` 排除。若是重新 clone，請先補齊 `data_raw/video_rec_dataset/` 與 `data_raw/shortvideo_tiny/interaction.csv`，或重新跑資料準備流程。

## 環境設定

建議使用 Python 3.10。

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.lock
pip install scikit-learn lightgbm pyarrow

python scripts/check_env.py
```

## 快速重跑 Pipeline C

前提：已存在 `data_processed/behavior/meta.parquet`、`data_raw/video_rec_dataset/` 與 `data_raw/shortvideo_tiny/interaction.csv`。

```bash
# 建立異質圖
.venv/bin/python scripts/build_hetero_graph.py

# 訓練 GraphSAGE 版本的 HeteroGNN
.venv/bin/python scripts/train_hetero.py --model sage --epochs 300 --seed 42

# 對資料集中既有影片推論
.venv/bin/python scripts/predict_hetero.py by-pid --pid 84199269992

# 對新影片做 cold-start 推論，使用封面圖
.venv/bin/python scripts/predict_hetero.py from-raw \
  --cover /path/to/cover.jpg \
  --title "video title" \
  --duration 60
```

若要用 SBERT 文字特徵重建圖：

```bash
.venv/bin/python scripts/build_hetero_graph.py --use-sbert
.venv/bin/python scripts/train_hetero.py --model sage --epochs 300 --seed 42
```

## 從原始互動資料重建 Pipeline B

```bash
# 1. 聚合影片層級行為標籤
.venv/bin/python scripts/build_behavior_labels.py \
  --interaction data_raw/shortvideo_tiny/interaction.csv \
  --out data_processed/behavior/video_behavior_labels.parquet \
  --min-impressions 10

# 2. 對齊視覺、文字特徵與標籤
.venv/bin/python scripts/align_features_labels.py \
  --labels data_processed/behavior/video_behavior_labels.parquet \
  --visual-matrix data_raw/video_rec_dataset/image_feat.npy \
  --visual-pid-list data_raw/video_rec_dataset/pids.txt \
  --text-matrix data_raw/video_rec_dataset/text_feat.npy \
  --text-pid-list data_raw/video_rec_dataset/pids.txt \
  --out-dir data_processed/behavior

# 3. 訓練 LightGBM baseline
.venv/bin/python scripts/train_behavior_model.py \
  --in-dir data_processed/behavior \
  --out-dir data_processed/behavior/models \
  --test-frac 0.1 \
  --seed 20260519
```

## Pipeline A 簡述

Pipeline A 使用 MMRec 跑推薦系統模型，目標是使用者對影片的 Top-K 推薦。

```bash
# 準備 MMRec video dataset
.venv/bin/python scripts/prepare_video_yaml.py

# 在 MMRec 中跑模型
cd MMRec/src
../../.venv/bin/python main.py --model LightGCN --dataset video --epochs 1 --gpu_id -1
../../.venv/bin/python main.py --model VBPR --dataset video --epochs 1 --gpu_id -1
../../.venv/bin/python main.py --model MMGCN --dataset video --epochs 5 --gpu_id -1
```

Pipeline A 主要是推薦系統對照；若目標是新影片表現預測，優先看 Pipeline C。

## 重要輸出

| 檔案 | 說明 |
|---|---|
| `data_processed/behavior/video_behavior_labels.parquet` | 每部影片的 8 個行為標籤 |
| `data_processed/behavior/meta.parquet` | 工作集 pid、曝光數與標籤 |
| `data_processed/behavior/models/metrics.json` | LightGBM baseline 指標 |
| `data_processed/behavior/hetero_graph.pt` | Pipeline C 異質圖，可重建 |
| `data_processed/behavior/models_gnn/metrics_gnn.json` | HeteroGNN 測試集指標 |

## 下一步

- 用 `--use-sbert` 重跑 Pipeline C，確認文字語意是否改善排序。
- 跑 `--model hgt` 和 GraphSAGE 做 ablation。
- 將更多影片屬性加入 LightGBM，建立更公平的非圖 baseline。
- 儲存 author/category 原始 ID 對應，讓 cold-start 推論能接上作者與分類邊，而不是只靠 kNN。
