# Video → Behavior Prediction Tool

A content-driven predictor: given a short video's visual + text features,
estimate the per-video aggregate user behaviors.

Status: scaffolding ready; awaiting download of the ShortVideo **tiny**
dataset (the one that contains `interaction_filtered.csv` with the full
behavior schema).

## What this tool predicts

Input
- 視覺特徵：ViT-B/16 抽出（4 frame × 768 = 3072 維），或沿用 Tsinghua
  Dropbox processed 版本的 768 維 ResNet+ViT 降維特徵
- 文字特徵：title / ASR 經 GloVe-50d 平均池化
- (可選) 影片屬性：duration、3-level category 等（目前 pipeline 暫不收）

Output（每部影片 8 個數字）

| Target | Range | 意義 |
|---|---|---|
| `like_rate` | [0, 1] | 看過此影片的使用者中按讚比例 |
| `comment_rate` | [0, 1] | 留言比例 |
| `follow_rate` | [0, 1] | 看完追蹤作者比例 |
| `collect_rate` | [0, 1] | 收藏比例 |
| `forward_rate` | [0, 1] | 轉發比例 |
| `hate_rate` | [0, 1] | hate 比例 |
| `effective_view_rate` | [0, 1] | 觀看 ≥ 3 秒比例 |
| `mean_watch_time` | 秒 | 平均觀看秒數 |

注意：這是「影片在整個平台上會引發的平均反應」，不是某個特定 user 的行為。

## Data source

### Tiny version (建議先用這份)

Dropbox 連結（在 ShortVideo_dataset/README.md 第 10 行）：

```
https://www.dropbox.com/scl/fo/5z7pwp6xjkrr1926vreu6/AFYter5C6BDTOCpxkxF0k9Y?rlkey=p28j6u1fl1ubb7bufiq16onbl&st=7aktff85&dl=0
```

下載完整目錄到 `data_raw/shortvideo_tiny/`，預期至少含：

```
data_raw/shortvideo_tiny/
├── interaction_filtered.csv     ← 行為主檔（必要）
├── video_feature_total/         ← 每部影片一個 .npy（必要）
│   ├── 1.npy
│   └── ...
├── title_en/                    ← 英文標題（建議）
│   └── {pid}.txt
├── asr_en/                      ← 英文 ASR（可選）
└── category_cn_en.csv           ← 分類資訊（可選）
```

如果該目錄沒有 `video_feature_total/`，就要先用
`ShortVideo_dataset/video_feature_process.py` 對 `raw_file/` 內每部影片
單獨抽 ViT 特徵存成同名 .npy。

### 何時要回去抓完整版

完整版（Tsinghua server `http://fi.ee.tsinghua.edu.cn/datasets/short-video-dataset`，帳號 `videodata`、密碼 `ShortVideo@10000`）資料量大很多，
等 tiny 版 PoC 跑通了再考慮搬上去。

## Pipeline

### Step 1 — 聚合 per-video labels

```bash
.venv/bin/python scripts/build_behavior_labels.py \
    --interaction data_raw/shortvideo_tiny/interaction_filtered.csv \
    --out data_processed/behavior/video_behavior_labels.parquet \
    --min-impressions 10
```

產出：`data_processed/behavior/video_behavior_labels.parquet`
（每列一個 pid，含 7 個 rate + `mean_watch_time` + `n_impressions`）

### Step 2 — 對齊特徵與 labels

```bash
.venv/bin/python scripts/align_features_labels.py \
    --labels data_processed/behavior/video_behavior_labels.parquet \
    --visual-dir data_raw/shortvideo_tiny/video_feature_total \
    --text-dir data_raw/shortvideo_tiny/title_en \
    --glove-dim 50 \
    --out-dir data_processed/behavior
```

產出：`data_processed/behavior/X.npy`, `Y.npy`, `weights.npy`, `meta.parquet`

如果沒有 `title_en/`，省略 `--text-dir` 會用零向量補 text 部分，模型還是
能跑（但只看 visual）。如果想用 Tsinghua 預抽的 50 維 text，可改用：

```bash
--text-matrix data_raw/video_rec_dataset/text_feat.npy \
--text-pid-list data_raw/video_rec_dataset/pids.txt
```

（需要先準備 `pids.txt`，每列一個 pid，對應 `text_feat.npy` 行索引。）

### Step 3 — 訓練

```bash
.venv/bin/pip install lightgbm scikit-learn scipy --break-system-packages
.venv/bin/python scripts/train_behavior_model.py \
    --in-dir data_processed/behavior \
    --out-dir data_processed/behavior/models \
    --test-frac 0.1 --val-frac 0.1 --seed 20260519
```

產出：
- `data_processed/behavior/models/{target}.txt`（每個 target 一個 booster）
- `data_processed/behavior/models/metrics.json`（test set MAE/RMSE/Spearman/AUC）
- `data_processed/behavior/models/test_predictions.parquet`

### Step 4 — 推論

模式 A：用 dataset 裡已有的 pid

```bash
.venv/bin/python scripts/predict_behavior.py by-pid \
    --pid 12345 \
    --in-dir data_processed/behavior \
    --models-dir data_processed/behavior/models
```

模式 B：對新影片（.mp4 + 標題 + 可選 ASR 文字）

```bash
.venv/bin/python scripts/predict_behavior.py from-raw \
    --video /path/to/new_clip.mp4 \
    --title "Underwater photography tips" \
    --asr-text "today we are going to film fish ..." \
    --models-dir data_processed/behavior/models \
    --glove-dim 50
```

輸出 JSON：

```json
{
  "video": "/path/to/new_clip.mp4",
  "title": "Underwater photography tips",
  "predictions": {
    "like_rate": 0.0823,
    "comment_rate": 0.0041,
    ...
    "mean_watch_time": 17.4
  }
}
```

## 模型設計

- **每個 target 各自一棵 LightGBM Regressor**（gradient-boosted tree），
  共享 input feature，輸出獨立。比共享 backbone 的 MLP 在 818-維特徵 +
  數千～數萬樣本的情境下更穩、tune cost 更低。
- `mean_watch_time` 用 `log1p` 轉換後訓練，預測時再 `expm1`，避免長尾。
- 訓練時用 `n_impressions` 當 sample_weight，曝光少的影片噪音大就權重低。
- 切分：random train/val/test = 0.8 / 0.1 / 0.1，early stop 看 val。

## 評估指標

- 連續：MAE / RMSE
- 排序：Spearman（更貼近實際用途——「哪部影片較會被讚」）
- 分類視角：AUC@above-median（把高於中位數的影片視為 positive）

預期合理範圍（憑經驗，沒有官方 benchmark 對齊）：
- `effective_view_rate`、`mean_watch_time`：Spearman 0.3 ~ 0.6
- `like_rate`、`collect_rate`：Spearman 0.2 ~ 0.4
- `comment_rate`、`follow_rate`、`hate_rate`：偏難，Spearman 0.1 ~ 0.3

## 常見問題

**Q1. 為什麼這條 pipeline 跟 MMRec/MMGCN 一點關係都沒有？**
因為任務本質不同。MMGCN 做的是「user → ranked items」協同過濾；這個工具
做的是「video features → behavior aggregates」的內容回歸。前者需要 user
歷史，後者完全 content-only。

**Q2. 可不可以同時利用既有 MMGCN 學到的 item embedding？**
可以。把 MMGCN 訓練後的 `item_embedding[v]` 當成額外特徵 concat 進 X。
但這會把模型綁回到「該 item 必須在訓練圖裡」的 transductive 限制，
新影片就用不上，違反這支工具的設計初衷。建議純內容特徵。

**Q3. 預測值會超出 [0, 1] 嗎？**
LightGBM regression 沒有 sigmoid，可能會輕微超出範圍。推論時可以
clip 到 [0, 1]，或用 `objective="binary"` 改成 logistic（但需要把
label 從 rate 改成個別觀測，數量會爆炸）。簡單 clip 即可。

**Q4. 跟 Tsinghua paper Table 1 的 BM3/MMGCN 等指標能對齊嗎？**
不能。那張表是 Recall@K / NDCG@K 的 ranking 評估，本工具是 regression
評估，是不同任務的不同 metric，不能直接比。
