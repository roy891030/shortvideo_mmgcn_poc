# Pipeline C 資料檢查記錄

檢查日期：2026-05-31

---

## 1. interaction.csv

路徑：`data_raw/shortvideo_tiny/interaction.csv`

- 行數：6,816,699（含標題行）
- 欄位（28 個）：
  ```
  user_id, pid, author_id, category_id, category_level, parent_id, root_id,
  exposed_time, author_fans_count, watch_time, duration, cvm_like, click,
  comment, follow, collect, forward, hate, tag_name, title, p_hour, p_date,
  gender, age, mod_price, fre_city, fre_community_type, fre_city_level
  ```
- **注意**：沒有 `title_en` 或 `asr_en`，只有中文 `title`；文字特徵若用 SBERT 須直接對 `title` 欄處理
- 唯一 pid 數：153,561
- 唯一 author_id 數：81,870
- 唯一 category_id 數：700（含三級結構 category_level / parent_id / root_id）
- tag_name：每列一個中文標籤字串，同一 pid 可能多列（需 groupby 聚合）
- duration 範圍：2.7 秒 ～ 1,734.9 秒，中位數 ≈ 82 秒

## 2. 視覺 / 文字特徵矩陣

路徑：`data_raw/video_rec_dataset/`

| 檔案 | Shape | dtype | 說明 |
|---|---|---|---|
| `image_feat.npy` | (153561, **768**) | float32 | ViT-B/16 平均，**已是 768 維**，非 3072 |
| `text_feat.npy` | (153561, **50**) | float64 | 舊版 50 維文字特徵（非 SBERT） |

- `image_feat` 列 i 對應 `pids.txt` 第 i 行的 pid，對齊無誤
- X.npy = concat(image 768 + text 50) = **818 維**，由 `align_features_labels.py` 確認

## 3. pids.txt

路徑：`data_raw/video_rec_dataset/pids.txt`

- 總行數：153,561（與 image_feat 列數吻合，差異為 0）
- 有效 pid 數（非 -1）：**86,578**
- 格式：每行一個整數 pid，`-1` 表示該列無對應影片

## 4. data_processed/behavior/ 既有產物

| 檔案 | Shape | 說明 |
|---|---|---|
| `video_behavior_labels.parquet` | (98,999, 12) | 所有有標籤的 pid，欄位含 pid + 8指標 + median_watch_time/std_watch_time/n_impressions |
| `X.npy` | (54,088, 818) | **工作集**：同時有 label 又在 pids.txt 有效的 pid |
| `Y.npy` | (54,088, 8) | 對應 8 指標 |
| `meta.parquet` | (54,088, 10) | pid + n_impressions + 8指標 |
| `weights.npy` | (54,088,) | n_impressions（曝光次數，訓練加權用） |

**Pipeline C 工作集 = 54,088 影片**（labels 98,999 與 pids.txt 86,578 有效的交集）

## 5. 環境

- `.venv` Python 版本：**3.10.20**
- 已安裝關鍵套件：
  - torch 2.12.0
  - torch-geometric 2.7.0
  - lightgbm 4.6.0
  - scipy 1.15.3
  - numpy 2.2.6
  - pandas 2.3.3
- **缺少，需安裝**：`sentence-transformers`、`faiss-cpu`（或退回 sklearn）、`opencv-python`

---

## 關鍵設計決策（由真實資料推導）

1. **視覺特徵**：直接用 `image_feat.npy` 的 768 維，無需重抽（已是 ViT 平均）
2. **文字特徵**：預設用現有 50 維 `text_feat.npy`；可切換 SBERT（對中文 `title` 用 `paraphrase-multilingual-MiniLM-L12-v2` 得 384 維）
3. **video 節點特徵維度**：768 + 50(或384) + 1(log_duration) = **819 維**（預設）或 1153 維（SBERT 模式）
4. **category 節點**：三級結構（category_id / parent_id / root_id），各自 `nn.Embedding`
5. **tag 節點**（選用）：需 groupby pid 聚合多個 tag_name
6. **工作集 pid 對應**：以 `meta.parquet` 的 54,088 個 pid 為主，透過 `pids.txt` 查 image_feat 列索引
