# ShortVideo MMGCN PoC — 完整教學說明

> **寫給誰看的？** 這份文件試著讓任何人都能看懂這個專案在做什麼——包括對 AI 只有基本認識的讀者。
> 如果你是第一次接觸推薦系統或機器學習，從頭讀起就對了。

---

## 目錄

1. [這個專案在解決什麼問題？](#1-這個專案在解決什麼問題)
2. [兩個平行的研究方向](#2-兩個平行的研究方向)
3. [背後的理論：推薦系統怎麼運作？](#3-背後的理論推薦系統怎麼運作)
4. [背後的理論：圖神經網路（GNN）是什麼？](#4-背後的理論圖神經網路gnn是什麼)
5. [資料集完整介紹](#5-資料集完整介紹)
6. [專案目錄結構](#6-專案目錄結構)
7. [環境設定](#7-環境設定)
8. [Pipeline A：推薦系統（MMGCN）](#8-pipeline-a推薦系統mmgcn)
9. [Pipeline B：行為預測（LightGBM）](#9-pipeline-b行為預測lightgbm)
10. [目前實驗結果與解讀](#10-目前實驗結果與解讀)
11. [遇到的技術問題與解法](#11-遇到的技術問題與解法)
12. [下一步計畫](#12-下一步計畫)

---

## 1. 這個專案在解決什麼問題？

### 你一定遇過這個情境

你打開抖音、TikTok、YouTube Shorts，下滑一下，系統就自動播放一支又一支的短影片。有時候你心想：「這系統怎麼這麼懂我？」

這不是魔法，是演算法。而這個演算法的核心問題叫做 **推薦系統（Recommendation System）**。

### 推薦系統要解決的問題

平台上有幾百萬支影片，每個使用者只看得完其中的一小部分。推薦系統的任務是：

> **「給這個使用者，在這個時間點，推薦他最可能喜歡看的影片。」**

怎麼判斷「喜歡」？最直接的方式是看 **過去的行為**——他看過哪些影片、按過哪些讚、追蹤過哪些作者。

### 短影片的特殊挑戰

短影片和傳統的電影推薦、商品推薦不一樣，有幾個特殊之處：

| 特點 | 說明 |
|---|---|
| **影片本身有內容** | 每支影片有畫面、聲音、文字說明，這些「內容特徵」可以幫助推薦 |
| **行為訊號非常豐富** | 不只「看沒看」，還有按讚、留言、追蹤、收藏、轉發、討厭、觀看秒數 |
| **冷啟動問題嚴重** | 新上傳的影片完全沒有歷史互動資料，要怎麼推薦它？ |
| **資料量龐大** | 幾百萬使用者 × 幾百萬影片，組合數量超大 |

這個專案想做的事情，就是把「影片的視覺和文字內容」結合「使用者的歷史行為」，建立一個更聰明的推薦模型。

---

## 2. 兩個平行的研究方向

這個專案同時探索兩個不同的問題，可以獨立理解：

```
這個專案
├── Pipeline A：推薦系統
│   問題：「哪部影片應該推給哪個使用者？」
│   方法：MMGCN（多模態圖神經網路）
│   輸入：使用者歷史行為 + 影片視覺/文字特徵
│   輸出：對每個使用者，預測他最可能互動的影片排名
│
└── Pipeline B：行為預測
    問題：「這部影片在平台上會引發多少互動？」
    方法：LightGBM（梯度提升樹）
    輸入：影片的視覺特徵 + 文字特徵
    輸出：8 個行為指標（按讚率、留言率、觀看秒數...）
```

**兩者的差別**：
- Pipeline A 預測「哪個 **使用者** 喜歡這部影片」——需要使用者的個人資訊
- Pipeline B 預測「這部影片在 **整個平台** 上會有多少互動」——只需要影片本身

---

## 3. 背後的理論：推薦系統怎麼運作？

### 3.1 最簡單的方法：協同過濾

想像你和同學都喜歡看烹飪影片，你看過 A、B、C 三支，他看過 A、B、D、E。系統推斷你們品味相近，就把 D、E 推薦給你。

這叫 **協同過濾（Collaborative Filtering）**——「和你相似的人喜歡什麼，你也可能喜歡」。

問題：如果一支影片是全新上傳，沒人看過，系統就完全不知道要推薦給誰。這叫 **冷啟動問題**。

### 3.2 加入影片內容：多模態推薦

解決冷啟動的方法之一，是讓系統「看懂」影片本身。

影片有兩種主要的「感官」資訊：
- **視覺模態（Visual Modality）**：影片的畫面，用 AI 抽取出數百維的數字向量
- **文字模態（Text Modality）**：影片的標題、字幕（ASR），用 AI 轉成數字

這種同時使用多種資料類型的方法，叫做 **多模態（Multimodal）**。

### 3.3 本專案使用的模型：LightGCN 和 MMGCN

我們測試了三個模型：

#### LightGCN
- 一個純粹基於互動行為的圖神經網路
- 不看影片內容，只看「誰看了什麼」
- 基線（Baseline）：用來比較「加入影片內容有沒有幫助」

#### VBPR（Visual Bayesian Personalized Ranking）
- 把視覺特徵加入傳統推薦模型
- 2015 年的經典方法，用來當作較老的多模態 baseline

#### MMGCN（Multi-Modal Graph Convolutional Network）
- 這是我們的主要目標
- 同時利用視覺、文字、互動行為三種資訊
- 用圖神經網路學習「哪些影片和哪些使用者是有關聯的」

---

## 4. 背後的理論：圖神經網路（GNN）是什麼？

### 4.1 什麼是「圖」？

這裡的「圖（Graph）」不是長條圖或折線圖，而是數學上的圖論概念：

```
使用者 A ──看過──► 影片 1
使用者 A ──看過──► 影片 3
使用者 B ──看過──► 影片 1
使用者 B ──看過──► 影片 2
使用者 C ──看過──► 影片 2
使用者 C ──看過──► 影片 3
```

把使用者和影片當成圖中的**節點（Node）**，把「看過」關係當成**邊（Edge）**，就形成了一張互動圖。

### 4.2 圖神經網路怎麼學習？

GNN 的核心思想叫做 **訊息傳遞（Message Passing）**：

1. 每個節點一開始有一個隨機的「代表向量」（就像一個人有自己的 ID）
2. 每個節點「問問鄰居」——把相連節點的代表向量平均起來
3. 重複幾輪，每個節點都逐漸「知道」它的鄰居是誰

經過幾輪傳遞後：
- **使用者節點**的向量，會反映出他看過哪些影片的特徵
- **影片節點**的向量，會反映出哪些使用者看過它

最後用使用者向量和影片向量的「相似度」來預測「使用者是否會喜歡這部影片」。

### 4.3 MMGCN 加了什麼？

MMGCN 在普通 GCN 的基礎上加了兩條資訊流：

```
普通 GCN：
    行為互動圖 → 圖卷積 → 使用者/影片表示

MMGCN：
    視覺特徵圖 → 圖卷積 ─────┐
    文字特徵圖 → 圖卷積 ─────┼─→ 融合 → 使用者/影片表示
    行為互動圖 → 圖卷積 ─────┘
```

每個模態都有自己的圖卷積，最後把三個模態的資訊融合起來預測。

---

## 5. 資料集完整介紹

這個專案使用 **ShortVideo Dataset**，來自清華大學，2025 年發表在 WWW 會議。

### 5.1 資料集的兩個版本

#### 版本一：Tiny 版本（我們主要在用的）

從 Dropbox 下載，解壓後在 `data_raw/shortvideo_tiny/`：

```
data_raw/shortvideo_tiny/
├── interaction.csv          ← 主要行為日誌（6,767,010 行！）
├── video_feature_total/     ← 10 支範例影片的視覺特徵
│   ├── 1.npy                   每個檔案 shape: (8, 256) = 8幀 × 256維
│   └── ...（共 10 個）
├── raw_file/                ← 10 支範例影片的原始 .mp4
│   └── 1.mp4 ... 10.mp4
├── title_en/                ← 10 支影片的英文標題
├── asr_en/                  ← 10 支影片的英文語音辨識字幕
├── asr_zn/                  ← 10 支影片的中文字幕
└── categories_cn_en.csv     ← 分類標籤對照表
```

**重要注意**：`video_feature_total/` 的 10 個檔案（1.npy～10.npy）是示範用的樣本影片，
它們的「pid」是 1、2、...、10，**和 interaction.csv 裡的真實 pid（60億量級的數字）沒有直接對應**。
詳細說明見 [第 11 節](#11-遇到的技術問題與解法)。

#### 版本二：Processed 推薦版本（從 Dropbox 下載的）

存放在 `data_raw/video_rec_dataset/`：

```
data_raw/video_rec_dataset/
├── video.inter              ← 重新映射過的互動檔（294,355 行）
├── image_feat.npy           ← 所有影片的視覺特徵 (153,561, 768)
└── text_feat.npy            ← 所有影片的文字特徵 (153,561, 50)
```

這個版本是給推薦系統（MMRec 框架）直接使用的，itemID 是從 0 開始的整數。

### 5.2 interaction.csv 欄位詳解

這是最重要的資料檔，每一行代表「一個使用者看過一支影片的一次曝光記錄」：

| 欄位名稱 | 型態 | 說明 | 例子 |
|---|---|---|---|
| `user_id` | 數字 | 使用者 ID（已雜湊保護隱私） | 7313 |
| `pid` | 數字 | 影片 ID（已雜湊） | 84199269992 |
| `exposed_time` | 數字 | 曝光的 Unix 時間戳 | 1663333100 |
| `p_date` | 數字 | 日期（YYYYMMDD 格式） | 20220916 |
| `p_hour` | 數字 | 小時（0-23） | 21 |
| `watch_time` | 數字 | 這個使用者看了幾秒 | 136 |
| `duration` | 數字 | 這支影片的總長（秒） | 91.9 |
| `cvm_like` | 布林 | 有沒有按讚 | True |
| `click` | 布林 | 有沒有點擊進去看 | True |
| `comment` | 布林 | 有沒有留言 | False |
| `follow` | 布林 | 有沒有追蹤作者 | False |
| `collect` | 布林 | 有沒有收藏 | False |
| `forward` | 布林 | 有沒有轉發 | False |
| `hate` | 布林 | 有沒有點「不喜歡」 | False |
| `category_id` | 數字 | 影片的分類 ID | 36 |
| `author_id` | 數字 | 作者 ID | 193159093 |
| `author_fans_count` | 數字 | 作者的粉絲數 | 46761 |
| `tag_name` | 文字 | 內容標籤 | 正能量 |
| `title` | 文字 | 影片標題 | 现如今为啥 |
| `gender` | 文字 | 使用者性別 | M |
| `age` | 數字 | 使用者年齡 | 42 |
| `fre_city` | 文字 | 使用者所在城市 | 邯郸 |

### 5.3 資料規模

| 項目 | 數量 |
|---|---|
| 總行數（曝光記錄） | 6,767,010 |
| 獨立使用者數 | 10,000 |
| 獨立影片數 | 153,561 |
| 平均每人看片數 | 676.7 部 |
| 有 ≥ 10 次曝光的影片 | 98,999 部 |

### 5.4 影片視覺特徵是怎麼來的？

原始影片先用 AI 模型抽取「視覺特徵」。這個過程叫做 **特徵抽取（Feature Extraction）**：

```
原始影片 (.mp4)
    ↓
從影片裡均勻取 N 幀畫面
    ↓
每幀畫面送進 ViT（Vision Transformer，一種影像 AI 模型）
    ↓
每幀產生一個 768 維的數字向量（相當於影像的「指紋」）
    ↓
把多幀的結果平均，得到這支影片的視覺特徵向量
```

最終每支影片用一個 768 維的向量代表，存成 .npy 檔案。
Tiny 版本的 10 支範例影片是 8 幀 × 256 維（不同的模型設定）。

---

## 6. 專案目錄結構

```
shortvideo_mmgcn_poc/
│
├── README.md                      ← 你正在讀的這份文件
├── README_REPRO.md                ← Pipeline A 推薦系統的執行紀錄
├── README_BEHAVIOR.md             ← Pipeline B 行為預測的舊版說明
├── requirements.lock              ← Python 套件版本清單
│
├── MMRec/                         ← 推薦系統框架（從 GitHub clone 下來）
│   ├── src/
│   │   ├── main.py                   ← 訓練入口
│   │   ├── models/
│   │   │   ├── lightgcn.py           ← LightGCN 模型（已修改相容性）
│   │   │   ├── mmgcn.py              ← MMGCN 模型（已修改相容性）
│   │   │   └── vbpr.py               ← VBPR 模型
│   │   └── configs/dataset/
│   │       └── video.yaml            ← 我們的資料集設定檔
│   └── data/video/                   ← MMRec 讀取資料的位置（symlink）
│
├── ShortVideo_dataset/            ← 論文原始 repo（從 GitHub clone）
│   ├── README.md
│   └── video_feature_process.py   ← 官方的特徵抽取腳本
│
├── data_raw/
│   ├── shortvideo_tiny/           ← Tiny 版資料集（從 Dropbox 下載）
│   │   ├── interaction.csv            ← 6.7M 行的行為日誌
│   │   ├── video_feature_total/       ← 10 支範例影片特徵
│   │   ├── raw_file/                  ← 10 支範例 .mp4
│   │   ├── title_en/ asr_en/ asr_zn/  ← 標題和字幕
│   │   └── categories_cn_en.csv
│   │
│   └── video_rec_dataset/         ← 推薦用已處理資料集（從 Dropbox 下載）
│       ├── video.inter                ← 重映射的互動資料
│       ├── image_feat.npy             ← (153561, 768) 視覺特徵矩陣
│       ├── text_feat.npy              ← (153561, 50) 文字特徵矩陣
│       ├── pids.txt                   ← [我們生成的] itemID → 真實 pid 對照
│       └── itemid_to_pid.parquet      ← [我們生成的] 同上的完整版本
│
├── data_processed/
│   ├── video/                     ← MMRec 的 smoke-test 子集資料
│   │   ├── video.inter
│   │   ├── image_feat.npy         ← (2822, 768) 子集視覺特徵
│   │   └── text_feat.npy          ← (2822, 50) 子集文字特徵
│   │
│   └── behavior/                  ← Pipeline B 的中間產物
│       ├── video_behavior_labels.parquet  ← 每部影片的行為標籤（98,999 部）
│       ├── X.npy                          ← 特徵矩陣 (54088, 818)
│       ├── Y.npy                          ← 標籤矩陣 (54088, 8)
│       ├── weights.npy                    ← 每部影片的曝光次數（訓練權重）
│       ├── meta.parquet                   ← pid 和標籤的對照表
│       └── models/                        ← LightGBM 訓練好的模型
│           ├── like_rate.txt
│           ├── comment_rate.txt
│           ├── ... （共 8 個）
│           └── metrics.json               ← 評估結果
│
├── scripts/                       ← 我們撰寫的所有 Python 腳本
│   ├── build_behavior_labels.py   ← Step B1：從 interaction.csv 聚合標籤
│   ├── align_features_labels.py   ← Step B2：對齊特徵和標籤
│   ├── train_behavior_model.py    ← Step B3：訓練 LightGBM
│   ├── predict_behavior.py        ← Step B4：對新影片做預測
│   ├── prepare_video_yaml.py      ← 生成 MMRec smoke-test 子集
│   ├── check_env.py               ← 環境檢查
│   └── inspect_shortvideo_data.py ← 資料檢視
│
└── logs/                          ← 所有執行過的 log 檔
    ├── lightgcn_smoke.log
    ├── vbpr_smoke.log
    ├── mmgcn_smoke.log
    ├── mmgcn_5epoch.log
    ├── build_labels.log
    ├── align_features.log
    └── train_behavior.log
```

---

## 7. 環境設定

### 7.1 本機規格

| 項目 | 說明 |
|---|---|
| 作業系統 | macOS（Apple Silicon, arm64） |
| Python | 3.10.20 |
| PyTorch | 2.12.0 |
| torch-geometric | 2.7.0 |
| GPU | 無 CUDA（使用 CPU，MPS 可用但未啟用） |

### 7.2 建立 Python 環境

```bash
# 用 conda 建立 Python 3.10 基礎環境
conda create -p /Users/luoyi/Desktop/shortvideo_mmgcn_poc/.python310 python=3.10 -y

# 建立隔離的虛擬環境
/Users/luoyi/Desktop/shortvideo_mmgcn_poc/.python310/bin/python -m venv \
    /Users/luoyi/Desktop/shortvideo_mmgcn_poc/.venv

# 安裝所有套件
/Users/luoyi/Desktop/shortvideo_mmgcn_poc/.venv/bin/pip install --upgrade pip
/Users/luoyi/Desktop/shortvideo_mmgcn_poc/.venv/bin/pip install \
    torch torchvision torchaudio \
    torch_geometric \
    numpy pandas scipy pyyaml lmdb tqdm matplotlib \
    lightgbm scikit-learn pyarrow
```

### 7.3 為什麼要用虛擬環境？

虛擬環境就像一個「獨立的 Python 房間」。不同專案需要不同版本的套件，如果全裝在同一個地方會衝突。虛擬環境讓每個專案有自己乾淨的套件空間。

---

## 8. Pipeline A：推薦系統（MMGCN）

> **目標**：給每個使用者，預測他最可能互動的影片清單（排名）

### 8.1 整體流程

```
原始資料
    ↓
[prepare_video_yaml.py]
建立 smoke-test 子集（69 users, 2822 items, 3033 interactions）
    ↓
[MMRec/src/main.py]
訓練模型（LightGCN / VBPR / MMGCN）
    ↓
評估指標：Recall@10, Recall@20, NDCG@10, NDCG@20
```

### 8.2 為什麼要用「子集」而不是全部資料？

完整資料集有 153,561 部影片，直接在筆電 CPU 上跑會非常慢（幾天起跳）。
我們用腳本切出一個 **確定性子集（deterministic subset）**：固定隨機種子，選出 69 個活躍使用者，
再保留這些使用者互動過的影片，這樣可以在幾分鐘內完成訓練，驗證整條 pipeline 是否通。

### 8.3 子集資料統計

| 項目 | 數量 |
|---|---|
| 使用者數 | 69 |
| 影片數 | 2,822 |
| 總互動數 | 3,033 |
| 訓練集互動 | 2,452 |
| 驗證集互動 | 276 |
| 測試集互動 | 305 |
| 稀疏度 | 98.44%（代表大部分 user-item 組合都沒有互動） |

**「稀疏度」是什麼意思？**
69 位使用者 × 2,822 部影片 = 194,718 個可能的「使用者-影片」組合。
但實際上只有 3,033 個互動，也就是說 98.44% 的組合都是空白的。
推薦系統的核心挑戰就是：從這片空白裡，找出哪些位置「應該」是 1（使用者會喜歡）。

### 8.4 評估指標說明

推薦系統用「排名品質」來評估，不用普通的準確率：

#### Recall@K（召回率@K）
- 概念：在系統推薦的前 K 個影片裡，有多少比例是使用者真的喜歡的
- 例子：使用者喜歡 10 部影片，系統推薦 10 個，其中 2 個在使用者喜歡的清單裡 → Recall@10 = 0.2
- 越高越好

#### NDCG@K（Normalized Discounted Cumulative Gain）
- 概念：不只看「有沒有推對」，還看「推對的東西排在第幾位」
- 如果使用者喜歡的影片排在第 1 名，比排在第 10 名分數更高
- 越高越好

### 8.5 MMRec 框架修改

MMRec 是清華大學開源的多模態推薦框架，但有些地方和我們的環境（Python 3.10 + 新版 PyTorch/SciPy）不相容，我們做了以下修改：

| 檔案 | 問題 | 修改方式 |
|---|---|---|
| `utils/metrics.py` | `np.float` 在 NumPy 2.0 被移除 | 改成 `float` |
| `models/lightgcn.py` | SciPy 新版不支援 DOK sparse matrix 的 `_update` 方法 | 改用 COO 格式建立鄰接矩陣 |
| `models/mmgcn.py` | PyG 2.7 的 `MessagePassing.message()` 參數簽章改變 | 更新參數名 |
| `main.py` | `--epochs 1` CLI 參數沒有正確覆蓋 config | 加入 CLI override 邏輯 |
| `MMGCN.yaml` 等 | 原本設定要跑完整超參數網格搜尋 | 改成單一組合，節省時間 |

### 8.6 執行方式

**Step 1：生成子集資料**
```bash
cd /Users/luoyi/Desktop/shortvideo_mmgcn_poc
.venv/bin/python scripts/prepare_video_yaml.py
```

**Step 2：訓練 LightGCN（基線）**
```bash
cd MMRec/src
../../.venv/bin/python main.py --model LightGCN --dataset video --epochs 1 \
    2>&1 | tee ../../logs/lightgcn_smoke.log
```

**Step 3：訓練 VBPR（視覺基線）**
```bash
../../.venv/bin/python main.py --model VBPR --dataset video --epochs 1 \
    2>&1 | tee ../../logs/vbpr_smoke.log
```

**Step 4：訓練 MMGCN（主要模型）**
```bash
../../.venv/bin/python main.py --model MMGCN --dataset video --epochs 5 \
    2>&1 | tee ../../logs/mmgcn_5epoch.log
```

### 8.7 MMGCN 的模型架構（用白話解釋）

```
影片的視覺特徵 (768維)
    ↓
[MLP 降維] 768 → 256
    ↓
[圖卷積 1] 256 → 64  ← 把互動圖中的鄰居資訊融進來
    ↓
[圖卷積 2] 64 → 64
    ↓
[圖卷積 3] 64 → 64
    ↓
視覺模態的影片表示 (64維)
```

```
影片的文字特徵 (50維)
    ↓
[圖卷積 1] 50 → 64
    ↓
[圖卷積 2] 64 → 64
    ↓
[圖卷積 3] 64 → 64
    ↓
文字模態的影片表示 (64維)
```

```
視覺模態影片表示 (64維)
文字模態影片表示 (64維)
    ↓
[加權融合]
    ↓
最終影片表示 (64維)

最終使用者表示 (64維)
    ↓
使用者 · 影片 的內積 = 預測分數
（分數越高 = 越可能互動）
```

整個模型可訓練的參數數量：**378,564 個**。

---

## 9. Pipeline B：行為預測（LightGBM）

> **目標**：只看影片的視覺和文字特徵，預測這部影片在平台上的平均互動率

### 9.1 整體流程

```
interaction.csv（6.7M 行）
    ↓
[Step B1: build_behavior_labels.py]
每部影片聚合成一行：計算各行為的平均比例
    ↓
video_behavior_labels.parquet（98,999 部影片，每部 8 個標籤）

image_feat.npy (153,561 × 768)       video_behavior_labels.parquet
pids.txt（itemID → pid 對照）
    ↓
[Step B2: align_features_labels.py]
把特徵和標籤對齊到同一批影片
    ↓
X.npy (54,088 × 818)   Y.npy (54,088 × 8)

    ↓
[Step B3: train_behavior_model.py]
對每個目標，分別訓練一個 LightGBM 模型
    ↓
8 個模型檔案 + metrics.json

    ↓
[Step B4: predict_behavior.py]
對新影片做預測（可以吃新 .mp4 或已有 pid）
```

### 9.2 Step B1 詳解：從行為日誌到每部影片的標籤

**輸入**：`interaction.csv`，6,767,010 行，每行是一次曝光

**做什麼**：用 `groupby("pid")` 把同一部影片的所有曝光「壓縮」成一行

例子：
```
原始資料（影片 84199269992 的所有曝光記錄）：
user=7313, watch_time=136, cvm_like=True,  comment=False, ...
user=8821, watch_time=45,  cvm_like=False, comment=True,  ...
user=2204, watch_time=12,  cvm_like=False, comment=False, ...
...（共 50 次曝光）

聚合後（一行代表這部影片）：
pid=84199269992
like_rate = 1/50 = 0.02
comment_rate = 1/50 = 0.02
follow_rate = 0
effective_view_rate = 45/50 = 0.90  （watch_time ≥ 3秒的比例）
mean_watch_time = 64.3 秒
n_impressions = 50
```

**注意事項**：
- `effective_view` 欄位在 tiny 資料中不存在，根據官方 README 定義為 `watch_time >= 3 秒`，我們自行推導
- 曝光次數 < 10 次的影片被過濾掉（太少資料，標籤不穩定）

**輸出**：98,999 部影片的標籤，共 8 個目標：

| 目標 | 說明 | 典型值（中位數） |
|---|---|---|
| `like_rate` | 按讚率 | 0.000（大多數人不按讚） |
| `comment_rate` | 留言率 | 0.000 |
| `follow_rate` | 追蹤率 | 0.000 |
| `collect_rate` | 收藏率 | 0.000 |
| `forward_rate` | 轉發率 | 0.000 |
| `hate_rate` | 討厭率 | 0.000 |
| `effective_view_rate` | 有效觀看率 | — |
| `mean_watch_time` | 平均觀看秒數 | 17.4 秒 |

中位數幾乎都是 0 代表這些行為的分佈極度偏斜——大部分影片的互動率很低，只有少數爆紅影片才高。

**執行方式**：
```bash
cd /Users/luoyi/Desktop/shortvideo_mmgcn_poc
.venv/bin/python scripts/build_behavior_labels.py \
    --interaction data_raw/shortvideo_tiny/interaction.csv \
    --out data_processed/behavior/video_behavior_labels.parquet \
    --min-impressions 10
```

### 9.3 Step B2 詳解：特徵與標籤對齊

**問題**：標籤有 98,999 部影片（用真實 pid），但特徵矩陣的行用的是 itemID（0~153560）。
兩者需要對應起來才能訓練。

**解法：反推 itemID → pid 的映射**

推薦資料集和原始行為資料是同一批影片的不同處理版本。我們利用「兩份資料都有時間戳（timestamp）」這個特性：

```
1. 從推薦資料集取出 (userID, timestamp) 對
2. 在 interaction.csv 裡找同樣的 timestamp，得到 (user_id, pid) 對
3. 配對：userID=0 在推薦資料集出現 timestamp=T，interaction.csv 裡
   同樣 timestamp=T 的人是 user_id=3524
4. 那 userID=0 就是 user_id=3524
5. 有了 user 的對應關係後，再找 itemID 對應的 pid
```

結果：9,609/9,609 個使用者對應成功（100%），86,578/88,098 個影片對應成功（98.3%）。
未對應的影片保留在 `pids.txt` 裡標記為 -1，不用於訓練。

**最終對齊結果**：
- 98,999 個有標籤的影片
- 其中 54,088 個同時有標籤和特徵（被兩份資料都覆蓋）
- 特徵維度：768（視覺）+ 50（文字）= 818 維

**執行方式**：
```bash
.venv/bin/python scripts/align_features_labels.py \
    --labels data_processed/behavior/video_behavior_labels.parquet \
    --visual-matrix data_raw/video_rec_dataset/image_feat.npy \
    --visual-pid-list data_raw/video_rec_dataset/pids.txt \
    --text-matrix data_raw/video_rec_dataset/text_feat.npy \
    --text-pid-list data_raw/video_rec_dataset/pids.txt \
    --out-dir data_processed/behavior
```

### 9.4 Step B3 詳解：訓練 LightGBM 模型

**什麼是 LightGBM？**

LightGBM 是一種叫做「梯度提升樹（Gradient Boosting Tree）」的機器學習演算法，由微軟開發。
簡單來說：

1. 先訓練一棵決策樹，它的預測不夠準
2. 訓練第二棵樹，專門補第一棵樹犯的錯
3. 訓練第三棵樹，專門補前兩棵犯的錯
4. 重複幾百次，最後把所有樹的預測加起來

這種「把弱模型疊加成強模型」的方法非常有效，在各種資料科學競賽中都是熱門選擇。

**訓練設定**：

| 參數 | 值 | 說明 |
|---|---|---|
| 訓練/驗證/測試比例 | 80% / 10% / 10% | 隨機切分 |
| 訓練集大小 | 43,270 部影片 | |
| 驗證集大小 | 5,409 部影片 | 用來早停，防止過擬合 |
| 測試集大小 | 5,409 部影片 | 只在最後評估，不參與訓練 |
| 決策樹數量（最大） | 2,000 棵 | |
| 早停輪數 | 50 | 如果驗證誤差 50 輪沒改善就停 |
| 訓練權重 | 每部影片的曝光次數 | 曝光多的影片標籤更可靠，訓練時給更高權重 |

對 `mean_watch_time` 目標做了 `log1p` 轉換：因為觀看時間分佈很偏斜（有些影片超長，有些很短），取對數讓分佈更接近正態，訓練更穩定。

**執行方式**：
```bash
.venv/bin/python scripts/train_behavior_model.py \
    --in-dir data_processed/behavior \
    --out-dir data_processed/behavior/models \
    --test-frac 0.1 --seed 20260519
```

### 9.5 Step B4：對新影片做預測

訓練好之後，可以對任意影片做預測：

**方式 A：已知 pid（資料集內的影片）**
```bash
.venv/bin/python scripts/predict_behavior.py by-pid \
    --pid 84199269992 \
    --in-dir data_processed/behavior \
    --models-dir data_processed/behavior/models
```

輸出範例：
```json
{
  "pid": 84199269992,
  "predictions": {
    "like_rate": 0.023,
    "comment_rate": 0.008,
    "follow_rate": 0.005,
    "collect_rate": 0.011,
    "forward_rate": 0.004,
    "hate_rate": 0.002,
    "effective_view_rate": 0.87,
    "mean_watch_time": 45.3
  }
}
```

**方式 B：全新的 .mp4 影片（尚未在資料集中）**
```bash
.venv/bin/python scripts/predict_behavior.py from-raw \
    --video /path/to/new_video.mp4 \
    --title "影片標題" \
    --asr-text "字幕文字..." \
    --models-dir data_processed/behavior/models
```

這個模式會即時用 ViT-B/16 抽取影片特徵，再做預測。需要安裝 `opencv-python`。

---

## 10. 目前實驗結果與解讀

### 10.1 Pipeline A 結果：推薦系統

所有實驗在 69 users、2,822 items 的 smoke-test 子集上跑，CPU 訓練。

#### LightGCN 1 epoch

| 資料集 | Recall@10 | Recall@20 | NDCG@10 | NDCG@20 |
|---|---|---|---|---|
| 驗證集 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 測試集 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

#### VBPR 1 epoch

| 資料集 | Recall@10 | Recall@20 | NDCG@10 | NDCG@20 |
|---|---|---|---|---|
| 驗證集 | 0.0000 | 0.0036 | 0.0000 | 0.0016 |
| 測試集 | 0.0036 | 0.0036 | 0.0037 | 0.0036 |

#### MMGCN 5 epochs（最佳驗證結果）

| 資料集 | Recall@10 | Recall@20 | NDCG@10 | NDCG@20 |
|---|---|---|---|---|
| 驗證集 | 0.0000 | 0.0048 | 0.0000 | 0.0018 |
| 測試集 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

**為什麼數字這麼低？**

這些低數值完全在預期之內，原因有三：

1. **子集太小**：只有 69 個使用者，每人平均只有 4.4 次測試互動，隨機因素影響很大
2. **稀疏度 98.4%**：幾乎所有 user-item 組合都沒互動，模型很難學到規律
3. **只跑 1-5 個 epoch**：推薦模型通常需要幾十到幾百個 epoch 才能收斂

要看有意義的結果，需要用完整的 153k items 資料在 GPU 上跑幾十個 epoch。

### 10.2 Pipeline B 結果：行為預測

54,088 部影片，測試集 5,409 部。

| 目標 | MAE | RMSE | Spearman 相關 | AUC@中位數 |
|---|---|---|---|---|
| `like_rate` | 0.0441 | 0.0820 | -0.0013 | 0.498 |
| `comment_rate` | 0.0091 | 0.0366 | -0.0029 | 0.496 |
| `follow_rate` | 0.0079 | 0.0362 | -0.0085 | 0.487 |
| `collect_rate` | 0.0125 | 0.0414 | -0.0348 | 0.459 |
| `forward_rate` | 0.0046 | 0.0261 | -0.0115 | 0.480 |
| `hate_rate` | 0.0020 | 0.0136 | -0.0140 | 0.460 |
| `effective_view_rate` | 0.1425 | 0.1769 | **+0.016** | **0.508** |
| `mean_watch_time` | 23.63 秒 | 34.68 秒 | **+0.047** | — |

**指標說明**：
- **MAE（平均絕對誤差）**：預測值和真實值的平均差距，越低越好
- **RMSE（均方根誤差）**：類似 MAE 但對大誤差更敏感，越低越好
- **Spearman 相關**：預測排名和真實排名的相關性，1 = 完美，0 = 沒有關係，-1 = 完全相反。AUC = 0.5 = 跟猜硬幣一樣沒用

**結果解讀**：

Spearman 接近 0、AUC 接近 0.5，代表**目前的特徵對行為預測幾乎沒有幫助**。

`mean_watch_time` 的 Spearman = 0.047 是最高的，說明視覺特徵和「看多久」有一點點關聯，但仍然很弱。

**這是合理的！原因如下**：

1. **特徵本身的問題**：`image_feat.npy` 和 `text_feat.npy` 是為「推薦系統的協同過濾」設計的特徵，不是純粹的內容特徵。它們已經被 embedding 過，可能混入了使用者偏好資訊
2. **行為的高度偏斜**：like_rate 的中位數是 0，大部分影片幾乎沒人按讚，這讓分類邊界很難找到
3. **缺乏屬性特徵**：duration、category、author_fans_count 這些屬性特徵可能比視覺特徵更能預測行為（受歡迎的作者 → 更高互動率）

---

## 11. 遇到的技術問題與解法

### 問題 1：MMRec 的新版套件相容性

**症狀**：執行 `main.py` 出現 `AttributeError: module 'numpy' has no attribute 'float'`

**原因**：NumPy 2.0 移除了 `np.float`（應該用內建的 `float`），但 MMRec 是用舊版 NumPy 寫的

**解法**：在 `MMRec/src/utils/metrics.py` 把所有 `np.float` 替換成 `float`

---

### 問題 2：PyTorch Geometric 的訊息傳遞 API 改變

**症狀**：執行 MMGCN 時出現 `TypeError: message() got an unexpected keyword argument`

**原因**：PyTorch Geometric 2.7 更改了 `MessagePassing.message()` 的參數命名規則

**解法**：在 `MMRec/src/models/mmgcn.py` 更新 `message()` 方法的參數簽章

---

### 問題 3：tiny 資料集缺少 `effective_view` 欄位

**症狀**：`build_behavior_labels.py` 執行時報 `KeyError: missing required column: effective_view`

**原因**：tiny 資料集的 `interaction.csv` 有 `click` 和 `watch_time`，但沒有 `effective_view`

**解法**：根據官方 README 定義（觀看 ≥ 3 秒 = effective_view），自動推導：
```python
df["effective_view"] = (df["watch_time"].astype(float) >= 3)
```

---

### 問題 4：label 欄位名稱和 align script 不符

**症狀**：`align_features_labels.py` 執行時報 `KeyError: ['like_rate', 'comment_rate', ...] not in index`

**原因**：pandas 的 `groupby().agg()` 在混合單一和多重聚合時，會把欄名變成 `cvm_like_mean` 而非 `cvm_like`。
`RATE_RENAME` 字典想把 `cvm_like` 改名成 `like_rate`，但實際欄名已經是 `cvm_like_mean`，所以 rename 沒有效果

**解法**：同時建立帶 `_mean` 後綴的 rename 對應：
```python
rate_rename_flat = {f"{k}_mean": v for k, v in RATE_RENAME.items()}
```

---

### 問題 5：feature 的 pid 和 interaction.csv 的 pid 不對應

**症狀**：`align_features_labels.py` 說 `visual: kept 0 / 98999 pids`（沒有任何對應）

**原因**：
- `video_feature_total/` 的 10 個檔案名為 1.npy～10.npy（小數字 ID）
- `interaction.csv` 的 pid 是 60 億量級的 hash 值
- `video_rec_dataset/image_feat.npy` 的行索引是 0～153560 的整數（重映射後的 itemID）
- 三者用的 ID 系統完全不同

**解法**：反推 itemID → pid 的映射

推薦資料集和 interaction.csv 描述同一批影片，因此兩份資料的時間戳（timestamp）可以當作「鑰匙」：

```
推薦資料集：userID=0 在時間 T 看了 itemID=116244
interaction.csv：user_id=3524 在時間 T 看了 pid=82927753107

→ userID=0 就是 user_id=3524
→ itemID=116244 就是 pid=82927753107
```

對全部 294,355 筆互動做這個比對，得到 86,578/88,098 個 itemID 的對應 pid。
結果存為 `data_raw/video_rec_dataset/pids.txt`（第 i 行 = itemID=i 對應的真實 pid）。

---

## 12. 下一步計畫

### 短期（可以立刻做）

1. **加入屬性特徵**
   把 `interaction.csv` 裡的 duration、category_id、author_fans_count 等屬性加進 X 矩陣，
   這些特徵通常對行為預測很有幫助（受歡迎的作者 → 更高互動）

2. **用 asr_en / title_en 的文字特徵**
   目前使用的是推薦 pipeline 用的 text_feat.npy，是降維過的。
   直接用影片的英文標題和 ASR 字幕，用 TF-IDF 或 Sentence-BERT 重新抽取文字特徵，
   可能比現有的 50 維文字特徵更有代表性

3. **增大訓練子集**
   目前推薦 pipeline 只用了 69 個使用者，試試 `--target-interactions 30000`，
   看 Recall / NDCG 能不能提高

### 中期（需要 GPU 環境）

4. **在 RunPod 跑完整資料集**
   租一台 GPU 機器（例如 A100），用完整 153,561 items 訓練，
   預計需要 1-2 小時（相較於本機 CPU 要幾天）

5. **多 seed 比較**
   目前只跑了 seed=999，結果有隨機性。
   用 seed=1/2/3/999 各跑一次，取平均值，結果才有統計意義

6. **加入 MMGCN 的 attribute 支援**
   把影片的 category、author 等屬性也加入 MMGCN 的特徵，變成真正的多模態 + 多屬性模型

### 長期

7. **設計 ablation study**
   分別移除視覺特徵、文字特徵、屬性特徵，各跑一次，量化每種資訊對推薦效果的貢獻

8. **探索 cold-start 場景**
   測試：對從未出現在訓練資料裡的新影片，模型的推薦準確度如何？
   這是多模態推薦系統最重要的應用場景

---

## 附錄：常見問題

### Q：為什麼不直接下載 3.2 TB 的原始影片？
A：我們的目標是驗證 pipeline 是否可行，不需要重新抽取特徵。
官方已提供預處理好的 `image_feat.npy` 和 `text_feat.npy`，直接用就好。
3.2 TB 的原始影片只有在你需要「自己設計特徵抽取方式」時才需要下載。

### Q：Recall@10 = 0.0048 很差嗎？
A：在 **69 個使用者、2,822 部影片** 的極小子集上，這個數字意義不大。
正式的推薦系統評估需要在幾萬到幾百萬量級的資料上跑，才有統計意義。
這次的實驗目的只是驗證「整條 pipeline 可以跑通、能輸出數字」。

### Q：LightGBM 的行為預測 Spearman 接近 0，是模型的問題嗎？
A：不一定。更可能的原因是「特徵不夠好」。
目前的視覺/文字特徵是推薦系統用的 embedding，不一定能直接用來預測行為。
下一步應該加入屬性特徵（duration、category 等），這些通常是更強的信號。

### Q：predict_behavior.py 的 from-raw 模式需要什麼條件？
A：需要安裝 `opencv-python`，而且本機需要能存取 ViT-B/16 的 torchvision 預訓練權重。
第一次執行會自動下載模型權重（約 346 MB）。
注意：mode-B 抽取的是 4 幀 × 768 = 3072 維特徵，訓練時用的是 768 維，**維度不符**。
如果要用 mode-B，需要重新用 mode-B 的特徵抽法來訓練。

---

*最後更新：2026-05-19*
*論文來源：[A Large-scale Dataset with Behavior, Attributes, and Content of Mobile Short-video Platform](https://arxiv.org/pdf/2502.05922) (WWW 2025)*
*推薦框架：[enoche/MMRec](https://github.com/enoche/MMRec)*
*資料集：[tsinghua-fib-lab/ShortVideo_dataset](https://github.com/tsinghua-fib-lab/ShortVideo_dataset)*
