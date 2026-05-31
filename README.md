# ShortVideo MMGCN PoC — 完整教學說明

> **寫給誰看的？** 這份文件試著讓任何人都能看懂這個專案在做什麼——包括對 AI 只有基本認識的讀者。
> 如果你是第一次接觸推薦系統或機器學習，從頭讀起就對了。

---

## 目錄

1. [這個專案在解決什麼問題？](#1-這個專案在解決什麼問題)
2. [三個平行的研究方向](#2-三個平行的研究方向)
3. [背後的理論：推薦系統怎麼運作？](#3-背後的理論推薦系統怎麼運作)
4. [背後的理論：圖神經網路（GNN）是什麼？](#4-背後的理論圖神經網路gnn是什麼)
5. [資料集完整介紹](#5-資料集完整介紹)
6. [專案目錄結構](#6-專案目錄結構)
7. [環境設定](#7-環境設定)
8. [Pipeline A：推薦系統（MMGCN）](#8-pipeline-a推薦系統mmgcn)
9. [Pipeline B：行為預測（LightGBM）](#9-pipeline-b行為預測lightgbm)
10. [Pipeline C：異質圖 GNN 影片表現預測](#10-pipeline-c異質圖-gnn-影片表現預測)
11. [目前實驗結果與解讀](#11-目前實驗結果與解讀)
12. [遇到的技術問題與解法](#12-遇到的技術問題與解法)
13. [下一步計畫](#13-下一步計畫)
14. [完整資料集下載指南](#14-完整資料集下載指南)
15. [複現紀錄（技術細節）](#15-複現紀錄技術細節)

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

## 2. 三個平行的研究方向

這個專案同時探索三個不同的問題，可以獨立理解：

```
這個專案
├── Pipeline A：推薦系統
│   問題：「哪部影片應該推給哪個使用者？」
│   方法：MMGCN（多模態圖神經網路）
│   輸入：使用者歷史行為 + 影片視覺/文字特徵
│   輸出：對每個使用者，預測他最可能互動的影片排名
│
├── Pipeline B：行為預測
│   問題：「這部影片在平台上會引發多少互動？」
│   方法：LightGBM（梯度提升樹）
│   輸入：影片的視覺特徵 + 文字特徵
│   輸出：8 個行為指標（按讚率、留言率、觀看秒數...）
│
└── Pipeline C：影片表現預測（異質圖 GNN）
    問題：「全新的影片，在平台上表現會如何？（冷啟動）」
    方法：HeteroGNN（異質圖神經網路，以影片為中心）
    輸入：影片內容 + 作者背景 + 同分類/相似影片的圖結構
    輸出：8 個行為指標（與 B 相同，但能處理零互動的新影片）
```

**三者的核心差別**：

| | Pipeline A | Pipeline B | Pipeline C |
|---|---|---|---|
| **預測對象** | 特定使用者的偏好 | 影片的平台均值 | 影片的平台均值 |
| **有無 user 節點** | ✓ 有 | ✗ 無 | ✗ 無 |
| **能否冷啟動** | ✗ 不行 | △ 勉強 | ✓ 主要目標 |
| **能否利用圖結構** | ✓ 二部圖 | ✗ 不能 | ✓ 異質圖 |
| **模型類型** | 協同過濾式 GNN | 梯度提升樹 | 節點回歸 GNN |

Pipeline C 刻意不放 user 節點：放了就回到協同過濾框架，且新影片無法冷啟動。
GNN 優於 LightGBM 的地方在於：LightGBM 只能看影片自己的特徵向量；GNN 透過訊息傳遞
把「同作者其他影片、同分類熱門影片、內容相似影片」的信號用**圖結構**傳進來。

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

---

> **在讀這節之前，先明確一件事**：
> 這個資料集叫「tiny version」，但這只是指「原始影片只附了 10 支示範 .mp4」。
> **行為日誌（6.7M 行）和視覺特徵（153K 部影片）都是完整的**，
> 對 Pipeline B/C 來說已完全足夠。

---

### 5.1 所有檔案逐一說明

#### A. `data_raw/shortvideo_tiny/interaction.csv`（**最核心的檔案**）

**本質**：平台 7 天內的真實使用者行為日誌。每一行 = 一個使用者看了一支影片的一次曝光事件。

```
大小：1.42 GB
行數：6,767,010 行（含 header 共 6,767,011）
欄位：28 個
時間範圍：2022-09-16 ～ 2022-09-22（7 天）
唯一影片：153,561 部
唯一使用者：10,000 人
```

**28 個欄位的完整說明**（附真實第一行數值）：

| 欄位 | 型態 | 第一行範例 | 說明 |
|---|---|---|---|
| `user_id` | int | 7313 | 使用者 ID（已雜湊匿名） |
| `pid` | int | 84199269992 | 影片 ID（已雜湊，非流水號） |
| `author_id` | int | 193159093 | 作者 ID |
| `category_id` | int | 36 | 影片的三級分類 leaf ID |
| `category_level` | int | 1 | 此次出現的分類是哪層（1=根） |
| `parent_id` | int | 36 | 父分類 ID |
| `root_id` | int | 36 | 根分類 ID |
| `exposed_time` | int | 1663333100 | 曝光的 Unix 時間戳 |
| `author_fans_count` | int | 46761 | 作者粉絲數（範圍：0 ～ 2.16億） |
| `watch_time` | float | 136 | 此使用者觀看了幾秒（min=0, median=14, max=951） |
| `duration` | float | 91.9 | 影片總長（秒）（min=2.7, median=81.7, max=1734.9） |
| `cvm_like` | bool | False | 有沒有按讚（全體按讚率：2.57%） |
| `click` | bool | True | 有沒有點擊進去看 |
| `comment` | bool | False | 有沒有留言（全體留言率：0.38%） |
| `follow` | bool | False | 有沒有追蹤作者 |
| `collect` | bool | False | 有沒有收藏 |
| `forward` | bool | False | 有沒有轉發 |
| `hate` | bool | False | 有沒有點「不喜歡」 |
| `tag_name` | str | 正能量 | 影片標籤（同一影片可有多個，每行一個） |
| `title` | str | 现如今为啥 | 影片標題（中文） |
| `p_hour` | int | 21 | 曝光小時（0–23） |
| `p_date` | int | 20220916 | 曝光日期（YYYYMMDD） |
| `gender` | str | M | 使用者性別（M/F） |
| `age` | int | 42 | 使用者年齡 |
| `mod_price` | int | 2699 | 使用者手機價格（元） |
| `fre_city` | str | 邯郸 | 使用者所在城市 |
| `fre_community_type` | str | unknown | 居住類型 |
| `fre_city_level` | str | 三线城市 | 城市等級 |

> **注意**：`tag_name` 是每行一個標籤，所以同一部影片（pid）可能有多行，每行帶不同的 tag。
> 這是 interaction.csv 有 6.7M 行但只有 153K 個唯一 pid 的主要原因之一。

#### B. `data_raw/video_rec_dataset/image_feat.npy`（Pipeline C 視覺輸入）

**本質**：別人（資料集作者）已經替所有 153,561 部影片跑完 ViT-B/16 視覺特徵抽取，把結果打包成矩陣。

```
shape:   (153561, 768)
dtype:   float32
大小：   471.7 MB
每列：   1 支影片的 768 維視覺特徵向量
數值範圍：約 -4.6 ～ +4.2（標準化後的浮點數）
```

**怎麼讀懂這個矩陣**：

```
image_feat[0] = [ -0.7176, -0.1146, -0.2300, ... ]  ← pid=59821271563 的視覺特徵
image_feat[1] = [ -0.0481, -0.7161, -0.2116, ... ]  ← pid=80200361806 的視覺特徵
image_feat[2] = [  0.0000,  0.0000,  0.0000, ... ]  ← -1（無效）
image_feat[3] = [  0.0000,  0.0000,  0.0000, ... ]  ← -1（無效）
...
（第 i 列對應哪支影片，要查 pids.txt 第 i 行）
```

**特徵的意義**：
768 個數字不是人類可以直接解讀的，而是 ViT-B/16 模型的中間層輸出。
這 768 個數字共同代表了影片的「視覺語意」——兩支視覺風格相似的影片，
它們的 768 維向量在空間中的方向會很接近（cosine 相似度高）。

#### C. `data_raw/video_rec_dataset/text_feat.npy`（Pipeline C 文字輸入）

```
shape:   (153561, 50)
dtype:   float64
大小：   61.4 MB
```

**本質**：預先計算的 50 維文字特徵，與 image_feat.npy 的行一一對應（第 i 行 = 同一支影片的文字特徵）。確切的抽取方式不明，但推測是基於影片標題的詞向量（如 GloVe）平均後降維。

⚠️ **注意**：50 維是非常低的維度，文字語意保留有限。若要改善，
用 `--use-sbert` 選項重建圖可以提升至 384 維的 SBERT 表示。

#### D. `data_raw/video_rec_dataset/pids.txt`（矩陣行索引↔影片 ID 的對應表）

**本質**：這是連結「矩陣行號」和「真實影片 pid」的唯一橋樑。

```
格式：純文字，每行一個整數（pid 或 -1）
行數：153,561 行（= image_feat.npy 的行數）
```

```
第 0 行：59821271563   → image_feat[0] 是 pid=59821271563 的特徵
第 1 行：80200361806   → image_feat[1] 是 pid=80200361806 的特徵
第 2 行：-1            → image_feat[2] 無效（沒有對應影片）
第 3 行：-1            → image_feat[3] 無效
第 4 行：84664784964   → image_feat[4] 是 pid=84664784964 的特徵
...
```

**重要的資料質量問題**：

```
總行數（= 矩陣行數）：153,561
  其中 -1（無效）：      66,983（43.6%）
  其中有效 entries：     86,578
    但有 21,548 個 pid 重複出現於多行
    → 唯一有效 pid 數：  65,030（42.3% 的矩陣行有真實影片對應）
```

這是工作集只有 54,088 部影片的根本原因之一。

#### E. `data_raw/video_rec_dataset/video.inter`（Pipeline A 推薦系統用）

```
大小：7.7 MB
格式：TSV（Tab 分隔），header = userID \t itemID \t rating \t timestamp \t x_label
資料行：294,355 行
```

```
前3行：
0    116244    5    1663597856    0
0    37411     5    1663794733    0
0    6003      5    1663794733    0
```

⚠️ **特別注意**：這裡的 `userID` 和 `itemID` 是 **重新映射過的 0-based 整數**，
和 interaction.csv 的 `user_id`（如 7313）和 `pid`（如 84199269992）**完全不同**！
這個檔案只給 Pipeline A（MMGCN 推薦系統）使用，Pipeline B/C 不用它。

#### F. `data_raw/shortvideo_tiny/video_feature_total/`（⚠️ 舊格式，Pipeline C 不用）

```
檔案數：10 個（只有示範影片 1.mp4 ～ 10.mp4 的特徵）
每個檔案 shape：(8, 256) = 8 幀 × 256 維
```

**為什麼 Pipeline C 不用這個**：
- 這是舊版特徵，256 維，與 `image_feat.npy` 的 768 維體系**不相容**，不能混用
- 只有 10 支示範影片，遠遠不夠
- `image_feat.npy` 才是 153,561 部影片的完整現代特徵

#### G. `data_raw/shortvideo_tiny/title_en/ asr_en/ asr_zn/`（各 10 個 .txt 文字檔）

這三個資料夾各有 10 個 .txt 檔，分別是示範影片的英文標題、英文 ASR 字幕、中文 ASR 字幕。

```
title_en/10.txt  → "Exploring Secret Whitening Techniques for 5 Years"
asr_en/10.txt    → "I've tried so many products on the road to whitening..."
asr_zn/10.txt    → "有没有人跟我一样啊，就是我在美白祛斑这条路上真的走过太多弯路了..."
```

**關係到 Pipeline C**：目前 Pipeline C 訓練時用的是 `text_feat.npy` 的 50 維預計算文字特徵，
不是這些 .txt 檔。若要用 SBERT 重抽，是對 `interaction.csv` 裡的 `title` 欄位（中文）操作。

#### H. `data_raw/shortvideo_tiny/categories_cn_en.csv`（分類對照表）

```
shape：826 行 × 6 欄
欄位：category_level, category_id, category_name_cn, parent_id, root_id, category_name_en
level=1 的根分類：36 個（如「美妝」「美食」「知識」）
level=2 的次分類：307 個
level=3 的葉分類：483 個
```

Pipeline C 工作集中只有 **51 個** 唯一 `category_id`（不是 826），因為只有 54,088 部影片的分類被涵蓋。

---

### 5.2 資料處理完整鏈條：從原始資料到 54,088 部工作集

以下說明每個中間產物如何從原始資料生成：

```
┌─────────────────────────────────────────────────────────────────┐
│ 原始輸入                                                         │
│   interaction.csv（1.42 GB，6,767,010 行，153,561 個唯一 pid）   │
└──────────────────────────┬──────────────────────────────────────┘
                           │ scripts/build_behavior_labels.py
                           │ 對每個 pid 做 groupby，計算平均比例
                           │ 篩掉曝光 < 10 次的（標籤不穩定）
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ 中間產物 A                                                       │
│   video_behavior_labels.parquet（98,999 部影片 × 12 欄）         │
│   欄位：pid + 8個行為率 + 觀看時間統計 + n_impressions           │
│   說明：153,561 個 pid 中有 98,999 個的曝光次數 ≥ 10             │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           │ 需要對齊視覺特徵
                           │ (image_feat.npy 行號 ↔ pid)
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ 特徵矩陣                                                         │
│   image_feat.npy（153,561 × 768）                               │
│   text_feat.npy（153,561 × 50）                                 │
│   pids.txt（第 i 行 = 第 i 列對應的 pid）                       │
│                                                                  │
│   問題：pids.txt 有 66,983 個 -1 和 21,548 個重複                │
│   → 唯一有效 pid：65,030 個                                     │
└──────────────────────────┬──────────────────────────────────────┘
                           │ scripts/align_features_labels.py
                           │ 取交集：有標籤 AND 有有效特徵
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ 工作集（Pipeline B 和 C 都用這個）                               │
│   meta.parquet    54,088 × 10（pid + 8標籤 + n_impressions）    │
│   X.npy           54,088 × 818（768視覺 + 50文字）              │
│   Y.npy           54,088 × 8（8個目標標籤）                     │
│   weights.npy     54,088（曝光次數，訓練加權用）                 │
└──────────────────────────┬──────────────────────────────────────┘
                           │
         ┌─────────────────┴─────────────────┐
         │ Pipeline B                         │ Pipeline C
         │ 直接把 X 和 Y 送進 LightGBM        │ scripts/build_hetero_graph.py
         ▼                                    ▼
   8 個 LightGBM 模型               hetero_graph.pt（199 MB）
   models/metrics.json              54,088 影片節點 + 邊
                                    + 從 interaction.csv 補充
                                      作者/分類/duration 資訊
```

**工作集只有 54,088 部的原因視覺化**：

```
有行為標籤的影片：                有視覺特徵的影片：
╔═══════════════════╗            ╔══════════════╗
║  98,999 部         ║           ║  65,030 部    ║
║  (≥10次曝光)       ║           ║  (pids.txt    ║
║                   ╠═══════════╣  有效對應)    ║
║  有標籤但沒特徵：  ║  交集     ║              ║
║  44,911 部         ║  54,088   ║  有特徵但無   ║
║                   ║  Pipeline ║  充分標籤：   ║
║                   ║  C 工作集  ║  10,942 部    ║
╚═══════════════════╝           ╚══════════════╝

有標籤但沒特徵的 44,911 部影片：
→ 若下載主站 video_feature_total/ 並重建對應，工作集可擴大至 ~98,999 部
```

---

### 5.3 Pipeline C 的完整資料流（從特徵到圖到預測）

```
[video.x 每支影片的 819 維輸入]

  視覺特徵（768 維）：來自 image_feat.npy，ViT-B/16 預計算
  文字特徵（50 維）： 來自 text_feat.npy，預計算（低品質）
  時長（1 維）：      來自 interaction.csv 的 duration 欄，log1p 轉換
  ────────────────────────────────────────────────────────
  合計：768 + 50 + 1 = 819 維

[author.x 每位作者的 10 維輸入]

  log1p(粉絲數)（1 維）：         interaction.csv 的 author_fans_count
  訓練集影片數（1 維）：           只統計訓練集（43,270 部）的影片數
  8 個目標的訓練集均值（8 維）：   只統計訓練集的標籤均值（嚴防洩漏）
  ────────────────────────────────────────────────────────
  合計：1 + 1 + 8 = 10 維

[category 節點]

  無數值特徵，用可學習嵌入向量（訓練時自動更新），128 維
  分類體系：三級（root → parent → leaf），共 51 個唯一分類

[邊的建立]

  posted_by：interaction.csv 的 author_id 欄 → 影片到作者
  belongs_to：interaction.csv 的 category_id 欄 → 影片到分類
  similar_to：image_feat.npy 的餘弦 kNN（k=10）→ 影片到相似影片
              注意：只用視覺特徵計算相似度，不用任何標籤
```

### 5.4 為什麼現有資料足夠，以及真正的限制

**足夠的理由**：
- 訓練集 43,270 部影片，充分驗證了「圖結構是否有效」這個核心問題
- `mean_watch_time` Spearman 從 0.047（LightGBM）提升到 **0.256（HeteroGNN）**，RTX 4090 上驗證
- 端到端 pipeline（建圖→訓練→預測→from-raw 冷啟動）全部可運行

**真正的限制**：
- 工作集只覆蓋 54.6%（54K / 98K）的有行為標籤影片
- 文字特徵品質低（50 維，來源不明的降維特徵）
- 7 天資料時間跨度短，可能無法反映長期趨勢
- from-raw 模式的文字輸入是全零（無法重現 50 維的預計算特徵）

### 5.5 interaction.csv 欄位的數值統計

| 數值欄位 | 最小值 | 中位數 | 最大值 | 說明 |
|---|---|---|---|---|
| `watch_time` | 0 秒 | 14 秒 | 951 秒 | 中位數 14 秒遠低於時長中位數 82 秒 |
| `duration` | 2.7 秒 | 81.7 秒 | 1734.9 秒 | 短影音但有長達 29 分鐘的 |
| `author_fans_count` | 0 | 124,216 | 216,473,883 | 差距極大，需 log1p 壓縮 |
| `age` | — | — | — | 使用者年齡 |

| 布林欄位 | 全體比例 | 說明 |
|---|---|---|
| `cvm_like`（按讚） | **2.57%** | 每 39 次曝光才有 1 次按讚 |
| `comment`（留言） | **0.38%** | 每 263 次曝光才有 1 次留言 |

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
│       ├── models/                        ← LightGBM 訓練好的模型
│       │   ├── like_rate.txt
│       │   ├── ... （共 8 個）
│       │   └── metrics.json               ← Pipeline B 評估結果
│       ├── hetero_graph.pt                ← [C] 異質圖（PyG HeteroData）
│       └── models_gnn/                    ← [C] GNN 訓練產物
│           ├── best_model.pt              ← 最佳驗證 checkpoint
│           └── metrics_gnn.json           ← Pipeline C 評估結果
│
├── scripts/                       ← 我們撰寫的所有 Python 腳本
│   ├── build_behavior_labels.py   ← Step B1：從 interaction.csv 聚合標籤
│   ├── align_features_labels.py   ← Step B2：對齊特徵和標籤
│   ├── train_behavior_model.py    ← Step B3：訓練 LightGBM
│   ├── predict_behavior.py        ← Step B4：對新影片做預測
│   ├── build_hetero_graph.py      ← Step C1：建異質圖 (HeteroData .pt)
│   ├── hetero_gnn.py              ← Step C2：HeteroGNN 模型定義
│   ├── train_hetero.py            ← Step C3：訓練 + 評估 + 對比 LightGBM
│   ├── predict_hetero.py          ← Step C4：推論（by-pid / from-raw 冷啟動）
│   ├── prepare_video_yaml.py      ← 生成 MMRec smoke-test 子集
│   ├── check_env.py               ← 環境檢查
│   └── inspect_shortvideo_data.py ← 資料檢視
│
├── notes/
│   └── pipelineC_data_check.md   ← Pipeline C 資料檢查記錄（真實欄位/shape）
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

# Pipeline C 額外需要的套件（sentence-transformers / faiss-cpu / opencv）
.venv/bin/pip install sentence-transformers faiss-cpu opencv-python
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

## 10. Pipeline C：異質圖 GNN 影片表現預測

> **目標**：輸入一支影片的封面或影片本身（＋可得屬性），輸出 8 個平台表現指標。
> 主要應用場景：**冷啟動**——全新影片只有封面/影片、零互動紀錄。

### 10.1 任務定義：為什麼是「節點回歸」而不是「推薦」？

Pipeline A 在預測「這個使用者和這部影片的匹配度」，本質上是一個排名問題。
Pipeline C 在預測「這部影片在整個平台的表現」，與特定使用者完全無關——這叫 **節點回歸（Node Regression）**。

為什麼不放 user 節點？
- 放了就回到協同過濾框架（每筆預測都要有使用者），新影片無法冷啟動
- 我們想要的是：給一支全新的影片，不需要任何互動記錄，就能估計它的爆紅潛力

8 個預測目標（與 Pipeline B 一致）：
```
like_rate          按讚率
comment_rate       留言率
follow_rate        追蹤率
collect_rate       收藏率
forward_rate       轉發率
hate_rate          討厭率
effective_view_rate  有效觀看率
mean_watch_time    平均觀看秒數
```

### 10.2 建圖說明：以影片為中心的異質圖

**四種節點**：

| 節點類型 | 數量（工作集） | 節點特徵 |
|---|---|---|
| `video` | 54,088 | 視覺 768 維 + 文字 50 維 + log(duration) 1 維 = **819 維** |
| `author` | 34,939 | log(粉絲數) + 訓練集影片數 + 訓練集 8 指標均值 = **10 維** |
| `category` | 51 | 無數值特徵，用可學習 Embedding（128 維）表示 |

**四種邊（全部加雙向）**：

| 邊類型 | 定義 | 數量 |
|---|---|---|
| `(video, posted_by, author)` | 這支影片是這個作者發的 | 54,088 |
| `(video, belongs_to, category)` | 這支影片屬於這個分類 | 54,088 |
| `(video, similar_to, video)` | cosine 相似度最近的 10 支影片 | ≈ 540,896 |
| 以上三種的反向邊 | 由 `T.ToUndirected()` 自動生成 | 同上 |

**為什麼這樣建圖能解冷啟動？**

當一支全新影片進來（零互動），GNN 還是能透過圖結構學到信號：

```
新影片
├── 透過 similar_to 邊 → 找到內容相似的老影片 → 借它們的表現當參考
├── 透過 posted_by 邊  → 找到作者的歷史影片  → 借作者聲譽當參考
└── 透過 belongs_to 邊 → 找到同分類的熱門影片 → 借分類的基準當參考
```

`similar_to` 邊只用影片的**內容 embedding**（768 維視覺特徵）做 cosine kNN（k=10），
完全不涉及任何標籤或互動資料，確保不洩漏未來資訊。

**嚴防資料洩漏（leakage）**：

author 節點的「歷史均值」特徵只能用 **train split** 的標籤計算。
程式碼裡有明確的 assert：

```python
def build_author_features(meta_df, vid_attrs, train_mask):
    train_pids = set(meta_df.loc[train_mask, "pid"])
    # 確認沒有 val/test pid 混進來
    assert used_pids.issubset(train_pids), "Author stats use non-train pids!"
```

建圖結束時會印出 leakage 自檢結果：
```
[LEAKAGE CHECK] Author stats pid coverage in train: 100.0% (must be 100%)
[LEAKAGE CHECK] PASSED
```

### 10.3 模型架構：兩層異質訊息傳遞 + MLP 輸出頭

```
輸入層
├── video.x (819維)   → Linear(-1, 128) → 128維
├── author.x (10維)   → Linear(-1, 128) → 128維
└── category          → Embedding(51, 128) → 128維

         ↓ 第 1 層 HeteroConv（所有邊類型同步傳遞）↓
         ↓ ReLU + Dropout ↓
         ↓ 第 2 層 HeteroConv ↓
         ↓ ReLU + Dropout ↓

只取 video 節點 (128維)
         ↓
    MLP 輸出頭
    Linear(128→64) → ReLU → Linear(64→8)
         ↓
輸出欄 0-6：sigmoid   (7 個 rate 指標，值域 [0,1])
輸出欄  7 ：linear    (log-space mean_watch_time，預測後用 expm1 還原成秒)
```

**兩種可切換的訊息傳遞層**：

| 變體 | 方法 | 說明 |
|---|---|---|
| `--model sage` | GraphSAGE (`SAGEConv`) | 採樣鄰居後平均，inductive 設定，能處理新節點（推薦用於冷啟動） |
| `--model hgt` | Heterogeneous Graph Transformer (`HGTConv`) | 用 attention 機制加權不同類型的鄰居，GAT 的異質圖推廣（做 ablation 用） |

### 10.4 訊息傳遞白話解釋

以一支全新影片 v 為例，說明 2 層 GNN 怎麼把圖結構的信號傳進來：

**Layer 1（第 1 輪）：收直接鄰居的信號**

```
v 的表示 = ReLU(
    同作者的影片的特徵 × W_posted_by
  + 同分類的影片的特徵 × W_belongs_to
  + 內容相似的影片的特徵 × W_similar_to
)
```

用數學式表示（更新式）：

```
h_v^(1) = σ( Σ_r  W_r · Agg_{u ∈ N_r(v)} h_u^(0) )
```

- `r`：邊類型（posted_by / belongs_to / similar_to 三種關係）
- `N_r(v)`：在關係 r 下，v 的鄰居集合
- `Agg`：聚合函數（GraphSAGE 用 mean，HGT 用 attention-weighted sum）
- `W_r`：每種關係的可學習權重矩陣
- `σ`：ReLU 激活函數

白話翻譯：「把我的每種鄰居（作者、分類、相似影片）的特徵各自加權平均，再非線性變換一下，就是我的新表示。」

**Layer 2（第 2 輪）：收 2 跳鄰居的信號**

第 2 層的輸入是 Layer 1 輸出的 `h^(1)`，現在每個節點的向量已經融合了它的直接鄰居資訊。
所以 Layer 2 聚合的，是「鄰居的鄰居」的資訊（2 跳鄰域）。

**Readout（輸出）**：

```
MLP( h_v^(2) ) → 8 個數值
```

只有 video 節點參與輸出，author / category 節點的向量在訊息傳遞後就不再使用。

### 10.5 訓練細節

**loss 設計（有曝光次數加權）**：

曝光次數越多，該影片的標籤越可靠，訓練時給它更高的權重：

```
Loss = Σ_{v ∈ train} w_v × (
    Σ_{i=0}^{6} MSE(ŷ_i, y_i)     ← 7 個 rate 指標，MSE
  + Huber(ŷ_7, y_7, δ=1.0)        ← watch_time，Huber 對 outlier 更穩健
)
```

其中 `w_v = n_impressions_v / Σ n_impressions`（歸一化曝光次數）。

**評估指標**（test set，每個 target 分開計算）：

| 指標 | 說明 |
|---|---|
| MAE | 平均絕對誤差 |
| RMSE | 均方根誤差 |
| Spearman | 排名相關性（最重要，反映推薦場景的排序能力） |
| nDCG@10 | 以真實 rate 為 relevance、用預測值排序計算 |
| AUC@median | 二元分類（高於/低於中位數）的 AUC |

**早停**：監控 val loss，patience=50。

### 10.6 執行指令

**完整流程（從零開始）**：

```bash
cd /Users/luoyi/Desktop/shortvideo_mmgcn_poc

# Step C1：建異質圖（約 5-10 分鐘，主要時間在 CSV 讀取和 kNN）
.venv/bin/python scripts/build_hetero_graph.py
# 進階：使用 SBERT 文字特徵（384 維，首次會下載模型）
.venv/bin/python scripts/build_hetero_graph.py --use-sbert

# Step C2：訓練 HeteroGNN（默認 SAGE，~2-3 分鐘，早停於 66 epoch）
.venv/bin/python scripts/train_hetero.py --model sage --epochs 300 --seed 42
# 訓練 HGT（Attention 變體，做 ablation 比較）
.venv/bin/python scripts/train_hetero.py --model hgt  --epochs 300 --seed 42

# Step C3：對已有影片預測（by-pid 模式）
.venv/bin/python scripts/predict_hetero.py by-pid --pid 84199269992

# Step C3：對新影片冷啟動預測（from-raw 模式）
.venv/bin/python scripts/predict_hetero.py from-raw \
    --cover /path/to/cover.jpg --title "影片標題" --duration 90
# 或用影片檔
.venv/bin/python scripts/predict_hetero.py from-raw \
    --video /path/to/clip.mp4 --title "影片標題"
```

**macOS 注意事項**：faiss 在 macOS 上有 OpenMP 衝突問題，加上環境變數即可：
```bash
KMP_DUPLICATE_LIB_OK=TRUE .venv/bin/python scripts/build_hetero_graph.py
```

### 10.7 預期結果與解讀

```
──────────────────────────────────────────────────────────────────────
  HeteroGNN (SAGE) — test-set metrics (seed=42, 早停於 epoch 66)

Target                      MAE     RMSE   Spearman   nDCG@10   AUC@med
──────────────────────────────────────────────────────────────────────
  like_rate              0.0658   0.0921    -0.034     0.007       nan
  comment_rate           0.0452   0.0585     0.016     0.071       nan
  follow_rate            0.0399   0.0506    -0.040     0.000       nan
  collect_rate           0.0704   0.0810    -0.003     0.000       nan
  forward_rate           0.0809   0.0905    -0.011     0.003       nan
  hate_rate              0.0312   0.0372     0.018     0.000       nan
  effective_view_rate    0.1524   0.1933    -0.010     0.597     0.492
  mean_watch_time        505 s    22734 s    0.251     0.208     0.634
──────────────────────────────────────────────────────────────────────
```

**誠實的解讀**：

1. **`mean_watch_time` 是最有希望的指標**：Spearman = 0.251，遠高於 LightGBM 的 0.047。
   觀看時長最受影片內容驅動（吸引人的影片讓你多看幾秒），圖結構把相似內容的信號傳了進來。

2. **rate 類指標 Spearman ≈ 0**：這與 Pipeline B 的結論一致，也與理論預期相符。
   rate 的中位數幾乎都是 0（極度稀疏），絕大多數影片的按讚率趨近 0，
   少數爆紅影片則遠高於平均。從純內容預測這個「隨機爆紅效應」本就非常困難。

3. **AUC@median 為 nan**：rate 指標中位數 = 0，二元分類（高於/低於 0）中幾乎全是「低」，
   使 AUC 無法定義。這是正常的統計現象，不是程式錯誤。

4. **mean_watch_time 的 MAE 很大（505 秒）**：模型的 **排名能力** 不錯（Spearman 高），
   但 **絕對數值** 的校準還不足。推薦場景通常更在乎排名而非絕對值，所以 Spearman 是更重要的指標。

### 10.8 對照組比較

| 方法 | 種類 | `mean_watch_time` Spearman | 備注 |
|---|---|---|---|
| **Pipeline C HeteroGNN (SAGE)** | 異質圖 GNN | **0.251** | 本專案，圖結構信號 |
| Pipeline B LightGBM | 梯度提升樹（非圖） | 0.047 | 只看影片自身特徵 |
| MMVED / HMMVED | 內容人氣預測（Xie et al.） | 論文報告 | 有公開 code，建議做 baseline |

主要評估指標：**Spearman**（排名相關）與 **nDCG@10**（Top-10 質量）。
MAE/RMSE 受分佈偏斜影響大，僅作參考。

### 10.9 參考文獻

**方法基礎**：
- Gilmer et al., *Neural Message Passing for Quantum Chemistry*, ICML 2017（訊息傳遞框架）
- Kipf & Welling, *Semi-Supervised Classification with GCN*, ICLR 2017
- Hamilton et al., *GraphSAGE: Inductive Representation Learning on Graphs*, NeurIPS 2017（inductive，對應新影片冷啟動）
- Veličković et al., *Graph Attention Networks (GAT)*, ICLR 2018（attention 機制基礎）
- Hu et al., *Heterogeneous Graph Transformer (HGT)*, WWW 2020（異質 attention，本專案 HGT 變體所用）

**應用對標**：
- GraphInf（GCN 做短影片人氣預測，快手資料，勝過 SOTA）— 自建異質圖的正當性
- GraphTR（CIKM 2020，video–tag–user–media 異質網路解稀疏）— 異質圖原型
- Xie et al., *MMVED*, WWW 2020 ＋ HMMVED（IEEE TMM）— 內容人氣預測經典 baseline，有公開 code，建議對比
- Shang et al., *ShortVideo Dataset*, WWW 2025（本專案使用的資料集）

---

## 11. 目前實驗結果與解讀

### 11.1 Pipeline A 結果：推薦系統

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

### 11.2 Pipeline B 結果：行為預測

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

### 11.3 Pipeline C 結果：HeteroGNN

| 環境 | mean_watch_time Spearman | 訓練時間（66-68 epochs） |
|---|---|---|
| macOS CPU | 0.2505 | ~47 秒 |
| **Ubuntu RTX 4090** | **0.2559** | **~1 秒** |

重點：
- `mean_watch_time` Spearman = **0.256**（GNN）vs 0.047（LightGBM）：**GNN 明顯優勝**，圖結構把相似影片的觀看時長信號傳了進來
- RTX 4090 訓練速度比 CPU **快約 47 倍**，讓後續大模型實驗可行
- 6 個 rate 指標 Spearman ≈ 0：兩個模型都接近隨機，主因是 rate 分佈極度稀疏（中位數 = 0）
- 建議後續：加入 SBERT 文字特徵（`--use-sbert`）、加深層數（`--layers 3 --d 256`）、與 MMVED 對比

---

## 12. 遇到的技術問題與解法

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

## 13. 下一步計畫

### 短期（可以立刻做）

1. **Pipeline C：切換 SBERT 文字特徵**
   目前使用的是舊版 50 維文字特徵；改用 SBERT（384 維）預計對 `mean_watch_time` 有提升：
   ```bash
   .venv/bin/python scripts/build_hetero_graph.py --use-sbert
   .venv/bin/python scripts/train_hetero.py --model sage
   ```

2. **Pipeline C：HGT ablation**
   用 `--model hgt` 跑一次，對比 SAGE vs HGT 的 Spearman 差異，量化 attention 機制的貢獻

3. **Pipeline C：與 MMVED/HMMVED 比較**
   Xie et al. 有公開 code，在同一 test split 上跑，填入第 10.8 節的對照表

4. **Pipeline B：加入屬性特徵**
   把 `interaction.csv` 裡的 duration、category_id、author_fans_count 等屬性加進 LightGBM 的 X 矩陣

5. **Pipeline A：增大訓練子集**
   目前推薦 pipeline 只用了 69 個使用者，試試 `--target-interactions 30000`

### 中期（需要 GPU 環境）

6. **Pipeline C：在 GPU 上跑更多 epoch（更大 d）**
   `--d 256 --layers 3`，CPU 上跑很慢，在 A100/H100 上跑完整的超參搜尋

7. **Pipeline A：在 RunPod 跑完整資料集**
   租一台 GPU 機器（例如 A100），用完整 153,561 items 訓練

8. **多 seed 比較**
   C pipeline 目前只跑了 seed=42，用 seed=0/1/2/42 各跑一次，取平均值

### 長期

9. **Pipeline C：加入 tag 節點**
   `(video, has_tag, tag)` 邊目前是選用的，加入後可以引入 tag 語義信號

10. **設計跨 pipeline ablation study**
    定量比較「有無圖結構」（B vs C）和「有無 user 節點」（A vs C）對 `mean_watch_time` 預測的影響

11. **探索 Pipeline C 的真實冷啟動場景**
    在完全沒有出現在訓練資料的新影片上測試，評估模型的 inductive 泛化能力

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

### Q：LightGBM 預測出來的 rate 會超過 1 嗎？
A：可能。LightGBM 回歸沒有 sigmoid，極端情況下可能預測到負值或超過 1。推論時直接 `clip(pred, 0, 1)` 即可，不需要改成 binary classification（那樣需要把 rate 標籤轉成個別觀測，資料量會爆炸）。

### Q：Pipeline A 的 MMGCN embedding 可以拿來當 Pipeline C 的特徵嗎？
A：技術上可以，把 MMGCN 訓練後的 `item_embedding[v]` concat 進 Pipeline C 的 video.x。但這樣會讓 Pipeline C 綁回「該影片必須出現在 MMGCN 訓練圖裡」的限制，新影片就用不上，違背冷啟動設計初衷。建議 Pipeline C 保持純內容特徵。

### Q：Spearman 相關係數是什麼？
A：Spearman 相關係數衡量「排名的一致性」。若預測排第 1 名的影片真的是觀看最長的，Spearman = 1；若完全相反，Spearman = -1；若和猜硬幣一樣，Spearman ≈ 0。推薦場景最在意排名（哪支影片放第一），所以 Spearman 比 MAE 更重要。

---

## 14. 完整資料集下載指南

### 14.1 資料集結構總覽

本專案的資料來自**三個不同的下載源**，功能各不相同：

```
資料來源 1：Tiny 行為日誌（Dropbox）
  → interaction.csv（6.7M 行，153K 影片，7 天）
  → 10 支示範影片的特徵和文字

資料來源 2：推薦用預處理特徵（Dropbox）
  → image_feat.npy（153561 × 768 維視覺特徵，所有影片）
  → text_feat.npy（153561 × 50 維文字特徵，所有影片）

資料來源 3：清華主站（完整版，需帳密）
  → video_feature_total/（每支影片個別 .npy，可重抽特徵）
  → asr_en/ title_en/（153K 支影片的文字，SBERT 需要）
  → raw_file/（3.2 TB 原始 .mp4，非必要）
```

**Pipeline C 現在使用資料來源 1 + 2，已能完整運行。**

### 14.2 目前工作集的實際規模

「tiny」這個名字容易誤導。我們實際使用的資料量：

| 項目 | 數量 |
|---|---|
| 互動行為記錄 | **6,767,010 行** |
| 獨立使用者數 | **10,000 人** |
| 獨立影片數 | **153,561 部** |
| 有效行為標籤影片 | **98,999 部**（≥ 10 次曝光） |
| Pipeline C 工作集 | **54,088 部**（有標籤 ∩ 有預抽特徵） |

工作集只有 54,088 部的原因：另外 44,911 部影片雖然有行為標籤，但不在推薦資料集的 `image_feat.npy` 覆蓋範圍內（推薦資料集的 pid 對應關係只涵蓋其中一個子集）。要納入這些影片，需要從清華主站下載 `video_feature_total/`。

### 14.3 重新下載推薦預處理特徵（最常用）

```bash
# 在 SSH 環境（Ubuntu，不需要 sudo）
cd ~/shortvideo_mmgcn_poc

# 方法 A：用 wget（通常已預裝）
wget -O /tmp/rec_dataset.zip \
  "https://www.dropbox.com/scl/fo/ha0e0wolgqgg5qskr52l1/AHZUySejwWJzfyJy8WGo2k4?rlkey=xwpqosx7b906yb7g8nwy53oqh&st=slry1d1l&dl=1"
unzip /tmp/rec_dataset.zip -d data_raw/video_rec_dataset/
# 應得到 image_feat.npy (153561,768)、text_feat.npy (153561,50)、video.inter

# 方法 B：用 Python（不依賴任何系統工具）
.venv/bin/python - <<'EOF'
import urllib.request, pathlib
url  = "https://www.dropbox.com/scl/fo/ha0e0wolgqgg5qskr52l1/AHZUySejwWJzfyJy8WGo2k4?rlkey=xwpqosx7b906yb7g8nwy53oqh&st=slry1d1l&dl=1"
dest = pathlib.Path("/tmp/rec_dataset.zip")
print("Downloading...")
urllib.request.urlretrieve(url, dest)
print(f"Saved to {dest}")
EOF
```

### 14.4 從清華主站下載完整文字檔（SBERT 用，無連線數限制）

```bash
BASE="http://fi.ee.tsinghua.edu.cn/datasets/short-video-dataset"
AUTH="--user=videodata --password='ShortVideo@10000'"

# 下載互動主檔（.csv，無連線數限制）
wget $AUTH -c --show-progress \
     -O data_raw/shortvideo_full/interaciton_filtered.csv \
     "$BASE/interaciton_filtered.csv"

# 下載分類對照（小檔）
wget $AUTH -O data_raw/shortvideo_full/category_cn_en.csv \
     "$BASE/category_cn_en.csv"
```

**注意**：`.npy` 和 `.mp4` 大檔案限制 3 個同時連線，用 `-c` 斷點續傳：
```bash
wget $AUTH -c --limit-rate=10m -o /tmp/wget_img.log \
     -O data_raw/shortvideo_full/image_feat.npy \
     "$BASE/image_feat.npy" &
```

### 14.5 不需要下載即可改善特徵（推薦先試）

`interaction.csv` 裡已有 `title`（中文標題）欄位，可直接用 SBERT 重抽 384 維文字特徵：

```bash
# 重建圖（用 SBERT 替換 50 維舊文字特徵）
.venv/bin/python scripts/build_hetero_graph.py --use-sbert
# 重新訓練
.venv/bin/python scripts/train_hetero.py --model sage --epochs 300 --seed 42 \
    --out-dir data_processed/behavior/models_gnn_sbert
```

這不需要下載任何新資料，但首次執行會自動下載 SBERT 模型（約 400 MB）。

---

## 15. 複現紀錄（技術細節）

### 15.1 環境差異

| 環境 | Python | PyTorch | CUDA | GPU |
|---|---|---|---|---|
| macOS（開發） | 3.10.20 | 2.12.0 | 無 | — |
| Ubuntu 24.04（訓練） | 3.12 | 2.6.0+cu124 | 12.4 相容 | RTX 4090 24 GB |

### 15.2 MMRec 相容性修補清單

MMRec 原始碼針對較舊版本的 NumPy / SciPy / PyG，需要以下修補才能在現有環境下運行：

| 檔案 | 問題 | 修法 |
|---|---|---|
| `utils/metrics.py` | `np.float` 在 NumPy 2.0 移除 | 改為內建 `float` |
| `models/lightgcn.py` | SciPy 新版不支援 DOK sparse `_update` | 改用 COO 格式建鄰接矩陣 |
| `models/mmgcn.py` | PyG 2.7 的 `message()` 參數簽章改變 | 更新參數名稱 |
| `main.py` | `--epochs` CLI 參數未覆蓋 config | 加入 CLI override 邏輯 |
| `*.yaml` 設定檔 | 原始設定跑完整超參數網格 | 改為單一組合 smoke test |

### 15.3 處理資料集的 pid 對應關係

推薦資料集（`video.inter`）使用 0～153560 的整數 itemID，行為日誌（`interaction.csv`）使用雜湊過的真實 pid（如 `84199269992`）。對應方式：

```
推薦資料集的 userID=0 在時間 T 看了 itemID=116244
interaction.csv 中 user_id=3524 在時間 T 看了 pid=82927753107
→ userID=0 = user_id=3524，itemID=116244 = pid=82927753107
```

對全部 294,355 筆互動配對，成功對應 86,578 個 itemID，結果存於 `data_raw/video_rec_dataset/pids.txt`（第 i 行 = itemID i 的真實 pid，`-1` 表示無法對應）。

---

*最後更新：2026-05-31（新增 Pipeline C GPU 結果、完整資料下載指南、複現紀錄）*
*論文來源：[A Large-scale Dataset with Behavior, Attributes, and Content of Mobile Short-video Platform](https://arxiv.org/pdf/2502.05922) (WWW 2025)*
*推薦框架：[enoche/MMRec](https://github.com/enoche/MMRec)*
*資料集：[tsinghua-fib-lab/ShortVideo_dataset](https://github.com/tsinghua-fib-lab/ShortVideo_dataset)*
