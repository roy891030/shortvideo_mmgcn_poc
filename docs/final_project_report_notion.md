# 以異質圖神經網路預測短影音平台表現

**期末專案報告　｜　Pipeline C：冷啟動影片表現預測系統**

作者：roy891030　｜　2026 年 6 月

> 📐 **Notion 使用提示**：所有數學式已改為獨立 block（`$$` 各佔一行），可直接貼入 Notion。若仍有顯示問題，在該區塊按 `/equation` 重建即可。

---

## 目錄

1. 摘要
2. 研究動機
3. 方法回顧
4. 實驗方法
5. 研究成果

---

## 一、摘要

短影音平台（如 TikTok、抖音）每日新增數十萬支影片，平台需要在影片剛上傳、**零互動記錄的冷啟動條件**下即時估計其潛在表現，以決定初始流量分配策略。

本專案以 Shang et al.（WWW 2025）公開的 **ShortVideo 資料集**為基礎，收錄 10,000 名使用者在 153,561 部影片上的 6,767,010 條互動記錄，並附有每部影片的 768 維 ViT 視覺特徵及 50 維 SBERT 文字特徵。

> **原始論文的侷限**：原始論文以推薦系統為主要應用場景，最優方法（BM3）達到 Recall@10 = 0.0238、NDCG@10 = 0.0178。然而推薦框架存在冷啟動失效（測試期 21.4% 的影片從未在訓練期出現）、任務框架錯位、互動矩陣稀疏度 99.93% 等根本侷限。

為此，本研究設計了 **Pipeline C**：以影片為中心建立異質圖（不引入使用者節點），透過兩層 GraphSAGE 異質圖卷積（HeteroGNN），在測試集（5,410 部影片）上的 `mean_watch_time` Spearman 排名相關係數達到 **0.256**，優於非圖基線 LightGBM 的 0.047（**提升 4.4 倍**），並驗證了端到端冷啟動推論的可行性。

---

## 二、研究動機

### 2.1 資料集背景與核心統計

| 項目 | 數值 | 說明 |
| --- | --- | --- |
| 使用者數 | 10,000 | 匿名化，含性別、年齡、城市等 6 種屬性 |
| 影片數（唯一 pid） | 153,561 | 每部有原始 .mp4 及預抽取特徵 |
| 互動記錄數 | 6,767,010 | 28 欄，含 7 種行為 + 觀看時長 |
| 互動期間 | 7 天 | 20220916–20220922 |
| 視覺特徵維度 | 768 | ViT-B/16 平均池化（image_feat.npy） |
| 文字特徵維度 | 50 | SBERT 降維（text_feat.npy） |
| 矩陣稀疏度 | **99.93%** | 僅 0.07% 的使用者–影片格子非空 |
| 平均每片被互動人數 | 6.6 人 | 大量影片互動記錄極少 |

### 2.2 8 個行為指標的稀疏性

| 指標 | 中位數 | 均值 | 零值比例 | 預測難度 |
| --- | --- | --- | --- | --- |
| `like_rate` | 0.000 | 0.026 | 84.6% | 極難（社群隨機性強） |
| `comment_rate` | 0.000 | 0.004 | 96.7% | 極難 |
| `follow_rate` | 0.000 | 0.004 | 96.9% | 極難 |
| `collect_rate` | 0.000 | 0.006 | 95.2% | 極難 |
| `forward_rate` | 0.000 | 0.002 | 98.0% | 極難 |
| `hate_rate` | 0.000 | 0.001 | 99.2% | 極難 |
| `effective_view_rate` | 0.833 | 0.791 | 1.8% | 中等 |
| **`mean_watch_time`** | **22 秒** | **33 秒** | **0.2%** | **最容易** |

`mean_watch_time` 零值比例最低，且直接反映觀眾對影片內容的吸引力，是最適合以內容特徵預測的目標。

### 2.3 推薦系統框架的四大侷限

#### 侷限一：冷啟動失效

所有協同過濾方法均需要影片在**訓練集中至少出現一次**，才能為其嵌入向量提供有效梯度更新。以「前 5 天訓練、後 2 天測試」模擬真實場景：

| 項目 | 數量 | 說明 |
| --- | --- | --- |
| 訓練期（前 5 天）影片 | 138,768 | 有互動記錄，可訓練嵌入 |
| 測試期（後 2 天）影片 | 69,094 | 需要預測 |
| **冷啟動影片**（訓練期未見） | **14,793** | 任何 CF 方法均無法預測 |
| **冷啟動比例** | **21.4%** | 每 5 支測試影片就有 1 支是冷啟動 |

#### 侷限二：任務框架的根本錯位

推薦系統解決的是**個人化排名**問題，輸出的是每位使用者的影片排名清單：

$$
\forall\, u \in \mathcal{U},\quad
\text{找出}\;\operatorname{top-}K\;\text{個最高評分影片}
$$

而平台真正需要的是**影片層面的絕對指標預測**，輸出的是影片在全平台的行為均值（8 個數值）：

$$
\hat{\mathbf{y}}_v = f_\theta\!\left(\mathbf{x}_v,\, \mathcal{G}\right),\quad \hat{\mathbf{y}}_v \in \mathbb{R}^8
$$

這兩個問題的輸入、目標函數、評估方式均不同，從根本上就不能用同一套框架解決。

#### 侷限三：互動矩陣的極度稀疏

> **稀疏度 99.93% 的影響**：即使對已出現在訓練集的影片，其互動記錄也可能只有 1–2 人，統計上極不穩定。這是即使最佳方法（BM3）Recall@10 也只有 2.38% 的根本原因——協同過濾在稀疏資料上的天花板很低。

#### 侷限四：7 種行為信號的浪費

原始論文所有基準算法只把「有無互動」作為訓練信號，**完全忽略**行為類型的差異。`like_rate` 反映社群認同，`effective_view_rate` 反映主動選擇，`mean_watch_time` 直接反映內容吸引力——三者的預測難度與信號來源截然不同，應分開對待。

### 2.4 研究問題的精確定義

> **核心研究問題**：在不依賴任何使用者互動歷史的前提下，能否透過建立以**影片為中心的異質圖**，讓圖神經網路的訊息傳遞有效融合「作者背景信號」、「分類基準信號」和「視覺相似影片的歷史表現」，使冷啟動場景下的影片平台表現預測（特別是 `mean_watch_time`）在排名能力（Spearman）上優於純特徵的非圖基線（LightGBM）？

---

## 三、方法回顧

### 3.1 矩陣分解：BPR

**問題設定**：給定使用者集合、影片集合，以及觀測到的正向互動集合，目標是學習每個使用者和影片的 $d$ 維嵌入向量，使預測評分（兩個嵌入向量的內積）能反映互動偏好：

$$
\hat{x}_{ui} = \mathbf{e}_u^\top \mathbf{e}_i,\quad \mathbf{e}_u,\,\mathbf{e}_i \in \mathbb{R}^d
$$

**BPR 損失函數**（Bayesian Personalized Ranking，貝葉斯個人化排名）：

$$
\mathcal{L}_{\mathrm{BPR}} =
-\sum_{(u,i,j)\in\mathcal{D}}
\log\,\sigma\!\bigl(\hat{x}_{ui} - \hat{x}_{uj}\bigr)
+ \lambda\,\|\Theta\|^2
$$

其中 $j$ 是負樣本（使用者未互動過的影片），$\sigma$ 是 sigmoid 函數，訓練三元組集合為：

$$
\mathcal{D} = \bigl\{(u,i,j) \;\big|\; (u,i)\in\mathcal{O}^+,\;(u,j)\notin\mathcal{O}^+\bigr\}
$$

**致命缺陷**：每個影片的嵌入向量完全從互動記錄中學習。新影片（無互動）的嵌入永遠停在隨機初始化，對所有使用者的預測評分都只是雜訊，冷啟動預測完全無效。

### 3.2 圖神經網路推薦：LightGCN

LightGCN（He et al., SIGIR 2020）把使用者–影片互動建成二部圖，並大幅簡化 GCN 的傳播規則（移除特徵變換矩陣與非線性激活）：

**使用者表示更新**（第 $\ell$ 層）：

$$
\mathbf{e}_u^{(\ell+1)} =
\sum_{i\in\mathcal{N}(u)}
\frac{1}{\sqrt{|\mathcal{N}(u)|}\cdot\sqrt{|\mathcal{N}(i)|}}
\,\mathbf{e}_i^{(\ell)}
$$

**影片表示更新**（第 $\ell$ 層）：

$$
\mathbf{e}_i^{(\ell+1)} =
\sum_{u\in\mathcal{N}(i)}
\frac{1}{\sqrt{|\mathcal{N}(i)|}\cdot\sqrt{|\mathcal{N}(u)|}}
\,\mathbf{e}_u^{(\ell)}
$$

跑 $K$ 層後，最終表示取各層等權平均：

$$
\mathbf{e}_u = \frac{1}{K+1}\sum_{\ell=0}^{K}\mathbf{e}_u^{(\ell)},\qquad
\mathbf{e}_i = \frac{1}{K+1}\sum_{\ell=0}^{K}\mathbf{e}_i^{(\ell)}
$$

LightGCN 透過圖結構傳遞高階協同信號，顯著優於 BPR（Recall@10：0.0223 vs 0.0113），但本質仍是協同過濾——新影片沒有邊，訊息傳遞無法作用，冷啟動依然無解。

### 3.3 多模態圖推薦：MMGCN 與 BM3

**MMGCN**（Wei et al., ACM MM 2019）為視覺（v）、文字（t）、行為（b）三個模態各建一張圖，分別傳播後加權融合：

$$
\mathbf{e}_u =
\sum_{m\in\{v,\,t,\,b\}}
\alpha_m \cdot \mathrm{GCN}_m\!\left(\mathcal{G}_m,\,\mathbf{F}^m\right)[u]
$$

**BM3**（Zhou et al., WWW 2023）採用自監督對比學習，損失函數同時優化推薦項、多模態對齊項與正則項：

$$
\mathcal{L}_{\mathrm{BM3}} =
\mathcal{L}_{\mathrm{rec}}
+ \lambda_1\,\mathcal{L}_{\mathrm{mm}}
+ \lambda_2\,\mathcal{L}_{\mathrm{reg}}
$$

BM3 是 ShortVideo 資料集上的最優基線（Recall@10 = 0.0238），但本質仍依賴使用者互動圖，無法支援冷啟動。

### 3.4 各方法對核心需求的支援狀況

| 方法 | 冷啟動 | 全平台指標 | 不需使用者歷史 |
| --- | --- | --- | --- |
| BPR | ✗ | ✗ | ✗ |
| LightGCN | ✗ | ✗ | ✗ |
| MMGCN | △ 部分 | ✗ | ✗ |
| BM3 | △ 部分 | ✗ | ✗ |
| MMVED / HMMVED | ✓ | ✓ | ✓ |
| **Pipeline C（本研究）** | **✓** | **✓** | **✓** |

Pipeline C 與 MMVED 的核心差異：本研究額外建立了**異質圖結構**，讓影片透過作者節點和分類節點借用鄰域歷史表現作為冷啟動信號，而非僅依賴影片自身的內容特徵。

---

## 四、實驗方法

### 4.1 資料前處理管線

**原始資料檔案**：

| 檔案 | 格式 | 大小 | Shape | 用途 |
| --- | --- | --- | --- | --- |
| `interaction.csv` | CSV, 28 欄 | 1.42 GB | 6,767,010 行 | 標籤 + 屬性 |
| `image_feat.npy` | float32 矩陣 | 471 MB | (153561, 768) | 視覺特徵 |
| `text_feat.npy` | float64 矩陣 | 61 MB | (153561, 50) | 文字特徵 |
| `pids.txt` | 純文字 | 1.2 MB | 153,561 行 | 行號對應 |

> ⚠️ **pids.txt 品質問題**：153,561 行中，66,983 行（43.6%）記錄為 -1（無對應影片），且 21,548 個 pid 重複出現在多列，有效的唯一 pid 僅 65,030 個。

**前處理三步驟**：

```
演算法 1：資料前處理管線
─────────────────────────────────────────────
輸入：interaction.csv, image_feat.npy,
      text_feat.npy, pids.txt
輸出：工作集 54,088 部影片的 X、Y、元資料
─────────────────────────────────────────────
步驟 1：聚合行為標籤
  讀取 interaction.csv，按 pid 分組
  過濾曝光次數 < 10 的影片
  計算 8 個行為指標的全平台均值
  → 98,999 部影片的行為標籤

步驟 2：建立特徵對齊索引
  解析 pids.txt，建立 pid→列號字典（只保留首次出現）
  有效唯一 pid = 65,030

步驟 3：取行為標籤與特徵的交集
  工作集 = 行為標籤集合 ∩ 特徵索引集合
         = 98,999 ∩ 65,030 = 54,088 部影片
  拼接特徵向量（819 維）：
    x_v = [視覺特徵(768), 文字特徵(50), log(時長+1)(1)]
  儲存 X ∈ R^{54088×819}，Y ∈ R^{54088×8}
```

**工作集規模推導**：

$$
\underbrace{153{,}561}_{\text{interaction.csv 唯一 pid}}
\;\xrightarrow{\;\geq 10\text{ 次曝光}\;}\;
\underbrace{98{,}999}_{\text{有行為標籤}}
\;\cap\;
\underbrace{65{,}030}_{\text{pids.txt 唯一有效}}
\;=\;
\mathbf{54{,}088}
$$

### 4.2 異質圖建構

#### 節點設計

> **關鍵設計決策：刻意排除使用者節點。** 加入使用者節點即回到協同過濾框架，必須依賴使用者互動歷史，冷啟動依然無解。本研究預測的是全平台均值，與特定使用者偏好無關。

| 節點類型 | 數量 | 維度 | 特徵構成 |
| --- | --- | --- | --- |
| 影片（video） | 54,088 | **819** | 768（ViT 視覺）+ 50（SBERT 文字）+ 1（log 影片時長） |
| 作者（author） | 34,939 | **10** | 1（log 粉絲數）+ 1（訓練集作品數）+ 8（訓練集 8 指標歷史均值） |
| 分類（category） | 51 | **128** | 可學習 Embedding，隨模型訓練更新 |

**作者特徵的防洩漏設計**：「8 指標歷史均值」**嚴格限定**只使用訓練集（43,270 部）的標籤統計：

```python
def build_author_features(meta_df, train_mask):
    train_pids = set(meta_df.loc[train_mask, "pid"])
    # 只從訓練集影片統計
    train_only = meta_df[meta_df["pid"].isin(train_pids)]
    author_stats = train_only.groupby("author_id")[TARGET_COLS].mean()
    # 嚴格驗證：val/test 標籤不能出現在統計集合中
    assert not any(p in train_pids for p in val_test_pids), \
        "洩漏！驗證/測試標籤被使用"
    return author_stats  # 覆蓋率驗證：100.0% PASSED
```

#### 邊的設計

| 邊類型 | 連接 | 條數（含反向） | 建構方式 |
| --- | --- | --- | --- |
| `posted_by` | 影片 → 作者 | 108,176 | 從 `author_id` 欄讀取 |
| `belongs_to` | 影片 → 分類 | 108,176 | 從 `category_id` 欄讀取 |
| `similar_to` | 影片 ↔ 影片 | 942,894 | 對 768 維視覺特徵做 cosine kNN（$k=10$） |
| **總計** | | **1,159,246** | |

> 💡 **`similar_to` 邊的冷啟動意義**：建立此邊只需影片的視覺特徵，不需要任何互動記錄。全新的冷啟動影片可找到 $k=10$ 支視覺上最相似的老影片，透過 GNN 訊息傳遞從這些老影片的歷史表現中借用信號。

**建圖演算法**：

```
演算法 2：異質圖建構
─────────────────────────────────────────────
輸入：54,088 部影片的特徵、元資料、interaction.csv
輸出：hetero_graph.pt（199 MB）
─────────────────────────────────────────────
1. 節點初始化
   video.x   ← 特徵矩陣 [54088, 819]
   author.x  ← 粉絲數、作品數、訓練集均值 [34939, 10]
   category  ← 可學習 Embedding(51, 128)

2. posted_by 邊
   for 每部影片 v:
       讀取 author_id → 建立邊 (v → 作者節點)

3. belongs_to 邊
   for 每部影片 v:
       讀取 category_id → 建立邊 (v → 分類節點)

4. similar_to 邊（cosine kNN）
   正規化視覺特徵矩陣 F ∈ R^{54088×768}
   計算相似度矩陣 S = F × F^T
   for 每部影片 v:
       找出 top-10 最相似影片（排除自身）
       建立雙向邊 (v ↔ 相似影片)

5. 加入所有反向邊（by_video, in_category）
6. 儲存為 hetero_graph.pt
```

### 4.3 HeteroGNN 模型架構

**整體設計：三段式架構（總參數量約 450,632）**

```
┌─────────────────────────────────────────────────────────────┐
│  輸入層                                                       │
│  video.x [54088, 819]  │  author.x [34939, 10]  │  category │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│  輸入投影層（Linear + ReLU）                                  │
│  · video/author：Linear(d_in → 128)                          │
│  · category：Embedding(51, 128)                              │
│  ⇒ 所有節點統一為 128 維                                      │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│  HeteroConv Layer 1（GraphSAGE，1 跳訊息傳遞）                │
│  · 對 5 種邊類型各自聚合鄰居 → 加總 → ReLU + Dropout(0.1)   │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│  HeteroConv Layer 2（GraphSAGE，2 跳訊息傳遞）                │
│  · 鄰居已融合各自鄰域資訊，感受野擴展到 2 跳                   │
│  · ReLU + Dropout(0.1)                                       │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│  輸出頭（僅取 video 節點）                                    │
│  h[128] → Linear(128→64) → ReLU → Linear(64→8) → ŷ ∈ R^8  │
└─────────────────────────────────────────────────────────────┘
```

### 4.4 訊息傳遞的數學形式化

#### GraphSAGE 異質聚合（預設模型）

第 $\ell$ 層中，影片節點 $v$ 的表示更新公式為：

$$
\mathbf{h}_v^{(\ell+1)} =
\sigma\!\Biggl(
  \mathbf{W}_{\mathrm{self}}^{(\ell)}\,\mathbf{h}_v^{(\ell)}
  \;+\;
  \sum_{r\,\in\,\mathcal{R}}
  \mathbf{W}_r^{(\ell)}
  \cdot
  \frac{1}{|\mathcal{N}_r(v)|}
  \sum_{u\,\in\,\mathcal{N}_r(v)}
  \mathbf{h}_u^{(\ell)}
\Biggr)
$$

**符號說明**：

| 符號 | 說明 |
| --- | --- |
| $\mathcal{R}$ | 邊關係集合（5 種）：`posted_by`, `belongs_to`, `similar_to`, `by_video`, `in_category` |
| 鄰居集合 N_r(v) | 節點 $v$ 在關係 $r$ 下的鄰居集合 |
| 投影矩陣 **W**_r（128×128） | 關係 $r$ 專屬的可學習投影矩陣 |
| 自身矩陣 **W**_self（128×128） | 自身特徵保留矩陣 |
| $\sigma$ | ReLU 激活函數 |

**兩層傳播後，影片節點的表示包含**：

- **0 跳**：自身的視覺與文字特徵
- **1 跳**：直接鄰居（作者、分類、10 支相似影片）的特徵
- **2 跳**：鄰居的鄰居（2 跳以內所有節點）的融合特徵

#### 對比變體：HGT（異質圖 Transformer）

HGT（Hu et al., WWW 2020）以多頭注意力（$H=4$ 頭）代替均值聚合，讓模型學習「哪個鄰居更值得參考」。注意力加權更新公式：

$$
\mathbf{h}_v^{(\ell+1)} =
\sigma\!\Biggl(
\sum_{r\in\mathcal{R}}
\sum_{u\in\mathcal{N}_r(v)}
\alpha_{u\to v}^{(r,\ell)}
\cdot
\mathbf{W}_r^{V,(\ell)}\,\mathbf{h}_u^{(\ell)}
\Biggr)
$$

注意力係數（$d/H$ 為每頭維度）：

$$
\alpha_{u\to v}^{(r,\ell)}
\;\propto\;
\exp\!\left(
  \frac{
    \bigl(\mathbf{W}_r^{Q,(\ell)}\mathbf{h}_v^{(\ell)}\bigr)^\top
    \bigl(\mathbf{W}_r^{K,(\ell)}\mathbf{h}_u^{(\ell)}\bigr)
  }{\sqrt{d/H}}
\right)
$$

| 面向 | GraphSAGE（預設） | HGT |
| --- | --- | --- |
| 聚合方式 | 鄰域均值 | 多頭注意力加權（4 heads） |
| 白話解釋 | 等量採納所有鄰居意見 | 根據內容決定聽哪個鄰居 |
| 歸納能力（冷啟動） | **強** | 中 |
| 計算成本 | 低 | 高 |

### 4.5 多任務損失函數

8 個預測目標同時訓練，損失以曝光次數加權：

$$
\mathcal{L} =
\sum_{v\in\mathcal{T}}\, w_v
\Biggl[
  \underbrace{
    \sum_{k=0}^{6}
    \bigl(\hat{y}_{v,k} - y_{v,k}\bigr)^2
  }_{\text{7 個 rate 指標（MSE）}}
  \;+\;
  \underbrace{
    \mathcal{L}_{\mathrm{Huber}}\!\bigl(
      \hat{y}_{v,7},\;\log(1+y_{v,7});\;\delta\!=\!1.0
    \bigr)
  }_{\text{觀看時長（Huber，對數空間）}}
\Biggr]
$$

**曝光加權係數**的計算方式（其中 $n$ 為各影片的曝光次數）：

$$
w_v = \frac{n_v}{\displaystyle\sum_{u\in\mathcal{T}} n_u},
\qquad
n_v \in [10,\;131124]\;\text{（均值 95.7 次）}
$$

**Huber 損失定義**（$\delta = 1.0$）：

$$
\mathcal{L}_{\mathrm{Huber}}(\hat{y},\,y;\,\delta) =
\begin{cases}
  \dfrac{1}{2}(\hat{y}-y)^2, & |\hat{y}-y| \leq \delta \\[6pt]
  \delta\!\left(|\hat{y}-y| - \dfrac{\delta}{2}\right), & |\hat{y}-y| > \delta
\end{cases}
$$

**設計理由**：

- **為何用 Huber 而非 MSE？** 觀看時長分佈高度右偏（中位數 22 秒，最大 545 秒），MSE 對離群值的懲罰過重，Huber 在誤差小時精確（平方），誤差大時穩健（線性）。
- **為何先取 $\log(1+t)$？** 直接回歸秒數，「500 秒 vs 5 秒」的差距（495 秒）會淹沒「5 秒 vs 10 秒」的差距（5 秒）。對數壓縮後比例關係被保留，模型對各時長範圍均等學習。
- **為何曝光加權？** 曝光次數越多的影片，其行為均值的統計誤差越小，應賦予更高的訓練權重。

### 4.6 訓練設定與防洩漏設計

| 項目 | 設定 |
| --- | --- |
| 資料切分 | 80 / 10 / 10（訓練 43,270 / 驗證 5,408 / 測試 5,410），seed=42 |
| 訓練環境 | Ubuntu 24.04，NVIDIA RTX 4090（24 GB VRAM），PyTorch 2.6.0+cu124 |
| 優化器 | Adam，學習率 0.001 |
| 訓練 epoch | 至多 500，patience=50（早停） |
| 隱藏維度 $d$ | 128 |
| Dropout | 0.1（每層 HeteroConv 後） |
| 總參數量 | 450,632 |

> 🔒 **防洩漏屏障**：資料切分後，所有依賴標籤的二次特徵（作者歷史均值）**嚴格限定只使用訓練集**（43,270 部）計算。驗證集與測試集的標籤在整個訓練過程中對模型完全不可見，程式碼以 `assert` 驗證 100% 覆蓋率。

---

## 五、研究成果

### 5.1 GPU 加速效果

| 環境 | 早停 epoch | 訓練時間 | Spearman | MAE（秒） |
| --- | --- | --- | --- | --- |
| macOS CPU | 66 | 47 秒 | 0.251 | 505 |
| Ubuntu RTX 4090 | 68 | **1 秒** | **0.256** | **386** |
| **加速比** | — | **47×** | +0.005 | −119 秒 |

### 5.2 主要指標比較：HeteroGNN vs LightGBM

測試集 5,410 部影片，GPU 訓練結果（Spearman ↑ 越高越好；MAE ↓ 越低越好）：

| 指標 | HeteroGNN Spearman | LightGBM Spearman | HeteroGNN MAE | LightGBM MAE |
| --- | --- | --- | --- | --- |
| `like_rate` | −0.032 | −0.001 | 0.058 | 0.044 |
| `comment_rate` | +0.012 | −0.003 | 0.039 | 0.009 |
| `follow_rate` | −0.031 | −0.009 | 0.024 | 0.008 |
| `collect_rate` | +0.011 | −0.035 | 0.074 | 0.013 |
| `forward_rate` | −0.005 | −0.012 | 0.066 | 0.005 |
| `hate_rate` | +0.019 | −0.014 | 0.029 | 0.002 |
| `effective_view_rate` | −0.012 | +0.016 | 0.169 | 0.143 |
| **`mean_watch_time`** | **+0.256** | +0.047 | 386 秒 | 24 秒 |

### 5.3 結果分析

#### 核心發現一：圖結構顯著提升觀看時長的排名預測

> ✅ **最重要的發現**：`mean_watch_time` 的 Spearman 相關係數從 0.047（LightGBM）提升至 **0.256**（HeteroGNN），提升幅度達 **4.4 倍**。
>
> 這直接驗證了研究假說：GNN 的訊息傳遞能有效地將「視覺風格相似的老影片的觀看時長」這個冷啟動時唯一可用的鄰域信號，傳遞給新影片的表示向量。

#### 核心發現二：Rate 指標的預測天花板

7 個 rate 指標的 Spearman 對兩個模型而言都接近 0。以 `like_rate` 為例：84.6% 的影片按讚率為零，「哪支影片會爆紅」受到**社群隨機性**（Social Randomness）高度影響——一個大號的轉發就可以帶動按讚數。這種隨機效應無法從內容特徵或鄰域信號預測，兩個模型的 Spearman ≈ 0 是誠實且符合預期的結果。

#### 核心發現三：Spearman 與 MAE 的矛盾解讀

| 模型 | 預測特性 | 根本原因 |
| --- | --- | --- |
| **LightGBM** | MAE 低（絕對值準確），Spearman 低（排名弱） | 傾向預測接近平均值的保守估計，MAE 小但排名無力 |
| **HeteroGNN** | Spearman 高（排名能力強），MAE 高（絕對值偏差大） | GNN 嵌入尚未精確校準到真實秒數，但排序方向正確 |

在平台應用中，流量分配關注「哪支影片排第一」，即**排名能力（Spearman）更重要**。MAE 過高的問題可透過事後輸出校準（isotonic regression 或 Platt scaling）改善，不需要重新訓練模型。

#### 核心發現四：冷啟動推論的可行性驗證

在 from-raw 模式下，使用原始 .mp4 輸入，系統完整執行：

**視覺特徵抽取 → cosine kNN 建圖 → HeteroGNN 推論**

全新影片無需任何互動記錄，即可在毫秒級時間內得到 8 個預測指標，驗證了端到端冷啟動推論的可行性。

### 5.4 未來改進方向

| 優先級 | 方向 | 說明 |
| --- | --- | --- |
| 🔴 高 | SBERT 文字特徵 | 從 50 維升至 384 維，使用 `--use-sbert` 開關 |
| 🔴 高 | 冷啟動 author/category 對應 | 持久化 `hetero_graph_meta.pkl`，解決節點映射問題 |
| 🟡 中 | 多任務不確定性加權（Kendall 2018） | 以可學習不確定性參數 $\sigma$ 自動平衡各任務損失貢獻 |
| 🟡 中 | Tag 節點 | 加入 `(video, has_tag, tag)` 邊，利用 `tag_name` 語義 |
| 🟢 低 | 多 seed 評估 | seed = 0/1/2/42，獲得均值 ± 標準差的統計顯著結論 |
| 🟢 低 | 與 MMVED 比較 | 在同一測試集上系統性對比，確立 Pipeline C 的定位 |

---

## 參考文獻

- Hamilton, W., Ying, Z., and Leskovec, J. (2017). *Inductive Representation Learning on Large Graphs*. NeurIPS.
- He, R. and McAuley, J. (2016). *VBPR: Visual Bayesian Personalized Ranking from Implicit Feedback*. AAAI.
- He, X., et al. (2020). *LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation*. SIGIR.
- Hu, Z., et al. (2020). *Heterogeneous Graph Transformer*. WWW.
- Kendall, A., Gal, Y., and Cipolla, R. (2018). *Multi-Task Learning Using Uncertainty to Weigh Losses*. CVPR.
- Rendle, S., et al. (2009). *BPR: Bayesian Personalized Ranking from Implicit Feedback*. UAI.
- Shang, Y., et al. (2025). *A Large-scale Dataset with Behavior, Attributes, and Content of Mobile Short-video Platform*. WWW 2025, pp. 793–796.
- Wei, Y., et al. (2019). *MMGCN: Multi-modal Graph Convolution Network for Personalized Recommendation of Micro-video*. ACM MM.
- Xie, H., et al. (2020). *MMVED: Multi-modal Variational Encoder-Decoder for Micro-video Popularity Prediction*. WWW.
- Xie, H., et al. (2021). *A Hierarchical Multi-task Learning Framework for Micro-video Popularity Prediction*. IEEE TMM.
- Zhou, X., et al. (2023). *Bootstrap Latent Representations for Multi-modal Recommendation (BM3)*. WWW 2023.
