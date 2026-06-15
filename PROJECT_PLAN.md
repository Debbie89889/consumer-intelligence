# Consumer Intelligence 分析專案 — 規劃書

> **一句話定位**：用一份真實電商交易資料，打造「客群分群 × CLV × Next Best Offer × 購買預測 × LLM 洞察 Copilot」的端到端消費者智慧分析，並產品化為可互動的 dashboard。

---

## 1. 為什麼做這個專案（策略意圖）

這個專案刻意設計成一次達成三件事：

1. **補上目標職位（春樹科技 Consumer Intelligence Platform）缺的加分技能**：零售分析（Basket Analysis、CLV、Next Best Offer、Purchase Prediction、Audience Modeling）、行銷科學（Customer Segmentation、Propensity）、推薦系統。
2. **補齊必備卻較薄的 classic ML**：分群（Clustering）、分類（Classification）、Tree-based 模型——讓履歷上這幾項從「會」變成「做過」。
3. **重用既有強項**：產品化能力（模組化套件、pytest、CI、Docker、Streamlit）與 LLM grounding 設計（數字由 Python 算、LLM 只敘述）。

> 命名與最終架構刻意對齊 JD 寫的產品：**AI Insight Generator / AI Consumer Copilot / Audience Factory**。

---

## 2. 資料集

### 首選：Online Retail II（UCI id=502 / Kaggle）
- 英國線上禮品商 2009–2011 交易資料，約 **107 萬筆**。
- 欄位：`InvoiceNo`、`StockCode`、`Description`、`Quantity`、`InvoiceDate`、`UnitPrice`、`CustomerID`、`Country`。
- **同時有金額 + 客戶 ID + 重複購買** → RFM、CLV、購物籃分析可用同一份資料完成。
- 取得：`from ucimlrepo import fetch_ucirepo; d = fetch_ucirepo(id=502)`，或 Kaggle 下載 CSV。
- 授權：CC BY 4.0。

### 替代方案（依想強調的重點選用）
| 資料集 | 規模 | 特點 | 缺點 |
|---|---|---|---|
| **Instacart Market Basket**（Kaggle 競賽） | 20 萬+ 用戶、300 萬+ 訂單 | 有 reorder 標籤，適合大規模再購預測（分類） | 無價格 → 無法做 CLV |
| **Olist 巴西電商**（Kaggle） | 約 10 萬訂單、9 張關聯表 | 真實多表結構 + 顧客評論（可做 NLP／滿意度預測） | 多為一次性買家，RFM／CLV 訊號弱；join 較複雜 |

> **建議**：用 **Online Retail II** 當主軸把核心做完；行有餘力再拉 Olist 加「評論 NLP／關聯式資料」維度。**不要三個都做——深度優於廣度。**

---

## 3. 技術棧

- 資料處理：`pandas`、`numpy`、`pyarrow`（parquet）
- 分群／分類：`scikit-learn`、`lightgbm`、`xgboost`、`shap`
- CLV：`lifetimes`（BG/NBD + Gamma-Gamma）
- 購物籃：`mlxtend`（Apriori／FP-Growth）
- 視覺化／App：`plotly`、`streamlit`
- LLM Copilot：`openai`／`anthropic` API、`pydantic`（輸出驗證）
- 工程化：`pytest`、`ruff`、GitHub Actions、（選配）`docker`

---

## 4. 建議專案結構

```
consumer-intelligence/
├── README.md
├── CLAUDE.md
├── pyproject.toml
├── data/
│   ├── raw/            # 原始下載檔（gitignore）
│   └── processed/      # 清理後 parquet
├── src/consumer_intel/
│   ├── data/           # 載入 + 清理
│   ├── features/       # RFM、特徵工程
│   ├── segmentation/   # RFM 分數 + K-means
│   ├── clv/            # BG/NBD + Gamma-Gamma
│   ├── basket/         # 關聯規則 + Next Best Offer
│   ├── propensity/     # 購買／再購預測
│   ├── copilot/        # LLM 洞察層（數字僅來自 Python）
│   └── viz/            # 繪圖輔助
├── app/dashboard.py    # Streamlit
├── notebooks/          # 僅探索用；邏輯放 src/
├── tests/              # 每個模組對應一份 pytest
└── .github/workflows/ci.yml
```

---

## 5. 執行階段（每階段都對應 JD 技能）

### Phase 0 — 設定與商業框架 ｜ 半天
- [ ] 建 repo、README 開頭寫清楚商業問題（例：「如何用交易資料把對的 offer 給對的客群，提升再購率與 LTV」）
- [ ] 資料載入與清理：取消單（`InvoiceNo` 開頭為 `C`）、退貨（負數量）、缺 `CustomerID` 列、批發大戶離群值
- [ ] 基本 EDA（時間趨勢、國家分布、客單價）
- **對應**：Data Quality、Feature Engineering、商業思維

### Phase 1 — 客群分群：RFM + 分群〔核心〕 ｜ 1–2 天
- [ ] 計算每位客戶 Recency / Frequency / Monetary
- [ ] (a) RFM 分位分數 → 規則式分群
- [ ] (b) 對標準化（含 log 轉換）後的 RFM 跑 K-means，用 elbow／silhouette 選 k
- [ ] 描繪每群輪廓，用商業語言命名（Champions / At-risk / New / Hibernating…）並寫出「對每群該做什麼」
- **對應**：Customer Segmentation、Audience Modeling、必備 Clustering（履歷的「子群體分析」在此變成真正的消費者分群）

### Phase 2 — CLV〔核心〕 ｜ 1 天
- [ ] 歷史 CLV（簡單版）
- [ ] 預測型 CLV：`lifetimes` 的 BG/NBD（購買次數）+ Gamma-Gamma（金額），預測未來 N 個月價值
- [ ] 把 CLV 接回 Phase 1 分群，找出最值錢的客群
- **對應**：CLV；展現懂正規機率方法，而非只算平均

### Phase 3 — 購物籃分析 → Next Best Offer〔核心〕 ｜ 1–2 天
- [ ] 關聯規則（`mlxtend` Apriori／FP-Growth）：support / confidence / lift
- [ ] 簡單 Next-Best-Offer／交叉銷售推薦：給定購物籃或商品，依 lift 推薦下一個商品
- **對應**：Basket Analysis、Next Best Offer、Recommendation System

### Phase 4 — 購買預測／傾向模型〔補 tree-based 缺口〕 ｜ 2 天
- [ ] 定義目標（例：未來 90 天是否再購）
- [ ] 從 RFM + 行為特徵建快照特徵表
- [ ] Logistic Regression baseline + LightGBM／XGBoost
- [ ] 評估：ROC-AUC、PR-AUC、calibration；用 SHAP 解釋特徵重要性
- **對應**：必備 Classification + Tree-based、Propensity Modeling、Purchase Prediction
- *（若改用 Instacart，這步變成 reorder 預測，是更乾淨、更大的分類任務）*

### Phase 5 — 產品化 +（選配）LLM Insight Copilot〔差異化〕 ｜ 2–3 天
- [ ] Streamlit dashboard：分群探索、CLV 檢視、NBO 推薦、傾向分數
- [ ] pytest 單元測試 +（選配）Docker
- [ ] **（高槓桿選配）客群洞察 Copilot**：使用者選一個客群 → 數字全由 Python 算好 → LLM 只把結果翻成商業摘要與建議行動
  - 重用 SKU Copilot 的「數字在 Python、LLM 只敘述、grounded、防幻覺」設計
  - 用 Pydantic 驗證 LLM 輸出
- **對應**：MLOps、Product Thinking，直接對應 JD 的 AI Insight Generator／AI Consumer Copilot

---

## 6. 因果推論／Uplift（誠實處理）

公開資料沒有實驗組／對照組，**真正的 uplift／incrementality 做不出來，不要硬湊**。改在 README 加一小段「因果思維」：說明若有 treatment 資料，你會如何用 A/B test 設計與 uplift modeling 量測增量。展現方法論理解、又不 overclaim——定位成「正在學、懂原理」。

---

## 7. 範圍與時程

| 版本 | 內容 | 時程 |
|---|---|---|
| **最小可行（CP 值最高）** | Phase 0–3 | 2–3 個週末 |
| **建議** | + Phase 4 | +1 週末 |
| **差異化** | + Phase 5（含 LLM Copilot） | +1–2 週末 |

---

## 8. 放上履歷的方式

- 一個專案條目 + GitHub 連結 + 一行 Streamlit demo 說明
- 標題範例：**消費者智慧分析（個人專案）｜客群分群 × CLV × Next Best Offer × LLM Insight Copilot**
- bullet 用商業 framing（與其他專案一致：商業問題 → 方法 → 量化洞察）
- README 結構：**商業問題 → 方法 → 關鍵洞察（放一張圖）→ 如何執行**（招募方掃 README 很快）

---

## 9. 執行心法

1. **深度優於廣度**：一個有乾淨評估與清楚商業敘事的傾向模型，勝過每樣都淺嘗。
2. **每段開頭寫商業問題、結尾寫「所以該怎麼做」**：這就是 JD 要的商業敏感度與 Product Thinking。
3. **維持你已展現的工程嚴謹度**（模組化、測試、CI）：這是相對於「只有 notebook」的 portfolio 的差異化。
