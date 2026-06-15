# Consumer Intelligence 分析專案

> 用一份真實電商交易資料，打造「客群分群 × CLV × Next Best Offer × 購買預測 × LLM 洞察 Copilot」的端到端消費者智慧分析，並產品化為可互動的 dashboard。

---

## 商業問題

線上零售商手上有兩年、超過百萬筆的交易紀錄，但「資料」不等於「決策」。本專案要回答：

> **如何用交易資料，把對的 offer 給對的客群，提升再購率與顧客終身價值（LTV）？**

拆成四個可操作的子問題，對應後續各 Phase：

1. **客群長什麼樣？** 誰是高價值、誰快流失、誰是新客 → 客群分群（RFM + K-means）
2. **每位客戶未來值多少錢？** → 預測型 CLV（BG/NBD + Gamma-Gamma）
3. **下一個該推什麼商品？** → 購物籃關聯規則 → Next Best Offer
4. **誰最可能在 90 天內再購？** → 傾向模型（LightGBM / XGBoost）

最後把以上結果包成 Streamlit dashboard，並加一層 **LLM 洞察 Copilot**——數字一律由 Python 算好，LLM 只負責翻成商業摘要（grounded、防幻覺）。

---

## 方法（資料與清理）

**資料集**：Online Retail II（UCI id=502 / Kaggle，CC BY 4.0）。英國線上禮品商 2009–2011 交易，約 107 萬筆，同時含金額 + 客戶 ID + 重複購買，足以一份資料完成 RFM / CLV / 購物籃分析。

**Phase 0 清理規則**（見 `src/consumer_intel/data/clean.py`，每條都有對應 pytest）：

| 步驟 | 移除列數 | 理由 |
|---|---:|---|
| 完全重複列 | 34,335 | 重複匯出 |
| 取消單（`Invoice` 開頭 `C`） | 19,104 | 非實際銷售 |
| 非商品代碼（POST、M、BANK CHARGES…） | 4,791 | 郵資／手續費等管理用代碼，非商品 |
| 非正數量（退貨／調整） | 3,362 | 客戶層正向銷售才進主表 |
| 非正單價（免費品／錯誤） | 2,566 | 金額不可信 |
| 缺 `CustomerID` | 226,636 | 無法歸戶，做不了客戶層分析 |

> 清理後 **保留 776,577 / 1,067,371 列（72.8%）**，存成 `data/processed/transactions_clean.parquet`，下游一律讀清理後的檔。
> 設計取向：主表只保留「可歸戶的正向銷售」，作為 RFM / CLV / 購物籃的基礎；退貨與取消若日後要算淨營收，可由原始檔重新推導。

---

## 關鍵洞察（Phase 0）

清理後的資料輪廓（完整見 `reports/eda_phase0.md`）：

- **規模**：776,577 筆明細、**5,852** 位客戶、36,594 筆訂單、4,619 項商品、41 國，總營收約 **£17.1M**。
- **高度集中於英國**：UK 佔營收 **83.7%**；EIRE 名列第二但僅 3 位客戶貢獻 528 筆訂單——**典型批發／內部帳戶**，分群與 CLV 時需留意這類離群值（`CLAUDE.md` 已標註）。
- **客戶價值極度右偏**：中位客戶累積消費約 £856，但前 1% 達 £28,631、最高一位逾 £580,000。平均值（£2,917）被少數大戶拉高——**用平均描述客戶會誤導，這正是要做分群與機率型 CLV 的原因**。
- **訂單金額**：中位 £303、均值 £466，同樣右偏。
- **暢銷品**：以 REGENCY CAKESTAND、WHITE HANGING HEART T-LIGHT HOLDER、JUMBO BAG 系列等禮品／家居小物為主，購物籃分析應有足夠共購訊號。

> 圖表：`reports/monthly_revenue.html`（月營收 × 活躍客戶）、`reports/country_revenue.html`（各國營收）。

## 關鍵洞察（Phase 1：客群分群）

對 5,852 位客戶算 RFM，再用兩種互補的方法分群（完整見 `reports/phase1_segmentation.md`）：

- **規則式 RFM（11 個行銷語意分群）**：**Champions 佔 19% 客戶卻貢獻 65.5% 營收**（人均 £10,030）——清楚的 80/20 結構，行銷預算重點不言而喻。另有 **At Risk + Can't Lose Them** 共 543 位高價值但已沉睡的客戶（平均 342～491 天未購），是 win-back 的首要名單。
- **K-means（資料驅動，k=4，以 silhouette 在 3–8 區間選定）**：得到 High-Value Active / High-Value Lapsing / Recent Low-Value / Dormant Low-Value 四群；**High-Value Active 佔 20% 客戶、73% 營收**。
- **兩種方法互相驗證**：規則式與資料驅動都指向「約 1/5 的客戶撐起約 2/3 以上營收」。規則式給的是可直接溝通的行銷標籤，K-means 給的是這份資料自身的結構——dashboard 會同時呈現兩種視角。

每個分群都附了「該怎麼做」（見報告的 action 表），這正是 JD 要的商業敏感度。

> 圖表：`reports/rfm_segments_revenue.html`、`reports/kmeans_elbow.html`、`reports/kmeans_silhouette.html`、`reports/kmeans_scatter.html`。

---

## 因果思維（誠實處理）

本資料沒有實驗組／對照組，**無法做真正的 uplift / incrementality**，不在此 overclaim。若日後取得 treatment 資料（例如曾否收到某 offer），會以 A/B test 設計搭配 uplift modeling 量測增量效果——此處定位為「懂方法、不硬湊」。

---

## 如何執行

```bash
# 安裝（含 dev 相依）
pip install -e ".[dev]"

# 取得資料：下載 Online Retail II CSV 放到 data/raw/online_retail_II.csv
# （UCI id=502 或 Kaggle）

# 跑 Phase 0：清理 -> 存 parquet -> 產 EDA 報告與圖表
python scripts/run_phase0.py

# 測試與 lint
pytest
ruff check . && ruff format .
```

---

## 進度

- [x] **Phase 0** — 設定與商業框架、資料清理、EDA
- [x] **Phase 1** — 客群分群（RFM 規則式分群 + K-means）
- [ ] **Phase 2** — CLV（BG/NBD + Gamma-Gamma）
- [ ] **Phase 3** — 購物籃分析 → Next Best Offer
- [ ] **Phase 4** — 購買預測／傾向模型
- [ ] **Phase 5** — 產品化 + LLM Insight Copilot

詳細規劃見 `PROJECT_PLAN.md`；給 Claude Code 的工作守則見 `CLAUDE.md`。

## 專案結構

```
consumer-intelligence/
├── data/
│   ├── raw/                      # 原始 CSV（gitignore）
│   └── processed/                # 清理後 parquet
├── src/consumer_intel/
│   ├── config.py                 # 路徑與清理常數
│   ├── data/                     # 載入 + 清理
│   ├── features/                 # RFM 特徵
│   ├── segmentation/             # RFM 規則式分群 + K-means
│   └── eda/                      # EDA 摘要函式
├── scripts/
│   ├── run_phase0.py             # 清理 + EDA pipeline
│   └── run_phase1.py             # RFM + 分群 pipeline
├── reports/                      # 產出的 EDA 報告與圖表
├── tests/                        # 對應 data/ 與 eda/ 的 pytest
└── .github/workflows/ci.yml      # lint + test
```
