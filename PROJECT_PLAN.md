# Consumer Intelligence 分析專案 — 規劃書

> **一句話定位**:用一份真實電商交易資料,打造「客群分群 × CLV × Next Best Offer × 購買傾向預測 × grounded LLM 洞察」的端到端消費者智慧分析,並產品化為可部署的後端服務與互動儀表板。

本文件記錄專案的設計取向、資料選擇、技術棧、系統架構與各階段範圍。所有階段皆已完成;進度見最後一節。

---

## 1. 設計取向

這個專案刻意把「分析」與「產品化」一起做完,串成一條完整的價值鏈:

1. **零售分析的核心題目一次到位**:Basket Analysis、CLV、Next Best Offer、Purchase Prediction、Customer Segmentation,涵蓋分群(clustering)、分類(classification)、機率模型與推薦。
2. **方法要會驗證,不只 fit**:CLV 用 holdout 檢查預測量、傾向模型放 baseline 對照並做 calibration 與 SHAP——展示「懂評估」而非只跑出一個數字。
3. **工程嚴謹度**:模組化套件、單元測試、CI、容器化、可部署服務,而不是停在 notebook。
4. **LLM 的正確用法(grounding)**:數字由 Python/SQL 算好,LLM 只負責敘述並以 Pydantic 驗證——把 LLM 放在它該在的位置。

---

## 2. 資料集

### 採用:Online Retail II(UCI id=502 / Kaggle)

- 英國線上禮品商 2009–2011 交易資料,約 **107 萬筆**。
- 欄位:`Invoice`、`StockCode`、`Description`、`Quantity`、`InvoiceDate`、`Price`、`CustomerID`、`Country`。
- **同時有金額 + 客戶 ID + 重複購買** → RFM、CLV、購物籃分析可用同一份資料完成。
- 取得:`from ucimlrepo import fetch_ucirepo; d = fetch_ucirepo(id=502)`,或 Kaggle 下載 CSV。
- 授權:CC BY 4.0。

### 為何不選其他資料集

| 資料集 | 特點 | 不採用的原因 |
|---|---|---|
| Instacart Market Basket | 有 reorder 標籤,適合大規模再購預測 | 無價格 → 無法做 CLV |
| Olist 巴西電商 | 真實多表結構 + 評論文字 | 多為一次性買家,RFM／CLV 訊號弱;join 較複雜 |

> 取捨原則:**深度優於廣度**。以 Online Retail II 一份資料把核心題目做完整、做到可驗證,優於把多份資料都淺嘗一遍。

---

## 3. 技術棧

- 資料處理:`pandas`、`numpy`、`pyarrow`(parquet)
- 分群／分類:`scikit-learn`、`lightgbm`、`shap`
- CLV:`lifetimes`(BG/NBD + Gamma-Gamma)
- 購物籃:`mlxtend`(FP-Growth)
- 視覺化／前端:`plotly`、`streamlit`
- 後端服務:`fastapi`、`uvicorn`、`sqlalchemy`(PostgreSQL / SQLite)
- LLM Copilot:`langchain`(`init_chat_model`,OpenAI／Anthropic 可切換)、`pydantic`(輸出驗證)
- 工程化:`pytest`、`ruff`、GitHub Actions、`docker`(`docker-compose`、`render.yaml`)

---

## 4. 系統架構

```
原始 CSV ─► Phase 0 清理/EDA ─► transactions_clean.parquet
                                   ├─ Phase 1 分群    ─► customer_segments
                                   ├─ Phase 2 CLV     ─► customer_clv
                                   ├─ Phase 3 NBO     ─► association_rules
                                   ├─ Phase 4 傾向    ─► propensity_scores
                                   └─ build_summaries ─► product / monthly / country
                                              │
                                   load_db ─► PostgreSQL / SQLite(5 張表)
                                              │
                                   FastAPI(含 grounded LLM Copilot, /docs)
                                              │
                                   Streamlit 儀表板(呼叫 API)
```

> 業務邏輯集中在 `src/consumer_intel/`;前端是薄客戶端,只呼叫 API。SQL 採參數化、ANSI 相容,正式(Postgres)與測試(SQLite)共用同一份程式碼。

### 專案結構

```
consumer-intelligence/
├── README.md · CLAUDE.md · PROJECT_PLAN.md · pyproject.toml
├── data/processed/        # 各階段彙總 parquet(小檔已進版控)
├── src/consumer_intel/
│   ├── config.py · labels.py
│   ├── data/ · eda/ · features/
│   ├── segmentation/ · clv/ · basket/ · propensity/
│   ├── copilot/           # grounded LLM 洞察層
│   ├── db/                # engine、loader、SQL repository
│   └── api/               # FastAPI app、Pydantic models、依賴
├── app/dashboard.py       # Streamlit 前端
├── db/schema.sql · sql/analytics.sql
├── scripts/               # run_phase0~4、build_summaries、load_db、start_*.sh
├── Dockerfile · docker-compose.yml · render.yaml · .streamlit/config.toml
├── reports/               # 各階段 Markdown 報告
├── tests/                 # 每個模組對應一份 pytest
└── .github/workflows/ci.yml
```

---

## 5. 執行階段

每階段的計算邏輯都在 `src/` 並有對應 pytest;每階段產出一份 `reports/*.md`。

### Phase 0 — 設定與商業框架 ✅
- repo、商業問題框架、資料載入與清理(取消單、退貨、缺 `CustomerID`、非商品代碼、批發大戶離群值)、基本 EDA(時間趨勢、國家分布、客單價)。

### Phase 1 — 客群分群:RFM + 分群 ✅
- 計算每位客戶 R/F/M;(a) RFM 分位 → 規則式 11 分群;(b) 對 log 轉換後 RFM 跑 K-means,以 elbow／silhouette 選 k;描繪每群輪廓、用商業語言命名,並寫出「對每群該做什麼」。

### Phase 2 — CLV ✅
- 歷史 CLV 當基準;預測型 CLV 用 `lifetimes` 的 BG/NBD(次數)+ Gamma-Gamma(金額);以 calibration/holdout 驗證預測量;把 CLV 接回 Phase 1 分群,找出最值錢與正在流失的客群。

### Phase 3 — 購物籃分析 → Next Best Offer ✅
- 以 FP-Growth 挖頻繁項集,算 support／confidence／lift 關聯規則;Next Best Offer:給定商品或購物籃,依 lift 推薦下一個商品(排除已在籃中的品項)。

### Phase 4 — 購買傾向預測 ✅
- 目標:未來 90 天是否再購;**無洩漏的時間切分**(特徵只用 cutoff 前);Logistic Regression baseline + LightGBM;評估 ROC-AUC、PR-AUC、calibration;SHAP 解釋特徵重要性。誠實記錄「baseline 略勝樹模型」的結果。

### Phase 5 — 產品化 ✅
- **資料層**:各階段產出載入 5 張 SQL 表(customers／rules／products／monthly／country)。
- **API**:FastAPI 12 個端點 + Swagger `/docs`,Pydantic response model。
- **grounded LLM Copilot**:LangChain 敘述層,數字全由 Python 算好,Pydantic 驗證,離線/CI 退回確定性模板。
- **前端**:Streamlit 4 分頁(客群總覽／客戶分析／產品分析／趨勢與地區),含世界地圖。
- **部署**:Dockerfile、docker-compose、Render `render.yaml` Blueprint。

---

## 6. 因果推論／Uplift(誠實處理)

公開資料沒有實驗組／對照組,**真正的 uplift／incrementality 做不出來,不硬湊**。README 以方法論層次說明:若有 treatment 資料(例如曾否收到某 offer),會用 A/B test 設計搭配 uplift modeling 量測增量效果。定位是「懂方法、不 overclaim」。

---

## 7. 範圍說明

- **核心(Phase 0–3)**:分群、CLV、購物籃 NBO——一份資料即可完成的零售分析主幹。
- **加上 Phase 4**:補上 tree-based 分類與傾向建模,並做完整評估。
- **加上 Phase 5**:產品化為可部署服務與互動儀表板,並加 grounded LLM Copilot——把分析變成可操作的產品。
