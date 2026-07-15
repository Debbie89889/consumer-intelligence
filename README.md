# Consumer Intelligence｜消費者智慧分析平台

![CI](https://github.com/Debbie89889/consumer-intelligence/actions/workflows/ci.yml/badge.svg)

以一份真實電商交易資料,建構涵蓋**客群分群、顧客終身價值（CLV）、Next Best Offer、購買傾向預測**的端到端消費者智慧分析,並產品化為 **PostgreSQL + FastAPI + Streamlit** 服務,附一層 grounded LLM 洞察 workflow（LangChain／LangGraph）。

---

## 專案概述

線上零售商握有兩年、逾百萬筆交易紀錄,但「資料」不等於「決策」。本專案回答一個核心商業問題:

> 如何利用交易資料,辨識高價值客戶、預測未來價值、提供交叉銷售建議,並鎖定適合再行銷的客群?

這個問題拆成四個可操作的子問題,分別對應一個分析階段:客群輪廓(分群)、未來價值(CLV)、交叉銷售(Next Best Offer)、再購機率(傾向模型)。每個階段都有對應的驗證方法(holdout、baseline 對照、calibration、SHAP),不只是 fit 出一個數字就結案。

最後把四個階段整合為 SQL 資料層、FastAPI 服務與 Streamlit 儀表板,再加一層 grounded LLM Copilot——所有數字由 Python／SQL 計算,LLM 只負責把已算好的事實轉成自然語言敘述。

---

## 核心成果

**資料規模**:776,577 筆清理後交易・5,852 位客戶・36,594 筆訂單・4,619 項商品・41 個國家・約 £17.1M 總營收

| 指標 | 結果 |
| --- | --- |
| 客群結構 | Champions 佔 19% 客戶,貢獻 65.5% 營收 |
| CLV 預測驗證 | holdout 相關係數 **0.848**,總量誤差 **2.2%** |
| Next Best Offer | 562 項主力商品挖出 **944** 條關聯規則 |
| 購買傾向預測 | ROC-AUC **0.804**(Logistic Regression baseline) |
| Win-back 名單 | **543** 位高價值但已沉睡的客戶(At Risk + Can't Lose Them) |

---

## 主要功能

- **RFM segmentation**:規則式 RFM 分位,產生 11 個具行銷語意的客群(Champions、At Risk、Can't Lose Them…)。
- **K-means segmentation**:資料驅動的互補視角,以 silhouette 選 k,驗證「約兩成客戶撐起七成營收」的結構。
- **Probabilistic CLV**:BG/NBD + Gamma-Gamma 估計未來購買次數與金額,以 holdout 驗證預測量。
- **Next Best Offer**:FP-Growth 關聯規則,依 lift 為單一商品或購物籃推薦下一項商品。
- **Purchase propensity modeling**:預測 90 天內是否再購,Logistic Regression 與 LightGBM 對照,附 SHAP 解釋。
- **Customer / product analytics**:可查詢瀏覽的客戶與商品明細、月度與地區營收趨勢。
- **Grounded LLM customer insights**:單一客戶的繁體中文洞察敘述,數字全由 Python 算好。
- **LangGraph multi-turn customer Q&A**:跨輪對話狀態管理、平行資料擷取、SSE 串流。
- **Human-in-the-loop win-back campaign workflow**:草擬折扣與文案後 `interrupt()` 停下等人工審核,核准才寫入資料庫。

---

## 系統架構

```text
原始交易 CSV
  → 資料清理與特徵工程(Phase 0,含 RFM 特徵)
  → 客群分群 ／ CLV ／ Next Best Offer ／ 購買傾向(Phase 1–4)
  → 各階段彙總 Parquet(data/processed/)
  → PostgreSQL(正式)或 SQLite(本機／測試)
  → FastAPI 服務(/docs)
  → Streamlit 互動儀表板
  → Grounded LLM Copilot ／ LangGraph workflow
```

業務邏輯集中於 `src/consumer_intel/`,前端為薄客戶端,僅呼叫 API 呈現。SQL 採參數化、ANSI 相容查詢,正式(PostgreSQL)與測試(SQLite)共用同一份程式碼。

---

## Demo 與畫面截圖

本專案已有部署版本;公開 repository 不提供正式服務網址。以下為主要功能畫面,亦可依照下方「快速開始」,使用內附的 processed data 在本機啟動 API 與 Dashboard。

> 未設定 LLM API key 時,Copilot 會自動使用 deterministic fallback,核心功能仍可在無外部付費服務的情況下完整執行。

### 客群總覽

![客群總覽 — KPI、各客群營收與營收占比](assets/screenshots/overview-1.png)
![客群總覽 — 客戶數 × 平均 CLV 分布、預估 CLV 最高客戶](assets/screenshots/overview-2.png)

### 客戶分析

![客戶分析 — 查詢/瀏覽客戶與單一客戶 KPI](assets/screenshots/customer-1.png)
![客戶分析 — RFM、存活/回購儀表與 grounded AI 洞察](assets/screenshots/customer-2.png)

### 產品分析

![產品分析 — 商品 KPI 與營收 Top 15](assets/screenshots/product-1.png)
![產品分析 — 營收 × 銷售數量分布與商品瀏覽](assets/screenshots/product-2.png)
![產品分析 — 單品明細與 Next Best Offer](assets/screenshots/product-3.png)

### 趨勢與地區

![趨勢 — 月營收與月訂單/下單客戶趨勢](assets/screenshots/trends-1.png)
![地區 — 各國營收世界地圖](assets/screenshots/trends-2.png)

---

## 方法與模型摘要

下表為各模組方法與驗證結果的摘要;完整推導過程與圖表見對應的 `reports/*.md`。

| 模組 | 方法 | 驗證或結果 | 業務用途 |
| --- | --- | --- | --- |
| 資料清理(Phase 0) | 規則式清理(取消單、退貨、非商品代碼、缺 `CustomerID` 等) | 保留 776,577 / 1,067,371 列(72.8%) | 建立可信賴的分析基礎([報告](reports/eda_phase0.md)) |
| 客群分群(Phase 1) | RFM 規則式分群 + K-means(log 轉換後以 silhouette 選 k=4) | Champions 佔 19% 客戶、貢獻 65.5% 營收 | 依客群輪廓分配行銷資源([報告](reports/phase1_segmentation.md)) |
| CLV(Phase 2) | BG/NBD(購買次數)+ Gamma-Gamma(客單價) | holdout 相關係數 0.848,總量誤差 2.2% | 預測客戶未來價值,篩選 win-back 名單([報告](reports/phase2_clv.md)) |
| Next Best Offer(Phase 3) | FP-Growth 關聯規則(support／confidence／lift) | 562 項主力商品、944 條規則 | 依購物籃提供交叉銷售推薦([報告](reports/phase3_basket_nbo.md)) |
| 購買傾向(Phase 4) | Logistic Regression baseline + LightGBM + SHAP | ROC-AUC 0.804(baseline 勝出) | 預測 90 天再購機率,支援 targeting([報告](reports/phase4_propensity.md)) |
| 產品化(Phase 5) | PostgreSQL／SQLite + FastAPI + Streamlit + Alembic | 170+ pytest 測試、CI 自動化 | 把分析轉為可互動、可部署的服務 |
| Grounded LLM／LangGraph | LangChain LCEL + LangGraph StateGraph | 見下方兩節 | 多輪客戶問答、win-back campaign 人工審核 |

---

## Grounded LLM 設計

這個專案的核心差異化在於 **LLM 與數值計算解耦**:

- KPI、分群、CLV、lift、propensity score 等所有數字皆由 Python／SQL 計算,LLM 完全不參與運算。
- LLM 只負責把已算好的結構化事實,轉譯成自然語言(繁體中文)敘述。
- 輸出經 Pydantic(`Literal` 型別 + `Field` 約束)驗證,不合法的輸出會被擋下並退回確定性敘述。
- 敘述層以 LangChain 實作(`ChatPromptTemplate | model.with_structured_output(...)`),透過 `init_chat_model` 維持 provider-agnostic(OpenAI／Anthropic 可切換)。
- 未設定 `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` 時,自動退回確定性模板,確保離線與 CI 皆可運行。

---

## LangGraph Workflow 摘要

在既有的 LangChain LCEL Copilot(`copilot/`)之外,另有一套 **LangGraph agentic workflow**(`copilot_graph/`),涵蓋:

- **多輪客戶問答**:平行擷取 RFM／CLV／Next Best Offer／傾向分數,fan-in 後生成對話式回答;對話狀態存於 SQLAlchemy `messages` 表,跨請求維持。
- **SSE streaming**:`GET /chat/stream` 以 `astream_events` 轉成精簡的 SSE 事件(`node_start`／`token`／`final`…)。
- **Human-in-the-loop win-back campaign**:草擬折扣與文案後 `interrupt()` 停下等人工審核,核准／退回修改／終結三種決定,審核紀錄由 SQLAlchemy ORM 寫入、與 LangGraph checkpointer 各自獨立。
- **可恢復的 interrupt / resume**:審核流程可跨行程中斷與恢復,checkpointer 本機用 `SqliteSaver`、正式環境用 `PostgresSaver`。

完整的 state graph 節點設計、SSE 事件細節、效能 benchmark,以及與 LCEL 版本的逐項比較,見 [docs/langgraph-design.md](docs/langgraph-design.md)。

---

## 快速開始

### 啟動 API 與 Dashboard

倉庫已附各階段的彙總 parquet,**不需要重新執行模型 pipeline**,clone 後即可起服務:

```bash
pip install -e ".[api,app,copilot]"
python scripts/load_db.py
alembic upgrade head
uvicorn consumer_intel.api.app:app --reload
streamlit run app/dashboard.py
```

- API 文件位於 `http://localhost:8000/docs`。
- Dashboard 預設位於 `http://localhost:8501`。
- `alembic upgrade head` 只需執行一次,用於建立 win-back campaign 所需的資料表;不執行也不影響分群／CLV／NBO／傾向等既有功能。
- 預設使用 SQLite,未設定 `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` 時,Copilot 以確定性模板輸出。

### 從原始資料完整重建

```bash
pip install -e ".[ml,api,app,copilot,dev]"

# 1. 下載 Online Retail II CSV 至 data/raw/online_retail_II.csv(UCI id=502 或 Kaggle)
# 2. 依序執行各階段 pipeline
python scripts/run_phase0.py
python scripts/run_phase1.py
python scripts/run_phase2.py
python scripts/run_phase3.py
python scripts/run_phase4.py

# 3. 產生彙總並載入資料庫
python scripts/build_summaries.py
python scripts/load_db.py
```

---

## Docker

```bash
docker compose up
```

一次啟動 PostgreSQL、FastAPI 與 Streamlit(見 `docker-compose.yml`)。`render.yaml` 另提供 Render 一鍵部署 Blueprint(同樣是 PostgreSQL + FastAPI + Streamlit),雲端免費方案的條款請以 Render 官方文件為準。

---

## API

FastAPI 提供客戶、產品、分群、趨勢、Next Best Offer、Campaign 與對話功能,完整互動文件可於 `/docs` 查看。代表性端點:

```text
GET  /customers/{id}
GET  /customers/{id}/insight
GET  /products/{code}/next-best-offer
POST /campaigns/generate
POST /campaigns/{thread_id}/resume
GET  /chat/stream
```

---

## 測試與工程品質

- 170+ 個 pytest 測試,涵蓋資料清理、RFM、分群、CLV、關聯規則、傾向特徵與模型、SQL repository、FastAPI 端點,以及兩套 Copilot 實作的 grounding 與 Pydantic 約束。
- 測試以 SQLite 與 FastAPI `TestClient` 進行,Copilot 走確定性模板路徑,不依賴 PostgreSQL 或真實 LLM 金鑰。
- `ruff` 負責 lint 與格式化;GitHub Actions(`.github/workflows/ci.yml`)於每次 push 與 PR 執行 lint 與測試。
- FastAPI response model、SQLAlchemy ORM、Alembic 版本管理,SQLite(本機／測試)與 PostgreSQL(正式)共用同一套程式碼與 Docker 映像。

---

## 專案結構

```text
consumer-intelligence/
├── README.md · CLAUDE.md · PROJECT_PLAN.md · pyproject.toml
├── docs/                          # 長篇技術設計文件(如 LangGraph 設計筆記)
├── data/processed/                # 各階段彙總 parquet(小檔已進版控)
├── src/consumer_intel/
│   ├── config.py · labels.py      # 設定常數 · 中文顯示標籤
│   ├── data/ · eda/ · features/   # 清理 · EDA · RFM 特徵
│   ├── segmentation/ · clv/ · basket/ · propensity/
│   ├── copilot/                   # grounded LLM 洞察層(LangChain LCEL)
│   ├── copilot_graph/             # LangGraph agentic workflow
│   ├── db/                        # engine · loader · SQL repository · ORM models
│   └── api/                       # FastAPI app · Pydantic models · 依賴
├── app/dashboard.py                # Streamlit 前端
├── alembic/ · alembic.ini          # Copilot 業務表(conversations/campaign_approvals)版本管理
├── db/schema.sql · sql/analytics.sql
├── scripts/                        # run_phase0~4 · build_summaries · load_db · start_*.sh
├── Dockerfile · docker-compose.yml · render.yaml
├── reports/                        # 各階段 Markdown 報告 + 靜態圖表
├── tests/                          # 各模組對應的 pytest
└── .github/workflows/ci.yml
```

---

## 資料來源

Online Retail II, UCI Machine Learning Repository (id=502)。英國線上禮品商 2009–2011 交易資料,授權 CC BY 4.0。
