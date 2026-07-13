# CLAUDE.md

給 Claude Code 的專案說明。開始任何工作前先讀完本檔;詳細階段規劃見 `PROJECT_PLAN.md`。

## 專案是什麼

Consumer Intelligence 分析專案:用一份真實電商交易資料,做端到端的消費者智慧分析——**客群分群、CLV、Next Best Offer、購買傾向預測**——並已產品化為 **PostgreSQL + FastAPI + Docker + Streamlit** 的可互動服務,再加一層 **grounded LLM 洞察 Copilot**。個人 portfolio 專案,重點是把「資料 + ML + LLM 連結到商業價值並產品化」做完整且可驗證。

## 核心設計原則(最重要,務必遵守)

**LLM 與數值計算解耦(grounding / 防幻覺)**

- 所有 KPI、分群統計、CLV、lift、傾向分數等**數字一律由 Python/SQL 計算**並以結構化物件傳遞。
- **LLM 只負責把已算好的結構化結果翻成自然語言(繁體中文)**,絕不參與運算、絕不自己產生數字。
- LLM 的輸出一律用 **Pydantic(`Literal` 型別 + `Field` 約束)** 驗證;不合法的輸出要被擋下並退回友善的確定性敘述,不可讓系統崩潰。
- 任何 LLM 回應都要能對應回它所依據的工具結果(保留可追溯的 `grounding`)。

> 若你(Claude Code)發現自己想讓 LLM「順便算一下」某個數字——停下來,改成在 Python 算好再傳給它。

**資料持久化分層(checkpointer vs. SQLAlchemy ORM,不可混用)**

專案有兩種截然不同的「狀態」,故意用兩套機制持久化,面試常被問到這個分層:

- **LangGraph checkpointer**(`copilot_graph/`,見 `PROJECT_PLAN.md`／規劃中的 LangGraph 階段):負責 **graph 執行的技術性狀態**——目前跑到哪個節點、`interrupt()` 後如何從中斷點 `resume`。這是框架內部的執行序列快照,不是給人看的業務資料。
- **SQLAlchemy ORM**(`src/consumer_intel/db/models.py`,Alembic 管理):負責 **業務資料**——對話歷史(`conversations`／`messages`)、人工審核紀錄(`campaign_approvals`)。這些是有商業意義、需要查詢、需要稽核軌跡的資料,獨立於 graph 怎麼跑。
- 兩者**不共用同一張表、不互相依賴**:即使之後 checkpointer 存活期滿被清除,`campaign_approvals` 的審核歷史仍完整保留;反之 ORM 表的 schema 變動也不影響 checkpointer 的還原能力。

## 技術棧與慣例

- Python 3.11+。全面使用 type hints。
- 套件:pandas、numpy、scikit-learn、lightgbm、shap、lifetimes、mlxtend、plotly、streamlit、pydantic;後端與 LLM 層:fastapi、uvicorn、sqlalchemy、langchain(langchain-openai／langchain-anthropic)。
- **業務邏輯放 `src/consumer_intel/`**,前端 `app/dashboard.py` 只呼叫 API 呈現,不做運算。
- 函式小而單一職責,有 docstring;偏好純函式、易測試。
- 格式化與 lint 用 `ruff`。
- SQL 一律用 **參數化、ANSI 相容** 查詢,確保同一份程式碼在 PostgreSQL(正式)與 SQLite(本機/測試)都能跑。
- LLM 串接保持 **provider-agnostic**(LangChain `init_chat_model`,OpenAI／Anthropic 可切換)。
- **LLM orchestration 採用 LangChain**:Copilot 的敘述層用 `ChatPromptTemplate | model.with_structured_output(NarratedInsight)`,由 LangChain 負責 prompt 組裝與結構化輸出驗證。前提是**不破壞 grounding**——schema 只含可敘述的文字欄位,數字/segment/風險等級一律由 Python 算好,LLM 不參與運算。沒設 API key 或呼叫失敗時自動退回確定性模板,確保離線/CI 可跑。
  - (設計註:採用 LangChain 是為了展示主流 LLM 框架的使用,但 grounding 仍是不可妥協的核心。)

## 專案結構

```
src/consumer_intel/
  config.py        # 路徑與清理常數
  labels.py        # 中文顯示標籤(後端/前端共用)
  data/            # 載入 + 清理
  eda/             # EDA 摘要
  features/        # RFM、特徵工程
  segmentation/    # RFM 規則式分群 + K-means
  clv/             # 歷史 + BG/NBD + Gamma-Gamma、holdout 驗證
  basket/          # 關聯規則 + Next Best Offer
  propensity/      # 購買傾向:特徵、LogReg/LightGBM、SHAP
  copilot/         # grounded LLM 洞察層(context/narrator/schema)
  db/              # SQLAlchemy engine、loader、SQL repository
  api/             # FastAPI app、Pydantic models、DB 依賴
app/dashboard.py   # Streamlit 前端(呼叫 API)
db/schema.sql      # PostgreSQL DDL
sql/analytics.sql  # 分析查詢
scripts/           # run_phase0~4、build_summaries、load_db、start_*.sh
tests/             # 每個模組對應一份 pytest
```

## 常用指令

```bash
pip install -e ".[api,app,copilot,dev]"   # 安裝(serving extras + dev)

# 從原始資料完整重現
python scripts/run_phase0.py              # 清理 + EDA(其餘 run_phase1~4 類推)
python scripts/build_summaries.py         # 產品/月份/國家彙總
python scripts/load_db.py                 # 載入彙總進資料庫

# 起服務
uvicorn consumer_intel.api.app:app --reload   # API + /docs
streamlit run app/dashboard.py                # 前端
docker compose up                             # 或一鍵起 Postgres + API + 前端

pytest                                    # 跑測試
ruff check . && ruff format .             # lint + 格式化
```

> 注意:`build_summaries.py` 讀 `transactions_clean.parquet`(由 `run_phase0` 產生),`load_db.py` 讀各階段彙總 parquet。repo 已附彙總檔,clone 後可直接 `load_db` 起服務,不必下載原始資料。

## 測試要求

- 每個分析函式都要有對應的 pytest(尤其 RFM、CLV、關聯規則、特徵工程這類純計算)。
- 新增或修改計算邏輯時,**同一個 PR 內要補上／更新測試**。
- 評估指標(AUC、PR-AUC、calibration)要寫進測試或 reproducible 的腳本,不可只存在 notebook。
- API 與 DB 的測試走 SQLite + FastAPI `TestClient`,Copilot 走確定性模板路徑——測試不依賴 PostgreSQL 或真實 LLM key。

## 資料注意事項(Online Retail II)

- 取得:`from ucimlrepo import fetch_ucirepo; fetch_ucirepo(id=502)`,或 Kaggle CSV。
- 清理重點:取消單(`Invoice` 開頭 `C`)、退貨(負數量)、缺 `CustomerID`、非商品代碼、非正單價;部分客戶是批發大戶,是金額離群值,分群時要留意。
- 清理後存成 `data/processed/*.parquet`,下游一律讀清理後的檔。

## 名詞定義(保持一致用語)

- **RFM**:Recency(最近一次購買距今)、Frequency(購買次數)、Monetary(累積金額)。
- **CLV**:客戶終身價值;以 BG/NBD(預測購買次數)+ Gamma-Gamma(預測金額)估計預測型 CLV。
- **NBO(Next Best Offer)**:依購物籃關聯規則(lift／confidence)推薦的下一個商品。
- **Propensity**:客戶在未來時間窗內發生某行為(如再購)的機率。

## 不要做的事

- 不要在報告／Copilot 中產生未經 Python 計算的數字。
- 不要宣稱做了 uplift／incrementality／因果效果——本資料**沒有實驗組／對照組**。只能在 README 以方法論層次討論「若有 treatment 資料會如何做」。
- 不要把核心邏輯留在 notebook;不要讓前端做運算(前端只呼叫 API)。
- 不要在沒有測試的情況下合併計算邏輯的變更。
