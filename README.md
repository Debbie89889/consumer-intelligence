# Consumer Intelligence｜消費者智慧分析

> 用一份真實電商交易資料,做端到端的消費者智慧分析——**客群分群 × 顧客終身價值(CLV)× Next Best Offer × 購買傾向預測 × grounded LLM 洞察**——並產品化為 **PostgreSQL + FastAPI + Docker + Streamlit** 的可互動服務。

個人專案。從原始交易資料一路到可部署的後端服務與互動儀表板,重點放在**可驗證的方法**與**工程嚴謹度**(模組化、單元測試、CI、容器化)。

---

## 系統架構

```
原始交易 CSV
  └─ Phase 0  清理 + EDA ──────────────► transactions_clean.parquet
       ├─ Phase 1  RFM 規則式分群 + K-means ─► customer_segments.parquet
       ├─ Phase 2  CLV(BG/NBD + Gamma-Gamma)─► customer_clv.parquet
       ├─ Phase 3  購物籃關聯規則 → NBO ──────► association_rules.parquet
       ├─ Phase 4  購買傾向(LogReg / LightGBM)► propensity_scores.parquet
       └─ build_summaries  產品/月份/國家彙總 ─► product/monthly/country_summary.parquet
                                   │
                                   ▼
        load_db ──► PostgreSQL(正式)/ SQLite(本機・測試)
                                   │
                                   ▼
              FastAPI 服務(含 grounded LLM Copilot,/docs)
                                   │
                                   ▼
                  Streamlit 互動儀表板(呼叫 API)
```

**核心設計原則:LLM 與數值解耦(grounding / 防幻覺)**——所有 KPI、分群、CLV、lift、傾向分數等數字一律由 Python/SQL 算好,LLM 只把已算好的事實翻成繁體中文敘述,輸出以 Pydantic 驗證並保留可追溯的 grounding。

---

## 商業問題

線上零售商手上有兩年、超過百萬筆的交易紀錄,但「資料」不等於「決策」。本專案要回答:

> **如何用交易資料,把對的 offer 給對的客群,提升再購率與顧客終身價值(LTV)?**

拆成四個可操作的子問題,對應後續各 Phase:

1. **客群長什麼樣?** 誰是高價值、誰快流失、誰是新客 → 客群分群(RFM + K-means)
2. **每位客戶未來值多少錢?** → 預測型 CLV(BG/NBD + Gamma-Gamma)
3. **下一個該推什麼商品?** → 購物籃關聯規則 → Next Best Offer
4. **誰最可能在 90 天內再購?** → 傾向模型(LogReg / LightGBM)

最後把結果包成 FastAPI 服務與 Streamlit 儀表板,並加一層 **LLM 洞察 Copilot**——數字一律由 Python 算好,LLM 只負責翻成商業摘要(grounded、防幻覺)。

---

## 資料與清理(Phase 0)

**資料集**:Online Retail II(UCI id=502 / Kaggle,CC BY 4.0)。英國線上禮品商 2009–2011 交易,約 107 萬筆,同時含金額 + 客戶 ID + 重複購買,足以一份資料完成 RFM / CLV / 購物籃分析。

**清理規則**(見 `src/consumer_intel/data/clean.py`,每條都有對應 pytest):

| 步驟 | 移除列數 | 理由 |
|---|---:|---|
| 完全重複列 | 34,335 | 重複匯出 |
| 取消單(`Invoice` 開頭 `C`) | 19,104 | 非實際銷售 |
| 非商品代碼(POST、M、BANK CHARGES…) | 4,791 | 郵資／手續費等管理用代碼,非商品 |
| 非正數量(退貨／調整) | 3,362 | 客戶層正向銷售才進主表 |
| 非正單價(免費品／錯誤) | 2,566 | 金額不可信 |
| 缺 `CustomerID` | 226,636 | 無法歸戶,做不了客戶層分析 |

> 清理後**保留 776,577 / 1,067,371 列(72.8%)**,存成 `data/processed/transactions_clean.parquet`,下游一律讀清理後的檔。
> 設計取向:主表只保留「可歸戶的正向銷售」,作為 RFM / CLV / 購物籃的基礎;退貨與取消若日後要算淨營收,可由原始檔重新推導。

---

## 關鍵洞察(Phase 0:資料輪廓)

完整見 `reports/eda_phase0.md`。

- **規模**:776,577 筆明細、**5,852** 位客戶、36,594 筆訂單、4,619 項商品、41 國,總營收約 **£17.1M**。
- **高度集中於英國**:UK 佔營收 **83.7%**;EIRE 名列第二但僅 3 位客戶貢獻 528 筆訂單——**典型批發／內部帳戶**,分群與 CLV 時需留意這類離群值。
- **客戶價值極度右偏**:中位客戶累積消費約 £856,但前 1% 達 £28,631、最高一位逾 £580,000。平均值(£2,917)被少數大戶拉高——**用平均描述客戶會誤導,這正是要做分群與機率型 CLV 的原因**。
- **暢銷品**:以 REGENCY CAKESTAND、WHITE HANGING HEART T-LIGHT HOLDER、JUMBO BAG 系列等禮品／家居小物為主,購物籃分析應有足夠共購訊號。

## 關鍵洞察(Phase 1:客群分群)

對 5,852 位客戶算 RFM,再用兩種互補的方法分群(完整見 `reports/phase1_segmentation.md`)。

- **規則式 RFM(11 個行銷語意分群)**:**Champions 佔 19% 客戶卻貢獻 65.5% 營收**(人均 £10,030)——清楚的 80/20 結構,行銷預算重點不言而喻。另有 **At Risk + Can't Lose Them** 共 543 位高價值但已沉睡的客戶(平均 342～491 天未購),是 win-back 的首要名單。
- **K-means(資料驅動,k=4,以 silhouette 在 3–8 區間選定)**:得到 High-Value Active / High-Value Lapsing / Recent Low-Value / Dormant Low-Value 四群;**High-Value Active 佔 20% 客戶、73% 營收**。
- **兩種方法互相驗證**:都指向「約 1/5 的客戶撐起約 2/3 以上營收」。規則式給的是可直接溝通的行銷標籤,K-means 給的是資料自身的結構——儀表板會同時呈現兩種視角。

每個分群都附了「該怎麼做」(見報告的 action 表),把分析連回可執行的行銷決策。

## 關鍵洞察(Phase 2:CLV)

對 5,852 位客戶估計未來 3 個月的預測型 CLV(完整見 `reports/phase2_clv.md`)。

- **方法**:歷史 CLV 當基準,再用 **BG/NBD 估未來購買次數 + Gamma-Gamma 估每筆金額** 算機率型 CLV。Gamma-Gamma 需要重複購買,所以只在 **4,179 位重複客** 上 fit;1,673 位一次性買家以「BG/NBD 預測次數 × 母體平均客單價」做 fallback,並用 `clv_method` 欄位標記,兩族群不混用。
- **模型驗證(重點)**:用 calibration/holdout 切分(訓練 ≤ 2011-06-12、測試 180 天),預測 vs 實際購買次數**相關係數 0.848**,總量預測 7,549 對實際 7,717(**僅低估 2.2%**)。Gamma-Gamma 的「次數與金額不相關」假設也成立(相關 0.02)。展示的是**會驗證的機率方法**。
- **接回分群**:Champions 平均預測 CLV £1,346、存活機率 0.99;**Can't Lose Them 的平均存活機率只有 0.38**——模型獨立印證了這群高價值客正在流失。

> 小註:BG/NBD 對「只買過一次」的客戶會給存活機率 = 1(模型定義中,還沒重購就無從「流失」),所以規則式的「Lost」與 BG/NBD 的 prob_alive 可能不一致——兩者衡量的是不同東西,這是模型特性而非錯誤。

## 關鍵洞察(Phase 3:購物籃分析 → Next Best Offer)

對 36,594 筆訂單做關聯規則(完整見 `reports/phase3_basket_nbo.md`)。

- **方法**:先把長尾稀有商品剪掉(只留出現在 ≥1% 訂單的 562 項),用稀疏矩陣 + **FP-Growth** 挖頻繁項集,再算 support / confidence / **lift**,共得 **944 條規則**。
- **規則抓到真實的商品組合**:最高 lift 是同系列收藏品(Poppy's Playhouse 房間系列、點點派對紙餐具、點點杯藍↔粉)。lift 高代表「一起買的機率遠超隨機」,正是交叉銷售訊號。
- **Next Best Offer**:輸入一個商品,引擎依 lift 回傳最該推的下一個商品,並自動排除已在購物籃內的品項。推薦結果都落在同主題/同系列,符合 NBO 的商業直覺。

> 公開資料沒有實驗組,所以這裡只做關聯(association),不宣稱因果——「推了會不會真的提升銷售」需要 A/B test。

## 關鍵洞察(Phase 4:購買傾向預測)

預測「客戶會不會在未來 90 天內再次購買」(完整見 `reports/phase4_propensity.md`)。

- **無洩漏的時間切分**:取一個 cutoff(資料最後一天往前 90 天),**特徵只用 cutoff 之前的交易**,標籤是 cutoff 之後 90 天內是否再購;只有 cutoff 前有購買紀錄的客戶才進模型(5,256 位,正樣本率 43.6%)。這個切分是整個 phase 的重點——錯了就會得到漂亮但造假的分數。
- **特徵**:9 個來自 RFM + 行為的快照特徵(recency、frequency、monetary、tenure、平均客單價、平均購買間隔…),全部無 NaN。
- **誠實的模型比較**:Logistic 回歸 baseline 的 **ROC-AUC 0.804 / PR-AUC 0.781**,**略勝** LightGBM(0.784 / 0.759)。這是個成熟的結果而非失敗——在這個規模、又用了精心設計的特徵時,訊號大致是線性的,樹模型的額外複雜度沒帶來好處。**baseline 存在的意義就是讓樹模型必須「贏得」它的位置**;這裡它沒贏,就誠實寫出來。
- **可解釋性(SHAP)**:最重要的特徵是 recency、recency/tenure 比、平均購買間隔、monetary——「多近、多規律地買」最能預測會不會再買。
- **校準(calibration)**:預測機率與實際再購比率大致吻合,代表分數可當「機率」用於 targeting,而非只是排序。

> 各 pipeline 會在 `reports/` 產生 Markdown 報告(已進版控)與互動式 HTML 圖表(`.html`,未進版控,執行後產生)。

---

## Phase 5:產品化

把前四階段的產出整合成一個小型後端系統。

- **資料層(SQL)**:各階段產出載入 5 張表——`customers`(RFM/segment/CLV/propensity 合併)、`rules`、`products`、`monthly`、`country`。`db/schema.sql` 是 PostgreSQL DDL,`sql/analytics.sql` 放分析查詢。API 的查詢用參數化 SQL(`db/repository.py`),ANSI 相容,**同一份程式碼在 PostgreSQL(正式)與 SQLite(本機/測試)都能跑**。
- **API(FastAPI)**:12 個端點(見下),Pydantic 定義 response model,互動文件在 `/docs`(Swagger UI)。
- **grounded LLM Copilot**:**所有數字由 SQL/Python 算好**,風險等級也是 Python 依 `prob_alive` 分級;LLM 只把事實翻成**繁體中文**敘述。敘述層用 **LangChain**:`ChatPromptTemplate | model.with_structured_output(NarratedInsight)`,由 LangChain 負責 prompt 組裝與**結構化輸出驗證**,其 schema 只含可敘述的文字欄位,LLM 碰不到 grounded 數字。provider-agnostic(`init_chat_model`,以環境變數選 OpenAI/Anthropic);沒設 key 或呼叫失敗時自動退回**確定性模板**,所以離線/CI 也能跑。
- **容器化與部署**:`Dockerfile` + `docker-compose.yml`(Postgres + API + Streamlit),`render.yaml` 為 Render 的一鍵 Blueprint。
- **前端**:`app/dashboard.py` 是薄客戶端,只呼叫 API 並呈現。

## 互動儀表板(4 個分頁)

- **客群總覽**:總客戶/營收/平均 CLV/存活機率 KPI、各客群營收與占比、「客戶數 × 平均 CLV」泡泡圖、預估 CLV 最高的客戶。
- **客戶分析**:輸入或點選客戶 → RFM 概況、存活機率/回購傾向儀表、**繁中 AI 洞察**(含流失風險標籤);附消費前 100 名的客戶瀏覽表(可點選查詢)。
- **產品分析**:商品總數/總營收 KPI、營收 Top 15、營收 × 數量泡泡、商品瀏覽表(可點選),以及單一商品明細 + 下一步最佳推薦。
- **趨勢與地區**:月營收趨勢、月訂單/客戶數,以及**各國營收世界地圖**(可切換上色指標、可排除英國)。

## API 端點

| 方法 | 路徑 | 說明 |
|---|---|---|
| GET | `/health` | 服務狀態 + 客戶數 |
| GET | `/customers` | 客戶清單(瀏覽,依消費排序) |
| GET | `/customers/top-clv` | 預估 CLV 最高的客戶 |
| GET | `/customers/{id}` | 單一客戶的 RFM/CLV/傾向完整檔 |
| GET | `/customers/{id}/insight` | grounded LLM 洞察(Copilot) |
| GET | `/segments` | 各客群彙總 |
| GET | `/products` | 商品清單(依營收排序) |
| GET | `/products/{code}` | 單一商品彙總 |
| GET | `/products/{code}/next-best-offer` | 交叉銷售推薦 |
| GET | `/analytics/products-overview` | 全體商品總數/總營收 |
| GET | `/analytics/monthly` | 月營收/訂單/客戶趨勢 |
| GET | `/analytics/countries` | 各國營收/訂單/客戶 |

---

## 如何執行

```bash
# 安裝(serving 所需 extras;加 dev 跑測試)
pip install -e ".[api,app,copilot,dev]"
```

### 快速啟動(用 repo 內附的彙總檔,免下載原始資料)

repo 已附上各階段的彙總 parquet,clone 後可直接起服務:

```bash
python scripts/load_db.py                       # 載入彙總到 SQLite(預設)
uvicorn consumer_intel.api.app:app --reload     # API + 文件 /docs
streamlit run app/dashboard.py                  # 前端(另一個終端)
```

API 文件在 http://localhost:8000/docs;前端在 http://localhost:8501。

### 從原始資料完整重現

```bash
# 1. 下載 Online Retail II CSV 放到 data/raw/online_retail_II.csv(UCI id=502 或 Kaggle)
# 2. 依序跑各階段 pipeline(產生 transactions_clean 與各客戶/規則 parquet)
python scripts/run_phase0.py
python scripts/run_phase1.py
python scripts/run_phase2.py
python scripts/run_phase3.py
python scripts/run_phase4.py
# 3. 產生產品/月份/國家彙總,再載入資料庫
python scripts/build_summaries.py
python scripts/load_db.py
```

### Docker(Postgres + API + 前端一鍵啟動)

```bash
docker compose up
```

### 啟用真實 LLM 敘述(否則用確定性模板)

於 API 服務設環境變數 `OPENAI_API_KEY` 或 `ANTHROPIC_API_KEY`(可選 `LLM_MODEL`、`LLM_PROVIDER`)。未設定時 Copilot 會以確定性模板輸出,功能照常。

### 測試與 lint

```bash
pytest                          # 91 個測試
ruff check . && ruff format .
```

---

## 部署

`render.yaml` 是 Render 的 Blueprint:一次建立 managed PostgreSQL + API 服務 + Streamlit 服務(皆以同一 Docker 映像、不同啟動腳本)。API 開機時先 `load_db.py` 把彙總載入 Postgres 再啟動;前端服務以 `API_URL` 指向 API。

> 註:雲端免費方案常有閒置休眠(首次請求較慢)與資料庫期限等限制,且各家條款時有調整,部署前請查當下文件。


---

## 測試與工程

- **91 個 pytest**:涵蓋清理規則、RFM、分群、CLV、關聯規則、傾向特徵與模型、SQL repository、FastAPI 端點、Copilot 的 grounding 與 Pydantic 約束。
- repository 對 **SQLite**、API 用 FastAPI `TestClient`、Copilot 走確定性模板路徑——**全程不需 PostgreSQL 或真實 LLM key**。
- `ruff` 負責 lint 與格式化;GitHub Actions(`.github/workflows/ci.yml`)在每次 push/PR 跑 lint + 測試。

---

## 專案結構

```
consumer-intelligence/
├── README.md · CLAUDE.md · PROJECT_PLAN.md
├── pyproject.toml
├── data/processed/               # 各階段彙總 parquet(小檔已進版控;transactions_clean 未進版控)
├── src/consumer_intel/
│   ├── config.py                 # 路徑與清理常數
│   ├── labels.py                 # 中文顯示標籤(後端/前端共用)
│   ├── data/                     # 載入 + 清理
│   ├── eda/                      # EDA 摘要
│   ├── features/                 # RFM 特徵
│   ├── segmentation/             # RFM 規則式分群 + K-means
│   ├── clv/                      # 歷史 + BG/NBD + Gamma-Gamma、holdout 驗證
│   ├── basket/                   # FP-Growth 關聯規則 + Next Best Offer
│   ├── propensity/               # 傾向:特徵、LogReg/LightGBM、SHAP
│   ├── db/                       # SQLAlchemy engine、loader、SQL repository
│   ├── copilot/                  # grounded LLM Copilot(context/narrator/schema)
│   └── api/                      # FastAPI app、Pydantic models、DB 依賴
├── app/dashboard.py              # Streamlit 前端(呼叫 API)
├── db/schema.sql                 # PostgreSQL DDL
├── sql/analytics.sql             # 分析查詢(展示 SQL)
├── scripts/
│   ├── run_phase0.py ~ run_phase4.py   # 各階段 pipeline
│   ├── build_summaries.py        # 產品/月份/國家彙總
│   ├── load_db.py                # 載入彙總進資料庫
│   └── start_api.sh · start_dashboard.sh   # 容器啟動腳本
├── Dockerfile · docker-compose.yml · render.yaml
├── .streamlit/config.toml        # 前端主題
├── reports/                      # 各階段 Markdown 報告(.html 圖表未進版控)
├── tests/                        # 對應各模組的 pytest
└── .github/workflows/ci.yml      # lint + test
```
