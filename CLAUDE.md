# CLAUDE.md

給 Claude Code 的專案說明。開始任何工作前先讀完本檔；詳細階段規劃見 `PROJECT_PLAN.md`。

## 專案是什麼

Consumer Intelligence 分析專案：用一份真實電商交易資料，做端到端的消費者智慧分析——**客群分群、CLV、Next Best Offer、購買預測**，最後產品化為 Streamlit dashboard，並加一層 **LLM 洞察 Copilot**。這是個人 portfolio 專案，目標是展示「把資料 + ML + LLM 連結到商業價值並產品化」的能力。

## 核心設計原則（最重要，務必遵守）

**LLM 與數值計算解耦（grounding / 防幻覺）**
- 所有 KPI、分群統計、CLV、lift、傾向分數等**數字一律由 Python 計算**並以結構化物件傳遞。
- **LLM 只負責把已算好的結構化結果翻成自然語言**，絕不參與運算、絕不自己產生數字。
- LLM 的輸出一律用 **Pydantic（`Literal` 型別 + `Field` 約束）** 驗證；不合法的輸出要被擋下並回傳友善錯誤，不可讓系統崩潰。
- 任何 LLM 回應都要能對應回它所依據的工具結果（保留可追溯的 trace）。

> 若你（Claude Code）發現自己想讓 LLM「順便算一下」某個數字——停下來，改成在 Python 算好再傳給它。

## 技術棧與慣例

- Python 3.11+。全面使用 type hints。
- 套件：pandas、numpy、scikit-learn、lightgbm、xgboost、shap、lifetimes、mlxtend、plotly、streamlit、pydantic、openai／anthropic。
- **業務邏輯放 `src/consumer_intel/`，notebook 只做探索**，不可把核心邏輯留在 notebook。
- 函式小而單一職責，有 docstring；偏好純函式、易測試。
- 格式化與 lint 用 `ruff`。
- LLM 串接保持 **provider-agnostic**（OpenAI／Anthropic 可切換）。
- **不要為了用而用重量級框架**：除非有明確理由，否則不引入 LangChain／LlamaIndex；orchestration 以明確、可讀的自寫流程為主（這是刻意的設計取向）。

## 專案結構

```
src/consumer_intel/
  data/          # 載入 + 清理
  features/      # RFM、特徵工程
  segmentation/  # RFM 分數 + K-means
  clv/           # BG/NBD + Gamma-Gamma
  basket/        # 關聯規則 + Next Best Offer
  propensity/    # 購買／再購預測
  copilot/       # LLM 洞察層（數字僅來自 Python）
  viz/           # 繪圖輔助
app/dashboard.py # Streamlit
tests/           # 每個模組對應一份 pytest
```

## 常用指令

```bash
pip install -e ".[dev]"          # 安裝（含 dev 相依）
pytest                           # 跑測試
ruff check . && ruff format .    # lint + 格式化
streamlit run app/dashboard.py   # 啟動 dashboard
```

## 測試要求

- 每個分析函式都要有對應的 pytest 單元測試（尤其是 RFM、CLV、關聯規則、特徵工程這類純計算）。
- 新增或修改計算邏輯時，**同一個 PR 內要補上／更新測試**。
- 評估指標（AUC、PR-AUC、calibration）要寫進測試或 reproducible 的腳本，不可只存在 notebook。

## 資料注意事項（Online Retail II）

- 取得：`from ucimlrepo import fetch_ucirepo; fetch_ucirepo(id=502)`，或 Kaggle CSV。
- 清理重點：
  - `InvoiceNo` 開頭為 `C` = 取消單，需排除或單獨處理。
  - `Quantity` 為負 = 退貨。
  - 缺 `CustomerID` 的列無法做客戶層分析，需處理。
  - 部分客戶是批發大戶，是金額離群值，分群時要留意。
- 清理後存成 `data/processed/*.parquet`，下游一律讀清理後的檔。

## 名詞定義（保持一致用語）

- **RFM**：Recency（最近一次購買距今）、Frequency（購買次數）、Monetary（累積金額）。
- **CLV**：客戶終身價值；本專案以 BG/NBD（預測購買次數）+ Gamma-Gamma（預測金額）估計預測型 CLV。
- **NBO（Next Best Offer）**：依購物籃關聯規則（lift／confidence）推薦的下一個商品。
- **Propensity**：客戶在未來時間窗內發生某行為（如再購）的機率。

## 不要做的事

- 不要在報告／Copilot 中產生未經 Python 計算的數字。
- 不要宣稱做了 uplift／incrementality／因果效果——本資料**沒有實驗組／對照組**。只能在 README 以方法論層次討論「若有 treatment 資料會如何做」。
- 不要把核心邏輯留在 notebook。
- 不要在沒有測試的情況下合併計算邏輯的變更。
