# LangGraph Copilot 設計筆記

本文件是 README「LangGraph Copilot」一節的延伸，記錄完整的 state graph 設計、SSE 串流細節、win-back campaign 的 human-in-the-loop 流程、效能量測，以及與既有 LCEL 實作的比較。持久化分層（checkpointer vs. SQLAlchemy ORM）與「為何保留 LCEL、不直接改寫」的設計理由已寫在 [CLAUDE.md](../CLAUDE.md)，本文不重複。

---

## 1. 客戶洞察 StateGraph

```text
extract_context
  →（customer_id 可解析？）→ clarify ／ router
      →（客戶存在？）→ not_found
      →（存在）→ fetch_rfm ‖ fetch_clv ‖ fetch_nbo ‖ fetch_propensity（平行）
          → join → response_generator →（LLM 失敗時）→ fallback
```

- **`extract_context`**：從對話歷史（含代名詞，例如「他」「那位客戶」）解析出這一輪在問哪位客戶。多輪對話狀態存在 SQLAlchemy `messages` 表，由 `copilot_graph/chat.py` 的 `run_turn()` 每輪重建——這個 graph 本身**不使用 checkpointer**（客戶洞察是無狀態查詢，跨輪記憶靠 ORM 讀寫，不需要「從中斷點恢復」的能力）。
- **`clarify` / `not_found`**：客戶身分無法判斷、或查無客戶時，直接回傳確定性訊息，不進 LLM——這類情況沒有值得敘述的事實，讓 LLM 介入只會增加幻覺風險。
- **四個 `fetch_*` 節點平行執行**（RFM、CLV、Next Best Offer、propensity），fan-in 後由 `response_generator` 產生**對話式**的 grounded 回答（純文字，不是固定欄位的 `CustomerInsight` 結構——這是與 `copilot/`〔LCEL〕洞察端點的關鍵差異，後者回傳結構化 schema）。
- **`fallback`**：LLM 呼叫失敗會經過顯式的 `fallback` 節點退回模板，不是被外層 try/except 靜默吞掉。

## 2. 多輪對話 + SSE 串流

`GET /chat/stream?thread_id=...&message=...`，以 `graph.astream_events(...)` 轉成 SSE（`text/event-stream`）。事件經過 `copilot_graph/streaming.py` 篩選成精簡格式（`node_start`／`node_end`／`token`／`final`／`interrupt`），不直接把 LangChain 原始事件（含完整 state、非 JSON 的訊息物件）丟給前端。

前端 Streamlit 的「Copilot 對話」分頁以 `st.chat_input` 驅動，每個瀏覽器 session 對應一個 `thread_id`，可連續追問（例如「12345 這位客戶如何？」→「他為什麼被歸為 At Risk？」→「那該給他什麼 offer？」），不需要每次都重講一次客戶編號。

## 3. 平行 fan-out 的 benchmark

`python scripts/benchmark_copilot_graph.py`（30 位客戶、本機 SQLite、確定性模板路徑，排除 LLM 呼叫變異）：

| 版本 | p50 (ms) | p95 (ms) | mean (ms) |
|---|---:|---:|---:|
| 序列（逐一呼叫四次查詢） | 2.23 | 3.26 | 2.35 |
| LangGraph 平行 fan-out | 4.09 | 6.41 | 4.32 |

**誠實的結論：在本機 SQLite 上，平行版本反而較慢。** SQLite 查詢在毫秒等級、幾乎不涉及真正的網路 I/O 等待（GIL 不太會因此釋放），LangGraph 的執行緒調度與 superstep 管理開銷因此蓋過了任何平行化收益。這個 fan-out 模式預期在正式環境（跨網路的 PostgreSQL，每次查詢有真實往返延遲）會有實際效益，但本開發環境沒有 Docker/Postgres 可用，尚未量測——之後接上正式資料庫後可補上對照數字，不先估算或美化。數字會隨機器與負載波動，重跑 `scripts/benchmark_copilot_graph.py` 可自行驗證。

## 4. Win-back Campaign 產生器（Human-in-the-Loop）

**為什麼需要人工審核**：發折扣給真實客戶是有金錢成本、有品牌風險、且一旦寄出就不可逆的行為。這不是為了展示 LangGraph 的 `interrupt()` 技術而加的裝飾，而是這個動作本身的業務性質要求必須有人把關（詳見 CLAUDE.md「Human-in-the-Loop 的正當性在於業務，不在技術」）。

流程（`copilot_graph/campaign_graph.py`）：

```text
campaign_intent → build_candidates → match_offers → draft_campaign
  → persist_pending → await_approval(interrupt)
       ├─ 核准     → commit_campaign（寫入 DB）
       ├─ 退回修改 → draft_campaign（帶著審核意見與客戶名單編輯重新草擬）
       └─ 終結     → reject_campaign
```

- **候選名單**：`segment IN (At Risk, Can't Lose Them)`，對應 Phase 1 找到的 543 位高價值但已沉睡的客戶。
- **折扣建議**：依候選名單內 CLV 分位數分三層（前三分之一 20%、中間 15%、後三分之一 10%）——Python 純函式計算，可測試；LLM 完全不接觸個別客戶數字。
- **推薦商品**：沿用 Phase 1 的 `next_best_offers_for_customer`，依客戶最常購買商品配對關聯規則。
- **文案**：整個活動共用一段 LCEL 產生的文案（標題／訴求／賣點），LLM 只看 Python 算好的彙總統計（人數、平均 CLV、平均折扣），看不到任何個別客戶資料；無金鑰或呼叫失敗時退回確定性模板。
- **審核**：LangGraph `interrupt()` + `Command(resume=...)`，checkpointer 用 `SqliteSaver`（本機）／`PostgresSaver`（正式）；`campaign_approvals` 的審核紀錄（狀態、審核人、意見、時間）由 SQLAlchemy ORM 寫入，與 checkpointer 各自獨立（見 [CLAUDE.md](../CLAUDE.md)「資料持久化分層」）。

前端「待審核 Campaign」分頁：產生草稿、瀏覽／編輯候選客戶名單（可勾選剔除、修改個別折扣）、填寫審核意見、核准／退回修改／終結三選一；核准後可下載名單 CSV。

## 5. 一種 Copilot 實作的心得：LCEL vs. LangGraph

這裡原本想比較三種 orchestration（手刻／LCEL／LangGraph），但手刻版本是另一個舊專案，這次沒有帶著它的程式碼一起看，與其憑印象編資料，不如老實只比這個 repo 裡真的並存的兩種實作——數字是實際量出來的，不是估的。

| 面向 | LangChain LCEL（`copilot/`） | LangGraph（`copilot_graph/`） |
|---|---|---|
| 程式碼行數 | 275 行（3 個檔案） | 988 行（10 個檔案）+ 201 行資料層基礎設施（`checkpointer.py`／`campaign_repository.py`／`models.py`） |
| 測試數 | 13 個 | 55 個 |
| 並行支援 | 無，單次 `chain.invoke()` | 原生 fan-out（4 節點平行擷取）；但本機 SQLite 上因無真實網路 I/O，平行版本反而比序列慢（見上方 benchmark） |
| 執行持久化 | 無狀態，每次呼叫獨立 | 視流程而定：customer insight 對話歷史存 ORM、graph 本身不用 checkpointer；campaign 審核流程需要 checkpointer（`SqliteSaver`/`PostgresSaver`）才能 `interrupt()`/`resume()` |
| 中斷（恢復）HITL | 不支援，沒有機制可以「停下來等人」 | 原生支援，是這個專案唯一真的需要 HITL 的地方（win-back campaign 發真的折扣） |
| 可觀測性 | 黑盒；敘述失敗只能在外層包 try/except，看不到內部發生什麼 | `astream_events()` 給節點級別的開始／結束／token 事件；失敗路徑是圖上顯式的 `fallback` 節點，不是被吞掉的例外 |
| 學習曲線／維護成本 | 低，一個函式接一個函式，新人一看就懂 | 高，要理解 StateGraph、reducer、conditional edges、interrupt 語意、checkpointer 生命週期 |
| **什麼時候該用它** | 單一意圖、不需要中斷、不需要平行 I/O 的場景 | 需要 HITL、需要跨請求維持對話狀態、或有真正平行 I/O 可以重疊等待時間的場景 |

這個專案讓兩套敘述層在同一個資料庫、同一組事實上做同一件事，差異因此清楚：LCEL 產生一句話就夠用時，LangGraph 完全是額外成本——要多學 StateGraph、reducer、conditional edge，還會踩到不顯而易見的坑（例如 `interrupt()` 前的程式碼在 `resume()` 時會重跑，不注意會意外重複寫入 DB，這是先寫小實驗才發現的，不是憑經驗猜到的）。平行 fan-out 在本機 SQLite 上甚至比序列還慢，因為沒有真正的網路 I/O 可以重疊，執行緒調度的開銷反而倒貼一筆。真正回本的地方只有兩個：win-back campaign 需要對「發真的折扣給真的客戶」這種不可逆動作停下來等人核准，這是 LCEL 原生做不到的；以及多輪對話需要跨請求維持狀態、並可觀測每個節點的執行過程。沒有這兩個理由，LangGraph 只是用更貴的方式做同一件事。
