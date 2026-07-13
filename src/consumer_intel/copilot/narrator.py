"""把已算好的事實轉成商業敘述(繁體中文)。

兩種後端:不需網路的**模板**敘述,以及由 **LangChain** orchestration 的 LLM 敘述。
LLM 只負責「把事實用繁體中文講出來」並回傳結構化的 NarratedInsight(由
with_structured_output 驗證),碰不到 grounded 的數字、客群或風險等級——那些一律
由 Python 算好再組裝。沒設 provider key 或呼叫失敗時退回模板,確保離線/CI 可跑。

語言由 INSIGHT_LANGUAGE 環境變數控制,預設繁體中文。
provider 由 init_chat_model 以 LLM_PROVIDER／LLM_MODEL(或依 API key 推斷)選擇。
"""

from __future__ import annotations

import os

from consumer_intel.copilot.schema import CustomerInsight, InsightContext
from consumer_intel.labels import action_zh, risk_zh, segment_zh

Backend = str  # "auto" | "template" | "langchain"

INSIGHT_LANGUAGE = os.environ.get("INSIGHT_LANGUAGE", "繁體中文")

_SYSTEM = (
    "你是一位零售數據分析助理。你會收到關於某位顧客的『已計算好的事實』。"
    f"請用{INSIGHT_LANGUAGE}把這些事實改寫成簡潔的商業洞察,輸出 headline(字串)、"
    "observations(字串陣列)與 recommended_actions(字串陣列)。"
    "不要捏造或重新計算任何數字,只能根據提供的事實敘述。"
    "客群名稱請使用事實中的 segment_zh(中文)。"
)


def _money(x: float) -> str:
    return f"£{x:,.0f}"


def narrate_template(ctx: InsightContext) -> dict:
    """直接由事實產生的確定性敘述(不經 LLM),繁體中文。"""
    seg = segment_zh(ctx.segment)
    headline = (
        f"「{seg}」客群|流失風險{risk_zh(ctx.risk_level)}|預估價值 {_money(ctx.predicted_clv)}"
    )
    observations = [
        f"最近一次購買在 {ctx.recency_days} 天前,共 {ctx.frequency} 筆訂單,"
        f"累積消費 {_money(ctx.monetary)}。",
        f"仍為活躍客戶的機率:{ctx.prob_alive:.0%}。",
    ]
    if ctx.propensity is not None:
        observations.append(f"模型預估 90 天回購傾向:{ctx.propensity:.0%}。")

    actions: list[str] = []
    mapped = action_zh(ctx.recommended_action)
    if mapped:
        actions.append(mapped)
    if ctx.next_best_offers:
        actions.append("交叉銷售推薦:" + "、".join(ctx.next_best_offers[:3]) + "。")
    if not actions:
        actions = ["暫無特別建議,持續觀察即可。"]

    return {"headline": headline, "observations": observations, "recommended_actions": actions}


def _chat_model():
    """依環境變數建立 provider-agnostic 的 LangChain chat model。"""
    provider = os.environ.get("LLM_PROVIDER")
    model = os.environ.get("LLM_MODEL")
    if not provider:
        if os.environ.get("ANTHROPIC_API_KEY"):
            provider = "anthropic"
        elif os.environ.get("OPENAI_API_KEY"):
            provider = "openai"
        else:
            raise RuntimeError("尚未設定 LLM provider(請設 OPENAI/ANTHROPIC key)。")
    if not model:
        # 目前(2026 年中)的便宜模型;若被淘汰請用 LLM_MODEL 覆蓋。
        model = "claude-haiku-4-5" if provider == "anthropic" else "gpt-4.1-mini"

    from langchain.chat_models import init_chat_model

    return init_chat_model(model, model_provider=provider)


def narrate_langchain(ctx: InsightContext) -> dict:
    """由 LangChain orchestration 的 LLM 敘述,回傳已驗證的繁體中文文字。

    以 prompt | model.with_structured_output(NarratedInsight) 串接,讓輸出被 LangChain
    解析並符合 schema。無 provider/key 時會丟出例外(由呼叫端退回模板)。
    """
    import json

    from langchain_core.prompts import ChatPromptTemplate

    from consumer_intel.copilot.schema import NarratedInsight

    facts = ctx.model_dump()
    facts["segment_zh"] = segment_zh(ctx.segment)

    model = _chat_model()
    prompt = ChatPromptTemplate.from_messages(
        [("system", _SYSTEM), ("human", "事實(JSON):\n{facts}\n\n請撰寫洞察。")]
    )
    chain = prompt | model.with_structured_output(NarratedInsight)
    narrated: NarratedInsight = chain.invoke({"facts": json.dumps(facts, ensure_ascii=False)})
    return {
        "headline": narrated.headline,
        "observations": narrated.observations,
        "recommended_actions": narrated.recommended_actions,
    }


def _resolve_backend(backend: Backend) -> str:
    """決定實際後端;'auto' 僅在有 key 時用 LangChain,否則用模板。"""
    if backend != "auto":
        return backend
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY"):
        return "langchain"
    return "template"


def generate_insight_with_backend(
    ctx: InsightContext, backend: Backend = "auto"
) -> tuple[CustomerInsight, str]:
    """同 :func:`generate_insight`,但額外回報實際生效的敘述來源。

    回傳值為 ``"template"``(本來就選模板)、``"langchain"``(LLM 呼叫成功)
    或 ``"template_fallback"``(想用 LLM 但呼叫失敗,退回模板)。給
    copilot_graph 用來把原本隱性的失敗路徑變成可觀測的顯式節點;
    ``generate_insight`` 的對外行為與簽名完全不變。
    """
    chosen = _resolve_backend(backend)
    used = chosen
    try:
        parts = narrate_template(ctx) if chosen == "template" else narrate_langchain(ctx)
    except Exception:
        parts = narrate_template(ctx)
        used = "template_fallback"

    insight = CustomerInsight(
        customer_id=ctx.customer_id,
        segment=ctx.segment,
        risk_level=ctx.risk_level,
        headline=parts["headline"],
        observations=parts["observations"],
        recommended_actions=parts["recommended_actions"],
        grounding=ctx.model_dump(),
    )
    return insight, used


def generate_insight(ctx: InsightContext, backend: Backend = "auto") -> CustomerInsight:
    """由 grounded 事實產生已驗證的 CustomerInsight。

    id、客群、風險等級與 grounding 皆來自 Python,只有自由文字由敘述層產生;
    任何 LLM/LangChain 失敗都會退回模板。
    """
    insight, _used = generate_insight_with_backend(ctx, backend)
    return insight
