"""Pydantic schemas for the conversational customer-insight graph.

Two LLM-only extraction points, same grounding contract as everywhere else
in this project: the LLM fills these fields and nothing else — no numbers,
no segment/risk decisions, those stay Python's.

* ``ExtractedContext`` — which customer this turn is about, resolved from
  conversation history (including pronouns like "他"/"那"). Purely a
  routing signal; contains no facts about the customer.
* ``ChatAnswer`` — the free-text reply to whatever the user actually asked,
  grounded in the already-fetched ``tool_results`` handed to the prompt as
  context. The LLM may only restate/explain those facts, never invent or
  recompute one.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ExtractedContext(BaseModel):
    customer_id: str | None = Field(
        default=None,
        description=(
            "The customer ID this turn is about, resolved from the conversation "
            "(including pronouns referring to a previously mentioned customer). "
            "Null if genuinely unclear which customer is meant."
        ),
    )


class ChatAnswer(BaseModel):
    answer: str = Field(
        min_length=1,
        max_length=800,
        description="Grounded, Traditional Chinese answer to the user's latest question",
    )
