"""LangGraph agentic workflow for the Consumer Intelligence Copilot.

Sits alongside (not instead of) ``consumer_intel.copilot``, which remains the
plain LangChain LCEL implementation used by the existing
``/customers/{id}/insight`` endpoint. This package adds the pieces LCEL
alone can't express well: parallel tool fan-out, conditional routing,
human-in-the-loop approval, and multi-turn conversation state — see
``PROMPT_langgraph_copilot.md`` for the full phased plan.
"""

from __future__ import annotations
