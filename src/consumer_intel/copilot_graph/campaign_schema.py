"""Pydantic schema for the win-back campaign's LLM-generated copy.

Mirrors ``consumer_intel.copilot.schema.NarratedInsight``: the LLM may only
fill these free-text fields. Every number in the campaign (customer count,
CLV, discount) is computed in Python and handed to the prompt as already-
resolved context — none of it lives in this schema, so there's nothing
numeric for the model to get wrong.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class CampaignBrief(BaseModel):
    headline: str = Field(min_length=1, max_length=120, description="One-line campaign title")
    message: str = Field(min_length=1, max_length=600, description="Marketing angle / copy")
    selling_points: list[str] = Field(min_length=1, max_length=6, description="Key selling points")
