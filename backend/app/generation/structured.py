"""Schema-constrained financial answer generation with one bounded repair attempt."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.core.logging import get_logger
from app.generation.prompts import build_answer_messages

logger = get_logger(__name__)


class Comparison(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subject: str = Field(min_length=1, max_length=200)
    basis: str = Field(min_length=1, max_length=200)
    finding: str = Field(min_length=1, max_length=1000)


class Citation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_index: int = Field(ge=1)
    claim: str = Field(min_length=1, max_length=1000)


class StructuredAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    conclusion: str = Field(min_length=1, max_length=4000)
    facts: list[str] = Field(default_factory=list, max_length=20)
    comparisons: list[Comparison] = Field(default_factory=list, max_length=10)
    citations: list[Citation] = Field(default_factory=list, max_length=30)
    caveats: list[str] = Field(default_factory=list, max_length=10)
    confidence: float = Field(ge=0.0, le=1.0)


_STRUCTURED_SYSTEM = """Return one JSON object only, with this exact schema:
{
  "conclusion": "grounded conclusion",
  "facts": ["directly supported fact"],
  "comparisons": [{"subject":"item", "basis":"comparison basis", "finding":"result"}],
  "citations": [{"source_index":1, "claim":"claim supported by that source"}],
  "caveats": ["missing evidence or scope limit"],
  "confidence": 0.0
}
Use source_index values from the numbered knowledge blocks only. Never invent a source,
number, company, period, or causal explanation. confidence is evidence completeness, not
model self-confidence. Empty arrays are allowed; unknown facts must be caveats.
"""


async def generate_structured_answer(
    llm: Any,
    question: str,
    blocks: list[dict],
    *,
    history: list[dict] | None = None,
) -> StructuredAnswer | None:
    messages = build_answer_messages(question, blocks, history=history)
    messages.insert(1, {"role": "system", "content": _STRUCTURED_SYSTEM})
    for attempt in range(2):
        try:
            raw = await llm.chat(messages, response_format={"type": "json_object"})
            answer = StructuredAnswer.model_validate_json(raw)
            if any(citation.source_index > len(blocks) for citation in answer.citations):
                raise ValueError("citation source_index is outside supplied blocks")
            return answer
        except (ValidationError, ValueError, json.JSONDecodeError) as exc:
            logger.warning(
                "structured answer validation failed: attempt=%d type=%s",
                attempt + 1,
                type(exc).__name__,
            )
            if attempt == 0:
                messages.append(
                    {
                        "role": "user",
                        "content": "The previous output failed schema validation. Return a corrected JSON object only.",
                    }
                )
    return None


def structured_payload(answer: StructuredAnswer, blocks: list[dict]) -> dict:
    payload = answer.model_dump()
    for citation in payload["citations"]:
        source = blocks[citation["source_index"] - 1]
        citation["source"] = {
            "doc_id": source.get("doc_id"),
            "doc_name": source.get("doc_name"),
            "section_path": source.get("section_path"),
            "page": source.get("page"),
        }
    return payload


def render_structured_answer(answer: StructuredAnswer) -> str:
    lines = [answer.conclusion]
    if answer.facts:
        lines.extend(["", "事实：", *[f"- {fact}" for fact in answer.facts]])
    if answer.comparisons:
        lines.extend(
            [
                "",
                "比较：",
                *[
                    f"- {item.subject}（{item.basis}）：{item.finding}"
                    for item in answer.comparisons
                ],
            ]
        )
    if answer.caveats:
        lines.extend(["", "限制：", *[f"- {item}" for item in answer.caveats]])
    return "\n".join(lines)
