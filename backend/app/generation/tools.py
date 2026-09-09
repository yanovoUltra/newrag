"""Tenant-bound whitelist tools and a bounded OpenAI-compatible tool loop."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.core.logging import get_logger, safe_ref
from app.fields.metrics import METRIC_CATALOG
from app.store.registry import query_benchmark_fields, query_fields

logger = get_logger(__name__)

_METRIC_KEYS = {str(item["key"]) for item in METRIC_CATALOG}
SearchCallback = Callable[[str, int], Awaitable[list[dict[str, Any]]]]


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FieldQueryArgs(_Args):
    metric: str = Field(min_length=1, max_length=80)
    year: int | None = Field(default=None, ge=1990, le=2100)
    company: str | None = Field(default=None, max_length=200)
    limit: int = Field(default=10, ge=1, le=20)


class DocumentSearchArgs(_Args):
    query: str = Field(min_length=2, max_length=1000)
    limit: int = Field(default=5, ge=1, le=10)


class CalculationArgs(_Args):
    left: str = Field(max_length=80)
    operator: Literal["add", "subtract", "multiply", "divide", "percent_change"]
    right: str = Field(max_length=80)


class UnitConversionArgs(_Args):
    value: str = Field(max_length=80)
    from_unit: Literal["元", "万元", "亿元", "%", "bp"]
    to_unit: Literal["元", "万元", "亿元", "%", "bp"]


class CompanyCompareArgs(_Args):
    metric: str = Field(min_length=1, max_length=80)
    companies: list[str] = Field(min_length=2, max_length=5)
    years: list[int] = Field(min_length=1, max_length=5)


class ExportArgs(_Args):
    format: Literal["json", "csv"]
    title: str = Field(min_length=1, max_length=120)
    rows: list[dict[str, Any]] = Field(max_length=200)


TOOL_MODELS: dict[str, type[_Args]] = {
    "financial_field_query": FieldQueryArgs,
    "document_search": DocumentSearchArgs,
    "exact_calculation": CalculationArgs,
    "unit_conversion": UnitConversionArgs,
    "company_compare": CompanyCompareArgs,
    "export": ExportArgs,
}


def _tool(name: str, description: str, model: type[BaseModel]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": model.model_json_schema(),
        },
    }


TOOLS = [
    _tool("financial_field_query", "Query an authorized structured financial field.", FieldQueryArgs),
    _tool("document_search", "Search authorized report chunks in the current tenant.", DocumentSearchArgs),
    _tool("exact_calculation", "Perform exact decimal arithmetic without estimating missing inputs.", CalculationArgs),
    _tool("unit_conversion", "Convert supported financial units without exchange-rate conversion.", UnitConversionArgs),
    _tool("company_compare", "Compare one structured metric across authorized companies and years.", CompanyCompareArgs),
    _tool("export", "Prepare bounded JSON/CSV export data; never writes a server file.", ExportArgs),
]


@dataclass
class ToolContext:
    subject: str
    org_id: str
    visibility: str
    search: SearchCallback
    cache: dict[str, dict[str, Any]] = field(default_factory=dict)


def _decimal(value: str) -> Decimal:
    try:
        return Decimal(value.replace(",", "").strip())
    except InvalidOperation as exc:
        raise ValueError("invalid decimal input") from exc


async def execute_tool(name: str, raw_args: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    model = TOOL_MODELS.get(name)
    if model is None:
        raise ValueError("tool is not in the whitelist")
    args = model.model_validate(raw_args)
    canonical = json.dumps(args.model_dump(), ensure_ascii=False, sort_keys=True, default=str)
    if len(canonical.encode("utf-8")) > 50_000:
        raise ValueError("tool arguments are too large")
    idem = hashlib.sha256(
        f"{context.subject}\n{context.org_id}\n{name}\n{canonical}".encode("utf-8")
    ).hexdigest()
    if idem in context.cache:
        return context.cache[idem]
    logger.info("tool execute: name=%s actor=%s", name, safe_ref(context.subject))
    if isinstance(args, FieldQueryArgs):
        if args.metric not in _METRIC_KEYS:
            raise ValueError("unknown financial metric")
        rows = query_fields(
            [args.metric],
            context.org_id,
            context.visibility,
            year=args.year,
            company=args.company,
            limit=args.limit,
        )
        result = {
            "items": [
                {
                    "metric": row.metric,
                    "company": row.company,
                    "year": row.year,
                    "value": row.value,
                    "unit": row.unit,
                    "raw": row.raw,
                    "page": row.page,
                }
                for row in rows
            ]
        }
    elif isinstance(args, DocumentSearchArgs):
        result = {"items": await context.search(args.query, args.limit)}
    elif isinstance(args, CalculationArgs):
        left, right = _decimal(args.left), _decimal(args.right)
        if args.operator == "add":
            value = left + right
        elif args.operator == "subtract":
            value = left - right
        elif args.operator == "multiply":
            value = left * right
        elif args.operator == "divide":
            if right == 0:
                raise ValueError("division by zero")
            value = left / right
        else:
            if left == 0:
                raise ValueError("percent change base is zero")
            value = (right - left) / left * Decimal(100)
        result = {"value": format(value, "f")}
    elif isinstance(args, UnitConversionArgs):
        scales = {"元": Decimal(1), "万元": Decimal(10_000), "亿元": Decimal(100_000_000)}
        rates = {"%": Decimal(1), "bp": Decimal("0.01")}
        family = scales if args.from_unit in scales and args.to_unit in scales else rates
        if args.from_unit not in family or args.to_unit not in family:
            raise ValueError("cross-family unit conversion is forbidden")
        value = _decimal(args.value) * family[args.from_unit] / family[args.to_unit]
        result = {"value": format(value, "f"), "unit": args.to_unit}
    elif isinstance(args, CompanyCompareArgs):
        if args.metric not in _METRIC_KEYS:
            raise ValueError("unknown financial metric")
        rows = query_benchmark_fields(
            [args.metric],
            args.companies,
            args.years,
            context.org_id,
            context.visibility,
        )
        result = {
            "items": [
                {
                    "company": row.company,
                    "year": row.year,
                    "metric": row.metric,
                    "value": row.value,
                    "unit": row.unit,
                    "page": row.page,
                }
                for row in rows
            ]
        }
    elif isinstance(args, ExportArgs):
        result = {
            "format": args.format,
            "title": args.title,
            "rows": args.rows,
            "server_file_created": False,
        }
    else:  # pragma: no cover - exhaustive safety net
        raise ValueError("unsupported validated tool arguments")
    context.cache[idem] = result
    return result


async def run_tool_agent(
    llm: Any,
    messages: list[dict[str, Any]],
    context: ToolContext,
    *,
    max_calls: int = 3,
    timeout_seconds: float = 8.0,
) -> str | None:
    call_count = 0
    transcript = list(messages)
    for _round in range(max_calls):
        response = await llm.tool_chat(transcript, TOOLS)
        calls = response.get("tool_calls") or []
        if not calls:
            return str(response.get("content") or "") or None
        transcript.append(
            {
                "role": "assistant",
                "content": response.get("content") or "",
                "tool_calls": calls,
            }
        )
        for call in calls:
            if call_count >= max_calls:
                break
            call_count += 1
            function = call.get("function") or {}
            name = str(function.get("name") or "")
            try:
                raw = json.loads(function.get("arguments") or "{}")
                async with asyncio.timeout(timeout_seconds):
                    result = await execute_tool(name, raw, context)
                payload = {"ok": True, "result": result}
            except Exception as exc:
                logger.warning("tool rejected: name=%s type=%s", name, type(exc).__name__)
                payload = {"ok": False, "error": "tool_validation_or_execution_failed"}
            transcript.append(
                {
                    "role": "tool",
                    "tool_call_id": call.get("id"),
                    "content": json.dumps(payload, ensure_ascii=False, default=str),
                }
            )
    # No more tools after the bounded loop; force one grounded final answer.
    transcript.append(
        {
            "role": "system",
            "content": "Tool budget exhausted. Answer from the supplied tool results without further calls.",
        }
    )
    return await llm.chat(transcript)
