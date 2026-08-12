"""可审计的财报竞品对标编排器。

LLM 不参与取值、单位换算和公式计算。工作流对字段索引执行批量查询，随后按明确
规则完成归一、冲突检测、同比计算和证据核验；即使模型服务不可用也能稳定运行。
"""

from __future__ import annotations

import math
import uuid
from collections import defaultdict
from datetime import datetime, timezone

from app.benchmark.catalog import catalog_keys, metric_profile
from app.benchmark.schemas import BenchmarkAnalysis, BenchmarkCatalog, BenchmarkRequest
from app.models.entities import FinancialField
from app.store.registry import (
    benchmark_dimensions,
    get_documents_by_ids,
    query_benchmark_fields,
)


class BenchmarkInputError(ValueError):
    """请求范围与当前字段索引不匹配。"""


_AMOUNT_FACTORS = {
    "元": 1.0,
    "千元": 1e3,
    "万元": 1e4,
    "百万元": 1e6,
    "百万": 1e6,
    "亿元": 1e8,
    "十亿元": 1e9,
    "万亿元": 1e12,
}


def build_catalog(org_id: str, user_visibility: str) -> BenchmarkCatalog:
    companies, years, counts = benchmark_dimensions(org_id, user_visibility)
    options = []
    for key in catalog_keys():
        count = counts.get(key, 0)
        if count:
            options.append({**metric_profile(key), "available_records": count})
    options.sort(key=lambda item: (-item["available_records"], item["label"]))
    return BenchmarkCatalog(companies=companies, years=years, metrics=options)


def _candidate_rank(row: FinancialField) -> tuple[int, int, int, float]:
    created = row.created_at.timestamp() if row.created_at else 0.0
    return (
        1 if row.source == "table" else 0,
        1 if row.unit else 0,
        1 if row.source_chunk_id else 0,
        created,
    )


def _normalize(row: FinancialField, kind: str) -> tuple[float | None, str, bool, list[str]]:
    warnings: list[str] = []
    value = float(row.value)
    if not math.isfinite(value):
        return None, "—", False, ["数值不是有限数"]
    if kind == "percent":
        return value, f"{value:,.2f}%", True, warnings
    if kind == "per_share":
        if not row.unit:
            return value, f"{value:,.4g} / 股", False, ["每股指标缺少币种或金额单位，未参与跨公司排名"]
        return value, f"{value:,.4g} {row.unit}/股", True, warnings

    factor = _AMOUNT_FACTORS.get((row.unit or "").strip())
    if factor is None:
        unit = row.unit or "单位缺失"
        return None, f"{value:,.4g} {unit}", False, ["金额单位未知，未参与换算和排名"]
    normalized = value * factor / 1e8
    return normalized, f"{normalized:,.2f} 亿元", True, warnings


def _is_conflict(values: list[float]) -> bool:
    if len(values) < 2:
        return False
    anchor = values[0]
    return any(not math.isclose(anchor, value, rel_tol=1e-6, abs_tol=1e-9) for value in values[1:])


def _default_name(req: BenchmarkRequest) -> str:
    company_text = " vs ".join(req.companies)
    year_text = str(req.years[0]) if len(req.years) == 1 else f"{req.years[0]}–{req.years[-1]}"
    return f"{company_text} · {year_text} 对标"


def analyze(req: BenchmarkRequest) -> BenchmarkAnalysis:
    catalog = build_catalog(req.org_id, req.user_visibility)
    available_companies = set(catalog.companies)
    unknown_companies = [name for name in req.companies if name not in available_companies]
    if unknown_companies:
        raise BenchmarkInputError(f"字段索引中不存在公司：{'、'.join(unknown_companies)}")

    available_metrics = {item.key for item in catalog.metrics}
    unknown_metrics = [key for key in req.metrics if key not in catalog_keys()]
    if unknown_metrics:
        raise BenchmarkInputError(f"未知指标：{', '.join(unknown_metrics)}")
    empty_metrics = [key for key in req.metrics if key not in available_metrics]
    if empty_metrics:
        raise BenchmarkInputError(f"当前机构没有这些指标的数据：{', '.join(empty_metrics)}")

    rows = query_benchmark_fields(
        req.metrics,
        req.companies,
        req.years,
        req.org_id,
        req.user_visibility,
    )
    documents = get_documents_by_ids([row.doc_id for row in rows])
    grouped: dict[tuple[str, str, int], list[FinancialField]] = defaultdict(list)
    for row in rows:
        grouped[(row.metric, row.company, row.year)].append(row)

    warnings: list[str] = []
    metric_results: list[dict] = []
    available_cells = comparable_cells = evidenced_cells = conflicts = 0
    has_amount_metrics = False

    for metric_key in req.metrics:
        profile = metric_profile(metric_key)
        has_amount_metrics = has_amount_metrics or profile["kind"] == "amount"
        observations: list[dict] = []
        by_company: dict[str, list[dict]] = defaultdict(list)
        for company in req.companies:
            for year in req.years:
                candidates = grouped.get((metric_key, company, year), [])
                if not candidates:
                    observation = {
                        "company": company,
                        "year": year,
                        "metric": metric_key,
                        "status": "missing",
                        "warnings": ["未抽取到该公司、年份和指标的字段"],
                    }
                    observations.append(observation)
                    by_company[company].append(observation)
                    continue

                available_cells += 1
                ordered = sorted(candidates, key=_candidate_rank, reverse=True)
                chosen = ordered[0]
                normalized, display, comparable, cell_warnings = _normalize(chosen, profile["kind"])
                comparable_values = []
                for candidate in ordered:
                    candidate_value, _, _, _ = _normalize(candidate, profile["kind"])
                    comparable_values.append(
                        candidate_value if candidate_value is not None else float(candidate.value)
                    )
                conflict = _is_conflict(comparable_values)
                if conflict:
                    conflicts += 1
                    comparable = False
                    cell_warnings.append(f"发现 {len(ordered)} 个不一致候选值，已优先采用表格/最新证据")
                if comparable:
                    comparable_cells += 1

                doc = documents.get(chosen.doc_id)
                evidence = {
                    "doc_id": chosen.doc_id,
                    "doc_name": doc.filename if doc else chosen.company,
                    "page": chosen.page,
                    "section_path": chosen.section_path,
                    "chunk_id": chosen.source_chunk_id,
                    "source": chosen.source,
                    "raw": chosen.raw,
                }
                if chosen.source_chunk_id:
                    evidenced_cells += 1
                else:
                    cell_warnings.append("缺少来源块标识，引用只能定位到文档页")

                observation = {
                    "company": company,
                    "year": year,
                    "metric": metric_key,
                    "status": "available",
                    "value": float(chosen.value),
                    "normalized_value": normalized,
                    "display_value": display,
                    "unit": chosen.unit,
                    "comparable": comparable,
                    "conflict": conflict,
                    "evidence": evidence,
                    "warnings": cell_warnings,
                }
                observations.append(observation)
                by_company[company].append(observation)

        insights: list[str] = []
        for company, company_rows in by_company.items():
            prior: dict | None = None
            for observation in company_rows:
                if observation["status"] != "available" or not observation["comparable"]:
                    continue
                if prior is not None:
                    current = observation["normalized_value"]
                    previous = prior["normalized_value"]
                    if current is not None and previous is not None:
                        if profile["kind"] == "percent":
                            observation["change_value"] = round(current - previous, 4)
                            observation["change_unit"] = "个百分点"
                        elif previous != 0:
                            observation["change_value"] = round((current - previous) / abs(previous) * 100, 4)
                            observation["change_unit"] = "%"
                prior = observation

        latest_year = req.years[-1]
        latest = [
            item for item in observations
            if item["year"] == latest_year
            and item["status"] == "available"
            and item["comparable"]
            and item["normalized_value"] is not None
        ]
        if len(latest) >= 2 and profile["direction"] != "neutral":
            reverse = profile["direction"] == "higher"
            leader = sorted(latest, key=lambda item: item["normalized_value"], reverse=reverse)[0]
            direction_text = "最高" if reverse else "最低"
            insights.append(
                f"{latest_year}年，{leader['company']}的{profile['label']}为"
                f"{leader['display_value']}，在本次可比公司中{direction_text}。"
            )

        for company, company_rows in by_company.items():
            comparable_rows = [
                item for item in company_rows
                if item["status"] == "available" and item["comparable"]
            ]
            if len(comparable_rows) >= 2:
                last = comparable_rows[-1]
                if last.get("change_value") is not None:
                    change = last["change_value"]
                    verb = "上升" if change > 0 else "下降" if change < 0 else "持平"
                    magnitude = f"{abs(change):,.2f}{last['change_unit']}" if change else ""
                    insights.append(
                        f"{company}的{profile['label']}在{last['year']}年较上一可用年度"
                        f"{verb}{magnitude}。"
                    )
                    break

        metric_results.append({**profile, "observations": observations, "insights": insights})

    requested_cells = len(req.metrics) * len(req.companies) * len(req.years)
    coverage = available_cells / requested_cells if requested_cells else 0.0
    comparable_coverage = comparable_cells / available_cells if available_cells else 0.0
    evidence_coverage = evidenced_cells / available_cells if available_cells else 0.0
    missing = requested_cells - available_cells
    if missing:
        warnings.append(f"{missing} 个对标单元格缺少结构化数据；结论不会对缺失值进行推断。")
    if available_cells - comparable_cells:
        warnings.append(
            f"{available_cells - comparable_cells} 个已有值因单位缺失或候选冲突未参与跨公司排名。"
        )
    if conflicts:
        warnings.append(f"检测到 {conflicts} 个冲突单元格，请在证据面板核对原表。")
    if has_amount_metrics:
        warnings.append(
            "金额已统一数量级，但当前字段索引尚未结构化保存币种、会计准则与合并范围；"
            "跨币种或跨口径公司的金额排名仅供初筛。"
        )

    if coverage >= 0.9 and evidence_coverage >= 0.95 and conflicts == 0 and not has_amount_metrics:
        confidence = "high"
    elif coverage >= 0.6 and evidence_coverage >= 0.7:
        confidence = "medium"
    else:
        confidence = "low"

    overall_insights = [insight for result in metric_results for insight in result["insights"]][:8]
    stages = [
        {"key": "plan", "label": "范围规划", "status": "complete", "detail": f"{len(req.companies)} 家公司 · {len(req.years)} 个财年 · {len(req.metrics)} 项指标"},
        {"key": "retrieve", "label": "字段取数", "status": "complete" if rows else "warning", "detail": f"批量读取 {len(rows)} 条候选记录"},
        {"key": "normalize", "label": "口径归一", "status": "complete" if comparable_cells == available_cells else "warning", "detail": f"{comparable_cells}/{available_cells} 个已有值可比"},
        {"key": "calculate", "label": "确定性计算", "status": "complete", "detail": "同比变化由代码计算，未使用模型心算"},
        {"key": "verify", "label": "证据核验", "status": "complete" if evidence_coverage >= 0.95 and conflicts == 0 else "warning", "detail": f"证据覆盖 {evidence_coverage:.1%} · 冲突 {conflicts}"},
        {"key": "report", "label": "生成报告", "status": "complete", "detail": f"形成 {len(overall_insights)} 条数据结论"},
    ]

    return BenchmarkAnalysis(
        id=uuid.uuid4().hex,
        name=req.name or _default_name(req),
        generated_at=datetime.now(timezone.utc),
        org_id=req.org_id,
        visibility=req.user_visibility,
        companies=req.companies,
        years=req.years,
        metrics=metric_results,
        summary={
            "requested_cells": requested_cells,
            "available_cells": available_cells,
            "comparable_cells": comparable_cells,
            "evidenced_cells": evidenced_cells,
            "coverage": round(coverage, 4),
            "comparable_coverage": round(comparable_coverage, 4),
            "evidence_coverage": round(evidence_coverage, 4),
            "conflicts": conflicts,
            "confidence": confidence,
        },
        insights=overall_insights,
        warnings=warnings,
        stages=stages,
    )
