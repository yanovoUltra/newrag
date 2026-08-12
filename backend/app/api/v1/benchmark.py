"""财报竞品对标分析 API。"""

from __future__ import annotations

import json
from typing import Literal

from fastapi import APIRouter, HTTPException, Query

from app.benchmark.schemas import (
    BenchmarkAnalysis,
    BenchmarkCatalog,
    BenchmarkRequest,
    BenchmarkRunSummary,
)
from app.benchmark.service import BenchmarkInputError, analyze, build_catalog
from app.store.registry import get_benchmark_run, list_benchmark_runs, save_benchmark_run

router = APIRouter(prefix="/benchmark", tags=["benchmark"])


@router.get("/catalog", response_model=BenchmarkCatalog)
def catalog(
    org_id: str = Query(default="default", min_length=1, max_length=64),
    user_visibility: Literal["public", "internal", "restricted"] = "public",
):
    return build_catalog(org_id, user_visibility)


@router.post("/analyze", response_model=BenchmarkAnalysis)
def run_analysis(req: BenchmarkRequest):
    try:
        result = analyze(req)
    except BenchmarkInputError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    payload = result.model_dump(mode="json")
    save_benchmark_run(
        run_id=result.id,
        org_id=req.org_id,
        visibility=req.user_visibility,
        name=result.name,
        request_data=req.model_dump(mode="json"),
        result_data=payload,
    )
    return result


@router.get("/runs", response_model=list[BenchmarkRunSummary])
def runs(
    org_id: str = Query(default="default", min_length=1, max_length=64),
    user_visibility: Literal["public", "internal", "restricted"] = "public",
    limit: int = Query(default=20, ge=1, le=50),
):
    output = []
    for run in list_benchmark_runs(org_id, user_visibility, limit):
        try:
            result = json.loads(run.result_json)
        except json.JSONDecodeError:
            continue
        output.append({
            "id": run.id,
            "name": run.name,
            "companies": result.get("companies", []),
            "years": result.get("years", []),
            "metric_count": len(result.get("metrics", [])),
            "coverage": result.get("summary", {}).get("coverage", 0.0),
            "confidence": result.get("summary", {}).get("confidence", "low"),
            "created_at": run.created_at,
        })
    return output


@router.get("/runs/{run_id}", response_model=BenchmarkAnalysis)
def run_detail(
    run_id: str,
    org_id: str = Query(default="default", min_length=1, max_length=64),
    user_visibility: Literal["public", "internal", "restricted"] = "public",
):
    run = get_benchmark_run(run_id, org_id, user_visibility)
    if run is None:
        raise HTTPException(status_code=404, detail="对标分析不存在或无权访问")
    try:
        return json.loads(run.result_json)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail="对标分析结果损坏") from exc
