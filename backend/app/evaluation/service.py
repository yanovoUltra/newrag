"""Publish sanitized, provenance-carrying evaluation summaries."""

from __future__ import annotations

import json
import re
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.core.runtime import effective_runtime_mode, model_feature_status
from app.models.entities import EvalResult
from app.store.registry import list_eval_results

_SNAPSHOT = Path(__file__).with_name("public_snapshot.json")
_SAFE_KEY = re.compile(r"^[a-zA-Z0-9_.@-]{1,64}$")


def _numeric_metrics(value: Any, *, depth: int = 0) -> Any:
    """Keep aggregate numeric metrics; drop strings, arrays and raw per-query data."""

    if depth > 4:
        return None
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if not isinstance(value, dict):
        return None
    output: dict[str, Any] = {}
    for key, child in value.items():
        key_text = str(key)
        if not _SAFE_KEY.fullmatch(key_text):
            continue
        cleaned = _numeric_metrics(child, depth=depth + 1)
        if cleaned is not None:
            output[key_text] = cleaned
    return output


def _scope_parameters(scope: str) -> dict[str, str | int | float | bool]:
    """Expose reproducibility parameters while withholding tenant identifiers."""

    allowed = {"top_k", "seed", "n", "summary", "candidate_k", "ndcg_k", "recall_k"}
    output: dict[str, str | int | float | bool] = {}
    for segment in scope.split(";"):
        key, separator, raw = segment.partition("=")
        if not separator or key not in allowed:
            continue
        raw = raw.strip()
        if raw.lower() in {"true", "false"}:
            output[key] = raw.lower() == "true"
            continue
        try:
            output[key] = int(raw)
        except ValueError:
            try:
                output[key] = float(raw)
            except ValueError:
                output[key] = raw[:64]
    return output


def _stored_run(record: EvalResult) -> dict[str, Any] | None:
    try:
        metrics = _numeric_metrics(json.loads(record.metrics))
    except (json.JSONDecodeError, TypeError):
        return None
    if not metrics:
        return None
    return {
        "id": record.id,
        "label": record.eval_name,
        "kind": "stored_evaluation",
        "scope": _scope_parameters(record.scope),
        "metrics": metrics,
        "created_at": record.created_at.isoformat(),
    }


def _ragas_version() -> str | None:
    try:
        return version("ragas")
    except PackageNotFoundError:
        return None


def evaluation_overview(org_id: str, *, limit: int = 12) -> dict[str, Any]:
    snapshot = json.loads(_SNAPSHOT.read_text(encoding="utf-8"))
    stored = []
    for record in list_eval_results(org_id=org_id, limit=limit):
        cleaned = _stored_run(record)
        if cleaned is not None:
            stored.append(cleaned)
    settings = get_settings()
    return {
        **snapshot,
        "runtime": {
            "mode": effective_runtime_mode(settings),
            "features": model_feature_status(settings),
            "note": "运行能力来自当前实例；发布指标来自带哈希的本地真实实验，二者不可混为一谈。",
        },
        "official_ragas": {
            "package": "ragas",
            "version": _ragas_version(),
            "integration": "official_package",
            "legacy_custom_results_labeled_separately": True,
        },
        "recent_stored_runs": stored,
    }
