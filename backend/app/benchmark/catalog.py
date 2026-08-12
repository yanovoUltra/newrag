"""对标指标的展示口径与比较方向。

字段抽取目录负责“能识别什么”，本模块只描述“如何比较和展示”，避免把
产品层排名规则混入底层抽取逻辑。
"""

from __future__ import annotations

from app.fields.metrics import METRIC_CATALOG

PERCENT_METRICS = {
    "roe",
    "roa",
    "gross_margin",
    "net_margin",
    "debt_ratio",
    "npl_ratio",
    "provision_coverage",
    "capital_adequacy",
}

PER_SHARE_METRICS = {"eps"}
LOWER_IS_BETTER = {"debt_ratio", "npl_ratio"}
NEUTRAL_METRICS = {"total_assets", "net_assets"}


def metric_profile(key: str) -> dict:
    catalog = next((item for item in METRIC_CATALOG if item["key"] == key), None)
    label = catalog["label"] if catalog else key
    if key in PERCENT_METRICS:
        kind, display_unit = "percent", "%"
    elif key in PER_SHARE_METRICS:
        kind, display_unit = "per_share", "每股"
    else:
        kind, display_unit = "amount", "亿元"

    if key in LOWER_IS_BETTER:
        direction = "lower"
    elif key in NEUTRAL_METRICS:
        direction = "neutral"
    else:
        direction = "higher"
    return {
        "key": key,
        "label": label,
        "kind": kind,
        "display_unit": display_unit,
        "direction": direction,
    }


def catalog_keys() -> set[str]:
    return {item["key"] for item in METRIC_CATALOG}
