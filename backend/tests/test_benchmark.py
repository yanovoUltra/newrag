"""竞品对标分析的计算、口径、证据与 API 回归测试。"""

from __future__ import annotations

from app.fields.extract import FieldRecord
from app.store.registry import create_document, replace_fields


def _seed_company(company: str, values: dict[tuple[str, int], tuple[float, str | None]]) -> str:
    doc = create_document(
        filename=f"{company}_annual_report.pdf",
        file_type="pdf",
        org_id="benchmark-test",
        visibility="public",
        size_bytes=100,
        fiscal_year=max(year for _, year in values),
    )
    records = []
    for (metric, year), (value, unit) in values.items():
        record = FieldRecord(
                doc_id=doc.id,
                metric=metric,
                metric_label=metric,
                year=year,
                value=value,
                unit=unit,
                raw=f"{company} {year} {value}{unit or ''}",
                company=company,
                source="table",
                page=year - 2000,
                section_path="主要财务数据",
            )
        record.source_chunk_id = f"{company}-{metric}-{year}"
        records.append(record)
    replace_fields(doc.id, records, "benchmark-test", "public")
    return doc.id


def _seed(client) -> None:
    marker = client.get(
        "/api/v1/benchmark/catalog",
        params={"org_id": "benchmark-test", "user_visibility": "public"},
    ).json()
    if "甲公司" in marker["companies"]:
        return
    _seed_company(
        "甲公司",
        {
            ("revenue", 2023): (100.0, "亿元"),
            ("revenue", 2024): (120.0, "亿元"),
            ("net_profit", 2023): (10.0, "亿元"),
            ("net_profit", 2024): (12.0, "亿元"),
            ("roe", 2024): (14.2, "%"),
        },
    )
    _seed_company(
        "乙公司",
        {
            ("revenue", 2023): (9000.0, "百万元"),
            ("revenue", 2024): (1_050_000.0, "万元"),
            ("net_profit", 2024): (9.0, None),
            ("roe", 2024): (11.5, "%"),
        },
    )


def test_catalog_exposes_only_available_metrics(client):
    _seed(client)
    response = client.get(
        "/api/v1/benchmark/catalog",
        params={"org_id": "benchmark-test", "user_visibility": "public"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["companies"] == ["乙公司", "甲公司"]
    assert body["years"] == [2024, 2023]
    metrics = {item["key"]: item for item in body["metrics"]}
    assert metrics["revenue"]["kind"] == "amount"
    assert metrics["roe"]["display_unit"] == "%"


def test_analysis_normalizes_units_calculates_growth_and_persists(client):
    _seed(client)
    response = client.post(
        "/api/v1/benchmark/analyze",
        json={
            "org_id": "benchmark-test",
            "user_visibility": "public",
            "companies": ["甲公司", "乙公司"],
            "years": [2023, 2024],
            "metrics": ["revenue", "net_profit", "roe"],
            "name": "银行同业对标",
        },
    )
    assert response.status_code == 200
    body = response.json()
    revenue = next(item for item in body["metrics"] if item["key"] == "revenue")
    values = {(item["company"], item["year"]): item for item in revenue["observations"]}
    assert values[("甲公司", 2024)]["normalized_value"] == 120.0
    assert values[("乙公司", 2024)]["normalized_value"] == 105.0
    assert values[("甲公司", 2024)]["change_value"] == 20.0
    assert values[("乙公司", 2024)]["change_value"] == round((105 - 90) / 90 * 100, 4)
    assert body["summary"]["requested_cells"] == 12
    assert body["summary"]["available_cells"] == 9
    assert body["summary"]["evidence_coverage"] == 1.0
    assert any("缺少结构化数据" in warning for warning in body["warnings"])
    assert any("尚未结构化保存币种" in warning for warning in body["warnings"])
    assert body["summary"]["confidence"] == "medium"

    detail = client.get(
        f"/api/v1/benchmark/runs/{body['id']}",
        params={"org_id": "benchmark-test", "user_visibility": "public"},
    )
    assert detail.status_code == 200
    assert detail.json()["name"] == "银行同业对标"
    runs = client.get(
        "/api/v1/benchmark/runs",
        params={"org_id": "benchmark-test", "user_visibility": "public"},
    ).json()
    assert any(item["id"] == body["id"] for item in runs)


def test_amount_without_unit_is_not_ranked(client):
    _seed(client)
    body = client.post(
        "/api/v1/benchmark/analyze",
        json={
            "org_id": "benchmark-test",
            "companies": ["甲公司", "乙公司"],
            "years": [2024],
            "metrics": ["net_profit"],
        },
    ).json()
    metric = body["metrics"][0]
    row = next(item for item in metric["observations"] if item["company"] == "乙公司")
    assert row["comparable"] is False
    assert row["normalized_value"] is None
    assert "单位" in row["warnings"][0]
    assert metric["insights"] == []


def test_amount_metrics_do_not_generate_cross_company_rankings(client):
    _seed(client)
    body = client.post(
        "/api/v1/benchmark/analyze",
        json={
            "org_id": "benchmark-test",
            "companies": ["甲公司", "乙公司"],
            "years": [2024],
            "metrics": ["revenue"],
        },
    ).json()
    assert body["metrics"][0]["insights"] == []
    assert any("不生成跨公司金额排名" in warning for warning in body["warnings"])


def test_analysis_rejects_unknown_company_and_duplicate_inputs(client):
    _seed(client)
    unknown = client.post(
        "/api/v1/benchmark/analyze",
        json={
            "org_id": "benchmark-test",
            "companies": ["甲公司", "不存在公司"],
            "years": [2024],
            "metrics": ["revenue"],
        },
    )
    assert unknown.status_code == 400
    duplicate = client.post(
        "/api/v1/benchmark/analyze",
        json={
            "org_id": "benchmark-test",
            "companies": ["甲公司", "甲公司"],
            "years": [2024],
            "metrics": ["revenue"],
        },
    )
    assert duplicate.status_code == 422
