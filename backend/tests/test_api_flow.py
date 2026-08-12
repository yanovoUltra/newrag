"""API 全链路集成测试：上传 → 解析入库 → 问答 SSE。需要本地 Qdrant。"""

from __future__ import annotations

import time
from datetime import date
from types import SimpleNamespace

import pytest

from tests.conftest import qdrant_available


def _wait_task(client, task_id, timeout=60, interval=0.5):
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get(f"/api/v1/tasks/{task_id}")
        assert r.status_code == 200
        data = r.json()
        if data["status"] in ("success", "failed"):
            return data
        time.sleep(interval)
    raise AssertionError(f"task {task_id} timeout")


@pytest.mark.skipif(not qdrant_available(), reason="需要本地 Qdrant")
def test_upload_ingest_chat_flow(client, sample_pdf):
    # 1) 上传
    with open(sample_pdf, "rb") as f:
        resp = client.post(
            "/api/v1/documents",
            files={"file": ("sample_report.pdf", f, "application/pdf")},
            data={"org_id": "orgA", "visibility": "public", "fiscal_year": 2023},
        )
    assert resp.status_code == 201
    body = resp.json()
    task_id, doc_id = body["task_id"], body["doc_id"]
    assert doc_id

    # 2) 轮询任务到完成
    task = _wait_task(client, task_id)
    assert task["status"] == "success", task
    assert task["stage"] == "index"

    # 3) 文档详情
    d = client.get(f"/api/v1/documents/{doc_id}").json()
    assert d["status"] == "indexed"
    assert d["chunk_count"] > 0
    assert d["fiscal_year"] == 2023

    # 4) 问答 SSE
    with client.stream("POST", "/api/v1/chat", json={
        "question": "2023年公司营业收入是多少？",
        "session_id": "s1",
        "org_id": "orgA",
        "user_visibility": "public",
    }) as resp:
        assert resp.status_code == 200
        text = "".join(resp.iter_text())
    assert "event: meta" in text
    assert "event: citation" in text
    assert "event: token" in text
    assert "event: done" in text

    # 5) 清理
    r = client.delete(f"/api/v1/documents/{doc_id}")
    assert r.status_code == 200


@pytest.mark.skipif(not qdrant_available(), reason="需要本地 Qdrant")
def test_permission_isolation(client, sample_pdf):
    """orgB 的数据对 orgA 不可见。"""
    with open(sample_pdf, "rb") as f:
        resp = client.post(
            "/api/v1/documents",
            files={"file": ("b_report.pdf", f, "application/pdf")},
            data={"org_id": "orgB", "visibility": "restricted"},
        )
    assert resp.status_code == 201
    body = resp.json()
    task = _wait_task(client, body["task_id"])
    assert task["status"] == "success", task
    doc_b = body["doc_id"]

    # orgA 检索：不应出现 orgB 的引用
    with client.stream("POST", "/api/v1/chat", json={
        "question": "营业收入是多少？",
        "org_id": "orgA",
        "user_visibility": "public",
    }) as resp:
        text = "".join(resp.iter_text())
    assert f'"doc_id": "{doc_b}"' not in text

    client.delete(f"/api/v1/documents/{doc_b}")


def test_upload_validation(client):
    # 非法类型
    resp = client.post(
        "/api/v1/documents",
        files={"file": ("a.exe", b"MZ...", "application/octet-stream")},
        data={"org_id": "orgA"},
    )
    assert resp.status_code == 400
    # 空文件
    resp = client.post(
        "/api/v1/documents",
        files={"file": ("a.pdf", b"", "application/pdf")},
        data={"org_id": "orgA"},
    )
    assert resp.status_code == 400


def test_current_year_helper():
    from app.api.v1.documents import _current_year

    assert _current_year(date(2032, 1, 1)) == 2032


@pytest.mark.parametrize("fiscal_year", [1990, date.today().year, None])
def test_upload_accepts_valid_fiscal_year(client, monkeypatch, fiscal_year):
    from app.api.v1 import documents as documents_api

    monkeypatch.setattr(documents_api, "_find_replace_target", lambda *args: ("none", None))
    monkeypatch.setattr(
        documents_api,
        "create_document",
        lambda **kwargs: SimpleNamespace(id=f"doc-{fiscal_year}"),
    )
    monkeypatch.setattr(
        documents_api,
        "create_task",
        lambda *args, **kwargs: SimpleNamespace(id=f"task-{fiscal_year}"),
    )
    monkeypatch.setattr(documents_api, "run_ingest", lambda *args, **kwargs: None)

    data = {"org_id": "orgA"}
    if fiscal_year is not None:
        data["fiscal_year"] = str(fiscal_year)
    resp = client.post(
        "/api/v1/documents",
        files={"file": ("year.pdf", b"valid", "application/pdf")},
        data=data,
    )

    assert resp.status_code == 201


@pytest.mark.parametrize("fiscal_year", [1989, date.today().year + 1])
def test_upload_rejects_fiscal_year_outside_reality(client, fiscal_year):
    resp = client.post(
        "/api/v1/documents",
        files={"file": ("year.pdf", b"valid", "application/pdf")},
        data={"org_id": "orgA", "fiscal_year": str(fiscal_year)},
    )

    assert resp.status_code == 400
    assert f"1990-{date.today().year}" in resp.json()["detail"]


def test_document_page_filters_and_contract(client, monkeypatch):
    from datetime import datetime

    from app.api.v1 import documents as documents_api

    captured = {}
    doc = SimpleNamespace(
        id="doc-page-1",
        filename="annual-report.pdf",
        file_type="pdf",
        status="indexed",
        org_id="orgA",
        visibility="public",
        fiscal_year=2025,
        fiscal_quarter=None,
        chunk_count=12,
        size_bytes=1024,
        error=None,
        created_at=datetime(2026, 1, 2, 3, 4, 5),
        updated_at=datetime(2026, 1, 2, 3, 4, 5),
    )

    def fake_query(**kwargs):
        captured.update(kwargs)
        return [doc], 37

    monkeypatch.setattr(documents_api, "query_documents_page", fake_query)
    response = client.get(
        "/api/v1/documents/page",
        params={
            "org_id": "orgA",
            "filename": "annual",
            "status": "indexed",
            "limit": 20,
            "offset": 20,
        },
    )

    assert response.status_code == 200
    assert response.json()["total"] == 37
    assert response.json()["items"][0]["id"] == "doc-page-1"
    assert captured == {
        "org_id": "orgA",
        "filename": "annual",
        "status": "indexed",
        "limit": 20,
        "offset": 20,
        "include_archived": False,
    }


@pytest.mark.parametrize(
    "params",
    [
        {"limit": 0},
        {"limit": 101},
        {"offset": -1},
    ],
)
def test_document_page_rejects_invalid_bounds(client, params):
    response = client.get("/api/v1/documents/page", params=params)
    assert response.status_code == 422


@pytest.mark.skipif(not qdrant_available(), reason="需要本地 Qdrant")
def test_upload_replace_archives_old_version(client, sample_pdf):
    """版本控制：同名文件替换 → 旧版标 archived（留档可追溯），list 默认隐藏归档。"""
    import os
    import tempfile

    import fitz

    # v1：sample_pdf
    with open(sample_pdf, "rb") as f:
        resp = client.post(
            "/api/v1/documents",
            files={"file": ("version_test.pdf", f, "application/pdf")},
            data={"org_id": "orgC"},
        )
    assert resp.status_code == 201
    v1_id = resp.json()["doc_id"]
    assert _wait_task(client, resp.json()["task_id"])["status"] == "success"

    # v2：同名不同内容
    tmp = os.path.join(tempfile.mkdtemp(), "version_test.pdf")
    doc = fitz.open()
    p = doc.new_page()
    p.insert_text((72, 80), "Revenue two hundred thirty four point five million", fontsize=11)
    doc.save(tmp)
    with open(tmp, "rb") as f:
        resp = client.post(
            "/api/v1/documents",
            files={"file": ("version_test.pdf", f, "application/pdf")},
            data={"org_id": "orgC"},
        )
    assert resp.status_code == 201
    v2_id = resp.json()["doc_id"]
    assert _wait_task(client, resp.json()["task_id"])["status"] == "success"

    # 旧版归档：状态 archived、list 默认不显示、include_archived=true 显示
    d1 = client.get(f"/api/v1/documents/{v1_id}").json()
    assert d1["status"] == "archived"
    listed = client.get("/api/v1/documents", params={"org_id": "orgC"}).json()
    assert v1_id not in [d["id"] for d in listed]
    assert v2_id in [d["id"] for d in listed]
    archived = client.get("/api/v1/documents", params={"org_id": "orgC", "include_archived": True}).json()
    ids = [d["id"] for d in archived]
    assert v1_id in ids and v2_id in ids

    # 清理
    client.delete(f"/api/v1/documents/{v1_id}")
    client.delete(f"/api/v1/documents/{v2_id}")
