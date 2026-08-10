"""API 全链路集成测试：上传 → 解析入库 → 问答 SSE。需要本地 Qdrant。"""

from __future__ import annotations

import time

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
