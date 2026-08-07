"""pytest 共享配置：使用 mock 嵌入/LLM 与临时数据目录，避免真实模型下载与污染。"""

from __future__ import annotations

import os
import tempfile

# 必须在导入 app 之前设置环境变量
_TMP = tempfile.mkdtemp(prefix="newrag_test_")
os.environ["EMBEDDING_BACKEND"] = "mock"
os.environ["LLM_PROVIDER"] = "mock"
os.environ["APP_ENV"] = "test"
os.environ["REGISTRY_DB"] = os.path.join(_TMP, "registry.db")
os.environ["UPLOAD_DIR"] = os.path.join(_TMP, "uploads")
os.environ["PIPELINE_DIR"] = os.path.join(_TMP, "pipeline")
os.environ["EMBEDDING_CACHE_DIR"] = os.path.join(_TMP, "models")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.main import app  # noqa: E402
from app.store.registry import init_db  # noqa: E402

from app.store import qdrant as qdrant_store  # noqa: E402


def qdrant_available() -> bool:
    try:
        qdrant_store.get_client().get_collections()
        return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def client():
    init_db()
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def sample_pdf():
    """生成合成财报 PDF（fixture 级）。"""
    import fitz

    settings = get_settings()
    path = os.path.join(_TMP, "sample_report.pdf")
    doc = fitz.open()
    p = doc.new_page()
    p.insert_text((72, 80), "一、营业收入", fontsize=15)
    p.insert_text((72, 110), "2023年度公司实现营业收入人民币86.5亿元，同比增长18.2%。", fontsize=11)
    p = doc.new_page()
    p.insert_text((72, 80), "二、净利润", fontsize=15)
    p.insert_text((72, 110), "2023年度实现归属于上市公司股东的净利润人民币7.9亿元，同比增长22.6%。", fontsize=11)
    doc.save(path)
    return path
