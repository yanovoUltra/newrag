"""OCR 层：百度智能云（逐页渲染识别）/ PaddleOCR AI Studio（ppocr，整文档异步任务）。

ppocr 流程：提交任务（multipart）→ 轮询状态 → 下载 JSONL → 按 layoutParsingResults 取每页 markdown 文本。
"""

from __future__ import annotations

import base64
import json
import re
import time
from pathlib import Path

import httpx

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_BAIDU_TOKEN_URL = "https://aip.baidubce.com/oauth/2.0/token"
_BAIDU_OCR_URL = "https://aip.baidubce.com/rest/2.0/ocr/v1/general_basic"

# 剔除 markdown / HTML 图片占位（如 ![](url)、<img .../>），避免图表链接混入向量库
_MD_IMAGE_RE = re.compile(r"\s*!\[[^\]]*\]\([^)]*\)\s*", re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _clean_md(text: str) -> str:
    """清理 OCR markdown：去 HTML 标签（含 <img>）、markdown 图片占位，压缩连续空行。"""
    text = _HTML_TAG_RE.sub("", text)
    text = _MD_IMAGE_RE.sub("", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


class BaseOcr:
    def recognize(self, image_path: Path) -> str:
        raise NotImplementedError


class BaiduOcr(BaseOcr):
    """百度智能云通用文字识别。"""

    def __init__(self, api_key: str, secret_key: str):
        self._api_key = api_key
        self._secret_key = secret_key
        self._token: str | None = None

    def _get_token(self) -> str:
        if self._token:
            return self._token
        resp = httpx.post(
            _BAIDU_TOKEN_URL,
            params={"grant_type": "client_credentials", "client_id": self._api_key, "client_secret": self._secret_key},
            timeout=30,
        )
        resp.raise_for_status()
        self._token = resp.json()["access_token"]
        return self._token

    def recognize(self, image_path: Path) -> str:
        token = self._get_token()
        b64 = base64.b64encode(image_path.read_bytes()).decode()
        resp = httpx.post(
            _BAIDU_OCR_URL,
            params={"access_token": token},
            data={"image": b64},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        if "words_result" not in data:
            logger.warning("baidu ocr unexpected response: %s", data)
            return ""
        return "\n".join(item["words"] for item in data["words_result"])


class MockOcr(BaseOcr):
    """开发用 mock：无密钥时返回空（不编造文本），仅打日志。"""

    def recognize(self, image_path: Path) -> str:
        logger.warning("OCR mock: 未配置 OCR 密钥，扫描页 %s 无法识别", image_path.name)
        return ""


class PaddleOcr:
    """PaddleOCR AI Studio 在线 API（PP-StructureV3）：整文档异步任务，按页返回 markdown 文本。"""

    def __init__(self, settings):
        self._settings = settings
        self._headers = {"Authorization": f"bearer {settings.ppocr_token}"}

    def document_pages(self, file_path: Path) -> list[str]:
        """提交整文件（PDF/图片）并轮询，返回按文档顺序的每页文本。"""
        settings = self._settings
        if not settings.ppocr_token:
            raise RuntimeError("未配置 PPOCR_TOKEN，无法调用 PaddleOCR 在线 API")
        with httpx.Client(timeout=settings.ppocr_timeout) as client:
            job_id = self._submit(client, file_path)
            logger.info("ppocr job submitted: %s (%s)", job_id, file_path.name)
            jsonl_url = self._wait_done(client, job_id)
            resp = client.get(jsonl_url)
            resp.raise_for_status()
        return self._parse_jsonl(resp.text)

    def _submit(self, client: httpx.Client, file_path: Path) -> str:
        settings = self._settings
        data = {
            "model": settings.ppocr_model,
            "optionalPayload": json.dumps(
                {
                    "useDocOrientationClassify": False,
                    "useDocUnwarping": False,
                    "useChartRecognition": False,
                }
            ),
        }
        with open(file_path, "rb") as f:
            resp = client.post(
                settings.ppocr_job_url, headers=self._headers, data=data, files={"file": f}
            )
        if resp.status_code != 200:
            raise RuntimeError(f"ppocr 任务提交失败: {resp.status_code} {resp.text[:300]}")
        return resp.json()["data"]["jobId"]

    def _wait_done(self, client: httpx.Client, job_id: str) -> str:
        settings = self._settings
        url = f"{settings.ppocr_job_url}/{job_id}"
        while True:
            resp = client.get(url, headers=self._headers)
            resp.raise_for_status()
            data = resp.json()["data"]
            state = data.get("state")
            if state == "done":
                return data["resultUrl"]["jsonUrl"]
            if state == "failed":
                raise RuntimeError(f"ppocr 任务失败: {data.get('errorMsg')}")
            time.sleep(settings.ppocr_poll_interval)

    @staticmethod
    def _parse_jsonl(jsonl_text: str) -> list[str]:
        """JSONL 每行一个 result，其 layoutParsingResults 按页给出 markdown 文本。"""
        texts: list[str] = []
        for line in jsonl_text.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            result = json.loads(line).get("result")
            for res in (result or {}).get("layoutParsingResults", []):
                md = res.get("markdown", {}).get("text", "")
                texts.append(_clean_md(md))
        return texts


def get_ocr() -> BaseOcr:
    settings = get_settings()
    if settings.ocr_provider == "baidu" and settings.ocr_api_key and settings.ocr_secret_key:
        return BaiduOcr(settings.ocr_api_key, settings.ocr_secret_key)
    return MockOcr()


def ocr_page_images(pdf_path: Path, page_nos: list[int]) -> dict[int, str]:
    """扫描页 OCR，返回 {page_no: text}。

    ppocr：整文档提交（API 按整文件处理），返回全部页文本，调用方仅消费扫描页；
    baidu：按页渲染图片逐张识别。
    """
    settings = get_settings()
    if settings.ocr_provider == "ppocr":
        pages = PaddleOcr(settings).document_pages(pdf_path)
        result = {i + 1: t for i, t in enumerate(pages) if t.strip()}
        logger.info("ppocr pages=%d (wanted=%s)", len(result), page_nos)
        return result

    ocr = get_ocr()
    result: dict[int, str] = {}
    import fitz

    with fitz.open(pdf_path) as doc:
        for page_no in page_nos:
            if page_no < 1 or page_no > doc.page_count:
                continue
            pix = doc[page_no - 1].get_pixmap(dpi=200)
            tmp = pdf_path.parent / f"_ocr_tmp_{page_no}.png"
            pix.save(tmp)
            try:
                text = ocr.recognize(tmp)
                if text:
                    result[page_no] = text
            finally:
                tmp.unlink(missing_ok=True)
    return result


def ocr_image_file(image_path: Path) -> str:
    """单张图片 OCR（png/jpg/jpeg 上传），返回全文。"""
    settings = get_settings()
    if settings.ocr_provider == "ppocr":
        pages = PaddleOcr(settings).document_pages(image_path)
        return "\n".join(pages).strip()
    return get_ocr().recognize(image_path)
