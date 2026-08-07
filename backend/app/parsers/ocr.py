"""OCR 层：百度智能云 OCR（按需触发）。未配置密钥时回退为 mock，不编造内容。"""

from __future__ import annotations

import base64
from pathlib import Path

import httpx

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_BAIDU_TOKEN_URL = "https://aip.baidubce.com/oauth/2.0/token"
_BAIDU_OCR_URL = "https://aip.baidubce.com/rest/2.0/ocr/v1/general_basic"


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
        logger.warning("OCR mock: 未配置百度 OCR 密钥，扫描页 %s 无法识别", image_path.name)
        return ""


def get_ocr() -> BaseOcr:
    settings = get_settings()
    if settings.ocr_provider == "baidu" and settings.ocr_api_key and settings.ocr_secret_key:
        return BaiduOcr(settings.ocr_api_key, settings.ocr_secret_key)
    return MockOcr()


def ocr_page_images(pdf_path: Path, page_nos: list[int]) -> dict[int, str]:
    """对指定页码渲染为图片并 OCR，返回 {page_no: text}。"""
    import fitz

    ocr = get_ocr()
    result: dict[int, str] = {}
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
