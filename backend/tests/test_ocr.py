"""OCR 层单元测试（仅纯逻辑，不触网）。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.parsers.ocr import PaddleOcr


def test_paddle_ocr_parse_jsonl():
    page = json.dumps(
        {
            "result": {
                "layoutParsingResults": [
                    {
                        "markdown": {
                            "text": '<div style="text-align:center;"><img src="imgs/x.jpg" alt="Image" width="77%" /></div>\n第一页正文 ![chart](http://x/y.png) 结束'
                        }
                    }
                ]
            }
        },
        ensure_ascii=False,
    )
    texts = PaddleOcr._parse_jsonl(f"{page}\n{page}")
    # HTML <img>/<div> 与 markdown 图片占位均被剔除
    assert texts == ["第一页正文结束", "第一页正文结束"]


def test_parse_layout_image_gated_by_ocr(monkeypatch):
    from app.parsers.base import parse_layout

    class FakeSettings:
        ocr_enabled = False

    monkeypatch.setattr("app.parsers.base.get_settings", lambda: FakeSettings())
    with pytest.raises(ValueError):
        parse_layout(Path("scan.png"))
