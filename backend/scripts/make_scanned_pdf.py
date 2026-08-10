"""构造"扫描版"PDF 用于实测 OCR 管线。

把源 PDF 的前 N 页渲染成图片，重建一个**无文本层**的纯图片 PDF（模拟扫描件），
使布局层提取率为 0，从而触发 ingest 的 OCR 分支（PaddleOCR API）。

用法（backend/ 下）：python scripts/make_scanned_pdf.py --src <pdf> --out <pdf> [--pages 5]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import fitz  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="构造扫描版 PDF")
    parser.add_argument("--src", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--pages", type=int, default=5)
    args = parser.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    doc = fitz.open(str(src))
    total = min(args.pages, doc.page_count)
    new_doc = fitz.open()
    for i in range(total):
        page = doc.load_page(i)
        pix = page.get_pixmap(dpi=150)
        img_bytes = pix.tobytes("png")
        new_page = new_doc.new_page(width=page.rect.width, height=page.rect.height)
        new_page.insert_image(new_page.rect, stream=img_bytes)
    out.parent.mkdir(parents=True, exist_ok=True)
    new_doc.save(str(out))
    new_doc.close()
    doc.close()
    print(f"已生成扫描版 PDF：{out}（{total} 页，无文本层）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())