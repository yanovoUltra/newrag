"""从公开源下载财报（供入库评测）。

- A 股：巨潮资讯 cninfo 全文搜索 → 取对应股票的年度报告 PDF 直链并下载。
- 美股：SEC EDGAR 提交记录 → 取最新 10-K 的 HTML，转成 .docx 下载（python-docx）。

用法（backend/ 下）：python scripts/fetch_reports.py [--outdir data/uploads]
返回成功/失败清单，失败的会打印原因，便于人工补。
"""

from __future__ import annotations

import argparse
import html
import re
import sys
import time
from pathlib import Path

import httpx

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# ---------------- A 股清单：code -> (是否为招股说明书) ----------------
# True=招股说明书（新上市暂无年报）；False=年度报告。
ASHARE = [
    ("300750", False),  # 宁德时代 年报
    ("002594", False),  # 比亚迪 年报
    ("688825", True),   # 长鑫科技 招股说明书（2026-07 上市，暂无年报）
    ("600519", False),  # 贵州茅台 年报
    ("600036", False),  # 招商银行 年报
    ("601318", False),  # 中国平安 年报
    ("688981", False),  # 中芯国际 年报
    ("000858", False),  # 五粮液 年报
    ("002415", False),  # 海康威视 年报
    ("300124", False),  # 汇川技术 年报
]

# ---------------- 美股清单：公司名 -> 检索关键词 ----------------
US = [
    ("TSLA", "Tesla, Inc."),
    ("SPCX", "SpaceX"),
]

CNINFO_BASE = "http://static.cninfo.com.cn"
CNINFO_SEARCH = "http://www.cninfo.com.cn/new/fulltextSearch/full"
SEC_H = {"User-Agent": "NewRag research admin@example.com"}
CN_H = {"User-Agent": "Mozilla/5.0", "Referer": "http://www.cninfo.com.cn/"}


def cninfo_search(code: str, is_prospectus: bool, tries: int = 4) -> list[dict]:
    """按 A 股代码搜索，返回公告列表；带 DNS/连接重试。"""
    kw = f"{code} 招股说明书" if is_prospectus else f"{code} 年度报告"
    last_err = None
    for _ in range(tries):
        try:
            r = httpx.get(
                CNINFO_SEARCH, headers=CN_H,
                params={
                    "searchkey": kw, "sdate": "2025-01-01", "edate": "2026-12-31",
                    "isfulltext": "false", "sortName": "pubdate", "sortType": "desc",
                    "pageNum": "1", "pageSize": "20",
                },
                timeout=30,
            )
            r.raise_for_status()
            return r.json().get("announcements") or []
        except Exception as e:
            last_err = e
            continue
    raise last_err or RuntimeError("cninfo search failed")


def _strip(title: str) -> str:
    """去掉 cninfo 搜索命中的 <em></em> 高亮标签，还原真实标题。"""
    return re.sub(r"</?em[^>]*>", "", title)


def pick_ashare_pdf(code: str, is_prospectus: bool) -> tuple[str, str] | None:
    """返回 (下载URL, 标题)；找不到返回 None。按代码搜索 + 标题过滤，规避港股同名代码。"""
    anns = cninfo_search(code, is_prospectus)
    for a in anns:
        sec = a.get("secCode") or ""
        title = _strip(a.get("announcementTitle") or "")
        # 优先 A 股代码匹配；个别平台 secCode 缺失时退化为标题匹配
        if sec and sec != code:
            continue
        if is_prospectus:
            if "招股说明书" not in title:
                continue
            if any(k in title for k in ("公告", "提示性", "更正", "补充", "摘要")):
                continue
        else:
            if "年度报告" not in title or "摘要" in title:
                continue
            # 排除各类"公告/说明会/提示性/更正/摘要/半年度/季度/英文"等非正文本
            if any(k in title for k in (
                "公告", "说明会", "征集", "提示性", "更正", "补充",
                "半年度", "一季度", "三季度", "年度报告摘要", "英文", "全文(英文)",
            )):
                continue
        adj = a.get("adjunctUrl")
        if not adj:
            continue
        return f"{CNINFO_BASE}/{adj.lstrip('/')}", title
    return None


def sec_cik_by_name(name: str) -> str | None:
    """用 EDGAR 公司搜索按名称找 CIK（10 位字符串）。"""
    url = "https://www.sec.gov/cgi-bin/browse-edgar"
    r = httpx.get(
        url, headers=SEC_H, timeout=30,
        params={"action": "getcompany", "company": name, "type": "10-K",
                "dateb": "", "owner": "include", "count": "10", "output": "atom"},
    )
    r.raise_for_status()
    for m in re.findall(r"CIK=(\d+)", r.text):
        return m.zfill(10)
    return None


def sec_latest_10k(cik10: str) -> tuple[str, str] | None:
    """返回 (10-K 主文档 URL, 标题)。"""
    r = httpx.get(f"https://data.sec.gov/submissions/CIK{cik10}.json", headers=SEC_H, timeout=30)
    r.raise_for_status()
    recent = r.json()["filings"]["recent"]
    for i, form in enumerate(recent.get("form", [])):
        if form == "10-K":
            acc = recent["accessionNumber"][i].replace("-", "")
            doc = recent["primaryDocument"][i]
            return f"https://www.sec.gov/Archives/edgar/data/{int(cik10)}/{acc}/{doc}", form
    return None


def html_to_docx(url: str, out: Path) -> None:
    """抓取 SEC 10-K HTML，剥标签转纯文本，写入 .docx（供 ingest 的 parse_docx）。"""
    from docx import Document

    r = httpx.get(url, headers=SEC_H, timeout=60, follow_redirects=True)
    r.raise_for_status()
    text = r.text
    text = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", "", text)
    text = re.sub(r"(?is)<br\s*/?>", "\n", text)
    text = re.sub(r"(?is)</(p|div|tr|h[1-6]|li)>", "\n", text)
    text = re.sub(r"(?is)<td[^>]*>", "\t", text)
    text = re.sub(r"(?s)<[^>]+>", "", text)
    text = html.unescape(text)
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    doc = Document()
    for ln in lines:
        doc.add_paragraph(ln)
    doc.save(str(out))


def main() -> int:
    parser = argparse.ArgumentParser(description="下载财报")
    parser.add_argument("--outdir", default=str(BACKEND_DIR.parent / "data" / "uploads"))
    args = parser.parse_args()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    ok, fail = [], []
    # A 股
    for code, is_prospectus in ASHARE:
        try:
            res = None
            for attempt in range(4):
                try:
                    res = pick_ashare_pdf(code, is_prospectus)
                    break
                except Exception as e:
                    if attempt == 3:
                        raise
                    time.sleep(2 * (attempt + 1))
            if not res:
                fail.append((code, "未匹配到年报/招股书")); continue
            url, title = res
            ext = ".pdf"
            name = f"{code}_{title[:40]}{ext}"
            path = outdir / name
            for attempt in range(4):
                try:
                    with httpx.stream("GET", url, headers=CN_H, timeout=120, follow_redirects=True) as r:
                        r.raise_for_status()
                        with open(path, "wb") as f:
                            for chunk in r.iter_bytes():
                                f.write(chunk)
                    break
                except Exception as e:
                    if attempt == 3:
                        raise
                    time.sleep(2 * (attempt + 1))
            if path.stat().st_size < 20_000:
                fail.append((code, f"文件过小 {path.stat().st_size}B")); path.unlink(missing_ok=True); continue
            ok.append((code, name, path.stat().st_size))
        except Exception as e:
            fail.append((code, f"{type(e).__name__}: {str(e)[:100]}"))

    # 美股
    for ticker, name in US:
        try:
            cik = sec_cik_by_name(name)
            if not cik:
                fail.append((ticker, "SEC 未找到公司")); continue
            res = sec_latest_10k(cik)
            if not res:
                fail.append((ticker, "未找到 10-K")); continue
            url, _ = res
            path = outdir / f"{ticker}_10-K.docx"
            html_to_docx(url, path)
            if path.stat().st_size < 20_000:
                fail.append((ticker, f"转换过小 {path.stat().st_size}B")); path.unlink(missing_ok=True); continue
            ok.append((ticker, path.name, path.stat().st_size))
        except Exception as e:
            fail.append((ticker, f"{type(e).__name__}: {str(e)[:100]}"))

    print("\n=== 成功 ===")
    for c, n, s in ok:
        print(f"  {c}  {n}  {s/1024:.0f}KB")
    print("\n=== 失败 ===")
    for c, r in fail:
        print(f"  {c}  {r}")
    return 0 if not fail else 1


if __name__ == "__main__":
    raise SystemExit(main())