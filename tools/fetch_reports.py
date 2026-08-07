"""财报数据获取工具（离线数据准备，非实时行情）。

支持三路数据源，输出统一落到 data/raw_reports/ 下，之后走系统现有上传入库管线：
  - sec AAPL    ：SEC EDGAR 官方 API，下载 10-K/10-Q 年报原文 → 转 DOCX
  - yahoo AAPL   ：yfinance（Yahoo Finance），三大报表 → 多 sheet Excel
  - cninfo 600000：巨潮资讯公告接口，下载 A 股年报 PDF 原文

合规说明：
- 数据仅用于个人研究 / 本地 RAG 知识库，不得商用再分发（Yahoo 数据源条款）。
- 仅调用公开数据接口，不含任何登录态 / 会话 cookie。


"""

from __future__ import annotations

import argparse
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx

# SEC 要求 UA 附带联系方式，否则 403
SEC_UA = {
    "User-Agent": "financial-report-rag research demo contact: admin@example.com",
    "Accept-Encoding": "gzip, deflate",
}
SEC_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{padded:010d}.json"
SEC_BROWSE = "https://www.sec.gov/cgi-bin/browse-edgar"
SEC_ARCHIVES = "https://www.sec.gov/Archives/edgar/data/{cik}/"

CNINFO_QUERY = "http://www.cninfo.com.cn/new/hisAnnouncement/query"
CNINFO_STATIC = "http://static.cninfo.com.cn/{path}"
CNINFO_STOCKLIST = "http://www.cninfo.com.cn/new/data/szse_stock.json"

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "data" / "raw_reports"


# ---------------------------------------------------------------- 公共
def _out(sub: str) -> Path:
    d = DEFAULT_OUT / sub
    d.mkdir(parents=True, exist_ok=True)
    return d


def _safe_name(name: str) -> str:
    return re.sub(r"[^\w\-.（）()一-龥]", "_", name)


def _extract_tables(soup) -> list[list[list[str]]]:
    """把 HTML <table> 转为结构化行（对齐 ParsedTable.to_text 的管道格式），并移除表格节点避免文本重复。"""
    tables: list[list[list[str]]] = []
    for tbl in soup.find_all("table"):
        rows: list[list[str]] = []
        for tr in tbl.find_all("tr"):
            cells = [" ".join(c.get_text(" ", strip=True).split()) for c in tr.find_all(["td", "th"])]
            if any(cells):
                rows.append(cells)
        # 过滤布局表：不足 2 行或不足 2 列多为页面排版而非数据表
        if len(rows) >= 2 and len(rows[0]) >= 2:
            tables.append(rows)
        tbl.decompose()
    return tables


# SEC 原始披露中的页眉/页脚/目录残留（逐行剔除）
_SEC_NOISE_LINE = re.compile(
    r"^table of contents$|^10-[kq]\s+[\w\-]*\.?htm$|^https?://\S+$|^page\s*\d+\s*(of\s*\d+)?$",
    re.IGNORECASE,
)


def _html_to_content(html: str) -> tuple[str, list[list[list[str]]]]:
    """HTML → (正文文本, 表格)。正文剔除表格区域，表格转结构化行。"""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    tables = _extract_tables(soup)
    lines: list[str] = []
    for ln in soup.get_text("\n").split("\n"):
        s = ln.strip()
        if not s or _SEC_NOISE_LINE.match(s):
            continue
        lines.append(s)
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(lines))
    return text.strip(), tables


def _write_docx(path: Path, title: str, body: str, tables: list[list[list[str]]] | None = None) -> None:
    from docx import Document

    doc = Document()
    doc.add_heading(title, level=0)
    for para in body.split("\n"):
        para = para.strip()
        if not para:
            continue
        if re.match(r"^[0-9]+\.$", para):  # 页码
            continue
        doc.add_paragraph(para)
    for rows in tables or []:
        n_cols = max(len(r) for r in rows)
        t = doc.add_table(rows=len(rows), cols=n_cols)
        for i, row in enumerate(rows):
            for j, cell in enumerate(row):
                t.cell(i, j).text = cell
    doc.save(str(path))


# ---------------------------------------------------------------- SEC EDGAR
def _lookup_cik(ticker: str) -> str:
    resp = httpx.get(
        SEC_BROWSE,
        params={
            "action": "getcompany",
            "CIK": ticker,
            "type": "10-K",
            "dateb": "",
            "owner": "include",
            "count": "1",
            "output": "atom",
        },
        headers=SEC_UA,
        timeout=30,
    )
    resp.raise_for_status()
    m = re.search(r"<cik>(\d+)</cik>", resp.text)
    if not m:
        raise RuntimeError(f"未找到 {ticker} 的 CIK")
    return m.group(1)


def fetch_sec(ticker: str, form: str = "10-K", year: int | None = None) -> list[Path]:
    """下载指定表格（默认 10-K）年报原文并转 DOCX。返回生成的文件路径。"""
    cik = _lookup_cik(ticker)
    data = httpx.get(SEC_SUBMISSIONS.format(padded=int(cik)), headers=SEC_UA, timeout=30).json()
    recent = data["filings"]["recent"]
    found = []
    for i, f in enumerate(recent["form"]):
        if f != form:
            continue
        filed = recent["filingDate"][i]
        if year is not None and not filed.startswith(str(year)):
            continue
        acc = recent["accessionNumber"][i]
        primary = recent["primaryDocument"][i]
        doc_url = f"{SEC_ARCHIVES.format(cik=int(cik))}{acc.replace('-', '')}/{primary}"
        resp = httpx.get(doc_url, headers=SEC_UA, timeout=60)
        resp.raise_for_status()
        body, tables = _html_to_content(resp.text)
        if len(body) < 2000:  # 首页/封面，跳过
            continue
        # 表格过多时（如含 XBRL 明细），超量表格转管道文本并入正文，避免 DOCX 体积失控
        MAX_TABLES = 200
        if len(tables) > MAX_TABLES:
            pipe_lines = [" | ".join(r) for t in tables[MAX_TABLES:] for r in t]
            body = f"{body}\n\n" + "\n".join(pipe_lines) if body else "\n".join(pipe_lines)
            tables = tables[:MAX_TABLES]
        title = f"{ticker.upper()} {form} 年报（SEC 原始披露，filed {filed}）"
        out_path = _out("sec") / f"{ticker.upper()}_{form}_{filed}.docx"
        _write_docx(out_path, title, body, tables)
        print(f"[sec] {ticker} {form} {filed} → {out_path}（{len(body)} 字符，{len(tables)} 表格）")
        found.append(out_path)
        if year is not None:  # 指定年份只取最近一份
            break
        if len(found) >= 2:
            break
        time.sleep(0.2)  # SEC 限速
    if not found:
        print(f"[sec] {ticker} 未找到 {form}{f'/{year}' if year else ''} 申报记录")
    return found


# ---------------------------------------------------------------- yfinance
def fetch_yahoo(ticker: str) -> Path:
    """三大报表（年度+季度）→ 多 sheet Excel。"""
    import yfinance as yf

    t = yf.Ticker(ticker)
    sheets = {
        "利润表_年度": t.income_stmt,
        "资产负债表_年度": t.balance_sheet,
        "现金流量表_年度": t.cashflow,
        "利润表_季度": t.quarterly_income_stmt,
        "资产负债表_季度": t.quarterly_balance_sheet,
        "现金流量表_季度": t.quarterly_cashflow,
    }
    from openpyxl import Workbook

    wb = Workbook()
    wb.remove(wb.active)
    n = 0
    for name, df in sheets.items():
        if df is None or df.empty:
            continue
        ws = wb.create_sheet(name[:31])
        # 转置：行 = 报告期，列 = 指标
        td = df.T
        ws.append(["报告期"] + [str(c) for c in td.columns])
        for idx, row in td.iterrows():
            ws.append([str(idx)] + ["" if v is None else str(v) for v in row])
        n += 1
    if n == 0:
        raise RuntimeError(
            f"{ticker} 无财务数据。若此前能联网，很可能是 Yahoo Finance 对当前网络（IP/地区）返回 403 封锁，"
            f"请换用 sec 子命令（SEC 年报原文）或代理网络后重试。"
        )
    out_path = _out("yahoo") / f"{ticker.upper()}_financials.xlsx"
    wb.save(str(out_path))
    print(f"[yahoo] {ticker} 三大报表 → {out_path}（{n} 个 sheet）")
    return out_path


# ---------------------------------------------------------------- 巨潮资讯
def _cninfo_org_id(stock_code: str) -> tuple[str, str]:
    """从巨潮股票列表查询 orgId 与市场列（'sse'/'szse'）。"""
    resp = httpx.get(CNINFO_STOCKLIST, headers={"User-Agent": SEC_UA["User-Agent"]}, timeout=30)
    resp.raise_for_status()
    for row in resp.json().get("stockList", []):
        if row.get("code") == stock_code:
            org = row.get("orgId", "")
            return org, ("sse" if org.startswith("gssh") else "szse")
    raise RuntimeError(f"巨潮资讯未收录代码 {stock_code}")


def fetch_cninfo(stock_code: str, year: int | None = None) -> Path | None:
    """按代码（6 位）搜索并下载最近一期年报 PDF。"""
    today = datetime.now()
    if year is None:
        year = today.year - 1  # 年报通常在次年 3-4 月披露
    se_date = f"{year}-01-01~{year + 1}-06-30"
    org_id, column = _cninfo_org_id(stock_code)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        "X-Requested-With": "XMLHttpRequest",
        "Referer": f"http://www.cninfo.com.cn/new/disclosure/stock?stockCode={stock_code}",
    }
    data = {
        "pageNum": "1",
        "pageSize": "30",
        "column": column,
        "tabName": "fulltext",
        "plate": "",
        "stock": f"{stock_code},{org_id}",  # 必须 secCode,orgId 双值，否则过滤器失效
        "searchkey": "",
        "secid": "",
        "category": "category_ndbg_szsh",
        "trade": "",
        "seDate": se_date,
        "sortName": "",
        "sortType": "",
        "isHLtitle": "true",
    }
    resp = httpx.post(CNINFO_QUERY, data=data, headers=headers, timeout=30)
    resp.raise_for_status()
    anns = resp.json().get("announcements", [])
    # 优先纯年报（排除摘要/英文版）
    target = None
    for a in anns:
        title = a.get("announcementTitle", "")
        if f"{year}年年度报告" in title and "摘要" not in title and "英文" not in title:
            target = a
            break
    if target is None:
        print(f"[cninfo] {stock_code} 未找到 {year} 年年报")
        return None
    pdf_url = CNINFO_STATIC.format(path=target["adjunctUrl"])
    title = target["announcementTitle"]
    pdf = httpx.get(pdf_url, headers=headers, timeout=120)
    pdf.raise_for_status()
    out_path = _out("cninfo") / f"{stock_code}_{year}年报.pdf"
    out_path.write_bytes(pdf.content)
    print(f"[cninfo] {title} → {out_path}（{len(pdf.content) // 1024} KB）")
    return out_path


# ---------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="财报数据获取工具（SEC / yfinance / 巨潮），输出到 data/raw_reports/"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_sec = sub.add_parser("sec", help="SEC EDGAR 年报原文 → DOCX")
    p_sec.add_argument("ticker")
    p_sec.add_argument("--form", choices=["10-K", "10-Q"], default="10-K")
    p_sec.add_argument("--year", type=int, default=None)
    p_sec.set_defaults(func=lambda a: fetch_sec(a.ticker, a.form, a.year))

    p_yahoo = sub.add_parser("yahoo", help="yfinance 三大报表 → Excel")
    p_yahoo.add_argument("ticker")
    p_yahoo.set_defaults(func=lambda a: fetch_yahoo(a.ticker))

    p_cninfo = sub.add_parser("cninfo", help="巨潮资讯 A 股年报 PDF")
    p_cninfo.add_argument("stock_code")
    p_cninfo.add_argument("--year", type=int, default=None)
    p_cninfo.set_defaults(func=lambda a: fetch_cninfo(a.stock_code, a.year))

    args = parser.parse_args(argv)
    try:
        args.func(args)
    except Exception as e:
        print(f"[错误] {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
