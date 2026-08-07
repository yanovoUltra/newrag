"""生成一份合成财报 PDF 作为演示/测试样例。

用法: python scripts/gen_sample_report.py
输出: data/samples/sample_report.pdf（100 页以内，含章节与表格）
"""

from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF

OUT = Path(__file__).resolve().parents[1] / "data" / "samples" / "sample_report.pdf"

SECTIONS = [
    ("一、公司概况", [
        "本公司成立于2010年，主营业务为智能硬件研发、生产与销售，产品覆盖消费电子与工业物联网两大领域。",
        "截至2023年末，公司员工总数12,800人，其中研发人员占比32%。",
    ]),
    ("二、营业收入", [
        "2023年度公司实现营业收入人民币86.5亿元，同比增长18.2%；其中海外收入占比41.5%，同比增长24.7%。",
        "分产品来看：智能终端收入52.3亿元，同比增长12.1%；物联网解决方案收入25.8亿元，同比增长31.9%；其他业务收入8.4亿元。",
    ]),
    ("三、净利润", [
        "2023年度实现归属于上市公司股东的净利润人民币7.9亿元，同比增长22.6%。",
        "扣除非经常性损益后的净利润为6.8亿元，同比增长19.4%。每股收益1.24元，同比增长22.8%。",
    ]),
    ("四、资产负债表", [
        "截至2023年12月31日，公司总资产人民币112.6亿元，总负债42.3亿元，资产负债率37.6%。",
        "货币资金19.8亿元，应收账款14.2亿元，存货11.5亿元；短期借款6.3亿元，长期借款8.9亿元。",
    ]),
    ("五、现金流量", [
        "2023年度经营活动产生的现金流量净额12.4亿元，同比增长35.9%；投资活动现金流量净额-6.1亿元；筹资活动现金流量净额-3.2亿元。",
    ]),
    ("六、主要财务指标", [
        "毛利率29.6%，净利率9.1%，净资产收益率12.8%，研发投入占营收比例8.9%。",
    ]),
]

# 三张表格：按季度拆分与资产负债表简表
TABLES = [
    {
        "title": "表1 分季度主要财务数据",
        "headers": ["季度", "营业收入(亿元)", "净利润(亿元)", "同比增长率"],
        "rows": [
            ["Q1", "18.9", "1.6", "14.2%"],
            ["Q2", "21.3", "1.9", "17.5%"],
            ["Q3", "22.8", "2.1", "22.9%"],
            ["Q4", "23.5", "2.3", "26.8%"],
        ],
    },
    {
        "title": "表2 分产品毛利率",
        "headers": ["产品", "收入(亿元)", "毛利率", "同比变化"],
        "rows": [
            ["智能终端", "52.3", "25.1%", "-1.2pct"],
            ["物联网解决方案", "25.8", "38.4%", "+2.3pct"],
            ["其他", "8.4", "22.0%", "+0.5pct"],
        ],
    },
]


def build() -> Path:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    page_no = 1

    def new_page() -> fitz.Page:
        nonlocal page_no
        p = doc.new_page()
        page_no += 1
        return p

    # 封面
    p = new_page()
    p.insert_text((72, 100), "XX智能科技股份有限公司", fontsize=20)
    p.insert_text((72, 130), "2023年年度报告（示例）", fontsize=16)
    p.insert_text((72, 160), "报告期：2023年1月1日至2023年12月31日", fontsize=11)

    for title, paras in SECTIONS:
        p = new_page()
        y = 72
        p.insert_text((72, y), title, fontsize=15)
        y += 24
        for para in paras:
            p.insert_text((72, y), para, fontsize=11)
            y += 18

    # 表格页
    for t in TABLES:
        p = new_page()
        y = 72
        p.insert_text((72, y), t["title"], fontsize=14)
        y += 20
        x0 = 72
        # 表头
        for x, h in zip(_x_positions(), t["headers"]):
            p.insert_text((x, y), h, fontsize=10)
        y += 16
        for row in t["rows"]:
            for col, cell in zip(_x_positions(), row):
                p.insert_text((col, y), cell, fontsize=10)
            y += 15

    doc.save(OUT)
    print(f"生成样例财报: {OUT} (共 {page_no - 1} 页)")
    return OUT


def _x_positions() -> list[float]:
    return [72, 190, 300, 410, 490]


if __name__ == "__main__":
    build()
