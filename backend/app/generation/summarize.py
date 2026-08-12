"""入库阶段 LLM 提炼（阶段四验证）：章节父块摘要 + 文档级综述。

设计：摘要块**只做检索锚点**——检索命中摘要块后由 search 层展开为原文父块
（原文供生成，citation 指向原文），摘要本身不进生成上下文，防 LLM 编造数字/引用。

- `summarize_section`：章节父块 → 100-200 字摘要（保留公司/财年/关键数字/结论）
- `summarize_document`：整份财报 → 200-300 字综述（指标概览/业务/风险/章节结构）
- `summarize_doc_sections`：同步批量入口（ingest / 独立脚本用），内部并发 + 限流
"""

from __future__ import annotations

import asyncio
import time

from app.core.config import get_settings
from app.core.logging import get_logger
from app.generation.llm import get_llm

logger = get_logger(__name__)

SECTION_SUMMARY_PROMPT = (
    "你是财报分析助手。请用不超过 200 字提炼以下财报章节的核心内容，"
    "必须保留：公司/机构名、财年、关键数字与指标、结论性表述。"
    "只输出提炼结果本身，不要前缀或解释。\n\n"
    "章节：{title}\n内容：\n{content}"
)

DOC_SUMMARY_PROMPT = (
    "你是财报分析助手。请用不超过 300 字综述这份财报，覆盖：公司/机构与财年、"
    "核心财务指标概览、主要业务与经营亮点、风险与展望、章节结构。"
    "只输出综述本身，不要前缀或解释。\n\n"
    "文档：{doc_name}\n章节概览：\n{chapters}"
)

_INPUT_MAX_CHARS = 6000  # 摘要输入截断（防超长章节超出 LLM 上下文）
_OUTPUT_MAX_TOKENS = 256


def _truncate(text: str, n: int = _INPUT_MAX_CHARS) -> str:
    if len(text) <= n:
        return text
    return text[:n] + "……（已截断）"


async def summarize_section(title: str, content: str) -> str:
    """章节父块 → 摘要（异步，内部走 LLM 限流与重试）。"""
    llm = get_llm()
    prompt = SECTION_SUMMARY_PROMPT.format(title=title or "（无标题）", content=_truncate(content))
    resp = await llm.chat(
        [{"role": "user", "content": prompt}],
        model=get_settings().llm_model,
    )
    return resp.strip()


async def summarize_document(doc_name: str, chapters: str) -> str:
    """整份财报 → 综述（chapters=章节标题概览串）。"""
    llm = get_llm()
    prompt = DOC_SUMMARY_PROMPT.format(doc_name=doc_name, chapters=_truncate(chapters))
    resp = await llm.chat(
        [{"role": "user", "content": prompt}],
        model=get_settings().llm_model,
    )
    return resp.strip()


async def _summarize_all(
    sections: list[tuple[str, str]],
    doc_name: str,
    doc_chapters: str,
) -> tuple[dict[str, str], str]:
    """并发生成全部章节摘要 + 文档综述（LLM 内部 FairSemaphore 限流）。"""
    results: list[tuple[str, str]] = []
    doc_summary = ""
    try:
        outs = await asyncio.gather(
            *[summarize_section(t, c) for t, c in sections]
        )
        results = [(sections[i][0], outs[i]) for i in range(len(sections))]
    except Exception as e:  # noqa: BLE001 摘要失败不阻断入库（检索有原文兜底）
        logger.warning("章节摘要批量生成失败（跳过）: %s", e)
    try:
        doc_summary = await summarize_document(doc_name, doc_chapters)
    except Exception as e:  # noqa: BLE001
        logger.warning("文档综述生成失败（跳过）: %s", e)
    return results, doc_summary


def summarize_doc_sections(
    sections: list[tuple[str, str]],
    doc_name: str,
    doc_chapters: str,
) -> tuple[list[tuple[str, str]], str]:
    """同步批量入口：sections=[(父块 chunk_id, 原文)], 返回 ([(chunk_id, 摘要)], 文档综述)。
    失败块自动跳过（返回时过滤空摘要），耗时随章节数线性（并发=LLM 限流）。"""
    t0 = time.perf_counter()
    results, doc_summary = asyncio.run(_summarize_all(sections, doc_name, doc_chapters))
    ok = [(cid, s) for cid, s in results if s]
    logger.info(
        "文档提炼完成: %s 章节成功 %d/%d，文档综述 %d 字，耗时 %.1fs",
        doc_name, len(ok), len(sections), len(doc_summary), time.perf_counter() - t0,
    )
    return ok, doc_summary
