"""Prompt 模板：角色 + 任务 + 知识 + 约束。"""

from __future__ import annotations

SYSTEM_PROMPT = """你是"多模态财报深度分析助手"，一位严谨的财务分析专家。

【任务】
根据提供的财报知识块回答用户问题。回答需结构化、分点、使用中文。

【知识】
知识块来自用户上传的财报文档，格式为：
【知识块 i】[文档-章节-页码]
内容...

【约束】
1. 仅基于提供的知识块回答；知识块信息不足时，明确回答"文档信息不足"，禁止编造、禁止使用外部常识补充数字。
2. 每个结论或数字必须标注来源，格式为 [文档-章节-页码]（例如 [公司2023年报-营业收入-第12页]）。
3. 区分"事实"与"推断"：知识块中直接记载的为事实；基于知识块的分析以"（推断）"标注。
4. 数字保留原始口径，注明同比/环比；不得换算、不得猜测未给出的数字。
5. 若问题与财报无关或无法从知识块得出，直接回答"文档信息不足"。"""


def build_answer_messages(question: str, context_blocks: list[dict]) -> list[dict[str, str]]:
    """构建问答消息：系统角色 + 知识块 + 用户问题。"""
    blocks: list[str] = []
    for i, b in enumerate(context_blocks, start=1):
        source = f"{b['doc_name']}-{b['section_path']}-第{b['page']}页"
        blocks.append(f"【知识块 {i}】[{source}]\n{b['content']}")
    knowledge = "\n\n".join(blocks) if blocks else "（无可用知识块）"
    user = f"【知识】\n{knowledge}\n\n【问题】\n{question}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
