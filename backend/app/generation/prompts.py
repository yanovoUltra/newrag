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

# 意图路由 + 查询重写 + 多跳拆分（JSON mode，阶段二）
ROUTER_PROMPT = """你是检索路由引擎。根据用户问题，输出 JSON（不要输出任何其他内容），字段如下：
{
  "intent": "factual | abstract | multi_hop | table",
  "complexity": "simple | complex",
  "rewritten_query": "改写后的精准检索查询（去除口语、补充指代），
  "sub_queries": ["多跳问题时拆分的 2~3 个子查询；非多跳时为空数组"],
  "needs_hyde": true | false
}

意图判定规则：
- factual（简单事实型）：问题直接对应某数字/事实，可直接检索。needs_hyde=false。
- abstract（抽象推理型）：需要基于多块知识推理/总结/对比。needs_hyde=true。
- multi_hop（多跳型）：需要先回答中间问题才能得到最终答案，拆分为 2~3 个子查询。needs_hyde=false。
- table（表格型）：问题针对表格数据（如明细表、构成表）。优先表格检索。needs_hyde=false。

complexity：问题简单（单数字/单事实）为 simple，涉及推理/多文档/多步骤为 complex。"""

HYDE_PROMPT = """请根据问题写一段约 80~150 字的"假设文档"，内容为一篇虚构财报中可能的原文表述（直接陈述数字、指标与口径，不包含提问、分析或"根据以上"等元话语）。仅输出假设文档正文。

问题：{question}"""


def build_answer_messages(
    question: str,
    context_blocks: list[dict],
    history: list[dict] | None = None,
) -> list[dict[str, str]]:
    """构建问答消息：系统角色 + 历史对话（可选） + 知识块 + 当前问题。

    context_blocks 可含 parent_content（所属章节父块），作为该块的章节上下文附加。
    history 为 [{role: user|assistant, content}]，按顺序插入系统消息之后、当前问题之前。
    """
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in history or []:
        if h.get("role") in ("user", "assistant"):
            messages.append({"role": h["role"], "content": h["content"]})
    blocks: list[str] = []
    for i, b in enumerate(context_blocks, start=1):
        source = f"{b['doc_name']}-{b['section_path']}-第{b['page']}页"
        content = b["content"]
        parent = b.get("parent_content") or ""
        if parent and b.get("chunk_type") != "section":
            blocks.append(
                f"【知识块 {i}】[{source}]\n{content}\n\n【该块所属章节上下文】\n{parent}"
            )
        else:
            blocks.append(f"【知识块 {i}】[{source}]\n{content}")
    knowledge = "\n\n".join(blocks) if blocks else "（无可用知识块）"
    user = f"【知识】\n{knowledge}\n\n【问题】\n{question}"
    messages.append({"role": "user", "content": user})
    return messages
