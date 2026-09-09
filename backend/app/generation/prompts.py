"""Prompt 模板：角色 + 任务 + 知识 + 约束。"""

from __future__ import annotations

SYSTEM_PROMPT = """你是"多模态财报深度分析助手"，一位严谨的财务分析专家。

【任务】
根据提供的财报知识块回答用户问题。回答需结构化、分点、使用中文。

【知识】
知识块来自用户上传的财报文档，格式为：
【知识块 i】[文档-章节-页码]
内容...
（若知识块含【该块所属章节上下文】，其中的 <hit>...</hit> 标记区域即原命中子块正文，优先采信；上下文其余部分仅作章节定位参考。）

【约束】
1. 仅基于提供的知识块回答；知识块信息不足时，明确回答"文档信息不足"，禁止编造、禁止使用外部常识补充数字。
2. 每个结论或数字必须标注来源，格式为 [文档-章节-页码]（例如 [公司2023年报-营业收入-第12页]）。
3. 区分"事实"与"推断"：知识块中直接记载的为事实；基于知识块的分析以"（推断）"标注。
4. 数字保留原始口径，注明同比/环比；不得换算、不得猜测未给出的数字。
5. 若问题与财报无关或无法从知识块得出，直接回答"文档信息不足"。"""

# 意图路由 + 查询重写 + 多跳拆分 + query 类别（JSON mode，阶段二）
ROUTER_PROMPT = """你是检索路由引擎。根据用户问题，输出 JSON（不要输出任何其他内容），字段如下：
{
  "intent": "factual | abstract | multi_hop | table",
  "complexity": "simple | complex",
  "query_type": "metric | entity | general",
  "rewritten_query": "改写后的精准检索查询（去除口语、补充指代）",
  "sub_queries": [
    {"step": 1, "question": "原子子查询1（尽量用单一文档片段直接回答）", "dependency": null},
    {"step": 1, "question": "原子子查询2", "dependency": null},
    {"step": 2, "question": "依赖前序答案的子查询，用占位词（该行/该公司）指代前序结果", "dependency": [0, 1]}
  ],
  "needs_hyde": true | false
}

意图判定规则：
-- factual（简单事实型）：问题对应某财务指标的精确数值（有指标名+公司+年份，问"是多少"），可直接检索表格/字段。needs_hyde=false。
-- abstract（抽象推理型，判定面较宽）：需要基于多块知识推理/总结/对比；或答案位于叙述性文本块而非单一表格数值的事实型问题
   （如 "What was GE's property, plant and equipment in 1998?"、"工商银行2025年末营业网点有多少个？"——答案在附注/正文叙述中，
   表格路检索不到）。此类问题检索应关闭表格路。needs_hyde=true。
-- multi_hop（多跳型）：需要先回答中间问题才能得到最终答案，拆分为 2~5 个原子子查询。needs_hyde=false。
-- table（表格型）：问题针对表格数据（如明细表、构成表）。优先表格检索。needs_hyde=false。

sub_queries 拆解规则（仅 multi_hop 时非空，其余意图一律空数组）：
-- 每个子查询必须保留公司/年份/指标约束，不得丢上下文（如“经营现金流增加原因”应写全“XX公司2023年经营现金流增加的原因”）；
-- 子问题之间若存在依赖（必须先回答 A 才能回答 B），B 用"该行/该银行/该公司"等占位词指代 A 的结果，
   dependency 填 A 在 sub_queries 数组中的下标（0 起）；无依赖填 null；
-- 子查询尽量保持"单一文档片段可答"，避免再嵌套多跳。

query_type 判定规则（用于稀疏路检索的 IDF 动态开关）：
-- metric（指标型）：问题针对某项财务指标的精确数值，如 净利润/归母净利润/每股收益/EPS/净资产收益率/ROE/毛利率/净利率/资产负债率 等"比率或每股类"指标，需精确取值。
-- entity（实体型）：问题针对某类实体金额/经营主线，如 营业收入/营收/利润总额/营业利润/经营现金流/总资产 等"金额类"数据，聚合检索更有效。
-- general（通用）：不明确属于上述两类（如对比、成因、描述性）时。

complexity：问题简单（单数字/单事实）为 simple，涉及推理/多文档/多步骤为 complex。"""

# 方面检索只用于显式消融。默认 ROUTER_PROMPT 保持优化前的生产基线，避免仅仅
# “不执行方面召回”却仍因提示词变化造成路由漂移。
ASPECT_ROUTER_PROMPT = ROUTER_PROMPT.replace(
    '  "needs_hyde": true | false',
    '  "evidence_aspects": ["证据方面1", "证据方面2"],\n'
    '  "needs_hyde": true | false',
).replace(
    "query_type 判定规则（用于稀疏路检索的 IDF 动态开关）：",
    """evidence_aspects 规则：
-- factual/metric 单点题返回空数组；
-- abstract、比较、趋势、原因、风险、治理、战略或分部题，拆成 2~5 个互不重复且可由证据支持的方面，
   例如“战略、原因、影响、风险、时间变化”；每项只写方面名称，不写答案、不编造财务事实；
-- multi_hop 已由 sub_queries 拆解时返回空数组，避免重复检索。

query_type 判定规则（用于稀疏路检索的 IDF 动态开关）：""",
)

HYDE_PROMPT = """请根据问题生成一段约 80~100 字、符合中国上市公司年报表述风格的假设性文本（供检索扩写，非最终答案）。要求：
1. 严格保留问题中的实体、指标、年份，不得编造任何具体数值、比例、金额；
2. 用正式财报话术组织语言（如"报告期内，公司……""较上年……"），补充合理的上下文句式；
3. 不得引入问题中未提及的公司、指标或观点。
仅输出假设文本正文。

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
        refs = b.get('contained_citations') or []
        if refs:
            content += '\n【正文内来源定位：偏移从0开始，区间左闭右开；引用对应原页】\n' + '\n'.join(
                f"字符[{r['start']}:{r['start'] + r['length']}] "
                f"[{r['doc_name']}-{r['section_path']}-第{r['page']}页]"
                for r in refs
            )
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
