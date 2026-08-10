"""结构块检测：识别财务报表"表头/页眉/版式说明"类 chunk。

背景（见稀疏路诊断）：这些块含 query 的泛化结构词（`2024`/`本集团`/`财务报表`/`人民币百万元`），
在稀疏/稠密检索里 score 高但非答案所在，挤占 top_k。与 clean 层页眉剔除互补——clean 处理行级
页眉/页脚，这里面向"表格块的表头/版式说明"这类结构化噪音。

保守规则（多条件 AND，避免误伤有信息量的块）：
- 内容较短（token < 阈值）；
- 且命中强版式特征（金额列示说明 / 止年度财务报表 / 单位说明 / 本集团 本行 表头）。
"""

from __future__ import annotations

import re

from app.splitter.tokens import count_tokens

# 强版式/表头特征（命中其一且短即判）
_STRONG_RE = [
    r"除另有标明外",
    r"金额(均)?(以|按)?人民币",
    r"单位[:：]?\s*(?:人民币)?(?:百万元|千元|万元|元)",
    r"止年度财务报表",
    r"本集团\s*本行",
    r"截至\s*(?:19|20)\d{2}\s*年(?:\d{1,2}\s*月)?\s*\d{1,2}\s*日?止",
]
# 弱特征（需与强特征或互相叠加才判）
_WEAK_RE = [
    r"财务报表",
    r"合并\s*(?:利润表|资产负债表|现金流量表)",
    r"列示",
    r"人民币百万元",
]

_STRONG_P = [re.compile(x) for x in _STRONG_RE]
_WEAK_P = [re.compile(x) for x in _WEAK_RE]

_MAX_TOKENS = 90  # token 上限：结构块通常极短


def is_structural_chunk(content: str) -> bool:
    """判断一个 chunk 是否为"财务表头/页眉/版式说明"类结构噪音。"""
    if not content or not content.strip():
        return False
    if count_tokens(content) > _MAX_TOKENS:
        return False
    strong = sum(1 for p in _STRONG_P if p.search(content))
    if strong >= 1:
        return True
    weak = sum(1 for p in _WEAK_P if p.search(content))
    return weak >= 2  # 多个弱特征叠加（如"财务报表"+“列示”）也判为结构噪音