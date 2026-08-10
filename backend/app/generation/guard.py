"""Agent 防线（Guard 层）：Prompt 注入检测 + 会话循环检测。

轻量版只做两个最实用的守卫，挂在 answer 管线入口：
  1. 注入检测：规则 + 关键词命中 → 拒绝回答（发出 warning + 固定拒答文案，不调用检索/LLM）；
  2. 循环检测：同会话内同一问题归一化重复 >= guard_loop_max_repeats 次 → 提示用户（不阻断）。
完整版（重量）才需要：令牌桶、PII NER 脱敏、语义循环向量判定、死胡同回溯、工具注册表。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.core.config import get_settings

# 注入特征：指令覆盖 / 越权提示 / 泄露诱导 常见片段
_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions",
    r"ignore\s+your\s+(system\s+)?prompt",
    r"disregard\s+(all\s+)?(previous|previous\s+instructions)",
    r"you\s+are\s+now\s+(a\s+)?(do\s+)?anything|act\s+as\s+a\s+",  # "you are now..." 角色替换
    r"system\s+prompt|developer\s+prompt|system\s+instructions",
    r"reveal\s+your\s+prompt|show\s+your\s+system",
    r"泄露|公开.{0,6}(系统|system)?(提示词|指令|prompt)",
    r"忽略(上面|之前|以上)?(所有)?.{0,8}(指令|规则|约束|prompt|指令)",
    r"从现在开始(你|系统).{0,10}(扮演|忽略)",
    r"不要遵守|别管.{0,6}(上面|之前)",
    r"绕过.{0,6}(限制|安全|审核)",
]


@dataclass
class InjectionVerdict:
    flagged: bool
    matched: str = ""
    reasons: list[str] = field(default_factory=list)


def check_prompt_injection(text: str) -> InjectionVerdict:
    """Prompt 注入检测：正则特征 + 关键词。返回 (是否命中, 命中特征, 命中列表)。"""
    text = text or ""
    if not text.strip():
        return InjectionVerdict(flagged=False)
    reasons = []
    for pat in _INJECTION_PATTERNS:
        if re.search(pat, text, flags=re.IGNORECASE):
            reasons.append(pat)
    settings = get_settings()
    for w in settings.guard_injection_trigger_words.split(","):
        w = w.strip().lower()
        if w and w in text.lower():
            reasons.append(f"keyword:{w}")
    return InjectionVerdict(flagged=bool(reasons), reasons=reasons)


def _normalize_question(q: str) -> str:
    return re.sub(r"\s+", "", q or "").lower().strip("？?。. ")


def check_conversation_loop(history: list[dict] | None, current_question: str, max_repeats: int | None = None) -> bool:
    """循环检测：同会话内归一化问题重复出现 >= max_repeats 次判定为循环。

    history: 形如 [{"role": "user", "content": "..."}]（session 历史）。"""
    settings = get_settings()
    limit = max_repeats or settings.guard_loop_max_repeats
    cur = _normalize_question(current_question)
    if not cur:
        return False
    count = 1  # 当前问题自身
    for turn in history or []:
        if turn.get("role") == "user" and _normalize_question(turn.get("content", "")) == cur:
            count += 1
    return count >= limit
