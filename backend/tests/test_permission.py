"""权限可见性逻辑单测。"""

from __future__ import annotations

from app.store.qdrant import VISIBILITY_LEVELS, visible_levels


def test_visible_levels():
    assert VISIBILITY_LEVELS["public"] == 0
    assert visible_levels("public") == ["public"]
    assert visible_levels("internal") == ["public", "internal"]
    assert visible_levels("restricted") == ["public", "internal", "restricted"]
    # 非法等级兜底为 public
    assert visible_levels("") == ["public"]
