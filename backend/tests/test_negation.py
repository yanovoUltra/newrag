"""否定词感知（B 类对抗性用例）单测：解析"除了X之外"并改写查询。"""

from app.retrieval.search import (
    _parse_negated_entity,
    _remove_negation_clause,
)


def test_parse_chulei_zhwai():
    assert _parse_negated_entity("除了iPhone之外，Apple哪些产品的净销售额在增长？") == "iPhone"


def test_parse_chu_outer():
    assert _parse_negated_entity("除Mac外，Apple哪些产品表现如何？") == "Mac"


def test_parse_exclude():
    assert _parse_negated_entity("排除房地产板块后，公司主营业务有哪些？") == "房地产板块"


def test_parse_none():
    assert _parse_negated_entity("Apple的Mac和iPad净销售额在2024财年表现如何？") is None
    assert _parse_negated_entity("浦发银行2024年营业收入是多少？") is None


def test_parse_entity_too_long_ignored():
    # 误匹配长从句（>16 字符）放弃改写，防误伤
    assert _parse_negated_entity("除了管理层讨论与分析中的风险与合规管理段落之外，其余内容如何？") is None


def test_remove_clause_keeps_positive_body():
    assert (
        _remove_negation_clause("除了iPhone之外，Apple哪些产品的净销售额在增长？")
        == "Apple哪些产品的净销售额在增长？"
    )
    assert _remove_negation_clause("除Mac外，Apple哪些产品表现如何？") == "Apple哪些产品表现如何？"


def test_remove_clause_no_negation_unchanged():
    q = "浦发银行2024年营业收入是多少？"
    assert _remove_negation_clause(q) == q
