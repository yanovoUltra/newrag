"""意图路由 JSON 解析单测。"""

from app.retrieval.router import RoutePlan, _parse_plan


def test_parse_valid_json():
    plan = _parse_plan(
        '{"intent": "multi_hop", "complexity": "complex", '
        '"rewritten_query": "公司2024年各业务板块收入", "sub_queries": ["2024年营业收入", "分业务收入构成"], '
        '"needs_hyde": false}'
    )
    assert plan.intent == "multi_hop"
    assert plan.complexity == "complex"
    assert plan.effective_queries == ["2024年营业收入", "分业务收入构成"]


def test_parse_with_code_fence():
    raw = '```json\n{"intent": "abstract", "complexity": "complex", "rewritten_query": null, "sub_queries": [], "needs_hyde": true}\n```'
    plan = _parse_plan(raw)
    assert plan.intent == "abstract"
    assert plan.needs_hyde is True
    assert plan.rewritten_query is None
    assert plan.effective_queries == []


def test_parse_invalid_falls_back():
    plan = _parse_plan("抱歉，我无法输出JSON")
    assert plan == RoutePlan()


def test_parse_bad_intent_normalized():
    plan = _parse_plan('{"intent": "unknown", "complexity": "medium", "needs_hyde": false}')
    assert plan.intent == "factual"
    assert plan.complexity == "simple"


def test_sub_queries_limited_and_filtered():
    raw = '{"intent": "multi_hop", "sub_queries": ["a", "", "b", "c", "d", "e"], "needs_hyde": false}'
    plan = _parse_plan(raw)
    assert len(plan.sub_queries) == 3
