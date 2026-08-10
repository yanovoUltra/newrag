"""意图路由 JSON 解析单测。"""

from app.retrieval.router import (
    RoutePlan,
    _is_summary_query,
    _merge_query_type,
    _parse_plan,
    classify_query_type_keyword,
)


def test_parse_valid_json():
    plan = _parse_plan(
        '{"intent": "multi_hop", "complexity": "complex", "query_type": "entity", '
        '"rewritten_query": "公司2024年各业务板块收入", "sub_queries": ["2024年营业收入", "分业务收入构成"], '
        '"needs_hyde": false}'
    )
    assert plan.intent == "multi_hop"
    assert plan.complexity == "complex"
    assert plan.query_type == "entity"
    assert plan.effective_queries == ["2024年营业收入", "分业务收入构成"]


def test_parse_with_code_fence():
    raw = '```json\n{"intent": "abstract", "complexity": "complex", "query_type": "general", "rewritten_query": null, "sub_queries": [], "needs_hyde": true}\n```'
    plan = _parse_plan(raw)
    assert plan.intent == "abstract"
    assert plan.needs_hyde is True
    assert plan.rewritten_query is None
    assert plan.effective_queries == []


def test_parse_invalid_falls_back():
    plan = _parse_plan("抱歉，我无法输出JSON")
    assert plan == RoutePlan()


def test_parse_bad_intent_normalized():
    plan = _parse_plan('{"intent": "unknown", "complexity": "medium", "query_type": "weird", "needs_hyde": false}')
    assert plan.intent == "factual"
    assert plan.complexity == "simple"
    assert plan.query_type == "general"


def test_sub_queries_limited_and_filtered():
    raw = '{"intent": "multi_hop", "sub_queries": ["a", "", "b", "c", "d", "e"], "needs_hyde": false}'
    plan = _parse_plan(raw)
    assert len(plan.sub_queries) == 3


# ---------- query_type 关键词分类（IDF 动态开关） ----------

def test_classify_metric_keyword():
    assert classify_query_type_keyword("2024年每股收益是多少？") == "metric"
    assert classify_query_type_keyword("公司ROE是多少？") == "metric"
    assert classify_query_type_keyword("2024年归属于母公司股东的净利润？") == "metric"


def test_classify_entity_keyword():
    assert classify_query_type_keyword("2024年营业收入是多少？") == "entity"
    assert classify_query_type_keyword("全年营收情况") == "entity"


def test_classify_general():
    assert classify_query_type_keyword("公司为什么利润下滑？") == "general"


def test_merge_query_type_keyword_wins():
    # 关键词信号强于 LLM 结果
    assert _merge_query_type("entity", "metric") == "metric"
    assert _merge_query_type("metric", "entity") == "entity"
    # 关键词 general 时回退 LLM
    assert _merge_query_type("metric", "general") == "metric"
    assert _merge_query_type("general", "general") == "general"


# ---------- HyDE 触发放宽（综述/总结/分析型） ----------

def test_summary_query_triggers():
    assert _is_summary_query("请总结管理层讨论与分析（MD&A）中关于风险的观点")
    assert _is_summary_query("请概述公司2024年的经营情况")
    assert _is_summary_query("分析一下浦发银行的资产质量")
    assert _is_summary_query("Summarize the risk factors in MD&A")
    assert not _is_summary_query("浦发银行2024年营业收入是多少？")
    assert not _is_summary_query("Apple的Mac和iPad净销售额在2024财年表现如何？")
