from app.fields.subject import find_mentioned_companies, find_subject_company

COMPANIES = [
    "中国工商银行",
    "中国建设银行",
    "General Electric",
    "宜宾五粮液",
    "杭州海康威视数字技术股份有限公司",
]


def test_analysis_question_falls_back_to_explicit_company_mention():
    question = "请综合梳理工商银行近三年的风险管理变化"
    assert find_subject_company(question, COMPANIES) == "中国工商银行"


def test_company_abbreviations_cover_english_and_chinese_names():
    assert find_mentioned_companies("Summarize GE's industrial businesses", COMPANIES) == [
        "General Electric"
    ]
    assert find_mentioned_companies("海康的研发投入趋势如何？", COMPANIES) == [
        "杭州海康威视数字技术股份有限公司"
    ]


def test_comparison_question_keeps_all_explicit_companies():
    assert find_mentioned_companies("比较工行和建行的资本充足率", COMPANIES) == [
        "中国工商银行",
        "中国建设银行",
    ]


def test_regional_prefix_can_be_omitted():
    assert find_mentioned_companies("五粮液渠道改革有哪些变化？", COMPANIES) == ["宜宾五粮液"]
