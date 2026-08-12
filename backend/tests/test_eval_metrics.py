from scripts.eval_metrics import _all_hit, _num_eq


def test_all_hit_requires_every_relevant_item_within_k():
    ranked = ["a", "x", "b", "c"]

    assert _all_hit(ranked, {"a", "b"}, 3) == 1.0
    assert _all_hit(ranked, {"a", "b", "c"}, 3) == 0.0


def test_all_hit_is_unreachable_when_relevant_set_exceeds_k():
    assert _all_hit(["a", "b"], {"a", "b", "c"}, 2) == 0.0


def test_numeric_match_accepts_percent_golden_snippets():
    assert _num_eq(15.31, "15.31%")
    assert _num_eq(-112.71, "-112.71%")
