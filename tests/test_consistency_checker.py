from gtm_boardroom.guardrails.consistency_checker import ConsistencyChecker

SHAP_INFO = {"price_gap_pct": 0.42, "marketing_intensity": -0.11}
MARKET_CONTEXT = {
    "signals": {"price_sensitivity": "Hyper-sensitive"},
    "wargame_alerts": {"competitor_price_war": True},
}
OPT_RESULTS = {"lift_percent": 12.0, "optimized_price": 799.8}


def test_valid_response_has_no_hallucinations():
    text = "We project a 12% lift at an optimized price near 800."
    result = ConsistencyChecker.validate_response(text, SHAP_INFO, MARKET_CONTEXT, OPT_RESULTS)
    assert result["is_valid"] is True
    assert result["hallucinated_values"] == []


def test_invented_number_is_flagged_as_hallucination():
    text = "We project a 47% lift, far above ground truth."
    result = ConsistencyChecker.validate_response(text, SHAP_INFO, MARKET_CONTEXT, OPT_RESULTS)
    assert result["is_valid"] is False
    assert 47.0 in result["hallucinated_values"]


def test_percentage_scaling_llm_states_whole_number_for_fractional_ground_truth():
    # Ground truth has price_gap_pct=0.42; LLM writing "42%" should be treated as a match.
    text = "Price gap stands at 42%."
    result = ConsistencyChecker.validate_response(text, SHAP_INFO, MARKET_CONTEXT, OPT_RESULTS)
    assert result["is_valid"] is True


def test_inverse_scaling_llm_states_fraction_for_whole_number_ground_truth():
    # Ground truth has lift_percent=12.0; LLM writing "0.12" should be treated as a match.
    text = "Lift of 0.12 expected."
    result = ConsistencyChecker.validate_response(text, SHAP_INFO, MARKET_CONTEXT, OPT_RESULTS)
    assert result["is_valid"] is True


def test_common_small_integers_are_ignored():
    text = "1. First point. 2. Second point. 3. Third point."
    result = ConsistencyChecker.validate_response(text, SHAP_INFO, MARKET_CONTEXT, OPT_RESULTS)
    assert result["is_valid"] is True


K_OPT_RESULTS = {"optimized_search": 176432.11, "optimized_retail": 195575.07}


def test_k_notation_citation_of_real_pool_value_is_accepted():
    # 176k for 176,432.11 (round) and 195k for 195,575.07 (truncation) are
    # both within one unit of the last written digit.
    for text in ("Search spend totals 176k this cycle.",
                 "Retail gets 195k over the horizon.",
                 "Search spend totals 176 thousand this cycle.",
                 "Search runs at 176.4k per the optimizer."):
        result = ConsistencyChecker.validate_response(text, {}, {}, K_OPT_RESULTS)
        assert result["is_valid"] is True, text


def test_k_notation_is_context_gated_and_precision_aware():
    # A bare "176" (no suffix) must NOT silently match 176,432.11...
    result = ConsistencyChecker.validate_response(
        "The figure is 176.", {}, {}, K_OPT_RESULTS)
    assert 176.0 in result["hallucinated_values"]
    # ...a fabricated k value must still flag...
    result = ConsistencyChecker.validate_response(
        "Allocate 84k to Social.", {}, {}, K_OPT_RESULTS)
    assert 84.0 in result["hallucinated_values"]
    # ...aggressive rounding beyond one k-unit flags (130k for 126,279)...
    result = ConsistencyChecker.validate_response(
        "Retail takes 130k.", {}, {}, {"optimized_retail": 126279.0})
    assert 130.0 in result["hallucinated_values"]
    # ...and higher written precision earns a tighter window.
    result = ConsistencyChecker.validate_response(
        "Search runs at 176.9k.", {}, {}, K_OPT_RESULTS)
    assert 176.9 in result["hallucinated_values"]
