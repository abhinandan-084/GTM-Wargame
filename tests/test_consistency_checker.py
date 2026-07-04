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
