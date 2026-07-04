import pytest

OPT_HORIZON = 2
OPT_BUDGET = 300_000


@pytest.fixture(scope="module")
def opt_results(driver_engine):
    return driver_engine.optimize_strategy(budget_limit=OPT_BUDGET, horizon=OPT_HORIZON)


def test_get_diagnostics_returns_shap_value_per_feature(driver_engine):
    import numbers

    shap_info = driver_engine.get_diagnostics()
    assert set(shap_info.keys()) == set(driver_engine.feat_cols)
    assert all(isinstance(v, numbers.Real) for v in shap_info.values())


def test_get_market_context_returns_expected_top_level_keys(driver_engine):
    context = driver_engine.get_market_context()
    assert set(context.keys()) == {"market_regime", "signals", "wargame_alerts"}
    assert set(context["wargame_alerts"].keys()) == {
        "promotion_fatigue",
        "imminent_stock_pressure",
        "competitor_price_war",
    }


def test_optimize_strategy_runs_and_respects_budget(opt_results):
    total_spend = (
        sum(opt_results["weekly_forecasted_search"])
        + sum(opt_results["weekly_forecasted_social"])
        + sum(opt_results["weekly_forecasted_retail"])
    )
    assert total_spend <= OPT_BUDGET + 1e-3
    assert opt_results["forecasted_total_sales"] >= 0
    assert len(opt_results["weekly_forecasted_sales"]) == OPT_HORIZON


def test_compare_optimized_vs_actual_returns_expected_metrics(driver_engine, opt_results):
    comparison_df = driver_engine.compare_optimized_vs_actual(opt_results, horizon=OPT_HORIZON)

    assert set(comparison_df["Metric"]) == {
        "Cumulative Sales",
        "Total Budget Spent",
        "Average Price",
        "Average Market Leader Price",
    }
