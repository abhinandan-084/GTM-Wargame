import pathlib

import pandas as pd

from gtm_boardroom.data.config import get_tier_config
from gtm_boardroom.data.source import CSVDataSource, SyntheticDataSource
from gtm_boardroom.diagnostics.driver_engine import GTM_DriverEngine

CSV_PATH = pathlib.Path(__file__).resolve().parent.parent / "data" / "simulated_sales_data_rank_3.csv"

REQUIRED_COLUMNS = {
    "week",
    "sales",
    "list_price",
    "market_leader_price",
    "search_spend",
    "social_spend",
    "retail_support_spend",
    "search_adstock",
    "social_adstock",
}


def test_synthetic_data_source_loads_dataframe():
    data_cfg, oem_cfg, coeffs = get_tier_config("upstart")
    source = SyntheticDataSource(data_cfg, oem_cfg, coeffs)

    df = source.load()

    assert isinstance(df, pd.DataFrame)
    assert REQUIRED_COLUMNS.issubset(df.columns)


def test_csv_data_source_loads_dataframe_with_parsed_week_column():
    source = CSVDataSource(CSV_PATH)

    df = source.load()

    assert isinstance(df, pd.DataFrame)
    assert REQUIRED_COLUMNS.issubset(df.columns)
    assert pd.api.types.is_datetime64_any_dtype(df["week"])


def test_driver_engine_runs_against_csv_backed_data_source():
    source = CSVDataSource(CSV_PATH)
    df = source.load()

    engine = GTM_DriverEngine(df, current_week_idx=-5)
    shap_info = engine.get_diagnostics()
    market_context = engine.get_market_context()

    assert set(shap_info.keys()) == set(engine.feat_cols)
    assert set(market_context.keys()) == {"market_regime", "signals", "wargame_alerts"}


def test_synthetic_and_csv_sources_are_interchangeable_for_driver_engine():
    data_cfg, oem_cfg, coeffs = get_tier_config("upstart")
    synthetic_df = SyntheticDataSource(data_cfg, oem_cfg, coeffs).load()
    csv_df = CSVDataSource(CSV_PATH).load()

    assert list(synthetic_df.columns) == list(csv_df.columns)
