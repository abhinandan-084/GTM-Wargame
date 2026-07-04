import numpy as np

from data_generator import GTM_DataGenerator
from schemas import DataConfig, OEMTierConfig

REQUIRED_COLUMNS = {
    "week",
    "month",
    "base_volume",
    "oem_launch_spike",
    "comp_a_impact",
    "comp_b_impact",
    "retail_support_spend",
    "search_spend",
    "social_spend",
    "search_adstock",
    "social_adstock",
    "list_price",
    "promo_rebate_pct",
    "effective_price",
    "price_effect",
    "market_leader_price",
    "sales",
}


def test_generate_has_expected_shape_and_columns(generated_df, simulation_config):
    weeks = simulation_config["simulation_config"]["weeks"]
    assert len(generated_df) == weeks
    assert REQUIRED_COLUMNS.issubset(generated_df.columns)


def test_generate_has_no_nulls_in_key_columns(generated_df):
    for col in ["sales", "list_price", "market_leader_price", "search_spend", "social_spend"]:
        assert generated_df[col].notna().all(), f"{col} contains NaNs"


def test_sales_are_non_negative(generated_df):
    assert (generated_df["sales"] >= 0).all()


def test_generation_is_deterministic_given_same_seed(simulation_config):
    tier_data = simulation_config["oem_tiers"]["upstart"]
    data_cfg = DataConfig(**simulation_config["simulation_config"])
    oem_cfg = OEMTierConfig(
        rank=tier_data["rank"], hill_k=tier_data["hill_k"], hill_n=tier_data["hill_n"]
    )

    df_a = GTM_DataGenerator(data_cfg, oem_cfg, tier_data["coeffs"]).generate()
    df_b = GTM_DataGenerator(data_cfg, oem_cfg, tier_data["coeffs"]).generate()

    assert np.array_equal(df_a["sales"].to_numpy(), df_b["sales"].to_numpy())
