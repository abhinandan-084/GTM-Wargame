import importlib.resources
from typing import Any, Tuple

import yaml

from gtm_boardroom.data.schemas import DataConfig, OEMTierConfig


def load_simulation_config() -> dict[str, Any]:
    config_text = (
        importlib.resources.files("gtm_boardroom.data")
        .joinpath("simulation_config.yaml")
        .read_text()
    )
    return yaml.safe_load(config_text)


def get_tier_config(tier_name: str) -> Tuple[DataConfig, OEMTierConfig, dict]:
    full_config = load_simulation_config()
    data_cfg = DataConfig(**full_config["simulation_config"])

    tier_data = full_config["oem_tiers"][tier_name]
    oem_cfg = OEMTierConfig(
        rank=tier_data["rank"], hill_k=tier_data["hill_k"], hill_n=tier_data["hill_n"]
    )
    return data_cfg, oem_cfg, tier_data["coeffs"]
