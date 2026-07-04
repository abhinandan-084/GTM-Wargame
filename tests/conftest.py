import pathlib

import pytest
import yaml

from data_generator import GTM_DataGenerator
from driver_engine import GTM_DriverEngine
from schemas import DataConfig, OEMTierConfig

CONFIG_PATH = pathlib.Path(__file__).resolve().parent.parent / "src" / "simulation_config.yaml"


@pytest.fixture(scope="session")
def simulation_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="session")
def upstart_generator(simulation_config):
    tier_data = simulation_config["oem_tiers"]["upstart"]
    data_cfg = DataConfig(**simulation_config["simulation_config"])
    oem_cfg = OEMTierConfig(
        rank=tier_data["rank"], hill_k=tier_data["hill_k"], hill_n=tier_data["hill_n"]
    )
    return GTM_DataGenerator(data_cfg, oem_cfg, tier_data["coeffs"])


@pytest.fixture(scope="session")
def generated_df(upstart_generator):
    return upstart_generator.generate()


@pytest.fixture(scope="module")
def driver_engine(generated_df):
    return GTM_DriverEngine(generated_df, current_week_idx=-5)
