import pytest

from gtm_boardroom.data.config import get_tier_config, load_simulation_config
from gtm_boardroom.data.generator import GTM_DataGenerator
from gtm_boardroom.diagnostics.driver_engine import GTM_DriverEngine


@pytest.fixture(scope="session")
def simulation_config():
    return load_simulation_config()


@pytest.fixture(scope="session")
def upstart_generator():
    data_cfg, oem_cfg, coeffs = get_tier_config("upstart")
    return GTM_DataGenerator(data_cfg, oem_cfg, coeffs)


@pytest.fixture(scope="session")
def generated_df(upstart_generator):
    return upstart_generator.generate()


@pytest.fixture(scope="module")
def driver_engine(generated_df):
    return GTM_DriverEngine(generated_df, current_week_idx=-5)
