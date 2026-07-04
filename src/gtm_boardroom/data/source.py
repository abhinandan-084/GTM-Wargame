from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd

from gtm_boardroom.data.generator import GTM_DataGenerator
from gtm_boardroom.data.schemas import DataConfig, OEMTierConfig, PromoEvent


class DataSource(ABC):
    """Common interface for anything that can produce the weekly GTM sales/marketing dataframe.

    GTM_DriverEngine only cares that .load() returns a dataframe with the raw columns it
    expects (sales, list_price, market_leader_price, search_spend, etc). Any upstream
    source - synthetic, CSV export, SQL/Snowflake query - can sit behind this interface.
    """

    @abstractmethod
    def load(self) -> pd.DataFrame:
        ...


class SyntheticDataSource(DataSource):
    """Wraps GTM_DataGenerator so synthetic data is exposed through the same interface as real sources."""

    def __init__(
        self,
        data_config: DataConfig,
        oem_config: OEMTierConfig,
        coeffs: dict,
        promos: Optional[List[PromoEvent]] = None,
    ):
        self._generator = GTM_DataGenerator(data_config, oem_config, coeffs, promos=promos)

    def load(self) -> pd.DataFrame:
        return self._generator.generate()


class CSVDataSource(DataSource):
    """Loads a pre-generated or externally exported weekly GTM dataframe from a CSV file."""

    def __init__(self, path: Union[str, Path]):
        self._path = Path(path)

    def load(self) -> pd.DataFrame:
        df = pd.read_csv(self._path)
        df["week"] = pd.to_datetime(df["week"])
        return df
