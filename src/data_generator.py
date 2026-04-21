# Library Imports
from datetime import date
import pandas as pd
import numpy as np
from typing import List, Optional, Dict
from schemas import DataConfig, OEMTierConfig, PromoEvent

class GTM_DataGenerator:
    def __init__(self, data_config: DataConfig, oem_config: OEMTierConfig, coeffs: dict, promos: Optional[List[PromoEvent]] = None):
        self.rank = oem_config.rank
        self.start_date = data_config.start_date
        self.weeks = data_config.weeks
        self.launch_spike = data_config.launch_spike
        self.market_leader_price_decay_factor = data_config.market_leader_price_decay_factor
        self.hill_k = oem_config.hill_k
        self.hill_n = oem_config.hill_n
        self.rng = np.random.default_rng(data_config.random_state)
        self.params = coeffs
        self.promos = promos or []

    def _apply_hill_function(self, spend: np.ndarray) -> np.ndarray:
        K = self.hill_k
        n = self.hill_n
        return (spend**n) / (spend**n + K**n)

    def _init_df(self):
        dates = pd.date_range(start=self.start_date, periods=self.weeks, freq='W')
        df = pd.DataFrame({'week': dates})
        df['month'] = df['week'].dt.month
        return df

    def _base_volume_simulation(self, df):
        df['week_no'] = df['week'].dt.isocalendar().week

        # Defining Baseline Trend (Long-term growth)
        # To mimic growth, we can use a linear trend with some noise. The growth_rate can be adjusted to simulate faster or slower growth.
        time_index = np.arange(len(df))
        growth_rate = 5  # This means we expect an increase of 5 units in baseline demand each week, which is a reasonable growth for a new product in a growing market.
        trend = 20000 + (time_index * growth_rate)

        # Seasonal Multipliers Map (Rational: Q4 high, July low, Q1 steady)
        # These represent % of baseline. 1.0 = average.
        monthly_map = {
            1: 0.90, 2: 0.90, 3: 1.0,  # Q1 Launch Peak for iPhone/Pixel
            4: 1.00, 5: 0.95, 6: 0.90,  # Q2 Decay
            7: 0.85, 8: 1.20, 9: 0.90,  # July Slump -> Aug/Sept : New Launches
            10: 0.90, 11: 1.10, 12: 1.10 # Q4 Holiday Peak
        }
        df['seasonality_factor'] = df['month'].map(monthly_map)

        # High-impact Holiday events like Black Friday/New Year Sales and post cannibalization.
        df['event_multiplier'] = 1.0
        for i in range(len(df)):
            # Black Friday (Week 47)
            if df.iloc[i]['week_no'] == 47:
                df.at[i, 'event_multiplier'] = 1.30
                if i+1 < len(df): df.at[i+1, 'event_multiplier'] = 0.85 # Post-BF "Payback" dip

            # Christmas/New Year (Week 52)
            if (df.iloc[i]['week_no']==52) | (df.iloc[i]['week_no']==51): # Extended Holiday Impact
                df.at[i, 'event_multiplier'] = 1.25
                if i+1 < len(df): df.at[i+1, 'event_multiplier'] = 0.90 # Early Jan dip

        # Combine: Trend * Seasonality * Holiday * Random Noise
        noise = self.rng.normal(1, 0.05, len(df)) # 5% relative noise is more realistic than fixed 50 units
        df['base_volume'] = trend * df['seasonality_factor'] * df['event_multiplier'] * noise
        df.drop(columns=['week_no', 'seasonality_factor', 'event_multiplier'], inplace=True)
        df['base_volume'] = df['base_volume'].astype(int)
        return df

    def _sales_simulation(self, df):
        df['oem_launch_spike'] = df['month'].apply(lambda x: self.launch_spike if x == self.params['launch_month'] else 1.0)
        df['comp_a_impact'] = df['month'].apply(lambda x: self.params['rank_a_impact']['impact'] if x == self.params['rank_a_impact']['launch_month'] else 1.0)
        df['comp_b_impact'] = df['month'].apply(lambda x: self.params['rank_b_impact']['impact'] if x == self.params['rank_b_impact']['launch_month'] else 1.0)
        return df

    def _marketing_spend_simulation(self, df):
        time_index = np.arange(len(df))
        monthly_multiplier  = np.array([self.params['monthly_seasonality'][m-1] for m in df['month']])

        # Structured retail support spend with trend, seasonality, and noise
        retail_noise = self.rng.normal(0, 5000, len(df))
        df['retail_support_spend'] = (self.params['base_retail_spend'] +
                                       (time_index * self.params['retail_spend_trend'])) * monthly_multiplier + retail_noise
        df['retail_support_spend'] = np.maximum(0, df['retail_support_spend']) # Ensure spend is non-negative

        # Apply Hill function for retail visibility and scale by retail_dependency
        retail_hill_impact = self._apply_hill_function(df['retail_support_spend'])
        df['retail_visibility_multiplier'] = 1 + (retail_hill_impact * (self.params['retail_dependency'] - 1))

        # Dampen retail_visibility_multiplier during OEM's launch months
        oem_launch_months = [self.params['launch_month']]
        df.loc[df['month'].isin(oem_launch_months), 'retail_visibility_multiplier'] *= 0.75

        # Structured search spend with trend, seasonality, and noise
        search_noise = self.rng.normal(0, 2000, len(df))
        df['search_spend'] = (self.params['base_search_spend'] +
                              (time_index * self.params['search_spend_trend'])) * monthly_multiplier + search_noise
        df['search_spend'] = np.maximum(0, df['search_spend']) # Ensure spend is non-negative

        # Structured social spend with trend, seasonality, and noise
        social_noise = self.rng.normal(0, 3000, len(df))
        df['social_spend'] = (self.params['base_social_spend'] +
                              (time_index * self.params['social_spend_trend'])) * monthly_multiplier + social_noise
        df['social_spend'] = np.maximum(0, df['social_spend']) # Ensure spend is non-negative

        if self.promos:
            for promo in self.promos:
                mask = (df['week'] >= pd.Timestamp(promo.start_date)) & (df['week'] <= pd.Timestamp(promo.end_date))
                df.loc[mask, 'social_spend'] *= promo.multiplier

        df.loc[df['month'] == self.params['launch_month'], 'search_spend'] *= 3

        mult = 1.3 if self.rank == 3 else (1.3 if self.rank == 2 else 1.1)
        df.loc[df['month'] == self.params['rank_a_impact']['launch_month'], 'social_spend'] *= mult
        df.loc[df['month'] == self.params['rank_b_impact']['launch_month'], 'social_spend'] *= mult

        df.loc[df['month'] == self.params['rank_a_impact']['launch_month'], 'search_spend'] *= mult
        df.loc[df['month'] == self.params['rank_b_impact']['launch_month'], 'search_spend'] *= mult

        df['search_adstock'] = df['search_spend'].ewm(alpha=0.7, adjust=False).mean()
        df['social_adstock'] = df['social_spend'].ewm(alpha=0.3, adjust=False).mean()

        df['search_base_impact'] = self._apply_hill_function(df['search_adstock'])
        df['social_base_impact'] = self._apply_hill_function(df['social_adstock'])

        df['media_synergy'] = np.where(df['social_adstock'] > 60000, 1.2, 1.0)
        df['atl_contribution'] = (df['search_base_impact'] * df['media_synergy']) + df['social_base_impact']
        return df

    def _generate_competitor_behavior(self, df):
        leader_base_price = 799.0
        leader_prices = np.zeros(len(df))
        leader_prices[0] = leader_base_price

        for i in range(1, len(df)):
            current_month = df.loc[i, 'month']
            # Reset to base price in month 9 (competitor's flagship launch)
            if current_month == self.params['rank_a_impact']['launch_month']:
                leader_prices[i] = leader_base_price
            else:
                leader_prices[i] = leader_prices[i - 1] * self.market_leader_price_decay_factor

        promos = self.rng.choice([0.9, 1.0], size=len(df), p=[0.1, 0.9])
        return leader_prices * promos

    def _price_decay_simulation(self, df):
        base_price = 799.0
        df['list_price'] = base_price
        for i in range(1, len(df)):
            if df.loc[i, 'month'] == self.params['launch_month']:
                df.loc[i, 'list_price'] = base_price
            else:
                df.loc[i, 'list_price'] = df.loc[i - 1, 'list_price'] * self.params['weekly_decay_factor']

        df['promo_rebate_pct'] = self.rng.uniform(0.02, 0.08, size=self.weeks)

        if self.rank == 3:
            idx = df['month'].isin([self.params['launch_month']])
            df.loc[idx, 'promo_rebate_pct'] = self.rng.uniform(0.10, 0.15, size=idx.sum())

        df['effective_price'] = df['list_price'] * (1 - df['promo_rebate_pct'])
        df['price_effect'] = np.exp(self.params['elasticity'] * (1 - (df['effective_price'] / base_price)))

        # Set price_effect to 1.0 during OEM's own launch months to prevent compounding with launch spikes
        oem_launch_months = [self.params['launch_month']]
        df.loc[df['month'].isin(oem_launch_months), 'price_effect'] = 1.0

        # Apply maximum cap on price_effect if specified
        if df['price_effect'].notnull().any():
            df['price_effect'] = np.clip(df['price_effect'], a_min= 1.0, a_max=2.0)

        if self.rank == 3:
            df['market_leader_price'] = self._generate_competitor_behavior(df)
            df['price_penalty'] = df.apply(lambda x: 0.9 if x['list_price'] > x['market_leader_price'] * 1.05 else 1.0, axis=1)
        else:
            df['price_penalty'] = 1.0

        # Apply leader_penalty only if market_leader_price decreases by more than 10%
        leader_price_prev = df.get('market_leader_price', pd.Series(1, index=df.index)).shift(1)
        current_leader_price = df.get('market_leader_price', pd.Series(1, index=df.index))
        df['leader_penalty'] = np.where((current_leader_price / leader_price_prev) <= 0.9, 0.8, 1.0)

        # Handle the first row or any division by zero if leader_price_prev is 0 or NaN
        df.loc[leader_price_prev.isna() | (leader_price_prev == 0), 'leader_penalty'] = 1.0
        return df

    def _final_sales(self, df):
        df['sales'] = (df['base_volume'] * df['oem_launch_spike'] * df['comp_a_impact']
        * df['comp_b_impact'] * df['retail_visibility_multiplier'] * df['price_effect']
        * df['price_penalty'] * df['leader_penalty']) * (1+df['atl_contribution'])
        df['sales'] = df['sales'].clip(lower=0).astype(int)
        return df

    def generate(self):
        df = self._init_df()
        df = self._base_volume_simulation(df)
        df = self._sales_simulation(df)
        df = self._marketing_spend_simulation(df)
        df = self._price_decay_simulation(df)
        df = self._final_sales(df)
        return df
    

