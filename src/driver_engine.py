import pandas as pd
import numpy as np
import xgboost as xgb
import shap
from typing import Any, List, Dict
from scipy.optimize import differential_evolution 

class GTM_DriverEngine:
    """
    Trains a sales prediction model, provides diagnostics, simulates future sales, and optimise marketing/pricing strategies.
    """
    def __init__(self, df, current_week_idx=-5):
        """
        Initializes the GTM_DriverEngine.
        Args:
            df : The input DataFrame containing historical sales and marketing data.
            current_week_idx (int): Index to treat as 'today' (e.g., -5 for the 5th to last week). This defines the split point for training data.
        """
        self.df = df.copy()
        # Normalize index to absolute position
        self.current_idx = len(df) + current_week_idx if current_week_idx < 0 else current_week_idx
        self._feat_engine()
        self._train_model()

    def _feat_engine(self):
        """
        Performs feature engineering on df. This includes creating various
        price, marketing, and competitive metrics, as well as time-based features.
        Sets `self.X_train` and `self.y_train` for model training.
        """
        # Feature Engineering : 
        self.df['price_gap_pct'] = (self.df['list_price'] - self.df['market_leader_price']) / self.df['market_leader_price']

        # This metric aims to capture the marketing effort per unit of price. Measure of how aggressively the product is being promoted
        self.df['marketing_intensity'] = (self.df['search_spend'] + self.df['social_spend']) / self.df['list_price']
        self.df['rolling_sales_4w'] = self.df['sales'].shift(1).rolling(window=4).mean().fillna(self.df['base_volume'])
        self.df['total_mkt_spend'] = self.df['retail_support_spend'] + self.df['search_spend'] + self.df['social_spend']
        self.df['leader_price_delta'] = self.df['market_leader_price'].diff(1).fillna(0)

        # % change in the market leader's price over the last 4 weeks to get an idea of competitor's and our pricing momentum, indicating aggressive moves or stability.
        self.df['leader_price_velocity'] = self.df['market_leader_price'].pct_change(periods=4).fillna(0)
        self.df['our_price_velocity'] = self.df['list_price'].pct_change(periods=4).fillna(0)
        self.df['relative_velocity'] = self.df['our_price_velocity'] - self.df['leader_price_velocity']

        # To capture an interaction effect: how marketing efforts (adstock) might amplify or mitigate the impact of price gap. 
        # # For example, high marketing spend might make a premium price more acceptable.
        self.df['price_marketing_synergy'] = self.df['price_gap_pct'] * (self.df['search_adstock'] + self.df['social_adstock'])

        self.df['is_launch'] = self.df['oem_launch_spike'] != 1.0
        self.df['is_comp_launch'] = self.df['month'].isin([1, 9]).astype(int)

        # Proxy for marketing efficiency: how many sales were generated per unit of marketing spend in the previous period. 
        # The +1 is to prevent division by zero. It's a key indicator of ROI.
        # effiviency momentum helps smooth out weekly fluctuations and indicates whether marketing effectiveness is generally 
        # improving or declining over time.

        self.df['mkt_efficiency_ratio'] = self.df['sales'].shift(1) / (self.df['total_mkt_spend'].shift(1) + 1)
        self.df.fillna({'mkt_efficiency_ratio':1.0},inplace=True)
        self.df['efficiency_momentum'] = self.df['mkt_efficiency_ratio'].rolling(4).mean().fillna(1.0)
        self.df['recent_promo_intensity'] = self.df['promo_rebate_pct'].rolling(window=4).mean().fillna(0)

        # Other Misc Features
        self.df['month_sin'] = np.sin(2 * np.pi * self.df['month']/12)
        self.df['month_cos'] = np.cos(2 * np.pi * self.df['month']/12)
        self.df['marketing_vs_comp_launch'] = (self.df['search_adstock'] + self.df['social_adstock']) / (self.df['is_comp_launch'] + 1)

        self.feat_cols = [
            'month','comp_a_impact', 'comp_b_impact', 'retail_support_spend','search_adstock','social_adstock',
            'list_price', 'market_leader_price', 'price_effect','price_penalty','leader_penalty',
            'price_gap_pct', 'marketing_intensity', 'rolling_sales_4w', 'total_mkt_spend',
            'leader_price_delta', 'leader_price_velocity', 'our_price_velocity', 'relative_velocity',
            'price_marketing_synergy', 'is_launch', 'is_comp_launch',
            'mkt_efficiency_ratio', 'efficiency_momentum', 'recent_promo_intensity',
            'month_sin', 'month_cos', 'marketing_vs_comp_launch'
        ]

        self.X_train = self.df.iloc[:self.current_idx][self.feat_cols]
        self.y_train = self.df.iloc[:self.current_idx]['sales']

    def _train_model(self):
        """
        Trains an XGBoost Regressor model using the engineered features and historical sales data.
        Also initializes a SHAP TreeExplainer for model diagnostics.
        """
        self.model = xgb.XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.8, random_state=42)
        self.model.fit(self.X_train, self.y_train)
        self.shap_explainer = shap.TreeExplainer(self.model)

    def get_diagnostics(self):
        """
        Generates SHAP values for the current week to provide model diagnostics and feature importance.
        Returns: Dict[str, Any]: A dictionary where keys are feature names and values are their SHAP values.
        """
        target_row = self.df.iloc[[self.current_idx]][self.feat_cols]
        shap_values = self.shap_explainer.shap_values(target_row)
        return dict(zip(self.feat_cols, shap_values[0]))

    def simulate_future_horizon(self, price, search_spends_per_week, social_spends_per_week, retail_spends_per_week, horizon=4, leader_price_override = None) -> List[float]:
        """
        Simulates sales over a future horizon based on proposed pricing and marketing spends.

        Args:
            price (float): The proposed list price for the product.
            search_spends_per_week (List[float]): List of search spends for each week in the horizon.
            social_spends_per_week (List[float]): List of social spends for each week in the horizon.
            retail_spends_per_week (List[float]): List of retail support spends for each week in the horizon.
            horizon (int): The number of weeks to simulate into the future.
            leader_price_override (float, optional): An optional override for the market leader's price to simulate competitive 'what-if' scenarios.
        Returns: List[float]: A list of forecasted weekly sales for the given horizon.
        """
        weekly_sales_predictions = []
        # Initialize adstock for the first week based on previous week's actuals
        # Or simplify by assuming current week's spend entirely dictates adstock effect for that week for optimization purposes.
        # For this simplified model, we will use the current week's spend to directly calculate its impact.
        for i in range(horizon):
            target_idx = self.current_idx + i + 1 # +1 because horizon starts from the next week
            if target_idx >= len(self.df): break
            scenario = self.df.iloc[[target_idx]].copy()
        
            # Update Leader Price if an override is provided (for stress testing)
            if leader_price_override:
                scenario['market_leader_price'] = leader_price_override

            # Recalculate the gap which drives the XGBoost model
            leader_p = scenario['market_leader_price'].values[0]
            scenario['list_price'] = price
            scenario['price_gap_pct'] = (price - leader_p) / leader_p
            
            # Apply optimized price and weekly spends
            scenario['search_adstock'] = search_spends_per_week[i] * 0.7  # Simplified adstock impact for the week
            scenario['social_adstock'] = social_spends_per_week[i] * 0.3  # Simplified adstock impact for the week
            scenario['retail_support_spend'] = retail_spends_per_week[i]

            # Recalculate dependent features for this specific week's scenario
            scenario['marketing_intensity'] = (search_spends_per_week[i] + social_spends_per_week[i]) / price # Use current week's spend and price
            # Note: other features like rolling_sales_4w, efficiency_momentum etc. are based on historical data 
            # which are already set in the scenario frame. For a true dynamic simulation, these would need 
            # to be updated sequentially.
            
            pred = self.model.predict(scenario[self.feat_cols])[0]
            weekly_sales_predictions.append(max(0, pred))
        return weekly_sales_predictions

    def optimize_strategy(self, budget_limit, horizon=4):
        """
        Optimizes pricing and marketing spend across channels to maximize total sales within a given budget.
        Uses differential evolution for optimization.
        Args:
            budget_limit (float): The total budget available for marketing spend over the horizon.
            horizon (int): The number of weeks for which to optimize the strategy.
        Returns:
            Dict[str, Any]: A dictionary containing optimization results including optimized price,
                            weekly spend allocations, forecasted sales, and lift percentage.
        """
        # 1. Get Actuals for this horizon to calculate Lift
        actual_data = self.get_historical_actuals(horizon=horizon)
        actual_sales_sum = actual_data['actual_total_sales']

        last_price = self.df.iloc[self.current_idx]['list_price']
        
        # Initial guess: current price, and budget evenly split per channel per week
        spend_per_channel_per_week = (budget_limit / 3) / horizon
        initial_guess = [
            last_price, 
            *[spend_per_channel_per_week] * horizon,  # Weekly search spend
            *[spend_per_channel_per_week] * horizon,  # Weekly social spend
            *[spend_per_channel_per_week] * horizon   # Weekly retail spend
        ]

        # Bounds: (min_price, max_price), then (min_spend, max_spend) for each weekly variable
        min_spend_per_week = 1000 # Minimum spend per channel per week
        max_spend_per_week = budget_limit # Max possible spend per channel per week, capped by total budget for simplicity

        bounds = [
            (max(199, last_price - 100), 999) # Price bounds
        ] + [
            (min_spend_per_week, max_spend_per_week)
        ] * (3 * horizon)

        def objective(x):
            price = x[0]
            search_spends = x[1 : 1 + horizon]
            social_spends = x[1 + horizon : 1 + 2 * horizon]
            retail_spends = x[1 + 2 * horizon : 1 + 3 * horizon]
            
            weekly_sales_predictions = self.simulate_future_horizon(
                price, list(search_spends), list(social_spends), list(retail_spends), horizon
            )
            sales_forecast = sum(weekly_sales_predictions)
            
            total_spend = sum(search_spends) + sum(social_spends) + sum(retail_spends)
            penalty = 1e6 * max(0, total_spend - budget_limit)
            
            return -sales_forecast + penalty

        res = differential_evolution(objective, bounds, seed=42)

        optimized_price = round(res.x[0], 2)
        optimized_search_weekly = [round(s, 2) for s in res.x[1 : 1 + horizon]]
        optimized_social_weekly = [round(s, 2) for s in res.x[1 + horizon : 1 + 2 * horizon]]
        optimized_retail_weekly = [round(s, 2) for s in res.x[1 + 2 * horizon : 1 + 3 * horizon]]
        
        forecasted_total_sales = round(-res.fun, 0)

        # Calculate Lift and Lift %
        # Lift = ((New Sales - Old Sales) / Old Sales) * 100
        lift_pct = ((forecasted_total_sales / actual_sales_sum) - 1) * 100 if actual_sales_sum > 0 else 0

        weekly_forecasted_sales = self.simulate_future_horizon(
            optimized_price, 
            optimized_search_weekly, 
            optimized_social_weekly, 
            optimized_retail_weekly, 
            horizon
        )

        return {
            "actual_historical_sales": round(actual_sales_sum, 0),
            "forecasted_total_sales": round(forecasted_total_sales, 0),
            "lift_percent": round(lift_pct, 2), # Now the Agent can see the "Value Add"
            "optimized_price": optimized_price,
            "optimized_search": sum(optimized_search_weekly), # Total for comparison
            "optimized_social": sum(optimized_social_weekly), # Total for comparison
            "optimized_retail": sum(optimized_retail_weekly), # Total for comparison
            "forecasted_total_sales": forecasted_total_sales,
            "weekly_forecasted_sales": [round(s, 0) for s in weekly_forecasted_sales],
            "weekly_forecasted_search": optimized_search_weekly,
            "weekly_forecasted_social": optimized_social_weekly,
            "weekly_forecasted_retail": optimized_retail_weekly,
            "actual_market_leader_price": actual_data['actual_market_leader_price']
        }

    def get_historical_actuals(self, horizon=4):
        """
        Returns the actual metrics (sales, spends, prices) for the weeks following the current reference point.
        Args: horizon (int): The number of weeks to retrieve historical actuals for.
        Returns: Dict[str, Any]: A dictionary containing various actual historical metrics.
        """
        future_df = self.df.iloc[self.current_idx + 1 : self.current_idx + 1 + horizon]

        actual_sales = future_df['sales'].sum()
        actual_search = future_df['search_spend'].sum()
        actual_social = future_df['social_spend'].sum()
        actual_retail = future_df['retail_support_spend'].sum()
        actual_total_spend = (future_df['search_spend'] + future_df['social_spend'] + future_df['retail_support_spend']).sum()
        actual_avg_price = future_df['list_price'].mean()
        actual_market_leader_price = future_df['market_leader_price'].mean()

        return {
            "actual_price": actual_avg_price,
            "actual_search": actual_search,
            "actual_social": actual_social,
            "actual_retail": actual_retail,
            "actual_total_spend": actual_total_spend,
            "actual_total_sales": actual_sales,
            "weekly_actual_sales": future_df['sales'].tolist(),
            "weekly_actual_search": future_df['search_spend'].tolist(),
            "weekly_actual_social": future_df['social_spend'].tolist(),
            "weekly_actual_retail": future_df['retail_support_spend'].tolist(),
            "actual_market_leader_price": actual_market_leader_price
        }

    def compare_optimized_vs_actual(self, opt_results, horizon=4):
        """
        Calculates and displays the delta between an optimized strategy and historical actuals
        using summary metrics.
        Args:
            opt_results (Dict[str, Any]): The results dictionary from the `optimize_strategy` method.
            horizon (int): The number of weeks considered for comparison.
        Returns:
            df: A DataFrame comparing actual vs. optimized metrics with deltas and lift percentages.
        """

        actuals = self.get_historical_actuals(horizon=horizon)

        actual_sales = actuals['actual_total_sales']
        actual_spend = actuals['actual_total_spend']
        actual_avg_price = actuals['actual_price']
        actual_market_leader_price = actuals['actual_market_leader_price']

        opt_sales = opt_results['forecasted_total_sales']
        opt_spend = sum(opt_results['weekly_forecasted_search']) + sum(opt_results['weekly_forecasted_social']) + sum(opt_results['weekly_forecasted_retail'])
        opt_price = opt_results['optimized_price']

        return pd.DataFrame({
            'Metric': ['Cumulative Sales', 'Total Budget Spent', 'Average Price', 'Average Market Leader Price'],
            'Actual (Historical)': [actual_sales, actual_spend, round(actual_avg_price, 2), round(actual_market_leader_price, 2)],
            'Optimized (Proposal)': [opt_sales, opt_spend, opt_price, round(actual_market_leader_price, 2)],
            'Delta': [
                opt_sales - actual_sales,
                opt_spend - actual_spend,
                round(opt_price - actual_avg_price, 2),
                round(actual_market_leader_price - actual_market_leader_price, 2)
            ],
            'Lift %': [
                round((opt_sales / actual_sales - 1) * 100, 2),
                round((opt_spend / actual_spend - 1) * 100, 2),
                round((opt_price / actual_avg_price - 1) * 100, 2),
                0.0 # Market leader price is not changed by optimization
            ]
        })

    def display_weekly_phased_strategy(self, opt_results, horizon=4):
        """
        Displays a week-by-week comparison of optimized vs. actual sales and spends.
        Args:
            opt_results (Dict[str, Any]): The results dictionary from the `optimize_strategy` method.
            horizon (int): The number of weeks for which to display the strategy.
        Returns:
            df: A DataFrame showing weekly actuals and optimized proposals for sales and spends.
        """

        actuals = self.get_historical_actuals(horizon=horizon)

        weekly_data = []
        for i in range(horizon):
            week_num = i + 1
            weekly_data.append({
                'Week': f'Week {week_num}',
                'Actual Sales': actuals['weekly_actual_sales'][i],
                'Optimized Sales': opt_results['weekly_forecasted_sales'][i],
                'Actual Search Spend': round(actuals['weekly_actual_search'][i], 2),
                'Optimized Search Spend': round(opt_results['weekly_forecasted_search'][i], 2),
                'Actual Social Spend': round(actuals['weekly_actual_social'][i], 2),
                'Optimized Social Spend': round(opt_results['weekly_forecasted_social'][i], 2),
                'Actual Retail Spend': round(actuals['weekly_actual_retail'][i], 2),
                'Optimized Retail Spend': round(opt_results['weekly_forecasted_retail'][i], 2)
            })
        return pd.DataFrame(weekly_data)

    def get_market_context(self) -> Dict[str, Any]:
        """
        Translates numerical signals into a strategic context for decision-making.
        Analyzes current market conditions based on price positioning, competitive activity,
        media efficiency, and product lifecycle.

        Returns:
            Dict[str, Any]: A dictionary providing strategic context, signals, and wargame alerts.
        """

        row = self.df.iloc[self.current_idx]
        prev_4w = self.df.iloc[self.current_idx-4 : self.current_idx]
        
        # 1. Price Elasticity Proxy (Current Week vs 4W Average)
        avg_price_4w = prev_4w['list_price'].mean()
        avg_sales_4w = prev_4w['sales'].mean()
        price_delta = (row['list_price'] - avg_price_4w) / avg_price_4w
        sales_delta = (row['sales'] - avg_sales_4w) / avg_sales_4w

        # 1. Price Positioning Context
        price_gap = row['price_gap_pct']
        if price_gap > 0.10:
            price_regime = "Premium (Significantly more expensive than Leader)"
        elif price_gap < -0.10:
            price_regime = "Under-cutter (Significantly cheaper than Leader)"
        else:
            price_regime = "Price Parity (Fighting head-to-head)"

        # 2. Competitive Stress Detection
        # Look at the leader's price velocity over the last 4 weeks
        comp_velocity = row['leader_price_velocity']
        if comp_velocity < -0.05:
            comp_context = "High Stress (Competitor is aggressively slashing prices)"
        else:
            comp_context = "Stable (No aggressive competitor pricing moves)"

        # 3. Media Efficiency Context
        # Check if our marketing ROI is improving or dying
        efficiency_trend = row['efficiency_momentum']
        if efficiency_trend > 1.10:
            media_state = "High Efficiency (Marketing is over-performing; likely room to scale)"
        elif efficiency_trend < 0.90:
            media_state = "Diminishing Returns (Marketing efficiency is dropping; spend with caution)"
        else:
            media_state = "Stable ROI"   

        # 4. Product Lifecycle Phase
        if row['is_launch']:
            lifecycle = "Launch Phase (Focus on volume and visibility over margins)"
        elif row['month'] in [11, 12]:  # This is hard-coded at the moment, can be updated
            lifecycle = "Peak Season (Holiday Season)"
        else:
            lifecycle = "Maturity/Maintain (Focus on sustaining baseline)"     

        # Heuristic for sensitivity: Did a small price change cause a big sales swing?
        sensitivity = abs(sales_delta / price_delta) if price_delta != 0 else 0
        
        # 2. Relative Share of Voice (SOV) Proxy
        # marketing_vs_comp_launch is (Our Adstock) / (Comp Launch Flag + 1)
        sov_signal = row['marketing_vs_comp_launch']
        
        # 3. Promotion Fatigue
        # High promo intensity but declining efficiency momentum
        promo_fatigue = row['recent_promo_intensity'] > 0.05 and row['efficiency_momentum'] < 0.95

        # 4. Synergy Health
        # Checks if marketing is actually amplifying the price gap impact
        synergy_score = row['price_marketing_synergy']

        context = {
            "market_regime": {
                "phase": "Product Launch" if row['is_launch'] else "Steady State",
                "competitor_activity": "Aggressive (Comp Launch Detected)" if row['is_comp_launch'] else "Quiet",
                "competitive_price_threat": comp_context,
                "price_position": price_regime,
                "lifecycle_stage": lifecycle,
            },
            "signals": {
                "price_sensitivity": "Hyper-sensitive" if sensitivity > 2.5 else "Inelastic/Brand-driven" if sensitivity < 0.8 else "Normal",
                "share_of_voice": "Dominant" if sov_signal > 1.5 else "Audible" if sov_signal > 0.8 else "Drowned Out",
                "marketing_synergy": "Strong Multiplier" if synergy_score < -0.05 else "Weak/Disconnected", # Negative SHAP/Value means price gap is being closed by marketing
                "efficiency_trend": media_state,
            },
            "wargame_alerts": {
                "promotion_fatigue": bool(promo_fatigue),
                "imminent_stock_pressure": bool(row['sales'] > row['rolling_sales_4w'] * 1.3),
                "competitor_price_war": bool(row['leader_price_velocity'] < -0.04)
            }
        }
        return context
