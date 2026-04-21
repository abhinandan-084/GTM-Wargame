import yaml

config_yaml = """
simulation_config:
  start_date: '2023-05-01'
  weeks: 156
  base_price: 799.0
  random_state: 42
  launch_spike: 2.0
  market_leader_price_decay_factor: 0.997

oem_tiers:
  leader:
    rank: 1
    hill_k: 30000
    hill_n: 1.1
    coeffs:
      baseline: 7.0
      elasticity: 0.3
      retail_dependency: 0.6
      ad_impact: 1.5
      launch_month: 9
      rank_a_impact: {impact: 0.90, launch_month: 1}
      rank_b_impact: {impact: 0.95, launch_month: 8}
      weekly_decay_factor: 0.997
      monthly_seasonality: [1.0, 1.0, 1.2, 1.1, 1.0, 0.85, 0.9, 1.0, 1.2, 1.0, 1.10, 1.10]
      base_retail_spend : 100000
      retail_spend_trend : 20
      base_search_spend : 70000
      search_spend_trend : 10
      base_social_spend : 80000
      social_spend_trend : 25
  challenger:
    rank: 2
    hill_k: 55000
    hill_n: 1.5
    coeffs:
      baseline: 2.0
      elasticity: 1.0
      retail_dependency: 1.4
      ad_impact: 1.2
      launch_month: 1
      rank_a_impact: {impact: 0.80, launch_month: 9}
      rank_b_impact: {impact: 0.90, launch_month: 8}
      weekly_decay_factor: 0.993
      monthly_seasonality: [1.2, 1.0, 1.2, 1.1, 1.0, 0.85, 0.9, 1.0, 1.0, 1.0, 1.10, 1.10]
      base_retail_spend : 70000
      retail_spend_trend : 30
      base_search_spend : 40000
      search_spend_trend : 15
      base_social_spend : 50000
      social_spend_trend : 20
  upstart:
    rank: 3
    hill_k: 80000
    hill_n: 2.2
    coeffs:
      baseline: 1.0
      elasticity: 1.8
      retail_dependency: 2.1
      ad_impact: 1.0
      launch_month: 8
      rank_a_impact: {impact: 0.80, launch_month: 9}
      rank_b_impact: {impact: 0.85, launch_month: 1}
      weekly_decay_factor: 0.990
      monthly_seasonality: [1.0, 1.0, 1.2, 1.1, 1.0, 0.85, 0.9, 1.2, 1.0, 1.0, 1.10, 1.10]
      base_retail_spend : 70000
      retail_spend_trend : 40
      base_search_spend : 40000
      search_spend_trend : 20
      base_social_spend : 50000
      social_spend_trend : 25
"""

with open('simulation_config.yaml', 'w') as f:
    f.write(config_yaml.strip())

with open('simulation_config.yaml', 'r') as f:
    loaded_config = yaml.safe_load(f)

print("Config file created and loaded successfully.")