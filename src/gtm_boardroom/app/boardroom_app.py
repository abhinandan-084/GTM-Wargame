# Import functions and libraries
import os
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from gtm_boardroom.diagnostics.driver_engine import GTM_DriverEngine
from gtm_boardroom.data.source import CSVDataSource, SyntheticDataSource
from gtm_boardroom.agents.gtm_agents import GTMBrain
from gtm_boardroom.guardrails.consistency_checker import ConsistencyChecker
from gtm_boardroom.data.config import get_tier_config
from gtm_boardroom.agents.providers import PROVIDER_ENV_VARS, detect_available_providers

# dotenv for reading env variables
from dotenv import load_dotenv
load_dotenv()  # reads variables from a .env file and sets them in os.environ

# The sample historical CSV lives in the repo's top-level data/ dir, not inside the
# installed package, since it's a committed dataset rather than package resource.
REPO_ROOT = Path(__file__).resolve().parents[3]
HISTORICAL_CSV_PATH = REPO_ROOT / "data" / "simulated_sales_data_rank_3.csv"

DATA_SOURCES = {
    "Synthetic Generator": "synthetic",
    "Historical CSV (data/simulated_sales_data_rank_3.csv)": "csv",
}

if __name__ == '__main__':
    # Initializing Session State : Ensures that the results stay on screen even if we move a slider after the simulation is done
    # and to stop streamlit's default behavior where any widget change triggers a full script rerun
    if 'run_sim' not in st.session_state:
        st.session_state['run_sim'] = None

    # Data Orchestrator Function : Encapsulates the entire simulation workflow
    def run_full_pipeline(data_source, llm_provider, api_key, horizon):
        """
        Runs the full GTM simulation pipeline, from data loading to agentic analysis.
        Args:
            data_source (DataSource): Upstream source providing the weekly GTM dataframe.
            llm_provider (str): The LLM provider to use (e.g., "gemini", "openai", "anthropic", "llamacpp").
            api_key (str): The API key for the LLM provider (if applicable).
            horizon (int): The planning horizon in weeks.
        Returns:
            dict: A dictionary containing all the results from the simulation, analysis, and strategy.
        """
        # 1. Load the weekly GTM dataframe from whichever upstream source was selected
        df = data_source.load()

        # 2. Driver & Optimization Engine 
        # Set current week index for historical data slicing
        current_week_idx = (horizon+1)*(-1)
        engine = GTM_DriverEngine(df, current_week_idx=current_week_idx)
        with st.spinner("📈 💰 Running Driver Engine & Optimiser..."):
            shap_info = engine.get_diagnostics()
            market_context = engine.get_market_context()
            opt_results = engine.optimize_strategy(budget_limit=budget_limit, horizon=horizon)
            comparison_df = engine.compare_optimized_vs_actual(opt_results, horizon=horizon)

        # 3. Agentic Layer
        # Initialize the GTM Brain with the selected LLM provider and API key
        brain = GTMBrain(provider=llm_provider, api_key=api_key)
        
        with st.spinner("🕵️ Analyst diagnosing drivers..."):
            analysis = brain.get_analyst_node(shap_info, market_context)
              
        with st.spinner("📈 Strategist evaluating spend..."):
            strategy = brain.get_strategist_node(opt_results, analysis, market_context) 
            
        with st.spinner("👔 GTM Manager finalizing playbook..."):
            summary = brain.get_gtm_manager_node(analysis, strategy, market_context)

        # 4. Consistency Checker 
        # Validate the generated summary against numerical results
        checker = ConsistencyChecker()

        if summary != None:
            validation = checker.validate_response(text=summary, shap_info=shap_info, market_context=market_context, opt_results=opt_results)
        else:
            validation = {'error_msg': 'Issue with Data Generation', 'is_valid':False}

        results = {
            "summary": summary, "analysis": analysis, "strategy": strategy,
            "opt_results": opt_results, "market_context": market_context,
            "comparison_df": comparison_df, "shap_info": shap_info,
            "valid_results": validation,
        }
        return results

    # Page Config : Streamlit page configuration
    st.set_page_config(
        page_title="GTM Wargame: Boardroom Simulator",
        page_icon="📱",
        layout="wide"
    )

    # Styling : Custom CSS styling for better aesthetics [Refer MDN Web Docs (Mozilla Developer Network) for more]
    st.markdown("""
        <style>
        .main { background-color: #f5f7f9; }
        .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
        .agent-box { padding: 20px; border-radius: 10px; margin-bottom: 20px; border-left: 5px solid #1f77b4; background-color: white; }
        </style>
        """, unsafe_allow_html=True)

    # Sidebar : Sidebar for simulation controls
    with st.sidebar:
        st.title("🕹️ Simulation Control")

        # Data source selection: pluggable upstream sources behind the DataSource interface.
        data_source_label = st.selectbox("Data Source", list(DATA_SOURCES.keys()))
        if DATA_SOURCES[data_source_label] == "synthetic":
            data_cfg, oem_cfg, coeffs = get_tier_config('upstart')
            data_source = SyntheticDataSource(data_cfg, oem_cfg, coeffs)
        else:
            data_source = CSVDataSource(HISTORICAL_CSV_PATH)

        st.divider()

        # LLM provider selection: only show providers whose API key is present in the
        # environment, plus local providers (e.g. llama.cpp) that don't need one.
        available_providers = detect_available_providers()
        llm_provider = st.selectbox("Intelligence Engine", available_providers)

        api_key = None
        if llm_provider in PROVIDER_ENV_VARS:
            env_var = PROVIDER_ENV_VARS[llm_provider]
            api_key_input = st.text_input(f"API Key ({env_var})", type="password")
            api_key = api_key_input or os.environ.get(env_var)

        st.divider()
        horizon = st.selectbox("Planning Horizon (Weeks)", [2, 4], index=0)

        # Budget limit slider, dependent on horizon
        if horizon == 2:
            budget_limit = st.slider("Total Strategy Budget ($)", 300000, 700000, 500000)
        else:
            budget_limit = st.slider("Total Strategy Budget ($)", 500000, 1000000, 750000)
            
        
        st.divider()
        if st.button("🚀 Run GTM Wargame", use_container_width=True):
            st.session_state['run_sim'] = run_full_pipeline(data_source, llm_provider, api_key, horizon)

    # Main UI Code
    # Display results if simulation has been run
    if st.session_state['run_sim']:
        results = st.session_state['run_sim']
        
        # Header for the main content area
        st.title("Boardroom Briefing: GTM Strategy Playbook")
        
        # Row 1 : Top Level Metrics
        # Display key metrics using Streamlit columns
        m1, m2, m3, m4 = st.columns(4)
        lift = results['opt_results']['lift_percent']
        m1.metric("Projected Sales Lift", f"{lift}%", delta_description = "vs Baseline")
        m2.metric("Market Regime", results['market_context']['market_regime']['phase'])
        m3.metric("Price Position", results['market_context']['market_regime']['price_position'].split('(')[0])
        m4.metric("Competitive Threat", results['market_context']['market_regime']['competitive_price_threat'].split('(')[0])

        # Tabs for different roles/viewpoints : Content organized into tabs for different perspectives
        tab_manager, tab_analytics, tab_optimization = st.tabs([
            "👔 Executive Playbook", "🧬 Driver Diagnostics", "📊 Spend Optimization"
        ])

        with tab_manager:
            col_left, col_right = st.columns([2, 1])
            with col_left:
                st.subheader("GTM Manager: Strategic Synthesis")
                with st.container(border=True):
                    st.write(results['summary'])
                
                st.subheader("Growth Strategist: Rationale")
                with st.container(border=True):
                    st.write(results['strategy'])
            
            with col_right:
                st.subheader("🛡️ Guardrail Status")
                if results['valid_results']['is_valid']:
                    st.success("✅ Numerical Consistency Verified")
                else:
                    st.error("⚠️ Hallucination Detected")
                    st.write(results['valid_results']['error_msg'])
                
                st.subheader("Wargame Alerts")
                for alert, active in results['market_context']['wargame_alerts'].items():
                    if active:
                        st.warning(f"🚨 {alert.replace('_', ' ').title()}")
                    else:
                        st.write(f"✅ {alert.replace('_', ' ').title()}: Stable")

        with tab_analytics:
            col_a, col_b = st.columns([1, 1])
            with col_a:
                st.subheader("Top Sales Drivers (SHAP)")
                # DataFrame for SHAP values and for visaulisation
                shap_df = pd.DataFrame(list(results['shap_info'].items()), columns=['Feature', 'Impact'])
                shap_df = shap_df.sort_values('Impact', ascending=False).head(10)
                fig_shap = px.bar(shap_df, x='Impact', y='Feature', orientation='h', color='Impact',
                                color_continuous_scale='RdBu_r')
                st.plotly_chart(fig_shap, use_container_width=True)
                
            with col_b:
                st.subheader("Market Signals")
                st.json(results['market_context']['signals'])
                st.markdown("---")
                st.subheader("Analyst Deep Dive:")
                with st.container(border=True):
                    st.markdown(results['analysis'])

        with tab_optimization:
            st.subheader("Actual (Historical) vs. Optimized (Proposal)")
            comp_df = results['comparison_df'].set_index('Metric')
            
            # Visualize the Cumulative Sales and Budget Spend in One Chart
            spend_metrics = ['Cumulative Sales', 'Total Budget Spent']
            fig_sales_spend = go.Figure(data=[
                go.Bar(name='Historical', x=spend_metrics, y=comp_df.loc[spend_metrics, 'Actual (Historical)']),
                go.Bar(name='Optimized', x=spend_metrics, y=comp_df.loc[spend_metrics, 'Optimized (Proposal)'])
            ])

            # Layout setting for sales and spend chart
            fig_sales_spend.update_layout(
                title='Sales Performance',
                xaxis_title='Metric name',
                yaxis_title='Metric value',
                template='plotly_white', # Cleaner look
                legend=dict(orientation="h",yanchor="bottom", y=1.02, xanchor="center", x=0.5)
            )

            st.plotly_chart(fig_sales_spend, use_container_width=True)

            # Visualize the Average Price and Market Leader Price
            spend_metrics = ['Average Price', 'Average Market Leader Price']
            fig_price = go.Figure(data=[
                go.Bar(name='Historical', x=spend_metrics, y=comp_df.loc[spend_metrics, 'Actual (Historical)']),
                go.Bar(name='Optimized', x=spend_metrics, y=comp_df.loc[spend_metrics, 'Optimized (Proposal)'])
            ])

            # Layout setting for Average Price and Market Leader Price
            fig_price.update_layout(
                title='Price Performance',
                xaxis_title='metric',
                yaxis_title='Price ($)',
                template='plotly_white', # Cleaner look
                legend=dict(orientation="h",yanchor="bottom", y=1.02, xanchor="center", x=0.5)
            )
            fig_price.update_yaxes(showgrid=False)

            st.plotly_chart(fig_price, use_container_width=True)

            # Displaying the comparison dataframe with custom formatting
            st.dataframe(comp_df
            .style
            .format(formatter='$ {:.2f}',
            subset=pd.IndexSlice[['Average Price','Average Market Leader Price'],['Actual (Historical)','Optimized (Proposal)','Delta']])
            .format(formatter=lambda x: f'{x/1e6:.1f}M' if x >= 1e6 else f'{x/1e3:.1f}K',
            subset=pd.IndexSlice['Cumulative Sales', ['Actual (Historical)','Optimized (Proposal)','Delta']])
            .format(formatter=lambda x: f'$ {x/1e6:.1f}M' if x >= 1e6 else f'$ {x/1e3:.1f}K',
            subset=pd.IndexSlice['Total Budget Spent', ['Actual (Historical)','Optimized (Proposal)','Delta']])
            .highlight_max(axis=1, subset=pd.IndexSlice['Cumulative Sales',['Actual (Historical)', 'Optimized (Proposal)']], color='#d4edda')
            .highlight_min(axis=1, subset=pd.IndexSlice[['Average Price','Total Budget Spent'],['Actual (Historical)', 'Optimized (Proposal)']], color='#d4edda')
            .set_properties(**{'text-align': 'center'}))

    else:
        # Landing State: Base state when the simulation has not been run
        st.info("👈 Select your LLM engine and budget in the sidebar, then click 'Run GTM Wargame' to start the simulation.")