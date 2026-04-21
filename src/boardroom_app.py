# Import functions and libraries
import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import yaml
from driver_engine import GTM_DriverEngine
from data_generator import GTM_DataGenerator
from gtm_agents import GTMBrain
from consistency_checker import ConsistencyChecker
from schemas import DataConfig, OEMTierConfig

# dotenv for reading env variables
from dotenv import load_dotenv
load_dotenv()  # reads variables from a .env file and sets them in os.environ

if __name__ == '__main__':

    # Initializing Session State : Ensures that the results stay on screen even if we move a slider after the simulation is done.
    if 'run_sim' not in st.session_state:
        st.session_state['run_sim'] = None

    # --- DATA ORCHESTRATION ---
    def run_full_pipeline(llm_provider, api_key, horizon):
        # 1. Generate Synthetic Data (using your finalized generator)
        # Load YAML
        with open('simulation_config.yaml', 'r') as f:
            full_config = yaml.safe_load(f)
            
        # Extract Data Objects
        s_cfg = full_config['simulation_config']
        data_cfg = DataConfig(**s_cfg)
        tier_data = full_config['oem_tiers']['upstart']
        oem_cfg = OEMTierConfig(rank=tier_data['rank'], hill_k=tier_data['hill_k'], hill_n=tier_data['hill_n'])
        coeffs = tier_data['coeffs']
        promos = []
        
        # Data Generator Class
        generator = GTM_DataGenerator(data_cfg, oem_cfg, coeffs, promos=promos) 
        df = generator.generate()

        # 2. Driver & Optimization Engine 
        current_week_idx = (horizon+1)*(-1)
        engine = GTM_DriverEngine(df, current_week_idx=current_week_idx)
        with st.spinner("📈 💰 Running Driver Engine & Optimiser..."):
            shap_info = engine.get_diagnostics()
            market_context = engine.get_market_context()
            opt_results = engine.optimize_strategy(budget_limit=budget_limit, horizon=horizon)
            comparison_df = engine.compare_optimized_vs_actual(opt_results, horizon=horizon)

        # 3. Agentic Layer
        brain = GTMBrain(provider=llm_provider, api_key=api_key)
        
        with st.spinner("🕵️ Analyst diagnosing drivers..."):
            analysis = brain.get_analyst_node(shap_info, market_context)
              
        with st.spinner("📈 Strategist evaluating spend..."):
            strategy = brain.get_strategist_node(opt_results, analysis, market_context) 
            
        with st.spinner("👔 GTM Manager finalizing playbook..."):
            summary = brain.get_gtm_manager_node(analysis, strategy, market_context)

        # Consistensy Checker
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

    # --- PAGE CONFIG ---
    st.set_page_config(
        page_title="GTM Wargame: Boardroom Simulator",
        page_icon="📱",
        layout="wide"
    )

    # --- STYLING ---
    st.markdown("""
        <style>
        .main { background-color: #f5f7f9; }
        .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
        .agent-box { padding: 20px; border-radius: 10px; margin-bottom: 20px; border-left: 5px solid #1f77b4; background-color: white; }
        </style>
        """, unsafe_allow_html=True)

    # --- SIDEBAR: CONTROLS ---
    with st.sidebar:
        st.title("🕹️ Simulation Control")
        llm_provider = st.selectbox("Intelligence Engine", ["gemini", "ollama"])
        api_key = st.text_input("API Key (if Gemini)", type="password")

        if api_key == "":
            api_key = os.environ.get('GOOGLE_API_KEY')

        st.divider()
        #budget_limit = st.slider("Total Strategy Budget ($)", 100000, 1000000, 500000)
        horizon = st.selectbox("Planning Horizon (Weeks)", [2, 4], index=0)

        if horizon == 2:
            budget_limit = st.slider("Total Strategy Budget ($)", 300000, 700000, 500000)
        else:
            budget_limit = st.slider("Total Strategy Budget ($)", 500000, 1000000, 750000)
            
        
        st.divider()
        if st.button("🚀 Run GTM Wargame", use_container_width=True):
            st.session_state['run_sim'] = run_full_pipeline(llm_provider, api_key, horizon)

    # --- MAIN UI ---
    if st.session_state['run_sim']:
        results = st.session_state['run_sim']
        
        # HEADER SECTION
        st.title("Boardroom Briefing: GTM Strategy Playbook")
        
        # --- ROW 1: TOP LEVEL METRICS ---
        m1, m2, m3, m4 = st.columns(4)
        lift = results['opt_results']['lift_percent']
        m1.metric("Projected Sales Lift", f"{lift}%", delta_description = "vs Baseline")
        m2.metric("Market Regime", results['market_context']['market_regime']['phase'])
        m3.metric("Price Position", results['market_context']['market_regime']['price_position'].split('(')[0])
        m4.metric("Competitive Threat", results['market_context']['market_regime']['competitive_price_threat'].split('(')[0])

        # --- TABS FOR DIFFERENT VIEWPOINTS ---
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
                #st.info(results['strategy'])
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

            # 2. Update Layout (The "Formatting" of the container)
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

            fig_price.update_layout(
                title='Price Performance',
                xaxis_title='metric',
                yaxis_title='Price ($)',
                template='plotly_white', # Cleaner look
                legend=dict(orientation="h",yanchor="bottom", y=1.02, xanchor="center", x=0.5)
            )
            fig_price.update_yaxes(showgrid=False)

            st.plotly_chart(fig_price, use_container_width=True)

            # Display the final dataframe 
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
        # Landing State
        st.info("👈 Select your LLM engine and budget in the sidebar, then click 'Run GTM Wargame' to start the simulation.")