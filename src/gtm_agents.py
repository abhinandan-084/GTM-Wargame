# Library Imports
from typing import List, Dict, Any, Union, Literal
import re

# LangChain Imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

class GTMBrain:
    def __init__(self, provider: Literal["gemini", "ollama"], api_key: str = None):
        self.provider = provider
        if self.provider == "gemini":
            # Gemini : temp = 0.1 for consistency 
            self.llm = ChatGoogleGenerativeAI(
                #model="gemini-2-flash-preview",  # model_names = gemini-3-flash-preview, gemma-4-31b-it
                model ="gemini-3-flash-preview",
                google_api_key=api_key, 
                temperature=0.1 
            )
        else:
            # Ollama (Qwen/Llama) : temp = 0.3-0.4 provides the 'entropy' needed to synthesize complex data without getting stuck in loops.
            self.llm = ChatOllama(
                model="qwen3.5:9b-q4_K_M", 
                temperature=0.4 # Raised for local model fluidity
            )
    def get_analyst_node(self, shap_values: Dict, market_context: Dict):
        """
        ROLE: Principal Data Scientist.
        GOAL: Diagnostics. Match SHAP drivers to Market Regimes.
        """

        template = """
        SYSTEM: You are a Principal Data Scientist for a Smartphone OEM. Your goal is to explain the 'Why' behind sales movement.
        INPUT_DATA (SHAP Values): {shap_info}
        BUSINESS_CONTEXT: {context}

        [STRICT FORMATTING RULES]
        - NO 'To:', 'From:', 'Subject:', or 'Date:' headers.
        - NO email salutations (e.g., 'Dear Team', 'Hi everyone').
        - NO conversational filler or signatures at the end.
        - START immediately with the first heading.
        - Use clean Markdown (### Headings).
        - Use ### for Section Headers.
        - Use - for bullet points.
        - Bold the text before the colon in every bullet point (e.g., **Category Name**: Description).
        - NEVER use asterisks (*) or underscores (_) for bold or italics except other than the text before colon in every bullet point
        - Ensure every section starts on a new line.

        TASK: 
        1. DRIVER ANALYSIS: Identify the top 3 SHAP drivers and explain WHY they are moving based on the 'signals' in the context
        2. SENSITIVITY CHECK: If SHAP shows 'price' is a driver, cross-reference with 'price_sensitivity'. Is the model detecting the regime correctly?
        3. REASONING: Connect these drivers to the BUSINESS_CONTEXT (e.g., if 'price_gap_pct' is high and sales are down, hypothesize price sensitivity).
        4. SYNERGY AUDIT: Look at 'marketing_synergy'. Is our spend actually amplifying our price position or is it disconnected?
        5. Check the wargame_alerts. If competitor_price_war is True, ignore small fluctuations in Retail Spend and focus your analysis on how the price_gap_pct is cannibalizing our baseline volume.
        
        OUTPUT FORMAT:
        - TOP DRIVERS: (List the drivers and their contextual significance)
        - SCIENTIFIC HYPOTHESIS: (e.g., "Despite the launch phase, price sensitivity is 'Hyper-sensitive', meaning our premium position is under threat.")
        - DATA ANOMALIES: (Mention any 'wargame_alerts' that contradict the drivers)

        RULES:
        - DO NOT invent numbers. 
        - NEVER mention a number not in the INPUT_DATA.
        - If 'promotion_fatigue' is True, highlight it as a risk to the drivers.
        """
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm
        
        if self.provider == "gemini": 
            return chain.invoke({"shap_info": shap_values, "context": market_context}).content[0].get('text')
        else:
            return chain.invoke({"shap_info": shap_values, "context": market_context}).content

    def get_strategist_node(self, opt_results: Dict, analyst_insight: str, market_context: Dict):
        """
        ROLE: Growth Strategist.
        GOAL: Tactics. Critique the Optimizer using the 'Share of Voice' and 'Efficiency' signals.
        """

        template = """
        SYSTEM: You are a Growth Strategist. You bridge the gap between ML Optimization and Boardroom Strategy.
        OPTIMIZER_RESULTS: {opt_results}
        ANALYST_INSIGHTS: {analyst_insight}
        MARKET_CONTEXT: {context}

        [STRICT FORMATTING RULES]
        - NO 'To:', 'From:', 'Subject:', or 'Date:' headers.
        - NO email salutations (e.g., 'Dear Team', 'Hi everyone').
        - NO conversational filler or signatures at the end.
        - START immediately with the first heading.
        - Use clean Markdown (### Headings).
        - Use ### for Section Headers.
        - Use - for bullet points.
        - Bold the text before the colon in every bullet point (e.g., **Category Name**: Description).
        - NEVER use asterisks (*) or underscores (_) for bold or italics except other than the text before colon in every bullet point
        - Ensure every section starts on a new line.

        TASK:
        1. OPTIMIZER CRITIQUE: The optimizer wants to shift budget. Evaluate this against Analyst's insight on efficiency. (e.g., If we are 'Drowned Out', does a budget increase even make sense?)
        2. RISK ASSESSMENT: Is the recommended 'Optimized Price' too close to the 'Market Leader Price'?
        3. EFFICIENCY EVALUATION: Look at 'efficiency_trend'. If ROI is 'Diminishing', justify why the optimizer is still suggesting spend, or suggest a 'Defensive Hold'.
        3. STRATEGIC TAGGING: Categorize the upcoming 2 weeks as 'Aggressive Expansion', 'Market Share Protection', or 'Margin Preservation'.

        OUTPUT FORMAT:
        - BUDGET RATIONALE: (Why we are moving money between Search, Social, and Retail)
        - STRATEGIC PLAY: (The Tag and the Logic)
        - RISK ASSESSMENT: (Specifically mention if 'competitor_price_war' is active)

        CONSTRAINTS:
        - You are bounded by the Budget Limit. 
        - If 'lift_percent' is negative, challenge the model's price suggestion.
        """
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm

        if self.provider == "gemini": 
            return chain.invoke({"opt_results": opt_results, "analyst_insight": analyst_insight, "context": market_context}).content[0].get('text')
        else:
            return chain.invoke({"opt_results": opt_results, "analyst_insight": analyst_insight, "context": market_context}).content

    # GTM Manager Analyst
    def get_gtm_manager_node(self, analysis: str, strategy: str, market_context: Dict):
        """
        ROLE: GTM Manager / Boardroom Lead.
        GOAL: Execution. Create the 'Wargame Playbook' and handle alerts.
        """

        template = """
        SYSTEM: You are the GTM Manager. You are leading a Wargame and make the final go/no-go decision.

        HISTORICAL_ANALYSIS: {analysis}
        PROPOSED_STRATEGY: {strategy}
        MARKET_CONTEXT: {context}

        [STRICT FORMATTING RULES]
        - NO 'To:', 'From:', 'Subject:', or 'Date:' headers.
        - NO email salutations (e.g., 'Dear Team', 'Hi everyone').
        - NO conversational filler or signatures at the end. 
        - START immediately with the first heading.
        - Use clean Markdown (### Headings).
        - Use ### for Section Headers.
        - Use - for bullet points.
        - Bold the text before the colon in every bullet point (e.g., **Category Name**: Description).
        - NEVER use asterisks (*) or underscores (_) for bold or italics except other than the text before colon in every bullet point
        - Ensure every section starts on a new line.

        TASK:
        1. EXECUTIVE SUMMARY: Synthesize the 'So-What' for the Director. Focus on the 'lifecycle_stage'
        2. THE WARGAME PLAY: Provide a 3-step execution plan for the sales team.
        3. CONTINGENCY: Use the 'wargame_alerts' to suggest what to do if 'imminent_stock_pressure' or a 'price_war' escalates.
        4. COUNTER-MOVE: What should we do if the competitor drops price by another 5%?

        RULES:
        - NO conversational filler.
        - Under no circumstances mention a number not listed in the Strategy or Analysis.
        - Be decisive.
        """
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm

        if self.provider == "gemini":
            return chain.invoke({"analysis": analysis, "strategy": strategy, "context": market_context}).content[0].get('text')
        else:
            return chain.invoke({"analysis": analysis, "strategy": strategy, "context": market_context}).content