# 🤖 **Agentic GTM Strategy Simulator**

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![LangChain](https://img.shields.io/badge/Framework-LangChain-green.svg)
![XGBoost](https://img.shields.io/badge/ML-XGBoost-orange.svg)
![Optimization](https://img.shields.io/badge/Optimization-SciPy-red.svg)
![Guardrails](https://img.shields.io/badge/Guardrails-Consistency--Checker-blueviolet)

This repository contains a modular **Agentic GTM (Go-To-Market) Strategy Simulator** that bridges Machine Learning Diagnostics (XGBoost/SHAP) and Executive Strategy (LangChain/LLM Agents). It is designed specifically for a OEM to simulate weekly sales performance and optimize marketing spend across Search, Social, and Retail channels. Unlike standard dashboards that merely report data, this system uses a multi-agent "Boardroom" architecture to perform Strategic Wargaming. It transforms XGBoost diagnostics and Scipy optimizations into actionable executive playbooks while maintaining strict numerical grounding.

### 💎 **Closing the "Signal-to-Action" Gap**

The gap between **Data Science** (ML outputs) and **GTM Strategy** (Boardroom decisions) is often weeks of manual interpretation. This project solves three critical "Principal-level" problems:

1. **The Contextualization Problem:** A SHAP value of `0.4` for Price is meaningless without knowing if we are in a *Price War* or a *Launch Window*. The **Programmatic Context Detector** ensures the AI "knows" the market regime before it speaks.

2. **The Summarizer-to-Thinker Transition:** Most GenAI apps simply re-state the data. This system uses **Chain-of-Thought (CoT)** prompting to force the Agents to *critique* the optimizer and hypothesize *why* drivers are moving, moving the AI from a "reporter" to a "strategic advisor."

3. **The Hallucination Guardrail:** For a businesses, a single hallucinated number destroys the credibility of the entire dashboard. The **Consistency Checker** provides the mathematical rigor required to deploy LLMs in high-stakes financial environments.
<br/><br/>
  
<video width="630" height="300" src="https://github.com/user-attachments/assets/c1876436-e99f-4f61-8b40-b120f6ddcfa5"></video>


## 🏛️ **Architecture Overview**

The system operates as a closed-loop "Reasoning Engine" spanning four distinct layers:

**1. The Simulation Engine (data_generator.py):** Generates high-variance weekly sales data considering price decay, seasonality, competitor launch spikes, and marketing adstock (Hill functions).

**2. Diagnostic Engine:** Employs XGBoost for sales forecasting and SHAP (SHapley Additive exPlanations) for local feature attribution on the most recent "Today" week. Leverages `scipy.optimize.differential_evolution` to find the global optimum for spend allocation across Social, Search, and Retail channels.

**3. Programmatic Context Layer:** A heuristic engine that translates raw metrics (Price Elasticity, Share of Voice, Efficiency Momentum) into strategic "Market Regimes."

**4. Agentic Reasoning Layer:** A Three-Agent consensus model (Analyst, Strategist, Manager) using LangChain with dual-provider support (Gemini 3 & Qwen 3.5:9b).


## 📂 **File System**

```text
├── boardroom_app.py        # Streamlit UI: The Boardroom Dashboard
├── gtm_agents.py           # Agentic Layer: LangChain Logic (Gemini/Ollama)
├── driver_analysis.py      # Core Engine: XGBoost, SHAP, SciPy Optimization
├── data_generator.py       # Simulation: Synthetic Market Data Factory
├── consistency_checker.py  # Guardrail: L3 Numerical Validation Module
├── schemas.py              # Type Definitions: DataConfig & OEMTierConfig
└── requirements.txt        # Dependency Manifest
```

## 🚀 **Key Technical Features**

**1. Programmatic Context Detection**

Unlike standard systems, this system uses a Market Context Detector within GTM_DriverEngine. It calculates:
- **Price Sensitivity**: Differentiates between "Hyper-sensitive" and "Brand-driven" regimes.
- **Share of Voice (SOV)**: Measures marketing dominance relative to competitor launch cycles.
- **Promotion Fatigue**: Detects diminishing returns on retail rebates.

**2. Multi-Agent Boardroom Logic**

The system implements Chain-of-Thought (CoT) reasoning across three personas:
- **Data Analyst**: Maps SHAP values to detected market signals.
- **Growth Strategist**: Critiques SciPy optimization results against the "Efficiency Momentum."
- **GTM Manager**: Synthesizes a "Wargame Playbook" with contingency plans for competitor price wars.

**3. L3 Numerical Guardrails**

The ConsistencyChecker prevents LLM hallucinations. It parses agent responses using regex and cross-references extracted numbers against the Ground Truth from the Optimization Engine, flagging mismatches. If the Agent suggests a "15% Lift" but the Scipy Optimizer outputted "12%", the system flags a Hallucination Alert in the UI, ensuring the boardroom only sees validated data.

**If a hallucination is detected, the UI flags the specific metric for boardroom review.**

**4. Hybrid LLM Support**

The system implements a **Hybrid LLM Architecture**, allowing the GTM Boardroom to run on either enterprise cloud-scale models or local privacy-first models.

* **Enterprise Cloud: Google Gemini 3**
    - **Use Case:** High-stakes boardroom simulations and complex strategic synthesis.
    - **Rationale:** Leverages massive parameter counts for high-fidelity **Chain-of-Thought (CoT)** reasoning. 
    - **Optimization:** Configured at `Temperature: 0.1` for maximum determinism. The high reasoning density allows the model to map disparate SHAP floats to qualitative market regimes with zero "logit drift."

* **Local Edge: Ollama (Qwen 3)**
    - **Use Case:** Rapid iterative testing, cost-sensitive simulations, and **Data Sovereignty** compliance.
    - **Rationale:** Ensures that sensitive internal sales data, pricing strategies, and competitor diagnostics never leave the corporate firewall
    - **Inference Tuning:** Unlike cloud models, local models (7B-14B) are susceptible to "logit starvation" at near-zero temperatures. We have optimized the local engine at `Temperature: 0.4` with `Top_P: 0.9` to ensure the Analyst Agent maintains fluid reasoning without looping or robotic stagnation.

## 🛠️ **Installation & Setup**

**1. Clone the Repository**
```bash
git clone https://github.com/abhinandan-084/GTM-Wargame.git
cd GTM-Wargame
```

**2. Prerequisites**

```bash
python -m venv venv
source venv/bin/activate
pip install streamlit langchain langchain-google-genai langchain-ollama xgboost shap pandas numpy scipy plotly
```

**3. Environment Setup**

```text
Gemini: Export your GOOGLE_API_KEY to your .env file
Ollama: Ensure ollama serve is running and the qwen3.5:9b model is pulled. Alternatively, you can choose any other model and change the name in the gtm_agents.py
```

**4. Execution**

You can choose between a local Ollama instance (for data privacy) or Google Gemini (for high-reasoning density).

```bash
streamlit run boardroom_app.py
```

## **📊 Methodology**

**1. Forecasting** : XGBoost Regressor (n_estimators=200, max_depth=5) trained on rolling historical features.

**2. Optimization**: scipy.optimize.differential_evolution used to maximize sales lift across three channels (Search, Social, Retail) constrained by a fixed budget.

**3. Attribution**: Tree-based SHAP values provide the "Why" for the most recent data point, enabling the Agent to explain sales variances.

## **📈 Key Boardroom Metrics**

- **Projected Sales Lift**: The Delta between the optimized proposal and historical baseline.

- **Marketing Synergy Score**: Programmatic detection of whether adstock is amplifying price-gap impact.

- **Promotion Fatigue Index**: Identifying diminishing returns in rebate strategies.



#### **Note**:

This framework is designed to be extensible. The GTM_DataGenerator can be replaced with a SQL/Snowflake connector, and the GTMBrain can be extended to include a 'CFO Agent' for margin-impact validation.
