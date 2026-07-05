# GTM-Wargame: Measured-Results Plan (ConsistencyChecker Benchmark)

Self-contained execution plan for a Claude Code session. Work through the phases
in order; each has acceptance criteria. Do not skip Phase 0.

---

## Context (read once)

Repo: `github.com/abhinandan-084/GTM-Wargame` — agentic GTM strategy simulator.
An XGBoost + SHAP + SciPy pipeline produces ground-truth numbers; a three-agent
LangChain "boardroom" (Analyst → Strategist → Manager) writes an executive memo
over them. `ConsistencyChecker` regex-extracts every number from the agents'
prose and validates it against a "ground truth pool" of real pipeline numbers;
unmatched numbers are flagged as hallucinations.

The repo currently *asserts* the guardrail works via one illustrative example.
The goal is **measured results** for a publication write-up:

- **Experiment B** — the checker's precision/recall as a classifier
  (deterministic, no LLM calls, CI-safe). **Build first.**
- **Experiment A** — fabrication rates of real LLMs with the guardrail
  observing (real API calls, run locally, never in CI).

### Framing for Experiment A (important — this shapes what gets logged)
This is **not a model leaderboard**. Two models play two rhetorical roles:

1. **Frontier model (Gemini)** = "even the best still does it": a frontier
   model, on clean synthetic data, with prompts that *explicitly forbid*
   inventing numbers, still fabricates in X% of runs. That's the headline.
2. **Local model (llama.cpp, e.g. Mistral-7B)** = "why the guardrail is
   load-bearing when you're forced local" (privacy/cost deployments). Its
   higher rate is the point, not an embarrassment.

Consequences for design:
- **Paired design**: identical seed list across all models, so differences are
  attributable to the model, not the scenario draw.
- **Raw counts, not just rates**: report "4 of 50 runs" + a 95% CI
  (Wilson interval), never a bare percentage. Small-N rates are noisy.
- Log everything needed to quote real flagged transcripts in the article.

### Honesty caveat (goes in docstrings + article)
This measures **numerical grounding against a known pool** — whether a number
*exists* in the ground truth — not semantic correctness (whether the agent used
the right number in a valid argument). State it explicitly.

---

## Phase 0 — Verify before writing anything

1. Read these files end-to-end:
   - `src/gtm_boardroom/guardrails/consistency_checker.py`
   - `src/gtm_boardroom/agents/gtm_agents.py`, `agents/providers.py`
   - `src/gtm_boardroom/diagnostics/driver_engine.py`
   - `src/gtm_boardroom/data/{config.py, source.py, generator.py, schemas.py, simulation_config.yaml}`
   - `tests/test_consistency_checker.py` (match its style)
   - `.github/workflows/ci.yml`
2. Confirm the signatures below still hold. If anything differs, adapt the plan
   to the source — the source wins.
3. `uv sync --group dev && uv run pytest -v` — baseline must be green before
   any changes.

### Expected signatures

```python
# guardrails
ConsistencyChecker.validate_response(text, shap_info, market_context, opt_results)
  -> {"is_valid": bool, "hallucinated_values": List[float], "error_msg": str|None}

# agents
GTMBrain(provider: str, api_key=None, **kwargs)
  .get_analyst_node(shap_values, market_context) -> str
  .get_strategist_node(opt_results, analyst_insight, market_context) -> str
  .get_gtm_manager_node(analysis, strategy, market_context) -> str
# providers: gemini | openai | anthropic | llamacpp ; detect_available_providers()

# diagnostics
GTM_DriverEngine(df, current_week_idx=-5)
  .get_diagnostics() -> dict        # shap_info
  .get_market_context() -> dict     # market_context
  .optimize_strategy(budget_limit, horizon=4) -> dict  # opt_results

# data
get_tier_config(tier_name) -> (DataConfig, OEMTierConfig, coeffs)
SyntheticDataSource(data_cfg, oem_cfg, coeffs).load() -> df
# DataConfig is pydantic; seed field = random_state
# re-seed: data_cfg.model_copy(update={"random_state": seed})
```

### Checker quirks that MUST shape the eval
- Pool = flattened **absolute values** of shap_info + market_context + opt_results
  (booleans become 0.0/1.0; NaN/Inf dropped).
- Numbers in `[0, 1, 2, 3, 4, 5]` are **skipped** entirely.
- Match tolerance `1.1` (absolute), AND matches on **×100 / ÷100 scaling**.
- Regex extracts *every* number in prose — years, week numbers, "Week 3", "Q4",
  percentages — so false positives are real and must be measured.

---

## Phase 1 — Experiment B: checker precision/recall (deterministic)

**Location:** `benchmark/eval_checker.py` + fixtures in `benchmark/fixtures/`
+ a thin pytest wrapper in `tests/test_eval_checker.py` so CI runs it.

### 1.1 Build the labelled case set
Generate real pipeline artifacts (one or two seeds, engine + optimizer, no LLM)
to get authentic pools. Then construct three case families of template texts
(realistic boardroom-memo style, numbers substituted programmatically):

- **CLEAN (~15 cases):** every number in the text traces to the pool (directly,
  rounded within tolerance, or via legitimate ×100/÷100 scaling).
  Expected: zero flags.
- **INJECTED (~15 cases):** clean text + one or more inserted numbers that are
  provably unmatchable. Expected: exactly those numbers flagged.
- **TRAP (~15 cases):** cases a naive checker false-positives on —
  `12` when pool holds `0.12`; `11.7` rounded to `12`; negative values
  (pool is abs()); `$1,234` comma/currency formatting; numbers in the
  skip-list used as bullet indices. Expected: zero flags.

### 1.2 ⚠️ The subtlety most likely to be botched
Before labelling any INJECTED value as a true hallucination, the harness must
**programmatically verify it is unmatchable** against that case's actual pool:
- not within 1.1 of any pool value,
- not within 1.1 of any pool value ×100 or ÷100,
- not in the skip list `[0..5]`.
Implement `is_unmatchable(value, pool) -> bool` and assert it for every injected
value at fixture-build time. A "random" fake that accidentally matches via
scaling silently corrupts recall. If a candidate fails, re-draw.

### 1.3 Score and report
Per number: TP (injected & flagged), FN (injected & missed), FP (legit &
flagged), TN (legit & passed). Report **precision, recall, FP-rate**, overall
and broken out by case family. Emit:
- `benchmark/results/eval_checker_report.md` (tables)
- `benchmark/results/eval_checker_raw.csv` (per-case rows)

### 1.4 Acceptance criteria
- Fixtures committed; eval fully reproducible offline with no API keys.
- `uv run pytest` green locally **and in CI** (the pytest wrapper asserts the
  eval runs and produces sane output — it should not hard-assert 100% scores;
  if the checker has genuine FN/FP, that's a *finding*, record it).
- No new CI steps needed beyond existing pytest; any new deps added via
  `uv add --group dev` so `uv.lock` stays consistent.

**Checkpoint: commit + push Phase 1 and confirm CI green before Phase 2.**

---## Phase 2 — Experiment A: fabrication rates with real models (local only)

**Location:** `benchmark/run.py`. **Never runs in CI.**

### 2.1 CLI spec
```
uv run python benchmark/run.py \
  --models gemini,llamacpp \
  --k 10 \
  --seeds 1000:1010 \            # or explicit comma list; identical across models
  --tier <valid tier from simulation_config.yaml> \
  --budget 500000 \              # sane default from notebooks; check
  --nodes manager \              # or analyst,strategist,manager
  --out benchmark/results/ \
  --dry-run                      # mock provider; no API calls; CI-import-safe
```
- Paired design: the same seed list is used for every model.
- `--dry-run` uses a mock provider returning canned text with known numbers, so
  the script is testable and importable without keys.
- Read keys from env (existing provider mechanism); fail fast with a clear
  message if a requested cloud provider's key is missing.
- Rough cost sanity check: print estimated call count (= models × seeds × nodes)
  before starting; require `--yes` or confirmation to proceed past 100 calls.

### 2.2 Per-run procedure (per model × seed)
1. Rebuild `DataConfig` with the seed → `SyntheticDataSource.load()` →
   `GTM_DriverEngine(df)`.
2. `shap_info = engine.get_diagnostics()`; `ctx = engine.get_market_context()`;
   `opt = engine.optimize_strategy(budget)`.
3. Run the agent chain: analyst → strategist → manager (manager consumes the
   other two, so all three run even if only manager is scored).
4. For each scored node output: `validate_response(text, shap_info, ctx, opt)`.
5. Log a JSON record: model, seed, node, full text, numbers extracted count,
   flagged values, is_valid, timestamp, model config (temperature etc.).
   Persist every record to `benchmark/results/runs/` — transcripts are article
   material.
6. Wrap each LLM call with retry (2–3 attempts, backoff) and a per-run
   try/except so one API failure doesn't kill the batch; record failures
   distinctly (excluded from denominators, reported separately).

### 2.3 Aggregate and report
`benchmark/results/experiment_a_report.md` containing:

| Model | Runs (K) | Runs w/ ≥1 flag | 95% CI | Numbers written | Numbers flagged | Per-number flag rate |

- **Wilson score interval** for the run-level rate (implement directly or via
  `statsmodels` if already a dep — prefer no new runtime deps).
- Raw counts alongside every rate.
- Copy 2–3 genuinely flagged transcripts (verbatim, with the flagged values
  highlighted) into `benchmark/results/examples/`.
- Framing note at the top of the report: frontier-model rate = "persists even
  at the frontier under explicit no-invention instructions"; local-model rate =
  "guardrail becomes load-bearing in local/private deployments". No
  leaderboard language.

### 2.4 Acceptance criteria
- `--dry-run` completes end-to-end with zero API calls and produces the full
  report pipeline (mock data).
- A real smoke run (`--k 2`, one cheap model) completes and writes records.
- Script is import-safe in CI (a trivial test may import it; no side effects at
  import time).

---

## Phase 3 — Wiring and hygiene

1. README: add a short "Measured results" section linking to both reports once
   real numbers exist (placeholder acceptable until the full local run).
2. Fix the known README inconsistency: intro prose says a **15%** illustrative
   hallucinated lift; the code example below it uses **47%**. Align them.
3. If any notebook is touched, keep outputs stripped
   (`uv run nbstripout --verify --keep-id notebooks/*.ipynb` must pass).
4. `benchmark/` layout at the end:
```
benchmark/
├── run.py                  # Experiment A driver (CLI above)
├── eval_checker.py         # Experiment B harness
├── fixtures/               # labelled cases (committed)
└── results/
    ├── eval_checker_report.md / eval_checker_raw.csv
    ├── experiment_a_report.md
    ├── runs/               # per-run JSON records
    └── examples/           # flagged transcripts for the article
```
5. Final: `uv run pytest -v` green; CI green on push.

---

## Constraints (hold throughout)
- `uv` for everything; new deps via `uv add` into the correct group.
- Experiment A never executes LLM calls in CI. Experiment B fully covered in CI.
- The source is the ground truth — where this plan and the code disagree, follow
  the code and note the deviation.
- Small, reviewable commits per phase, not one mega-commit.

## Execution order for the human (not Claude Code)
1. Let Claude Code deliver Phase 1 → confirm CI green.
2. Phase 2 in `--dry-run`, then smoke-run `--k 2` with Gemini.
3. Full paired run locally: Gemini + llamacpp (local Mistral-7B), identical
   seeds, `k=25–50` depending on cost/patience.
4. Bring `experiment_a_report.md` + `eval_checker_report.md` + example
   transcripts back to the article draft.