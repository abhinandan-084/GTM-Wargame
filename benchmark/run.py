"""Experiment A: fabrication rates of real LLMs with the guardrail observing.

Runs the full agent chain (Analyst -> Strategist -> Manager) over pipeline
artifacts rebuilt per seed, validates scored node outputs with the
ConsistencyChecker, and reports run-level and per-number flag rates with
Wilson 95% confidence intervals. Real API calls - run locally, never in CI.

This is not a model leaderboard, and a flag is not a fabrication verdict:
flag rates are an upper bound, audited per flag by benchmark/audit_flags.py
(corrected rates live in the report's flag-audit section). The corrected
local-model rate is what shows the guardrail is load-bearing in local/private
deployments.

Honesty caveat: this measures numerical grounding against a known pool -
whether a cited number *exists* in the ground truth - not semantic
correctness (whether the agent used the right number in a valid argument).

Usage:
    uv run python benchmark/run.py --models gemini,llamacpp --k 10 --dry-run
    uv run python benchmark/run.py --models gemini --k 2 --nodes manager
    uv run python benchmark/run.py --models gemini,llamacpp --seeds 1000:1050 \
        --nodes analyst,strategist,manager --yes
"""

import argparse
import json
import math
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

try:
    from benchmark.eval_checker import (
        _to_jsonable,
        extract_considered_numbers,
        is_unmatchable,
        pool_from_data,
    )
except ImportError:  # executed as a script: benchmark/ itself is on sys.path
    from eval_checker import (  # type: ignore
        _to_jsonable,
        extract_considered_numbers,
        is_unmatchable,
        pool_from_data,
    )

BENCHMARK_DIR = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = BENCHMARK_DIR / "results"

ALL_NODES = ("analyst", "strategist", "manager")
CLOUD_MODELS = ("gemini", "openai", "anthropic")
ALL_MODELS = CLOUD_MODELS + ("llamacpp",)
CALL_CONFIRM_THRESHOLD = 100

# Mirrors the defaults in gtm_boardroom.agents.providers, for the run records.
MODEL_TEMPERATURES = {"gemini": 0.1, "openai": 0.1, "anthropic": 0.1, "llamacpp": 0.4}


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple:
    """Wilson score 95% CI for a binomial proportion. Sane at small N."""
    if n == 0:
        return (0.0, 1.0)
    phat = successes / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    half = z * math.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def parse_seeds(spec: Optional[str], k: int) -> List[int]:
    """'1000:1010' -> [1000..1009]; '7,9,13' -> [7, 9, 13]; None -> [1000..1000+k-1]."""
    if spec is None:
        return list(range(1000, 1000 + k))
    if ":" in spec:
        start, end = spec.split(":", 1)
        seeds = list(range(int(start), int(end)))
    else:
        seeds = [int(s) for s in spec.split(",") if s.strip()]
    return seeds[:k] if k and k < len(seeds) else seeds


# ---------------------------------------------------------------------------
# Pipeline artifacts (shared across models: paired design)
# ---------------------------------------------------------------------------

def build_artifacts(seed: int, tier: str, budget: float, horizon: int) -> Dict:
    from gtm_boardroom.data.config import get_tier_config
    from gtm_boardroom.data.source import SyntheticDataSource
    from gtm_boardroom.diagnostics.driver_engine import GTM_DriverEngine

    data_cfg, oem_cfg, coeffs = get_tier_config(tier)
    data_cfg = data_cfg.model_copy(update={"random_state": seed})
    df = SyntheticDataSource(data_cfg, oem_cfg, coeffs).load()
    engine = GTM_DriverEngine(df, current_week_idx=-5)

    artifacts = _to_jsonable(
        {
            "shap_info": engine.get_diagnostics(),
            "market_context": engine.get_market_context(),
            "opt_results": engine.optimize_strategy(budget, horizon=horizon),
        }
    )
    artifacts["pool"] = pool_from_data(artifacts)
    return artifacts


# ---------------------------------------------------------------------------
# Dry-run mock provider (zero API calls)
# ---------------------------------------------------------------------------

class MockProvider:
    """Duck-types LLMProvider.invoke with canned memo text built from real pool numbers.

    On fabricating runs it also injects one number verified unmatchable against
    the run's actual pool, so the whole flag -> report -> examples pipeline is
    exercised deterministically without any API call.
    """

    def __init__(self, artifacts: Dict, fabricate: bool):
        self.name = "mock"
        opt = artifacts["opt_results"]
        self._citations = [
            f"{opt['optimized_price']:.2f}",
            f"{abs(opt['lift_percent']):.2f}%",
            f"{opt['forecasted_total_sales']:,.0f}",
        ]
        self._fabricated = None
        if fabricate:
            candidate = 987654.32
            while not is_unmatchable(round(candidate, 2), artifacts["pool"]):
                candidate += 1313.13
            self._fabricated = f"{round(candidate, 2):,}"

    def invoke(self, template: str, variables: Dict) -> str:
        lines = [
            "### Executive Summary",
            f"- **Optimized Price**: reposition at {self._citations[0]} for the coming cycle.",
            f"- **Projected Lift**: {self._citations[1]} against the trailing baseline.",
            f"- **Forecasted Volume**: {self._citations[2]} units over the horizon.",
        ]
        if self._fabricated is not None:
            lines.append(
                f"- **Incremental Upside**: an additional {self._fabricated} units from channel synergies."
            )
        return "\n".join(lines)


def make_brain(model: str, dry_run: bool, artifacts: Dict, seed: int):
    from gtm_boardroom.agents.gtm_agents import GTMBrain

    if dry_run:
        # Bypass create_provider without touching src: GTMBrain.__init__ only
        # sets self.provider, which MockProvider duck-types.
        brain = GTMBrain.__new__(GTMBrain)
        brain.provider = MockProvider(artifacts, fabricate=(seed % 2 == 1))
        return brain
    return GTMBrain(provider=model)


# ---------------------------------------------------------------------------
# Per-run procedure
# ---------------------------------------------------------------------------

def _call_with_retry(fn, label: str, retries: int = 3, backoff: float = 2.0):
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except Exception:
            if attempt == retries:
                raise
            print(f"    {label}: attempt {attempt} failed, retrying in {backoff * attempt:.0f}s...")
            time.sleep(backoff * attempt)


def run_one(model: str, seed: int, artifacts: Dict, nodes: List[str],
            dry_run: bool, run_config: Dict) -> List[Dict]:
    """One model x seed run: full chain, validate scored nodes, return records."""
    from gtm_boardroom.guardrails.consistency_checker import ConsistencyChecker

    shap_info = artifacts["shap_info"]
    ctx = artifacts["market_context"]
    opt = artifacts["opt_results"]

    brain = make_brain(model, dry_run, artifacts, seed)
    outputs = {}
    outputs["analyst"] = _call_with_retry(
        lambda: brain.get_analyst_node(shap_info, ctx), f"{model}/seed {seed}/analyst")
    outputs["strategist"] = _call_with_retry(
        lambda: brain.get_strategist_node(opt, outputs["analyst"], ctx), f"{model}/seed {seed}/strategist")
    outputs["manager"] = _call_with_retry(
        lambda: brain.get_gtm_manager_node(outputs["analyst"], outputs["strategist"], ctx),
        f"{model}/seed {seed}/manager")

    records = []
    for node in nodes:
        text = outputs[node]
        result = ConsistencyChecker.validate_response(text, shap_info, ctx, opt)
        records.append(
            {
                "model": model,
                "seed": seed,
                "node": node,
                "text": text,
                "numbers_extracted": len(extract_considered_numbers(text)),
                "flagged_values": result["hallucinated_values"],
                "is_valid": result["is_valid"],
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "config": {**run_config, "model": model,
                           "temperature": MODEL_TEMPERATURES.get(model)},
            }
        )
    return records


# ---------------------------------------------------------------------------
# Aggregation, report, examples
# ---------------------------------------------------------------------------

def aggregate(records: List[Dict], failures: List[Dict], models: List[str]) -> List[Dict]:
    rows = []
    for model in models:
        recs = [r for r in records if r["model"] == model]
        seeds_ok = sorted({r["seed"] for r in recs})
        flagged_seeds = sorted({r["seed"] for r in recs if r["flagged_values"]})
        numbers_written = sum(r["numbers_extracted"] for r in recs)
        numbers_flagged = sum(len(r["flagged_values"]) for r in recs)
        lo, hi = wilson_interval(len(flagged_seeds), len(seeds_ok))
        rows.append(
            {
                "model": model,
                "runs": len(seeds_ok),
                "runs_flagged": len(flagged_seeds),
                "ci_low": lo,
                "ci_high": hi,
                "numbers_written": numbers_written,
                "numbers_flagged": numbers_flagged,
                "failures": sum(1 for f in failures if f["model"] == model),
            }
        )
    return rows


def _highlight_flagged(text: str, flagged: List[float]) -> str:
    """Best-effort inline highlighting of flagged number occurrences."""
    for v in flagged:
        forms = {f"{v:g}", f"{v:.2f}", f"{v:.1f}", f"{v:,.2f}", f"{v:,.1f}"}
        if float(v).is_integer():
            forms.add(f"{int(v):,}")
            forms.add(str(int(v)))
        for form in sorted(forms, key=len, reverse=True):
            pattern = re.escape(form)
            # (?!...|,\d): never annotate the head of a comma-grouped number
            # ("25" inside "25,106.14" is not the flagged 25).
            text = re.sub(rf"(?<![\d.\w]){pattern}(?![\d.]|,\d)", f"**[FLAGGED: {form}]**", text)
    return text


def write_examples(records: List[Dict], examples_dir: Path, limit: int = 3) -> List[Path]:
    flagged_records = [r for r in records if r["flagged_values"]]
    # Prefer one example per model before taking seconds from the same model.
    flagged_records.sort(key=lambda r: (r["model"], r["seed"]))
    picked, seen_models = [], set()
    for r in flagged_records:
        if r["model"] not in seen_models:
            picked.append(r)
            seen_models.add(r["model"])
    for r in flagged_records:
        if len(picked) >= limit:
            break
        if r not in picked:
            picked.append(r)
    picked = picked[:limit]

    paths = []
    examples_dir.mkdir(parents=True, exist_ok=True)
    for r in picked:
        path = examples_dir / f"{r['model']}_seed{r['seed']}_{r['node']}.md"
        body = [
            f"# Flagged transcript: {r['model']}, seed {r['seed']}, {r['node']} node",
            "",
            f"Flagged values: {r['flagged_values']}",
            f"Timestamp: {r['timestamp']}",
            "",
            "---",
            "",
            _highlight_flagged(r["text"], r["flagged_values"]),
            "",
        ]
        path.write_text("\n".join(body))
        paths.append(path)
    return paths


def write_report(out_dir: Path, rows: List[Dict], failures: List[Dict],
                 run_config: Dict, example_paths: List[Path]) -> Path:
    lines = [
        "# LLM fabrication rates with the guardrail observing (Experiment A)",
        "",
        "**Framing:** this is not a model leaderboard, and a flag is not yet a",
        "fabrication: flag rates are an *upper bound* - a flag means a number failed",
        "pool matching, not necessarily that the model invented it. Every flag is",
        "audited and classified before it becomes a claim (the flag-audit section,",
        "maintained by `benchmark/audit_flags.py --write-report`); the fabrication",
        "rates there are what show the guardrail is load-bearing in local/private",
        "deployments.",
        "",
        "**Honesty caveat:** this measures numerical grounding against a known pool -",
        "whether a cited number *exists* in the ground truth - not semantic correctness.",
        "",
        f"**Design:** paired seeds (identical across models), tier `{run_config['tier']}`,",
        f"budget {run_config['budget']:,.0f}, horizon {run_config['horizon']},",
        f"scored nodes: {', '.join(run_config['nodes'])}."
        + (" **DRY RUN - mock provider, no API calls.**" if run_config["dry_run"] else ""),
        "",
        "| Model | Runs (K) | Runs w/ >=1 flag | 95% CI (run-level) | Numbers written | Numbers flagged | Per-number flag rate |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        rate = f"{r['runs_flagged']}/{r['runs']}"
        per_number = (
            f"{r['numbers_flagged']}/{r['numbers_written']}"
            f" ({r['numbers_flagged'] / r['numbers_written']:.1%})"
            if r["numbers_written"] else "n/a"
        )
        lines.append(
            f"| {r['model']} | {r['runs']} | {rate} | [{r['ci_low']:.1%}, {r['ci_high']:.1%}] "
            f"| {r['numbers_written']} | {r['numbers_flagged']} | {per_number} |"
        )

    lines += [
        "",
        "Raw counts are authoritative; rates at small K are noisy - read the CI.",
    ]

    if failures:
        lines += ["", "## Failures (excluded from denominators)", ""]
        for f in failures:
            lines.append(f"- {f['model']}, seed {f['seed']}: {f['error']}")

    lines += ["", "## Example flagged transcripts", ""]
    if example_paths:
        for p in example_paths:
            lines.append(f"- [`{p.name}`](examples/{p.name})")
    else:
        lines.append("None - no run produced a flag.")
    lines.append("")

    report_path = out_dir / "experiment_a_report.md"
    report_path.write_text("\n".join(lines))
    return report_path


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def check_prerequisites(models: List[str], dry_run: bool) -> None:
    """Fail fast, before any expensive work, if a requested model can't run."""
    if dry_run:
        return
    from gtm_boardroom.agents.providers import (
        DEFAULT_LLAMACPP_MODEL_PATH,
        LLAMACPP_MODEL_PATH_ENV_VAR,
        PROVIDER_ENV_VARS,
    )

    for model in models:
        if model in PROVIDER_ENV_VARS:
            env_var = PROVIDER_ENV_VARS[model]
            if not os.environ.get(env_var):
                sys.exit(f"Missing API key: set {env_var} (e.g. in .env) to run '{model}'.")
        elif model == "llamacpp":
            path = Path(os.path.expanduser(
                os.environ.get(LLAMACPP_MODEL_PATH_ENV_VAR, DEFAULT_LLAMACPP_MODEL_PATH)))
            if not path.is_file():
                sys.exit(
                    f"llama.cpp model file not found: {path}\n"
                    f"Set {LLAMACPP_MODEL_PATH_ENV_VAR} to a .gguf file to run 'llamacpp'."
                )


def run_experiment(models: List[str], seeds: List[int], tier: str, budget: float,
                   horizon: int, nodes: List[str], out_dir: Path,
                   dry_run: bool = False) -> Dict:
    run_config = {
        "tier": tier, "budget": budget, "horizon": horizon,
        "nodes": nodes, "dry_run": dry_run,
    }
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    records, failures = [], []
    for seed in seeds:
        print(f"Building pipeline artifacts for seed {seed}...")
        artifacts = build_artifacts(seed, tier, budget, horizon)
        for model in models:
            print(f"  Running {model} on seed {seed}...")
            try:
                seed_records = run_one(model, seed, artifacts, nodes, dry_run, run_config)
            except Exception as exc:  # one API failure must not kill the batch
                failure = {
                    "model": model, "seed": seed, "error": f"{type(exc).__name__}: {exc}",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                failures.append(failure)
                (runs_dir / f"{model}_seed{seed}_FAILURE.json").write_text(
                    json.dumps(failure, indent=2))
                print(f"    FAILED: {failure['error']}")
                continue
            for record in seed_records:
                path = runs_dir / f"{record['model']}_seed{record['seed']}_{record['node']}.json"
                path.write_text(json.dumps(record, indent=2))
            records.extend(seed_records)
            n_flags = sum(len(r["flagged_values"]) for r in seed_records)
            print(f"    done ({'no flags' if n_flags == 0 else f'{n_flags} flagged number(s)'})")

    rows = aggregate(records, failures, models)
    example_paths = write_examples(records, out_dir / "examples")
    report_path = write_report(out_dir, rows, failures, run_config, example_paths)
    print(f"\nReport: {report_path}")
    return {"rows": rows, "records": records, "failures": failures, "report": report_path}


def main() -> None:
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", default="gemini",
                        help=f"Comma list from {', '.join(ALL_MODELS)}")
    parser.add_argument("--k", type=int, default=10, help="Runs per model")
    parser.add_argument("--seeds", default=None,
                        help="'1000:1010' range or '7,9,13' list; identical across models")
    parser.add_argument("--tier", default="upstart", help="OEM tier from simulation_config.yaml")
    parser.add_argument("--budget", type=float, default=500_000.0)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--nodes", default="manager",
                        help="Comma list of nodes to score (the full chain always runs)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--dry-run", action="store_true",
                        help="Mock provider; zero API calls; full report pipeline")
    parser.add_argument("--yes", action="store_true",
                        help=f"Skip confirmation above {CALL_CONFIRM_THRESHOLD} API calls")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in models if m not in ALL_MODELS]
    if unknown:
        sys.exit(f"Unknown model(s) {unknown}; choose from {list(ALL_MODELS)}")
    nodes = [n.strip() for n in args.nodes.split(",") if n.strip()]
    unknown = [n for n in nodes if n not in ALL_NODES]
    if unknown:
        sys.exit(f"Unknown node(s) {unknown}; choose from {list(ALL_NODES)}")

    seeds = parse_seeds(args.seeds, args.k)
    check_prerequisites(models, args.dry_run)

    # The chain always runs all three nodes, whatever subset is scored.
    n_calls = len(models) * len(seeds) * len(ALL_NODES)
    print(f"Estimated LLM calls: {n_calls} ({len(models)} model(s) x {len(seeds)} seed(s) "
          f"x {len(ALL_NODES)} chain nodes){' - dry run, no API calls' if args.dry_run else ''}")
    if not args.dry_run and n_calls > CALL_CONFIRM_THRESHOLD and not args.yes:
        if not sys.stdin.isatty():
            sys.exit(f"{n_calls} calls exceeds {CALL_CONFIRM_THRESHOLD}; pass --yes to proceed.")
        if input(f"Proceed with {n_calls} API calls? [y/N] ").strip().lower() != "y":
            sys.exit("Aborted.")

    run_experiment(models, seeds, args.tier, args.budget, args.horizon,
                   nodes, args.out, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
