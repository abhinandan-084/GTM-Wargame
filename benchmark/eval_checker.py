"""Experiment B: measure ConsistencyChecker's precision/recall as a classifier.

Runs the checker against a committed set of labelled cases (CLEAN / INJECTED /
TRAP) built on top of authentic pipeline artifacts (real SHAP values, market
context, and optimizer results), and reports precision, recall, and
false-positive rate per number.

Honesty caveat: this measures numerical grounding against a known pool -
whether a number the text cites *exists* in the ground truth - not semantic
correctness (whether the agent used the right number in a valid argument).

Usage:
    uv run python benchmark/eval_checker.py                     # eval committed fixtures -> benchmark/results/
    uv run python benchmark/eval_checker.py --rebuild-fixtures  # regenerate fixtures (runs the real pipeline, no LLM)
"""

import argparse
import csv
import json
import random
import re
from pathlib import Path
from typing import Any

from gtm_boardroom.guardrails.consistency_checker import ConsistencyChecker

BENCHMARK_DIR = Path(__file__).resolve().parent
FIXTURES_DIR = BENCHMARK_DIR / "fixtures"
RESULTS_DIR = BENCHMARK_DIR / "results"

# These mirror ConsistencyChecker.validate_response and must stay in sync with it.
SKIP_VALUES = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
TOLERANCE = 1.1

SEEDS = (42, 1337)
TIER = "upstart"
BUDGET = 500_000.0
HORIZON = 4
CASES_PER_FAMILY = 15
FAMILIES = ("CLEAN", "INJECTED", "TRAP")
MASTER_RNG_SEED = 20260705


def pool_from_data(pool_data: dict) -> list[float]:
    """Ground truth pool exactly as the checker builds it: flattened absolute values."""
    raw = []
    for source in (pool_data["shap_info"], pool_data["market_context"], pool_data["opt_results"]):
        raw.extend(ConsistencyChecker._flatten_data(source))
    return [abs(v) for v in raw]


def matches_pool(value: float, pool: list[float]) -> bool:
    """True if the checker's matching rule (direct, x100, /100, tolerance 1.1) accepts value."""
    for gt in pool:
        if abs(value - gt) < TOLERANCE:
            return True
        if abs(value - gt * 100) < TOLERANCE:
            return True
        if abs(value - gt / 100) < TOLERANCE:
            return True
    return False


def matches_pool_k(value: float, pool: list[float], decimals: int = 0) -> bool:
    """The checker's context-gated k-notation rule: a value written with a k
    suffix matches a pool value within one unit of its last written digit
    ("176k" -> 176,000 +/- 1,000; "176.4k" -> +/- 100)."""
    window = 1000.0 * 10.0 ** (-decimals)
    return any(abs(value * 1000 - gt) < window for gt in pool)


def _k_decimals(token: str) -> int:
    digits = token.rstrip("kK")
    return len(digits.split(".")[1]) if "." in digits else 0


def is_matchable(value: float, pool: list[float]) -> bool:
    return value not in SKIP_VALUES and matches_pool(value, pool)


def is_unmatchable(value: float, pool: list[float]) -> bool:
    """True only if the checker could never legitimately match this value.

    An injected value must satisfy this at fixture-build time, otherwise a
    'fake' that accidentally matches via scaling silently corrupts recall.
    """
    return value not in SKIP_VALUES and not matches_pool(value, pool)


def extract_considered_numbers(text: str) -> list[float]:
    """Mirror of the checker's extraction (cleaning, regex, abs, skip list); used
    for true-negative accounting, since the checker only returns flags."""
    clean_text = text.replace(",", "").replace("$", "").replace("%", "")
    tokens = re.findall(r"[-+]?\d*\.\d+|\d+", clean_text)
    nums = [abs(float(t)) for t in tokens]
    return [n for n in nums if n not in SKIP_VALUES]


def _to_jsonable(data: Any) -> Any:
    import numpy as np

    if isinstance(data, dict):
        return {k: _to_jsonable(v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return [_to_jsonable(v) for v in data]
    if isinstance(data, np.bool_):
        return bool(data)
    if isinstance(data, np.integer):
        return int(data)
    if isinstance(data, (np.floating, np.number)):
        return float(data)
    return data


def _generate_pool_data(seed: int) -> dict:
    """Run the real pipeline (no LLM) for one seed and return jsonable artifacts."""
    from gtm_boardroom.data.config import get_tier_config
    from gtm_boardroom.data.source import SyntheticDataSource
    from gtm_boardroom.diagnostics.driver_engine import GTM_DriverEngine

    data_cfg, oem_cfg, coeffs = get_tier_config(TIER)
    data_cfg = data_cfg.model_copy(update={"random_state": seed})
    df = SyntheticDataSource(data_cfg, oem_cfg, coeffs).load()
    engine = GTM_DriverEngine(df, current_week_idx=-5)

    return _to_jsonable(
        {
            "shap_info": engine.get_diagnostics(),
            "market_context": engine.get_market_context(),
            "opt_results": engine.optimize_strategy(BUDGET, horizon=HORIZON),
        }
    )


_WEEK_WORDS = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth"]


def _numeric_entries(pool_data: dict) -> list[tuple[str, float]]:
    """(metric name, abs value) pairs with provenance, for realistic memo sentences.

    Metric names must not contain digits above the skip list, or they would leak
    unlabelled numbers into the case text (verified by the mirror assertion).
    """
    entries = []
    for feat, v in pool_data["shap_info"].items():
        entries.append((f"{feat} SHAP contribution", abs(float(v))))
    for key, v in pool_data["opt_results"].items():
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            entries.append((key, abs(float(v))))
        elif isinstance(v, list):
            for i, item in enumerate(v):
                entries.append((f"{key} ({_WEEK_WORDS[i]} week)", abs(float(item))))
    return entries


_TEMPLATES = [
    "- **{metric}**: the model attributes {tok} to this driver in the current cycle.",
    "- **{metric}**: holding at {tok} per the optimizer run.",
    "- **{metric}**: {tok}, which the boardroom should treat as the operative figure.",
    "The proposal puts {metric} at {tok}, in line with the diagnostic readout.",
    "Against the trailing baseline, {metric} lands at {tok}.",
    "Diagnostics place {metric} at {tok} for the review window.",
]


def _parse_token(token: str) -> float:
    cleaned = token.replace(",", "").replace("$", "").replace("%", "")
    if cleaned and cleaned[-1] in "kK":
        cleaned = cleaned[:-1]  # the checker's regex extracts the bare digits
    return abs(float(cleaned))


def _eligible_forms(g: float) -> list[str]:
    forms = []
    if g >= 0.01 and round(g, 2) not in SKIP_VALUES:
        forms.append("two_dp")
    if g >= 5.6:
        forms.append("int")
    if 0.056 <= g <= 0.999:
        forms.append("pct")
    if g >= 1000:
        forms.append("currency")
    return forms


def _render(g: float, form: str) -> str:
    if form == "two_dp":
        return f"{g:.2f}"
    if form == "int":
        return f"{round(g)}"
    if form == "pct":
        return f"{round(g * 100, 1):g}%"
    if form == "pct_bare":
        return f"{round(g * 100)}"
    if form == "frac":
        return f"{g / 100:.2f}"
    if form == "currency":
        return f"${round(g):,}"
    if form == "negative":
        return f"-{g:.2f}"
    if form == "k_int":
        return f"{round(g / 1000)}k"
    if form == "k_1dp":
        return f"{g / 1000:.1f}k"
    raise ValueError(f"Unknown form: {form}")


def _pick_clean_value(rng: random.Random, entries, pool, taken) -> tuple[str, str, float]:
    """Pick a pool entry and render it in a legitimately matchable form."""
    for _ in range(2000):
        metric, g = rng.choice(entries)
        forms = _eligible_forms(g)
        if not forms:
            continue
        token = _render(g, rng.choice(forms))
        value = _parse_token(token)
        if value in taken or value in SKIP_VALUES:
            continue
        if not is_matchable(value, pool):
            continue
        return metric, token, value
    raise RuntimeError("Could not pick a clean value; pool too degenerate")


def _draw_injected(rng: random.Random, base: float, pool, taken) -> tuple[str, float]:
    """Draw a provably-unmatchable value of realistic magnitude for the metric."""
    for _ in range(2000):
        if base < 1:
            cand = round(rng.uniform(6, 99), 1)
            token = f"{cand:g}%"
        elif base < 1000:
            cand = round(rng.uniform(6, 999), 2)
            token = f"{cand:.2f}"
        else:
            cand = float(int(rng.uniform(1500, 95000)))
            token = f"{int(cand):,}"
        value = _parse_token(token)
        if value in taken:
            continue
        if is_unmatchable(value, pool):
            return token, value
    raise RuntimeError("Could not draw an unmatchable injected value")


def _draw_injected_k(rng: random.Random, pool, taken) -> tuple[str, float]:
    """Draw a fabricated k-suffix value the checker must still flag.

    Must be unmatchable under the plain rules AND outside every pool value's
    k window - a fake that lands in either would rightly be accepted, which
    corrupts recall accounting."""
    for _ in range(2000):
        cand = float(rng.randint(6, 999))
        token = f"{int(cand)}k"
        if cand in taken:
            continue
        if is_unmatchable(cand, pool) and not matches_pool_k(cand, pool, 0):
            return token, cand
    raise RuntimeError("Could not draw an unmatchable injected k value")


_TRAP_SPECS = [
    # (name, predicate on pool value g, render form)
    ("pct_of_fraction", lambda g: 0.06 <= g <= 0.99, "pct_bare"),   # "12" when pool holds 0.12
    ("frac_of_large", lambda g: g >= 600, "frac"),                  # "7.99" when pool holds 799
    ("rounded", lambda g: 5.6 <= g <= 500 and abs(round(g) - g) >= 0.2, "int"),
    ("negative", lambda g: 0.01 <= g <= 999 and round(g, 2) not in SKIP_VALUES, "negative"),
    ("currency_comma", lambda g: g >= 1000, "currency"),
]

_SKIP_BULLET_TEXT = (
    "### Execution Plan\n"
    "1. Freeze the list price through the review window.\n"
    "2. Hold weekly search spend at the optimizer's allocation.\n"
    "3. Rebalance social toward the launch corridor.\n"
    "4. Revisit the playbook in Q4 if the price war escalates.\n"
)


def _build_case_text(lines: list[str]) -> str:
    return "### Boardroom Memo\n" + "\n".join(lines) + "\n"


def _assert_case_labels(case: dict, pool: list[float]) -> None:
    """Build-time verification that labels are correct and the text leaks no numbers."""
    for v in case["values"]:
        if v["label"] == "legit":
            assert is_matchable(v["value"], pool), f"legit value {v['value']} not matchable ({case['case_id']})"
        elif v["label"] == "legit_k":
            assert matches_pool_k(v["value"], pool, _k_decimals(v["token"])), (
                f"legit_k value {v['token']} not k-matchable ({case['case_id']})"
            )
        elif v["label"] == "injected":
            assert is_unmatchable(v["value"], pool), f"injected value {v['value']} is matchable ({case['case_id']})"
            if v["token"].rstrip("kK") != v["token"]:
                assert not matches_pool_k(v["value"], pool, _k_decimals(v["token"])), (
                    f"injected k value {v['token']} is k-matchable ({case['case_id']})"
                )
        elif v["label"] == "skipped":
            assert v["value"] in SKIP_VALUES, f"skipped value {v['value']} not in skip list ({case['case_id']})"
        else:
            raise AssertionError(f"unknown label {v['label']} in {case['case_id']}")
    considered = sorted(extract_considered_numbers(case["text"]))
    labelled = sorted(v["value"] for v in case["values"] if v["label"] != "skipped")
    assert considered == labelled, (
        f"text/label mismatch in {case['case_id']}: extracted {considered}, labelled {labelled}"
    )


def build_fixtures(fixtures_dir: Path = FIXTURES_DIR) -> None:
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(MASTER_RNG_SEED)

    pools_data = {}
    for seed in SEEDS:
        print(f"Generating pipeline artifacts for seed {seed} (tier={TIER}, budget={BUDGET:,.0f})...")
        pools_data[str(seed)] = _generate_pool_data(seed)
    pools = {s: pool_from_data(d) for s, d in pools_data.items()}

    cases = []

    def _suffix(idx: int) -> str:
        return chr(ord("a") + idx) if idx < 26 else "a" + chr(ord("a") + idx - 26)

    def add_case(family: str, idx: int, seed: int, lines: list[str], values: list[dict]) -> None:
        case = {
            "case_id": f"{family.lower()}-{_suffix(idx)}",
            "family": family,
            "seed": seed,
            "text": _build_case_text(lines) if lines else _SKIP_BULLET_TEXT,
            "values": values,
        }
        _assert_case_labels(case, pools[str(seed)])
        cases.append(case)

    for i in range(CASES_PER_FAMILY):
        seed = SEEDS[i % len(SEEDS)]
        pool = pools[str(seed)]
        entries = _numeric_entries(pools_data[str(seed)])

        # CLEAN: every number traces to the pool.
        taken, lines, values = set(), [], []
        for j in range(rng.randint(4, 5)):
            metric, token, value = _pick_clean_value(rng, entries, pool, taken)
            taken.add(value)
            lines.append(_TEMPLATES[(i + j) % len(_TEMPLATES)].format(metric=metric, tok=token))
            values.append({"token": token, "value": value, "label": "legit", "metric": metric})
        add_case("CLEAN", i, seed, lines, values)

        # INJECTED: clean base + provably unmatchable insertions.
        taken, lines, values = set(), [], []
        for j in range(3):
            metric, token, value = _pick_clean_value(rng, entries, pool, taken)
            taken.add(value)
            lines.append(_TEMPLATES[(i + j + 1) % len(_TEMPLATES)].format(metric=metric, tok=token))
            values.append({"token": token, "value": value, "label": "legit", "metric": metric})
        for j in range(rng.randint(1, 2)):
            metric, base = rng.choice(entries)
            token, value = _draw_injected(rng, base, pool, taken)
            taken.add(value)
            lines.append(_TEMPLATES[(i + j + 4) % len(_TEMPLATES)].format(metric=metric, tok=token))
            values.append({"token": token, "value": value, "label": "injected", "metric": metric})
        rng.shuffle(lines)
        add_case("INJECTED", i, seed, lines, values)

    # TRAP: false-positive bait a naive checker would flag. Cycle the specs,
    # plus one pure skip-list-bullets case per seed.
    trap_idx = 0
    for seed in SEEDS:
        add_case("TRAP", trap_idx, seed, [], [
            {"token": str(n), "value": float(n), "label": "skipped", "metric": "bullet index"}
            for n in (1, 2, 3, 4)
        ])
        trap_idx += 1
    spec_cycle = 0
    while trap_idx < CASES_PER_FAMILY:
        seed = SEEDS[trap_idx % len(SEEDS)]
        pool = pools[str(seed)]
        entries = _numeric_entries(pools_data[str(seed)])
        name, predicate, form = _TRAP_SPECS[spec_cycle % len(_TRAP_SPECS)]
        spec_cycle += 1

        taken, lines, values = set(), [], []
        candidates = [(m, g) for m, g in entries if predicate(g)]
        rng.shuffle(candidates)
        for metric, g in candidates:
            token = _render(g, form)
            value = _parse_token(token)
            if value in taken or value in SKIP_VALUES or not is_matchable(value, pool):
                continue
            taken.add(value)
            lines.append(_TEMPLATES[trap_idx % len(_TEMPLATES)].format(metric=metric, tok=token))
            values.append({"token": token, "value": value, "label": "legit", "metric": metric})
            break
        if not lines:
            continue  # no candidate for this spec in this pool; try the next spec
        metric, token, value = _pick_clean_value(rng, entries, pool, taken)
        lines.append(_TEMPLATES[(trap_idx + 3) % len(_TEMPLATES)].format(metric=metric, tok=token))
        values.append({"token": token, "value": value, "label": "legit", "metric": metric})
        add_case("TRAP", trap_idx, seed, lines, values)
        trap_idx += 1

    # k-notation cases, extending the same families: TRAP cases with correct
    # k-citations of thousand-scale pool values (a checker without the
    # context-gated x1000 rule flags every one), and INJECTED cases with
    # fabricated k-suffix values that must still flag under it.
    # (Built after the cases above so their RNG draws are unchanged.)
    k_forms = ("k_int", "k_1dp")

    def _pick_k_citation(rng, entries, pool, taken, form) -> tuple[str, str, float] | None:
        candidates = [(m, g) for m, g in entries if g >= 6000]
        rng.shuffle(candidates)
        for metric, g in candidates:
            token = _render(g, form)
            value = _parse_token(token)
            if value in taken or value in SKIP_VALUES:
                continue
            if is_matchable(value, pool):
                continue  # plain-matchable by accident: not a genuine k trap
            if not matches_pool_k(value, pool, _k_decimals(token)):
                continue
            return metric, token, value
        return None

    for i in range(CASES_PER_FAMILY):
        seed = SEEDS[i % len(SEEDS)]
        pool = pools[str(seed)]
        entries = _numeric_entries(pools_data[str(seed)])

        taken, lines, values = set(), [], []
        for j in range(2):
            picked = _pick_k_citation(rng, entries, pool, taken, k_forms[(i + j) % 2])
            if picked is None:
                continue
            metric, token, value = picked
            taken.add(value)
            lines.append(_TEMPLATES[(i + j) % len(_TEMPLATES)].format(metric=metric, tok=token))
            values.append({"token": token, "value": value, "label": "legit_k", "metric": metric})
        assert lines, f"no k-citable pool value for seed {seed}"
        metric, token, value = _pick_clean_value(rng, entries, pool, taken)
        lines.append(_TEMPLATES[(i + 2) % len(_TEMPLATES)].format(metric=metric, tok=token))
        values.append({"token": token, "value": value, "label": "legit", "metric": metric})
        add_case("TRAP", CASES_PER_FAMILY + i, seed, lines, values)

    for i in range(CASES_PER_FAMILY):
        seed = SEEDS[i % len(SEEDS)]
        pool = pools[str(seed)]
        entries = _numeric_entries(pools_data[str(seed)])

        taken, lines, values = set(), [], []
        picked = _pick_k_citation(rng, entries, pool, taken, k_forms[i % 2])
        if picked is not None:  # legit k citation alongside the fakes
            metric, token, value = picked
            taken.add(value)
            lines.append(_TEMPLATES[i % len(_TEMPLATES)].format(metric=metric, tok=token))
            values.append({"token": token, "value": value, "label": "legit_k", "metric": metric})
        metric, token, value = _pick_clean_value(rng, entries, pool, taken)
        taken.add(value)
        lines.append(_TEMPLATES[(i + 1) % len(_TEMPLATES)].format(metric=metric, tok=token))
        values.append({"token": token, "value": value, "label": "legit", "metric": metric})
        for j in range(rng.randint(1, 2)):
            metric, _ = rng.choice(entries)
            token, value = _draw_injected_k(rng, pool, taken)
            taken.add(value)
            lines.append(_TEMPLATES[(i + j + 3) % len(_TEMPLATES)].format(metric=metric, tok=token))
            values.append({"token": token, "value": value, "label": "injected", "metric": metric})
        rng.shuffle(lines)
        add_case("INJECTED", CASES_PER_FAMILY + i, seed, lines, values)

    counts = {f: sum(1 for c in cases if c["family"] == f) for f in FAMILIES}
    print(f"Built {len(cases)} cases: {counts}")

    (fixtures_dir / "pools.json").write_text(json.dumps(pools_data, indent=2))
    (fixtures_dir / "cases.json").write_text(json.dumps(cases, indent=2))
    print(f"Fixtures written to {fixtures_dir}")


def load_fixtures(fixtures_dir: Path = FIXTURES_DIR) -> tuple[dict, list[dict]]:
    pools_data = json.loads((fixtures_dir / "pools.json").read_text())
    cases = json.loads((fixtures_dir / "cases.json").read_text())
    return pools_data, cases


def score_case(case: dict, pool_data: dict) -> dict:
    result = ConsistencyChecker.validate_response(
        case["text"],
        pool_data["shap_info"],
        pool_data["market_context"],
        pool_data["opt_results"],
    )
    flagged = result["hallucinated_values"]
    injected = {v["value"] for v in case["values"] if v["label"] == "injected"}

    tp = sum(1 for f in flagged if f in injected)
    fp = len(flagged) - tp
    fn = len(injected) - tp
    n_considered = len(extract_considered_numbers(case["text"]))
    tn = (n_considered - len(injected)) - fp

    expected_flags = sorted(injected)
    verdict_match = sorted(set(flagged)) == expected_flags and fp == 0 and fn == 0
    return {
        "case_id": case["case_id"],
        "family": case["family"],
        "seed": case["seed"],
        "n_considered": n_considered,
        "n_injected": len(injected),
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "checker_is_valid": result["is_valid"],
        "flagged_values": flagged,
        "injected_values": expected_flags,
        "verdict_match": verdict_match,
    }


def _rates(tp: int, fn: int, fp: int, tn: int) -> dict[str, float | None]:
    return {
        "precision": tp / (tp + fp) if (tp + fp) else None,
        "recall": tp / (tp + fn) if (tp + fn) else None,
        "fp_rate": fp / (fp + tn) if (fp + tn) else None,
    }


def _aggregate(rows: list[dict]) -> dict:
    agg = {k: sum(r[k] for r in rows) for k in ("n_considered", "n_injected", "tp", "fn", "fp", "tn")}
    agg["cases"] = len(rows)
    agg["cases_matching_expectation"] = sum(1 for r in rows if r["verdict_match"])
    agg.update(_rates(agg["tp"], agg["fn"], agg["fp"], agg["tn"]))
    return agg


def _fmt(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def _write_report(out_dir: Path, overall: dict, by_family: dict[str, dict], rows: list[dict]) -> None:
    lines = [
        "# ConsistencyChecker precision/recall (Experiment B)",
        "",
        "Deterministic eval of the guardrail as a per-number classifier, over labelled",
        "cases built on authentic pipeline artifacts (real SHAP/context/optimizer pools,",
        f"seeds {list(SEEDS)}, tier `{TIER}`, budget {BUDGET:,.0f}, horizon {HORIZON}). No LLM involved.",
        "",
        "**Label rule:** a number is a true hallucination only if it is programmatically",
        "unmatchable against the case's pool - not within 1.1 of any pool value, its x100,",
        "or its /100, not a k-suffixed citation within one unit of a pool value's",
        "thousandth (`matches_pool_k`), and not in the skip list [0..5] (`is_unmatchable`).",
        "Verified for every injected value at fixture-build time.",
        "",
        "**Honesty caveat:** this measures numerical grounding against a known pool -",
        "whether a cited number *exists* in the ground truth - not semantic correctness",
        "(whether the agent used the right number in a valid argument).",
        "",
        "## Results",
        "",
        "| Scope | Cases | Cases as expected | Numbers | TP | FN | FP | TN | Precision | Recall | FP rate |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    def row(scope: str, a: dict) -> str:
        return (
            f"| {scope} | {a['cases']} | {a['cases_matching_expectation']} | {a['n_considered']} "
            f"| {a['tp']} | {a['fn']} | {a['fp']} | {a['tn']} "
            f"| {_fmt(a['precision'])} | {_fmt(a['recall'])} | {_fmt(a['fp_rate'])} |"
        )

    lines.append(row("**Overall**", overall))
    for family in FAMILIES:
        lines.append(row(family, by_family[family]))

    lines += [
        "",
        "- **CLEAN**: every number traces to the pool; expected zero flags.",
        "- **INJECTED**: contains provably unmatchable numbers - including fabricated",
        "  k-suffix values verified outside both the plain and k-notation windows;",
        "  expected exactly those flagged.",
        "- **TRAP**: naive-checker bait (x100//100 scaling, rounding, negatives, `$1,234`",
        "  formatting, skip-list bullet indices, and correct k-notation citations of real",
        "  pool values like `176k` for 176,432.11); expected zero flags.",
        "",
        "Precision/recall are only defined where injected numbers exist (the INJECTED",
        "family); the other families measure the false-positive rate.",
        "",
        "**Reading a perfect score:** labels are defined by the checker's documented matching",
        "rule, so this eval cannot disagree with that rule by construction - what it tests is",
        "everything layered around it: number extraction (regex, comma/currency/percent",
        "cleaning), the skip list, sign handling, and pool flattening. A perfect score means",
        f"those layers never contradict the matching rule across {overall['cases']} adversarial cases; it does",
        "not mean the checker is semantically infallible (see the honesty caveat above).",
        "",
    ]

    mismatches = [r for r in rows if not r["verdict_match"]]
    lines.append("## Findings (cases deviating from expectation)")
    lines.append("")
    if not mismatches:
        lines.append("None - every case produced exactly the expected flags.")
    else:
        for r in mismatches:
            lines.append(
                f"- `{r['case_id']}` ({r['family']}, seed {r['seed']}): "
                f"flagged {r['flagged_values']}, expected {r['injected_values']} "
                f"(TP={r['tp']}, FN={r['fn']}, FP={r['fp']})"
            )
    lines.append("")

    (out_dir / "eval_checker_report.md").write_text("\n".join(lines))


def _write_csv(out_dir: Path, rows: list[dict]) -> None:
    fields = [
        "case_id", "family", "seed", "n_considered", "n_injected",
        "tp", "fn", "fp", "tn", "checker_is_valid", "verdict_match",
        "flagged_values", "injected_values",
    ]
    with (out_dir / "eval_checker_raw.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            out = dict(r)
            out["flagged_values"] = json.dumps(r["flagged_values"])
            out["injected_values"] = json.dumps(r["injected_values"])
            writer.writerow(out)


def run_eval(fixtures_dir: Path = FIXTURES_DIR, out_dir: Path = RESULTS_DIR) -> dict:
    pools_data, cases = load_fixtures(fixtures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [score_case(case, pools_data[str(case["seed"])]) for case in cases]
    overall = _aggregate(rows)
    by_family = {f: _aggregate([r for r in rows if r["family"] == f]) for f in FAMILIES}

    _write_report(out_dir, overall, by_family, rows)
    _write_csv(out_dir, rows)
    return {"overall": overall, "by_family": by_family, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild-fixtures", action="store_true",
                        help="Regenerate pools and cases by running the real pipeline (no LLM calls)")
    parser.add_argument("--fixtures-dir", type=Path, default=FIXTURES_DIR)
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    if args.rebuild_fixtures:
        build_fixtures(args.fixtures_dir)

    metrics = run_eval(args.fixtures_dir, args.out_dir)
    overall = metrics["overall"]
    print(
        f"Scored {overall['cases']} cases / {overall['n_considered']} numbers: "
        f"precision={_fmt(overall['precision'])} recall={_fmt(overall['recall'])} "
        f"fp_rate={_fmt(overall['fp_rate'])}"
    )
    print(f"Report: {args.out_dir / 'eval_checker_report.md'}")


if __name__ == "__main__":
    main()
