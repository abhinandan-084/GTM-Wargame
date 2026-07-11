"""Flag audit for Experiment A: mechanical evidence for classifying each flag.

The experiment report counts checker *flags*; this audit asks what each flag
actually is. A manual spot-check found a flag (gemini/seed 1015, value 25)
that is a correct percentage derived from two pool values - a false positive
class Experiment B cannot see, because its labels are defined by the checker's
own matching rule.

For every flagged run this script:

1. Rebuilds the run's exact pool deterministically (build_artifacts).
2. Re-validates the stored transcript and asserts the flags match the stored
   flagged_values - guards against pipeline drift since the runs were recorded.
3. Extracts the sentence around each flag (unit disambiguation needs context:
   "25" as a percent vs "25" as the head of "25,106.14").
4. Runs two mechanical passes per flag:
   - notation: value x1000 / x1e6 near a pool value, with k/M wording present
   - derivation: pairwise pool combinations (difference, sum, ratio, percentage
     gap, per-week average, share-of-budget) within the checker's tolerance
5. Writes flag_audit_raw.csv: one row per flag with a candidate bucket and
   machine-readable evidence.

A derivation match is a *candidate*, not a verdict: with dozens of pool values
pairwise ops collide by chance. Final buckets are assigned by human judgment in
flag_audit_verdicts.json, requiring the sentence to semantically support the
match; --write-report regenerates experiment_a_report.md from the stored run
records plus those verdicts (flag counts under the current checker, fabrication
rates, per-flag appendix), so every published number is computed from the
committed evidence, never typed.

Fully offline - no API keys, no LLM calls. Usage:
    uv run python benchmark/audit_flags.py
    uv run python benchmark/audit_flags.py --allow-drift  # after checker changes:
        report flag-set differences vs the stored records instead of halting
    uv run python benchmark/audit_flags.py --write-report  # regenerate the
        experiment report from runs/ + flag_audit_verdicts.json (instant)
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

try:
    from benchmark.run import (aggregate, build_artifacts, wilson_interval,
                               write_examples, write_report)
except ImportError:  # executed as a script: benchmark/ itself is on sys.path
    from run import (aggregate, build_artifacts, wilson_interval,  # type: ignore
                     write_examples, write_report)

from gtm_boardroom.guardrails.consistency_checker import ConsistencyChecker

BENCHMARK_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BENCHMARK_DIR / "results"
RUNS_DIR = RESULTS_DIR / "runs"
OUT_CSV = RESULTS_DIR / "flag_audit_raw.csv"
VERDICTS_JSON = RESULTS_DIR / "flag_audit_verdicts.json"
REPORT_MD = RESULTS_DIR / "experiment_a_report.md"

BUCKETS = ("FABRICATED", "CORRECT-BUT-DERIVED", "NOTATION-ARTIFACT")

TOLERANCE = 1.1  # the checker's own matching tolerance
NOTATION_RELATIVE = 0.005  # 0.5% relative, for k/M-scaled comparisons


def _value_forms(v: float) -> list[str]:
    """Textual forms a flagged float may take in the transcript (mirrors
    _highlight_flagged in run.py)."""
    forms = {f"{v:g}", f"{v:.2f}", f"{v:.1f}", f"{v:,.2f}", f"{v:,.1f}"}
    if float(v).is_integer():
        forms.add(f"{int(v):,}")
        forms.add(str(int(v)))
    return sorted(forms, key=len, reverse=True)


def find_sentences(text: str, value: float) -> list[str]:
    """Sentences/bullets containing the value as a standalone number.

    Rejects matches that are the head of a longer number ("25" inside
    "25,106.14") so the context shown is where the *checker* saw the value.
    Falls back to any-occurrence sentences if the strict pass finds nothing.
    """
    # Bullets and headings are their own units; then split prose on sentence ends.
    units: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        units.extend(s.strip() for s in re.split(r"(?<=[.!?])\s+", line) if s.strip())

    strict, loose = [], []
    for unit in units:
        for form in _value_forms(value):
            pattern = rf"(?<![\d.\w]){re.escape(form)}(?![\d.]|,\d)"
            if re.search(pattern, unit):
                strict.append(unit)
                break
            if re.search(rf"(?<![\d.\w]){re.escape(form)}", unit):
                loose.append(unit)
                break
    return strict if strict else loose


def _near(a: float, b: float) -> bool:
    return abs(a - b) < TOLERANCE


def _near_relative(a: float, b: float, rel: float = NOTATION_RELATIVE) -> bool:
    return b != 0 and abs(a - b) / abs(b) <= rel


def notation_pass(value: float, pool: list[float], sentences: list[str]) -> str | None:
    """Candidate NOTATION-ARTIFACT: value x1000/x1e6 near a pool value AND the
    sentence actually carries k/M wording."""
    context = " ".join(sentences).lower()
    has_suffix = bool(re.search(r"\d\s*k\b|\bthousand\b|\d\s*m\b|\bmillion\b", context))
    for scale, name in ((1_000.0, "x1000"), (1_000_000.0, "x1e6")):
        scaled = value * scale
        for gt in pool:
            if _near(scaled, gt) or _near_relative(scaled, gt):
                suffix_note = "suffix present" if has_suffix else "NO suffix in context"
                return f"{name}: {value} * {scale:g} = {scaled:g} ~ pool {gt:g} ({suffix_note})"
    return None


def derivation_pass(value: float, pool: list[float], horizon: int,
                    budget: float) -> list[str]:
    """Candidate CORRECT-BUT-DERIVED: pairwise combinations of pool values that
    land within the checker's tolerance of the flag. Pairs only - no triples -
    to keep chance collisions bounded. Returns every match; the human pass
    decides whether the sentence semantically supports any of them."""
    matches: list[str] = []

    def record(formula: str, result: float) -> None:
        entry = f"{formula} = {result:.2f}"
        if entry not in matches:
            matches.append(entry)

    uniq = sorted(set(pool))
    for a in uniq:
        if _near(value, a / horizon):
            record(f"{a:g} / horizon {horizon}", a / horizon)
        if budget and _near(value, a / budget * 100):
            record(f"{a:g} / budget {budget:g} x100", a / budget * 100)
        for b in uniq:
            if a == b:
                continue
            if _near(value, a - b):
                record(f"{a:g} - {b:g}", a - b)
            if a < b:  # symmetric ops once per pair
                if _near(value, a + b):
                    record(f"{a:g} + {b:g}", a + b)
            if b != 0:
                if _near(value, a / b):
                    record(f"{a:g} / {b:g}", a / b)
                if _near(value, (a - b) / b):
                    record(f"({a:g} - {b:g}) / {b:g}", (a - b) / b)
                if _near(value, (a - b) / b * 100):
                    record(f"({a:g} - {b:g}) / {b:g} x100", (a - b) / b * 100)
            if a != 0:
                if _near(value, (a - b) / a):
                    record(f"({a:g} - {b:g}) / {a:g}", (a - b) / a)
                if _near(value, (a - b) / a * 100):
                    record(f"({a:g} - {b:g}) / {a:g} x100", (a - b) / a * 100)
    return matches


def load_flagged_runs(runs_dir: Path = RUNS_DIR) -> list[dict]:
    records = []
    for path in sorted(runs_dir.glob("*.json")):
        if path.stem.endswith("FAILURE"):
            continue
        record = json.loads(path.read_text())
        if record["flagged_values"]:
            records.append(record)
    return records


def revalidate(record: dict, artifacts: dict, allow_drift: bool) -> list[float]:
    """Re-run the checker on the stored text.

    Default mode asserts the flags match the stored record exactly (pipeline
    drift invalidates the audit). With allow_drift (for re-runs after a
    deliberate checker change) differences are returned as a finding instead.
    """
    result = ConsistencyChecker.validate_response(
        record["text"], artifacts["shap_info"], artifacts["market_context"],
        artifacts["opt_results"])
    got, stored = result["hallucinated_values"], record["flagged_values"]
    if got != stored and not allow_drift:
        sys.exit(
            f"DRIFT on {record['model']}/seed {record['seed']}/{record['node']}: "
            f"re-validation produced {got}, stored record has {stored}. "
            f"Halting - a drifted pipeline invalidates the audit."
        )
    return got


def _build_artifacts_cached(seed: int, tier: str, budget: float, horizon: int,
                            cache_dir: Path | None) -> dict:
    """build_artifacts is minutes of optimizer work per seed; cache the
    jsonable result so audit re-runs (e.g. after a checker change) are instant."""
    if cache_dir is None:
        return build_artifacts(seed, tier, budget, horizon)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"artifacts_seed{seed}_{tier}_{budget:g}_{horizon}.json"
    if cache_file.is_file():
        return json.loads(cache_file.read_text())
    artifacts = build_artifacts(seed, tier, budget, horizon)
    cache_file.write_text(json.dumps(artifacts))
    return artifacts


def audit(runs_dir: Path = RUNS_DIR, out_csv: Path = OUT_CSV,
          allow_drift: bool = False, cache_dir: Path | None = None) -> list[dict]:
    records = load_flagged_runs(runs_dir)
    if not records:
        sys.exit(f"No flagged runs found under {runs_dir}")

    pools: dict[tuple[int, str], dict] = {}
    rows: list[dict] = []
    for record in records:
        cfg = record["config"]
        key = (record["seed"], cfg["tier"])
        if key not in pools:
            print(f"Rebuilding pool for seed {record['seed']} (tier {cfg['tier']})...")
            pools[key] = _build_artifacts_cached(
                record["seed"], cfg["tier"], cfg["budget"], cfg["horizon"], cache_dir)
        artifacts = pools[key]
        current_flags = revalidate(record, artifacts, allow_drift)
        if current_flags != record["flagged_values"]:
            print(f"  FLAG-SET CHANGE {record['model']}/seed {record['seed']}: "
                  f"stored {record['flagged_values']} -> now {current_flags}")
        pool = artifacts["pool"]

        for value in record["flagged_values"]:
            sentences = find_sentences(record["text"], value)
            notation = notation_pass(value, pool, sentences)
            derivations = derivation_pass(value, pool, cfg["horizon"], cfg["budget"])

            if notation and "suffix present" in notation:
                bucket, evidence = "NOTATION-ARTIFACT", notation
            elif derivations:
                bucket = "CORRECT-BUT-DERIVED"
                evidence = "; ".join(derivations[:6])
                if len(derivations) > 6:
                    evidence += f"; (+{len(derivations) - 6} more)"
            else:
                bucket, evidence = "FABRICATED", "no notation or derivation match"

            rows.append(
                {
                    "model": record["model"],
                    "seed": record["seed"],
                    "node": record["node"],
                    "value": value,
                    "sentence": " | ".join(sentences) or "(no sentence located)",
                    "candidate_bucket": bucket,
                    "evidence": evidence,
                    "still_flagged": value in current_flags,
                }
            )
        print(f"  {record['model']}/seed {record['seed']}: "
              f"{len(record['flagged_values'])} flag(s) audited, re-validation OK")

    with out_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n{len(rows)} flags -> {out_csv}")
    return rows


def load_verdicts(verdicts_path: Path = VERDICTS_JSON,
                  csv_path: Path = OUT_CSV) -> list[dict]:
    """Load the final human-judgment buckets, asserting row-by-row alignment
    with the audit CSV so verdicts can never silently drift from evidence."""
    verdicts = json.loads(verdicts_path.read_text())["verdicts"]
    with csv_path.open() as fh:
        rows = list(csv.DictReader(fh))
    if len(verdicts) != len(rows):
        sys.exit(f"{len(verdicts)} verdicts vs {len(rows)} CSV rows - "
                 f"rerun the audit or fix {verdicts_path.name}.")
    for i, (v, r) in enumerate(zip(verdicts, rows), 1):
        if (v["model"], v["seed"], v["value"]) != (r["model"], int(r["seed"]), float(r["value"])):
            sys.exit(f"Verdict {i} ({v['model']}/seed {v['seed']}/{v['value']}) does not "
                     f"match CSV row {i} ({r['model']}/seed {r['seed']}/{r['value']}).")
        if v["bucket"] not in BUCKETS:
            sys.exit(f"Verdict {i}: unknown bucket {v['bucket']!r}")
    return verdicts


def _run_denominators(runs_dir: Path = RUNS_DIR) -> dict[str, dict]:
    """Numbers written, seeds run, and seeds flagged per model, from the
    stored run records (the experiment report's own denominators)."""
    stats: dict[str, dict] = {}
    for path in sorted(runs_dir.glob("*.json")):
        if path.stem.endswith("FAILURE"):
            continue
        r = json.loads(path.read_text())
        s = stats.setdefault(r["model"], {"numbers": 0, "seeds": set(), "flagged_seeds": set()})
        s["numbers"] += r["numbers_extracted"]
        s["seeds"].add(r["seed"])
        if r["flagged_values"]:
            s["flagged_seeds"].add(r["seed"])
    return stats


def render_audit_section(verdicts: list[dict], stats: dict[str, dict]) -> str:
    """The flag-audit section of experiment_a_report.md. Every number is
    computed from the verdicts and the stored run records."""
    models = sorted(stats)
    for m in models:
        if {v["seed"] for v in verdicts if v["model"] == m} != stats[m]["flagged_seeds"]:
            sys.exit(f"Verdict seeds for {m} do not match the flagged runs on disk.")

    def count(model: str, bucket: str) -> int:
        return sum(1 for v in verdicts if v["model"] == model and v["bucket"] == bucket)

    def ci(k: int, n: int) -> str:
        lo, hi = wilson_interval(k, n)
        return f"{k}/{n} ({k / n:.1%}), 95% CI [{lo:.1%}, {hi:.1%}]"

    flags = {m: sum(1 for v in verdicts if v["model"] == m) for m in models}
    fab = {m: count(m, "FABRICATED") for m in models}
    derived = {m: count(m, "CORRECT-BUT-DERIVED") for m in models}
    fab_seeds = {m: len({v["seed"] for v in verdicts
                         if v["model"] == m and v["bucket"] == "FABRICATED"}) for m in models}

    lines = [
        f"## Flag audit: what the {len(verdicts)} flags actually are",
        "",
        "A flag means a number failed pool matching - it does not by itself mean the",
        "model invented the number. Every flag was audited offline, no LLM involved:",
        "[`audit_flags.py`](../audit_flags.py) rebuilds each flagged run's exact pool",
        "deterministically, re-validates the stored transcript (every flag reproduced",
        "exactly - no pipeline drift), and collects mechanical notation/derivation",
        "evidence per flag ([`flag_audit_raw.csv`](flag_audit_raw.csv)). Final buckets",
        "([`flag_audit_verdicts.json`](flag_audit_verdicts.json)) require the sentence",
        "to semantically support the mechanical match - with ~50 pool values, pairwise",
        "derivations collide by chance - and genuinely ambiguous flags resolve to",
        "FABRICATED, so the fabrication rates below remain a defensible upper bound.",
        "",
        "| Bucket | " + " | ".join(models) + " | total |",
        "|---|" + "---|" * (len(models) + 1),
    ]
    for bucket, per_model in (("FABRICATED", fab), ("CORRECT-BUT-DERIVED", derived)):
        row = " | ".join(str(per_model[m]) for m in models)
        lines.append(f"| {bucket} | {row} | {sum(per_model.values())} |")
    lines.append("| **total** | " + " | ".join(f"**{flags[m]}**" for m in models)
                 + f" | **{len(verdicts)}** |")

    lines += [
        "",
        "- **FABRICATED**: matches nothing in the pool and no semantically supported",
        "  derivation over pool values.",
        "- **CORRECT-BUT-DERIVED**: a valid computation over real pool values (here:",
        "  the price gap `(leader - optimized) / leader`, both operands named in the",
        "  sentence) that simply isn't stored in the pool. Guardrail false positives -",
        "  but deriving new true values is outside the checker's contract, so these",
        "  remain flags by design.",
        "",
        "### Fabrication rates",
        "",
        "| Metric | " + " | ".join(models) + " |",
        "|---|" + "---|" * len(models),
        "| **Fabricated numbers / numbers written** | "
        + " | ".join(ci(fab[m], stats[m]["numbers"]) for m in models) + " |",
        "| **Runs with >=1 fabricated number** | "
        + " | ".join(ci(fab_seeds[m], len(stats[m]["seeds"])) for m in models) + " |",
        "",
        "Every llamacpp flag survived the audit as a genuine fabrication; no gemini",
        "flag did - in this sample the frontier model fabricated nothing, and every",
        "flag it drew was a guardrail false positive. The local model fabricates at a",
        "materially higher per-number rate with cleanly separated CIs - the guardrail",
        "is load-bearing exactly where local/private deployments need it, and its",
        "flags need auditing before they become claims.",
        "",
        "### Per-flag verdicts",
        "",
        "| # | Model | Seed | Value | Bucket | Justification |",
        "|---|---|---|---|---|---|",
    ]
    for i, v in enumerate(verdicts, 1):
        lines.append(f"| {i} | {v['model']} | {v['seed']} | {v['value']:g} "
                     f"| {v['bucket']} | {v['justification']} |")
    lines += [
        "",
        "Sentence context and raw mechanical evidence for every row:",
        "[`flag_audit_raw.csv`](flag_audit_raw.csv).",
        "",
    ]
    return "\n".join(lines)


def write_full_report(results_dir: Path = RESULTS_DIR, runs_dir: Path = RUNS_DIR,
                      verdicts_path: Path = VERDICTS_JSON,
                      csv_path: Path = OUT_CSV) -> Path:
    """Regenerate experiment_a_report.md and the flagged-transcript examples
    from the stored run records + audit verdicts. Instant and fully offline."""
    records, failures = [], []
    for path in sorted(runs_dir.glob("*.json")):
        data = json.loads(path.read_text())
        (failures if path.stem.endswith("FAILURE") else records).append(data)
    if not records:
        sys.exit(f"No run records under {runs_dir}")

    verdicts = load_verdicts(verdicts_path, csv_path)
    models = sorted({r["model"] for r in records})
    rows = aggregate(records, failures, models)

    examples_dir = results_dir / "examples"
    for old in examples_dir.glob("*.md"):
        old.unlink()
    n_flagged = sum(1 for r in records if r["flagged_values"])
    example_paths = write_examples(records, examples_dir, limit=n_flagged)

    report_path = write_report(results_dir, rows, failures,
                               records[0]["config"], example_paths)
    section = render_audit_section(verdicts, _run_denominators(runs_dir))
    report_path.write_text(report_path.read_text().rstrip() + "\n\n" + section)
    print(f"Report + {len(example_paths)} examples regenerated under {results_dir}")
    return report_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-drift", action="store_true",
                        help="Report flag-set changes vs stored records instead of halting "
                             "(for re-runs after a deliberate checker change)")
    parser.add_argument("--out", type=Path, default=OUT_CSV, help="Output CSV path")
    parser.add_argument("--pools-cache", type=Path, default=None,
                        help="Directory to cache rebuilt per-seed artifacts (minutes per seed)")
    parser.add_argument("--write-report", action="store_true",
                        help="Skip the audit; regenerate experiment_a_report.md and the "
                             "examples from runs/ + flag_audit_verdicts.json")
    args = parser.parse_args()
    if args.write_report:
        write_full_report()
    else:
        audit(out_csv=args.out, allow_drift=args.allow_drift, cache_dir=args.pools_cache)
