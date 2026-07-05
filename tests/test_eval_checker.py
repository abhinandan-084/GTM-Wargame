import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from benchmark.eval_checker import (  # noqa: E402
    CASES_PER_FAMILY,
    FAMILIES,
    FIXTURES_DIR,
    SKIP_VALUES,
    is_matchable,
    is_unmatchable,
    load_fixtures,
    pool_from_data,
    run_eval,
)


def _fixtures():
    pools_data, cases = load_fixtures(FIXTURES_DIR)
    pools = {seed: pool_from_data(data) for seed, data in pools_data.items()}
    return pools, cases


def test_fixtures_are_committed_and_balanced():
    pools, cases = _fixtures()
    assert len(pools) >= 1
    for family in FAMILIES:
        count = sum(1 for c in cases if c["family"] == family)
        assert count == CASES_PER_FAMILY, f"{family} has {count} cases, expected {CASES_PER_FAMILY}"


def test_every_injected_value_is_provably_unmatchable():
    # The core labelling guarantee: a 'fake' that accidentally matches the pool
    # (directly or via x100//100 scaling) would silently corrupt recall.
    pools, cases = _fixtures()
    injected_total = 0
    for case in cases:
        pool = pools[str(case["seed"])]
        for v in case["values"]:
            if v["label"] == "injected":
                injected_total += 1
                assert is_unmatchable(v["value"], pool), (
                    f"{case['case_id']}: injected value {v['value']} is matchable"
                )
    assert injected_total > 0


def test_every_legit_value_is_matchable_and_skipped_values_are_in_skip_list():
    pools, cases = _fixtures()
    for case in cases:
        pool = pools[str(case["seed"])]
        for v in case["values"]:
            if v["label"] == "legit":
                assert is_matchable(v["value"], pool), (
                    f"{case['case_id']}: legit value {v['value']} not matchable"
                )
            elif v["label"] == "skipped":
                assert v["value"] in SKIP_VALUES


def test_eval_runs_offline_and_produces_sane_report(tmp_path):
    # Asserts the eval runs end-to-end and its accounting is internally
    # consistent. Deliberately does NOT assert perfect scores: a genuine
    # FN/FP is a finding to record in the report, not a test failure.
    metrics = run_eval(FIXTURES_DIR, tmp_path)

    assert (tmp_path / "eval_checker_report.md").is_file()
    assert (tmp_path / "eval_checker_raw.csv").is_file()

    overall = metrics["overall"]
    assert overall["cases"] == CASES_PER_FAMILY * len(FAMILIES)
    assert overall["n_considered"] > 0
    assert overall["tp"] + overall["fn"] == overall["n_injected"]
    assert overall["tp"] + overall["fn"] + overall["fp"] + overall["tn"] == overall["n_considered"]
    for rate in ("precision", "recall", "fp_rate"):
        assert overall[rate] is None or 0.0 <= overall[rate] <= 1.0

    clean = metrics["by_family"]["CLEAN"]
    trap = metrics["by_family"]["TRAP"]
    injected = metrics["by_family"]["INJECTED"]
    assert clean["n_injected"] == 0 and trap["n_injected"] == 0
    assert injected["n_injected"] > 0
