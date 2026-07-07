import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from benchmark.eval_checker import (  # noqa: E402
    CASES_PER_FAMILY,
    FAMILIES,
    FIXTURES_DIR,
    SKIP_VALUES,
    _k_decimals,
    is_matchable,
    is_unmatchable,
    load_fixtures,
    matches_pool_k,
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
    # INJECTED and TRAP each hold a second batch of k-notation cases.
    expected = {"CLEAN": CASES_PER_FAMILY, "INJECTED": 2 * CASES_PER_FAMILY,
                "TRAP": 2 * CASES_PER_FAMILY}
    for family in FAMILIES:
        count = sum(1 for c in cases if c["family"] == family)
        assert count == expected[family], f"{family} has {count} cases, expected {expected[family]}"
    ids = [c["case_id"] for c in cases]
    assert len(ids) == len(set(ids))


def test_every_injected_value_is_provably_unmatchable():
    # The core labelling guarantee: a 'fake' that accidentally matches the pool
    # (directly, via x100//100 scaling, or - for k-suffixed tokens - via the
    # context-gated x1000 rule) would silently corrupt recall.
    pools, cases = _fixtures()
    injected_total, injected_k_total = 0, 0
    for case in cases:
        pool = pools[str(case["seed"])]
        for v in case["values"]:
            if v["label"] == "injected":
                injected_total += 1
                assert is_unmatchable(v["value"], pool), (
                    f"{case['case_id']}: injected value {v['value']} is matchable"
                )
                if v["token"].rstrip("kK") != v["token"]:
                    injected_k_total += 1
                    assert not matches_pool_k(v["value"], pool, _k_decimals(v["token"])), (
                        f"{case['case_id']}: injected k value {v['token']} is k-matchable"
                    )
    assert injected_total > 0
    assert injected_k_total > 0  # fabricated k-suffix values must be represented


def test_every_legit_k_citation_is_k_matchable():
    pools, cases = _fixtures()
    legit_k_total = 0
    for case in cases:
        pool = pools[str(case["seed"])]
        for v in case["values"]:
            if v["label"] == "legit_k":
                legit_k_total += 1
                assert matches_pool_k(v["value"], pool, _k_decimals(v["token"])), (
                    f"{case['case_id']}: legit_k value {v['token']} not k-matchable"
                )
    assert legit_k_total > 0


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
    assert overall["cases"] == 5 * CASES_PER_FAMILY  # CLEAN + 2x INJECTED + 2x TRAP
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
