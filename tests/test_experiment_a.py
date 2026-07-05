import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from benchmark.run import (  # noqa: E402  (import itself asserts import-safety: no side effects)
    MockProvider,
    parse_seeds,
    run_experiment,
    wilson_interval,
)


def test_wilson_interval_brackets_the_point_estimate():
    lo, hi = wilson_interval(4, 50)
    assert 0.0 <= lo < 4 / 50 < hi <= 1.0
    assert wilson_interval(0, 0) == (0.0, 1.0)
    lo_zero, hi_zero = wilson_interval(0, 50)
    assert lo_zero == 0.0 and 0.0 < hi_zero < 0.15


def test_parse_seeds_forms():
    assert parse_seeds("1000:1005", k=0) == [1000, 1001, 1002, 1003, 1004]
    assert parse_seeds("7,9,13", k=0) == [7, 9, 13]
    assert parse_seeds("1000:1010", k=3) == [1000, 1001, 1002]
    assert parse_seeds(None, k=2) == [1000, 1001]


def test_dry_run_completes_end_to_end_without_api_calls(tmp_path):
    # Odd seed -> the mock provider fabricates one provably-unmatchable number,
    # so this exercises the flag -> runs/ -> examples/ -> report pipeline too.
    result = run_experiment(
        models=["gemini"],
        seeds=[1001],
        tier="upstart",
        budget=300_000.0,
        horizon=2,
        nodes=["manager"],
        out_dir=tmp_path,
        dry_run=True,
    )

    assert (tmp_path / "experiment_a_report.md").is_file()
    assert (tmp_path / "runs" / "gemini_seed1001_manager.json").is_file()
    assert result["failures"] == []

    (row,) = result["rows"]
    assert row["runs"] == 1
    assert row["runs_flagged"] == 1
    assert row["numbers_flagged"] >= 1
    assert 0.0 <= row["ci_low"] <= row["ci_high"] <= 1.0

    examples = list((tmp_path / "examples").glob("*.md"))
    assert len(examples) == 1
    assert "FLAGGED" in examples[0].read_text()


def test_mock_provider_fabricated_number_is_unmatchable():
    artifacts = {
        "opt_results": {
            "optimized_price": 799.8,
            "lift_percent": 12.0,
            "forecasted_total_sales": 231087.0,
        },
        "pool": [799.8, 12.0, 231087.0],
    }
    provider = MockProvider(artifacts, fabricate=True)
    text = provider.invoke("ignored", {})
    assert "FLAGGED" not in text  # raw memo, no markers
    assert provider._fabricated is not None
    clean = MockProvider(artifacts, fabricate=False)
    assert clean._fabricated is None
