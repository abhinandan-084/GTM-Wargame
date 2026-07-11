import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from benchmark.audit_flags import (  # noqa: E402
    _run_denominators,
    load_verdicts,
    render_audit_section,
)
from benchmark.run import (  # noqa: E402  (import itself asserts import-safety: no side effects)
    MockProvider,
    _highlight_flagged,
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


def test_highlight_never_annotates_inside_comma_grouped_numbers():
    # Flagged 25 is "25 percent", not the head of "25,106.14" (real bug:
    # gemini seed 1015 example had the marker inside the spend figure).
    text = "Social spend is minimized to 25,106.14 given the 25 percent gap and 25% risk."
    out = _highlight_flagged(text, [25.0])
    assert "25,106.14" in out and "**[FLAGGED: 25]**,106.14" not in out
    assert "**[FLAGGED: 25]** percent" in out
    assert "**[FLAGGED: 25]**%" in out

    out = _highlight_flagged("Retail spend of 195,575.07 misses the 195k claim.", [195.0])
    assert "195,575.07" in out and "**[FLAGGED: 195]**,575" not in out
    assert "**[FLAGGED: 195]**k" in out


def test_highlight_never_double_wraps():
    # Duplicate flagged values must not nest markers (real bug: llamacpp
    # seed 1016 flagged 60 and 40 twice, producing **[FLAGGED: **[FLAGGED: 60]**]**).
    text = "Allocate 60% to Margin Preservation and the remaining 40% elsewhere."
    out = _highlight_flagged(text, [60.0, 40.0, 60.0, 40.0])
    assert "**[FLAGGED: 60]**%" in out
    assert "**[FLAGGED: 40]**%" in out
    assert "FLAGGED: **[FLAGGED" not in out

    # A second pass over already-annotated text is a no-op.
    assert _highlight_flagged(out, [60.0, 40.0]) == out


def test_flag_audit_section_is_computed_and_in_sync_with_the_report():
    # load_verdicts asserts row-by-row alignment with flag_audit_raw.csv, so a
    # verdict edited without rerunning the audit (or vice versa) fails here.
    verdicts = load_verdicts()
    section = render_audit_section(verdicts, _run_denominators())

    # The published taxonomy and headline rates, pinned to the committed
    # verdicts + run records rather than typed into prose.
    assert "| FABRICATED | 0 | 10 | 10 |" in section
    assert "| CORRECT-BUT-DERIVED | 3 | 0 | 3 |" in section
    assert "0/537 (0.0%), 95% CI [0.0%, 0.7%]" in section
    assert "10/138 (7.2%), 95% CI [4.0%, 12.8%]" in section

    # The committed report carries exactly this rendering: regenerating with
    # audit_flags.py --write-report after any evidence change keeps them in sync.
    report = REPO_ROOT / "benchmark" / "results" / "experiment_a_report.md"
    assert section in report.read_text()


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
