# Article numbers (paste-ready)

Source of truth for the TDS draft placeholders. Every figure below is
computed by the benchmark pipeline - do not retype or recompute:
[Experiment A + flag audit](benchmark/results/experiment_a_report.md) |
[Experiment B](benchmark/results/eval_checker_report.md).

## Experiment B (checker precision/recall)

75 labelled cases / 258 numbers (CLEAN / INJECTED / TRAP, including k-notation
citations and fabricated k-suffix values): **precision 1.000, recall 1.000,
false-positive rate 0.000**. All 43 injected fabrications caught.

## Experiment A flags (30 paired seeds, current checker)

| Metric | gemini | llamacpp |
|---|---|---|
| Flags per number written | 3/537 (0.6%) | 10/138 (7.2%) |
| Runs with >=1 flag | 3/30 (10.0%), 95% CI [3.5%, 25.6%] | 4/30 (13.3%), 95% CI [5.3%, 29.7%] |

## Flag audit (13 flags)

| Bucket | gemini | llamacpp | total |
|---|---|---|---|
| FABRICATED | 0 | 10 | 10 |
| CORRECT-BUT-DERIVED | 3 | 0 | 3 |
| **total** | **3** | **10** | **13** |

| Metric | gemini | llamacpp |
|---|---|---|
| **Fabricated numbers / numbers written** | 0/537 (0.0%), 95% CI [0.0%, 0.7%] | 10/138 (7.2%), 95% CI [4.0%, 12.8%] |
| **Runs with >=1 fabricated number** | 0/30 (0.0%), 95% CI [0.0%, 11.4%] | 4/30 (13.3%), 95% CI [5.3%, 29.7%] |

## Frontier-vs-local reading

Gemini fabricated nothing in 537 numbers across 30 runs: its only 3 flags are
correct derived percentages (the price gap `(leader - optimized) / leader`,
both operands real pool values named in the sentence) - guardrail false
positives by design, since deriving new true values is outside the checker's
contract. Every llama.cpp flag survived the audit as a genuine fabrication
(invented budget splits, action parameters, and a whole ROI table); at run
level the CIs overlap, but per number the CIs no longer do (10/138 sits
clearly above 0/537). Two honest caveats: 30 seeds bounds the frontier rate
at 0/537 with a CI upper edge, not at zero; and the flags still need auditing
before they become claims - the 3 derived-percentage false positives prove
the false-positive class exists.
