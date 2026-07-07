# LLM fabrication rates with the guardrail observing (Experiment A)

**Framing:** this is not a model leaderboard, and a flag is not yet a
fabrication: flag rates are an *upper bound* - a flag means a number failed
pool matching, not necessarily that the model invented it. Every flag is
audited and classified before it becomes a claim (the flag-audit section,
maintained by `benchmark/audit_flags.py --write-report`); the fabrication
rates there are what show the guardrail is load-bearing in local/private
deployments.

**Honesty caveat:** this measures numerical grounding against a known pool -
whether a cited number *exists* in the ground truth - not semantic correctness.

**Design:** paired seeds (identical across models), tier `upstart`,
budget 500,000, horizon 4,
scored nodes: analyst, strategist.

| Model | Runs (K) | Runs w/ >=1 flag | 95% CI (run-level) | Numbers written | Numbers flagged | Per-number flag rate |
|---|---|---|---|---|---|---|
| gemini | 30 | 3/30 | [3.5%, 25.6%] | 537 | 3 | 3/537 (0.6%) |
| llamacpp | 30 | 4/30 | [5.3%, 29.7%] | 138 | 10 | 10/138 (7.2%) |

Raw counts are authoritative; rates at small K are noisy - read the CI.

## Example flagged transcripts

- [`gemini_seed1015_strategist.md`](examples/gemini_seed1015_strategist.md)
- [`llamacpp_seed1015_strategist.md`](examples/llamacpp_seed1015_strategist.md)
- [`gemini_seed1019_strategist.md`](examples/gemini_seed1019_strategist.md)
- [`gemini_seed1023_strategist.md`](examples/gemini_seed1023_strategist.md)
- [`llamacpp_seed1016_strategist.md`](examples/llamacpp_seed1016_strategist.md)
- [`llamacpp_seed1017_strategist.md`](examples/llamacpp_seed1017_strategist.md)
- [`llamacpp_seed1029_strategist.md`](examples/llamacpp_seed1029_strategist.md)

## Flag audit: what the 13 flags actually are

A flag means a number failed pool matching - it does not by itself mean the
model invented the number. Every flag was audited offline, no LLM involved:
[`audit_flags.py`](../audit_flags.py) rebuilds each flagged run's exact pool
deterministically, re-validates the stored transcript (every flag reproduced
exactly - no pipeline drift), and collects mechanical notation/derivation
evidence per flag ([`flag_audit_raw.csv`](flag_audit_raw.csv)). Final buckets
([`flag_audit_verdicts.json`](flag_audit_verdicts.json)) require the sentence
to semantically support the mechanical match - with ~50 pool values, pairwise
derivations collide by chance - and genuinely ambiguous flags resolve to
FABRICATED, so the fabrication rates below remain a defensible upper bound.

| Bucket | gemini | llamacpp | total |
|---|---|---|---|
| FABRICATED | 0 | 10 | 10 |
| CORRECT-BUT-DERIVED | 3 | 0 | 3 |
| **total** | **3** | **10** | **13** |

- **FABRICATED**: matches nothing in the pool and no semantically supported
  derivation over pool values.
- **CORRECT-BUT-DERIVED**: a valid computation over real pool values (here:
  the price gap `(leader - optimized) / leader`, both operands named in the
  sentence) that simply isn't stored in the pool. Guardrail false positives -
  but deriving new true values is outside the checker's contract, so these
  remain flags by design.

### Fabrication rates

| Metric | gemini | llamacpp |
|---|---|---|
| **Fabricated numbers / numbers written** | 0/537 (0.0%), 95% CI [0.0%, 0.7%] | 10/138 (7.2%), 95% CI [4.0%, 12.8%] |
| **Runs with >=1 fabricated number** | 0/30 (0.0%), 95% CI [0.0%, 11.4%] | 4/30 (13.3%), 95% CI [5.3%, 29.7%] |

Every llamacpp flag survived the audit as a genuine fabrication; no gemini
flag did - in this sample the frontier model fabricated nothing, and every
flag it drew was a guardrail false positive. The local model fabricates at a
materially higher per-number rate with cleanly separated CIs - the guardrail
is load-bearing exactly where local/private deployments need it, and its
flags need auditing before they become claims.

### Per-flag verdicts

| # | Model | Seed | Value | Bucket | Justification |
|---|---|---|---|---|---|
| 1 | gemini | 1015 | 25 | CORRECT-BUT-DERIVED | '25 percent price gap relative to the leader (715.18)': (715.18 - 535.68) / 715.18 = 25.1%; both operands are pool values and the sentence names the derivation. |
| 2 | gemini | 1019 | 30 | CORRECT-BUT-DERIVED | '496.14 maintains a healthy 30% gap below the market leader price of 715.07': (715.07 - 496.14) / 715.07 = 30.6%, within the checker's own tolerance of 30; sentence names both operands. |
| 3 | gemini | 1023 | 28 | CORRECT-BUT-DERIVED | '515.32 ... 28 percent safety buffer' vs leader 715.07: (715.07 - 515.32) / 715.07 = 27.9%; sentence names both operands. |
| 4 | llamacpp | 1015 | 20 | FABRICATED | 'Reduce spending on Search by 20%': invented action parameter with no pool basis; derivation hits are chance collisions. |
| 5 | llamacpp | 1016 | 60 | FABRICATED | Invented '60% to Margin Preservation' budget split; no pool basis. |
| 6 | llamacpp | 1016 | 40 | FABRICATED | Invented '40% to Efficiency Enhancement' split; no pool basis. |
| 7 | llamacpp | 1016 | 60 | FABRICATED | Second occurrence of the invented 60% split (checker flags per occurrence). |
| 8 | llamacpp | 1016 | 40 | FABRICATED | Second occurrence of the invented 40% split. |
| 9 | llamacpp | 1017 | 20 | FABRICATED | 'Reduce budget by 20% to $X': invented parameter beside a literal unfilled placeholder. |
| 10 | llamacpp | 1017 | 30 | FABRICATED | 'Increase budget by 30% to $Y': same invented pattern. |
| 11 | llamacpp | 1029 | 100 | FABRICATED | Invented ROI table Spend cell ('$100k'); the pipeline computes no revenue or ROI; the only mechanical hits are degenerate (a-0)/a collisions. |
| 12 | llamacpp | 1029 | 500 | FABRICATED | Invented ROI table Revenue cell ('$500k'). Numerically coincides with the run budget 500,000, which is not itself a pool member - coincidence, not grounding. |
| 13 | llamacpp | 1029 | 50 | FABRICATED | Invented ROI table Spend cell ('$50k'); no pool value near 50,000. |

Sentence context and raw mechanical evidence for every row:
[`flag_audit_raw.csv`](flag_audit_raw.csv).
