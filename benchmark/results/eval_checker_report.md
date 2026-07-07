# ConsistencyChecker precision/recall (Experiment B)

Deterministic eval of the guardrail as a per-number classifier, over labelled
cases built on authentic pipeline artifacts (real SHAP/context/optimizer pools,
seeds [42, 1337], tier `upstart`, budget 500,000, horizon 4). No LLM involved.

**Label rule:** a number is a true hallucination only if it is programmatically
unmatchable against the case's pool - not within 1.1 of any pool value, its x100,
or its /100, not a k-suffixed citation within one unit of a pool value's
thousandth (`matches_pool_k`), and not in the skip list [0..5] (`is_unmatchable`).
Verified for every injected value at fixture-build time.

**Honesty caveat:** this measures numerical grounding against a known pool -
whether a cited number *exists* in the ground truth - not semantic correctness
(whether the agent used the right number in a valid argument).

## Results

| Scope | Cases | Cases as expected | Numbers | TP | FN | FP | TN | Precision | Recall | FP rate |
|---|---|---|---|---|---|---|---|---|---|---|
| **Overall** | 75 | 75 | 258 | 43 | 0 | 0 | 215 | 1.000 | 1.000 | 0.000 |
| CLEAN | 15 | 15 | 69 | 0 | 0 | 0 | 69 | n/a | n/a | 0.000 |
| INJECTED | 30 | 30 | 118 | 43 | 0 | 0 | 75 | 1.000 | 1.000 | 0.000 |
| TRAP | 30 | 30 | 71 | 0 | 0 | 0 | 71 | n/a | n/a | 0.000 |

- **CLEAN**: every number traces to the pool; expected zero flags.
- **INJECTED**: contains provably unmatchable numbers - including fabricated
  k-suffix values verified outside both the plain and k-notation windows;
  expected exactly those flagged.
- **TRAP**: naive-checker bait (x100//100 scaling, rounding, negatives, `$1,234`
  formatting, skip-list bullet indices, and correct k-notation citations of real
  pool values like `176k` for 176,432.11); expected zero flags.

Precision/recall are only defined where injected numbers exist (the INJECTED
family); the other families measure the false-positive rate.

**Reading a perfect score:** labels are defined by the checker's documented matching
rule, so this eval cannot disagree with that rule by construction - what it tests is
everything layered around it: number extraction (regex, comma/currency/percent
cleaning), the skip list, sign handling, and pool flattening. A perfect score means
those layers never contradict the matching rule across 75 adversarial cases; it does
not mean the checker is semantically infallible (see the honesty caveat above).

## Findings (cases deviating from expectation)

None - every case produced exactly the expected flags.
