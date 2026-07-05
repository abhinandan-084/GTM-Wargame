# LLM fabrication rates with the guardrail observing (Experiment A)

**Framing:** this is not a model leaderboard. A frontier model's non-zero rate
shows that fabrication persists even at the frontier, on clean synthetic data,
under prompts that explicitly forbid inventing numbers. A local model's higher
rate is the point, not an embarrassment: it shows why the guardrail is
load-bearing in local/private deployments.

**Honesty caveat:** this measures numerical grounding against a known pool -
whether a cited number *exists* in the ground truth - not semantic correctness.

**Design:** paired seeds (identical across models), tier `upstart`,
budget 500,000, horizon 4,
scored nodes: manager.

| Model | Runs (K) | Runs w/ >=1 flag | 95% CI (run-level) | Numbers written | Numbers flagged | Per-number flag rate |
|---|---|---|---|---|---|---|
| gemini | 2 | 0/2 | [0.0%, 65.8%] | 40 | 0 | 0/40 (0.0%) |

Raw counts are authoritative; rates at small K are noisy - read the CI.

## Example flagged transcripts

None - no run produced a flag.
