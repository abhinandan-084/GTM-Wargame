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
scored nodes: analyst, strategist.

| Model | Runs (K) | Runs w/ >=1 flag | 95% CI (run-level) | Numbers written | Numbers flagged | Per-number flag rate |
|---|---|---|---|---|---|---|
| gemini | 30 | 6/30 | [9.5%, 37.3%] | 537 | 13 | 13/537 (2.4%) |
| llamacpp | 30 | 4/30 | [5.3%, 29.7%] | 138 | 10 | 10/138 (7.2%) |

Raw counts are authoritative; rates at small K are noisy - read the CI.

## Example flagged transcripts

- [`gemini_seed1003_strategist.md`](examples/gemini_seed1003_strategist.md)
- [`llamacpp_seed1015_strategist.md`](examples/llamacpp_seed1015_strategist.md)
- [`gemini_seed1015_strategist.md`](examples/gemini_seed1015_strategist.md)
- [`gemini_seed1019_strategist.md`](examples/gemini_seed1019_strategist.md)
- [`gemini_seed1023_strategist.md`](examples/gemini_seed1023_strategist.md)
- [`gemini_seed1024_strategist.md`](examples/gemini_seed1024_strategist.md)
- [`gemini_seed1025_strategist.md`](examples/gemini_seed1025_strategist.md)
- [`llamacpp_seed1016_strategist.md`](examples/llamacpp_seed1016_strategist.md)
- [`llamacpp_seed1017_strategist.md`](examples/llamacpp_seed1017_strategist.md)
- [`llamacpp_seed1029_strategist.md`](examples/llamacpp_seed1029_strategist.md)
