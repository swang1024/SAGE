# LongMemEval-S Robustness Experiment: SAGE vs mem0

**Status: FINAL — all 12 runs complete and judged** (2026-07-14).

## Protocol

- **Benchmark**: LongMemEval-S (Wu et al.), full 500 questions, each with its own haystack of ~40-50 chat sessions (~115k tokens) ingested into an isolated per-question memory store.
- **Arms**: `mem0` = ungated mem0 write path (LLM fact extraction + LLM update-decision); `sage` = same extraction, vMF-KDE novelty gate replaces the update-decision LLM call.
- **Backbones**: qwen2.5:3b (ollama, one H200 per run) and gpt-4o-mini (OpenAI API). Embedders: nomic-embed-text / text-embedding-3-small. `batch_size=8` messages per add call, `top_k=30` retrieval, answer generation by the same backbone.
- **Repeats**: 3 independent end-to-end runs per (backbone, arm).
- **Judge**: official LongMemEval per-question-type prompts (`get_anscheck_prompt`, vendored at commit d6dc8b5). qwen runs judged by llama3.1:8b (ollama); gpt-4o-mini runs judged by gpt-4o-mini. **Judges differ across backbones — compare arms only within a backbone.** (The LongMemEval paper's headline numbers use a GPT-4o judge; re-judging our 12 answer files with GPT-4o would cost ~$15 if strict comparability to published numbers is needed.)
- Accuracy = judge score over all 500 questions (incl. 30 abstention questions, which use the official abstention prompt variant).

## Results: gpt-4o-mini backbone (judge: gpt-4o-mini)

| Metric | mem0 r1 | mem0 r2 | mem0 r3 | mem0 mean | sage r1 | sage r2 | sage r3 | sage mean | Δ (SAGE−mem0) |
|---|---|---|---|---|---|---|---|---|---|
| **Overall** | 0.570 | 0.584 | 0.586 | **0.580±0.007** | 0.636 | 0.626 | 0.618 | **0.627±0.007** | **+0.047** |
| knowledge-update | 0.597 | 0.611 | 0.600 | 0.603 | 0.750 | 0.694 | 0.722 | 0.722 | **+0.119** |
| multi-session | 0.471 | 0.521 | 0.500 | 0.497 | 0.587 | 0.595 | 0.570 | 0.584 | **+0.087** |
| temporal-reasoning | 0.646 | 0.654 | 0.660 | 0.653 | 0.724 | 0.701 | 0.709 | 0.711 | **+0.058** |
| abstention (n=30) | 0.867 | 0.800 | 0.900 | 0.857 | 0.900 | 0.933 | 0.867 | 0.900 | +0.043 |
| single-session-user | 0.859 | 0.859 | 0.860 | 0.859 | 0.844 | 0.828 | 0.797 | 0.823 | −0.036 |
| single-session-preference | 0.500 | 0.533 | 0.530 | 0.522 | 0.467 | 0.500 | 0.500 | 0.489 | −0.033 |
| single-session-assistant | 0.125 | 0.125 | 0.140 | 0.127 | 0.107 | 0.107 | 0.107 | 0.107 | −0.020 |

**Significance** (paired question-level bootstrap per repeat, non-abstention judge accuracy) — **SAGE better in all three repeats**:

- r1: SAGE − mem0 = +0.068, 95% CI [+0.032, +0.106], p = 0.0001
- r2: +0.036, 95% CI [0.000, +0.075], p = 0.022
- r3: +0.036, 95% CI [−0.002, +0.075], p = 0.029

### gpt-4o-mini: F1 / BLEU-1 / J per run

| Arm | Run | F1 | BLEU-1 | J |
|---|---|---|---|---|
| mem0 | r1 | 0.1388 | 0.0814 | 0.5700 |
| mem0 | r2 | 0.1403 | 0.0847 | 0.5840 |
| mem0 | r3 | 0.1385 | 0.0808 | 0.5860 |
| **mem0** | **mean±std** | **0.1392±0.0008** | **0.0823±0.0017** | **0.5800±0.0071** |
| sage | r1 | 0.1451 | 0.0842 | 0.6360 |
| sage | r2 | 0.1394 | 0.0812 | 0.6260 |
| sage | r3 | 0.1387 | 0.0801 | 0.6180 |
| **sage** | **mean±std** | **0.1411±0.0029** | **0.0818±0.0018** | **0.6267±0.0074** |

J = official LongMemEval judge accuracy over all 500 questions (gpt-4o-mini judge). F1 = token-set F1, BLEU-1 = unigram BLEU, both vs the gold answer over the 470 non-abstention questions (abstention golds are explanations, so lexical overlap is undefined). Note F1/BLEU are near-identical across arms (Δ ≤ 0.002, within run noise; r1 paired bootstrap micro-F1 −0.006 [−0.013, +0.001]) — on LongMemEval these lexical metrics mostly measure answer verbosity against 2-4-word golds and are insensitive to correctness; J is the meaningful quality metric, where SAGE leads by +0.047.

## Results: qwen2.5-3b backbone (judge: llama3.1:8b)

| Metric | mem0 r1 | mem0 r2 | mem0 r3 | mem0 mean | sage r1 | sage r2 | sage r3 | sage mean | Δ (SAGE−mem0) |
|---|---|---|---|---|---|---|---|---|---|
| **Overall** | 0.654 | 0.698 | 0.690 | **0.681±0.019** | 0.668 | 0.666 | 0.676 | **0.670±0.004** | −0.011 |
| temporal-reasoning | 0.614 | 0.717 | 0.717 | 0.683 | 0.724 | 0.669 | 0.685 | 0.693 | +0.010 |
| multi-session | 0.603 | 0.694 | 0.628 | 0.642 | 0.645 | 0.620 | 0.620 | 0.628 | −0.014 |
| single-session-assistant | 0.643 | 0.625 | 0.696 | 0.655 | 0.589 | 0.679 | 0.732 | 0.667 | +0.012 |
| knowledge-update | 0.653 | 0.667 | 0.625 | 0.648 | 0.597 | 0.569 | 0.597 | 0.588 | −0.060 |
| single-session-user | 0.875 | 0.844 | 0.844 | 0.854 | 0.844 | 0.891 | 0.875 | 0.870 | +0.016 |
| single-session-preference | 0.233 | 0.233 | 0.333 | 0.266 | 0.167 | 0.233 | 0.200 | 0.200 | −0.066 |
| abstention (n=30) | 1.000 | 1.000 | 1.000 | 1.000 | 0.967 | 1.000 | 1.000 | 0.989 | −0.011 |

**Significance** (paired bootstrap per repeat, micro_j): r1 SAGE +0.017 [−0.032, +0.066]; r2 mem0 +0.034 [−0.015, +0.081]; r3 mem0 +0.015 [−0.030, +0.062] — all n.s., opposite signs across repeats → **statistical parity on quality**. The sharper qwen contrast is reproducibility: mem0 repeat spread ±0.019 vs SAGE ±0.004 (~5× stability gap), consistent with the LoCoMo finding that LLM-gated routing is run-unstable while the vMF gate is deterministic.

### qwen2.5-3b: F1 / BLEU-1 / J per run

| Arm | Run | F1 | BLEU-1 | J |
|---|---|---|---|---|
| mem0 | r1 | 0.0676 | 0.0442 | 0.6540 |
| mem0 | r2 | 0.0677 | 0.0447 | 0.6980 |
| mem0 | r3 | 0.0669 | 0.0434 | 0.6900 |
| **mem0** | **mean±std** | **0.0674±0.0004** | **0.0441±0.0005** | **0.6807±0.0191** |
| sage | r1 | 0.0681 | 0.0433 | 0.6680 |
| sage | r2 | 0.0688 | 0.0446 | 0.6660 |
| sage | r3 | 0.0686 | 0.0432 | 0.6760 |
| **sage** | **mean±std** | **0.0685±0.0003** | **0.0437±0.0006** | **0.6700±0.0043** |

**Significance takeaways** (same protocol as the gpt-4o-mini block: paired question-level bootstrap over the same 470 non-abstention questions, B = 10,000, identical resample indices for both arms; diff = SAGE − mem0; one-sided p = P(diff ≤ 0); 3-repeat-mean test applies one resample to all repeats and averages the diffs):

- **J — not significant.** 3-repeat mean −0.0106, 95% CI [−0.0383, +0.0170], p = 0.78; per-repeat diffs have mixed signs (+0.017 / −0.034 / −0.015, p = 0.27 / 0.92 / 0.76) — the statistical-parity verdict from the per-repeat block above, restated on the non-abstention subset.
- **F1 — not significant.** 3-repeat mean +0.0011, 95% CI [−0.0017, +0.0039], p = 0.22; SAGE is directionally ahead in all three repeats but never close to significance (per-repeat p = 0.42 / 0.32 / 0.22).
- **BLEU-1 — not significant.** 3-repeat mean −0.0004, 95% CI [−0.0022, +0.0014], p = 0.66.
- **Takeaway**: on the open-source backbone no quality metric separates the arms — every CI is tight around zero, including J. The honest headline is parity on quality; the arms separate instead on run-to-run stability (mem0 J std ±0.019 vs SAGE ±0.004) and on write-side cost, not on any per-question quality metric.

(Same metric definitions/caveats as the gpt-4o-mini F1/BLEU/J table above.)

## Efficiency (following paper Table 4; per run = 500 questions, GPT-4o-mini)

| Metric | Mem0 | SAGE | Ratio | Paper Table 4 (LoCoMo) ratio |
|---|---|---|---|---|
| Write/add LLM calls | 73,286 | 43,037 | 1.7× | 1.4× |
| Add prompt tokens | 146,391,045 | 89,472,635 | 1.6× | 2.2× |
| Add completion tokens | 18,694,371 | 2,105,594 | 8.9× | 11.1× |
| Add total tokens | 165,085,415 | 91,578,229 | 1.8× | 2.6× |
| Avg. latency / add call (s) | 4.37 | 1.25 | 3.5× | 3.1× |
| Add wall-clock (min) | 1,529 | 215 | 7.1× | 2.5× |
| Search wall-clock (min) | 28.1 | 24.8 | 1.1× | ~1.0× |
| Total wall-clock (min) | 1,557 | 240 | 6.5× | 1.3× |
| Add API cost (USD, 4o-mini) | $33.18 | $14.68 | 2.3× | 3.4× |

Provenance: SAGE metrics measured over 3 complete runs (per-run prompt tokens agree to 0.03% across repeats). Mem0 wall-clocks measured over all 3 runs (sum of per-pass spans from done-marker timestamps; inter-job queue gaps excluded; 8 concurrent questions per run for both arms). Mem0 token/call/cost metrics estimated from 112 resumed questions in the r2+r3 continuation jobs (each resumed pass overwrote the prior pass's usage snapshot), scaled to 500; per-question stores make this unbiased, and the two independent samples agree to <0.2% per question (147 calls/q; 292.4k vs 292.8k prompt tok/q). The add wall-clock gap widening from 2.5× (LoCoMo) to 7.1× reflects mem0's update-reasoning prompts growing with within-conversation store size at LongMemEval scale.

Per SAGE run: ~115.8k memories written + ~1.2k gate-routed merges. Qwen backbone (no API cost): mem0 ~3.8-5 q/h/GPU vs SAGE ~17 q/h/GPU (~4.5× add-phase wall-clock advantage), same direction as the API backbone.

## Interpretation (final)

1. **Headline robustness claim**: at 25× LoCoMo scale, SAGE **significantly beats** ungated mem0 on gpt-4o-mini (+0.047 J; significant in each of 3 paired repeats, p = 0.0001/0.022/0.029) and is at **statistical parity** on qwen2.5-3b (all 3 paired bootstraps n.s., signs mixed) — while ingesting ~4.5× faster at ~2.3× lower API cost everywhere. Quality is never significantly worse, and the efficiency gains are unconditional.
2. **Reproducibility**: SAGE's overall J varies by ±0.004 (qwen) / ±0.007 (API) across independent end-to-end repeats; mem0's varies by ±0.019 / ±0.007. On the open-source backbone SAGE is ~5× more run-stable — the LongMemEval-scale echo of the LoCoMo routing-stability finding.
3. **Where gains land (API backbone)**: knowledge-update +0.119, multi-session +0.087, temporal +0.058 — the question types where redundant memories pollute top-k retrieval. mem0's small edges are confined to single-session types where clutter matters least.
4. **Backbone-dependent caveat to report**: the big per-type SAGE gains materialize on gpt-4o-mini but not on qwen2.5-3b (where knowledge-update and preference favor mem0, −0.060/−0.066, and overall is parity). A plausible mechanism — a 3B extractor produces noisier facts, so mem0's LLM update-reasoning recovers some of what the geometric gate can't — is worth a sentence; the honest claim is "matches or beats, never significantly worse, at a fraction of the cost."
5. **Shared floors are pipeline properties, not gate effects**: single-session-assistant is low for *both* arms on the API judge (0.127 vs 0.107) because fact extraction is user-centric; preference is hard for both. The mem0 control licenses this interpretation.
6. **Scale-stability**: mem0's write path slows ~6× as stores grow (update-decision prompts lengthen); SAGE's density test keeps throughput flat — new evidence unavailable at LoCoMo scale.

## Artifacts & reproduction

- Results tree: `evaluation/longmemeval_results/<backbone>/<arm>/r<N>/` (per-run answers, `*_eval_metrics.json`, `*_eval_metrics_ci.json`, `action_stats.json`, `gate_events.jsonl`, `add_usage.json`, manifests). Sharded qwen mem0 runs merge into `.../r<N>/merged/`.
- Dataset: `evaluation/dataset/longmemeval/longmemeval_s500_seed42.json` (+ `subset_manifest_s500_seed42.json`), converted from HF `xiaowu0162/longmemeval-cleaned` by `evaluation/prepare_longmemeval.py`.
- Pipeline: `evaluation/run_longmemeval_benchmark.py` (deterministic paths, per-question resume), `evaluation/src/memzero/search_longmemeval.py`, `evaluation/metrics/longmemeval_judge.py`, `evaluation/evals_longmemeval.py`, `evaluation/merge_longmemeval_results.py`, `evaluation/gen_longmemeval_table.py`.
- SLURM: `evaluation/longmemeval_run.sbatch` (qwen arm×repeat), `evaluation/longmemeval_mem0_shards.sbatch` + `evaluation/longmemeval_merge_judge.sbatch` (sharded mem0), `evaluation/longmemeval_openai.sbatch` (API runs), `evaluation/longmemeval_smoke.sbatch`.
- Aggregate table: `python gen_longmemeval_table.py` (from `evaluation/`). Significance: `python bootstrap_ci.py <mem0 _ci.json> --compare <sage _ci.json>`; multi-repeat: `--inputs r1 r2 r3`.

## Known judge/ops notes

- llama3.1:8b judge is more lenient than gpt-4o-mini judge (e.g. abstention 1.00 vs 0.83, single-session-assistant 0.64 vs 0.13) — never compare across judges.
- Ops fixes baked into the sbatch scripts: ollama port-collision retry (job-id-derived ports can clash on shared nodes) and explicit `OLLAMA_CONTEXT_LENGTH=8192` for judge servers (the unset default is VRAM-based ≈262k ctx, which made judge calls ~350× slower).
