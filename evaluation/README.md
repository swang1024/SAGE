# SAGE — LOCOMO evaluation

This directory evaluates **SAGE** against the **mem0** baseline on the
[LOCOMO](https://github.com/snap-research/locomo) long-conversation benchmark.

SAGE (adaptive novelty-gated memory) is the contribution of this repository, built
as a fork of [mem0](https://github.com/mem0ai/mem0). On each memory write, mem0
issues an LLM call to decide ADD / UPDATE / DELETE / NOOP; **SAGE replaces that per-write
LLM routing with a vector-math novelty gate** (a von Mises–Fisher KDE novelty score
against an adaptive per-scope threshold), keeping the same fact-extraction step. The
result is far fewer write-side LLM calls and generated tokens at comparable accuracy.

## What is new in this fork

| Component | Path | Purpose |
|---|---|---|
| Novelty gate (the contribution) | [`mem0/memory/novelty_gate.py`](../mem0/memory/novelty_gate.py) | vMF-KDE novelty scorer + adaptive threshold; gates writes without an LLM call. |
| Gate wiring | [`mem0/memory/main.py`](../mem0/memory/main.py) | `enable_sage` path that routes ADD/UPDATE/NOOP via the gate. |
| Write-side instrumentation | [`mem0/llms/usage_tracker.py`](../mem0/llms/usage_tracker.py) | Records LLM calls / tokens / latency into the run's `action_stats` `usage_stats`. |
| Benchmark runner | [`run_locomo_benchmark.py`](run_locomo_benchmark.py) | Unified entrypoint; dispatches on `--technique_type`. |
| Convenience drivers | `run_openai.sh`, `run_opensrc_full*.sh`, `score_openai.sh` | Arg-driven wrappers for the runs in the paper. |
| Table generators + data | `gen_table3_llm_calls.py`, `gen_token_table.py`, `make_results_tex.py`, [`paper_results/`](paper_results/) | Regenerate the paper's tables from committed result summaries. |

The two techniques compared in the paper are `sage` and `mem0`. The other
`--technique_type` values (`rag`, `langmem`, `zep`, `openai`) are inherited mem0
baselines and need extra assets/keys; they are not part of the SAGE results.

## Setup

Create a `.env` at the repo root with at least an OpenAI key (used for the
gpt-4o-mini backbone, embeddings, and the LLM-as-judge):

```
OPENAI_API_KEY="sk-..."
```

For the open-weight runs (Qwen/Llama/DeepSeek) you need a local
[Ollama](https://ollama.com) server instead of an OpenAI key. The `ollama` Python
package is only the client — install the Ollama **runtime** and pull the model
weights separately (neither ships with pip; both are multi-GB and need a **GPU**):

```bash
curl -fsSL https://ollama.com/install.sh | sh   # Linux runtime; macOS: brew install ollama
ollama pull qwen2.5:3b         # backbone LLM
ollama pull nomic-embed-text   # embedder (default --embedding_model for the ollama backend)
ollama pull llama3.1:8b        # open-weight LLM-as-judge (evals.py --judge_backend ollama)
```

The benchmark/judge talk to the server at `--ollama_base_url` (default
`http://127.0.0.1:11434`; override the host via `OLLAMA_HOST`). The one-command
[`reproduce_openweight_result.sh`](../reproduce_openweight_result.sh) starts the
server and pulls all three models for you — see **Reproducing the paper tables** below.

### Dataset

`dataset/` ships the LOCOMO splits used here:

- `locomo10.json` — full LOCOMO (all 10 conversations, all QA)
- `locomo10_smoke_20perc.json` — all 10 conversations, ~20% of the QA
- `locomo10_smoke.json` — 1% QA smoke test

## Step-by-step: reproduce the main result (gpt-4o-mini, SAGE vs mem0)

This reproduces the paper's headline **efficiency** and **accuracy** tables on full
LoCoMo with a gpt-4o-mini backbone and a gpt-4o-mini LLM-as-judge. Run everything
from `evaluation/`. The key setting is **`batch_size=8`** — the paper's
configuration (one shared fact-extraction call per `add`); the driver tags these
outputs with a `_bs8` suffix so they don't collide with other runs.

> **One command:** `bash reproduce_main_result.sh` (from the repo root) runs steps
> 2–4 below end-to-end. The individual steps follow for when you want more control.

**1. Install and key.** Install the package + eval extras (top-level README:
`uv pip install -e ".[eval]"`) and put your OpenAI key in a repo-root `.env`:

```
OPENAI_API_KEY="sk-..."
```

**2. Run SAGE and the mem0 baseline** (each does add **and** search):

```bash
# run_openai.sh <tech> <model> <split> [batch_size]
bash run_openai.sh mem0 gpt-4o-mini full 8
bash run_openai.sh sage gpt-4o-mini full 8
```

Each writes answers and an instrumented `*_action_stats_*.json` (carrying
`summary.usage_stats`) to `results_openai_full_bs8/{mem0,sage}_gpt-4o-mini/`.
Add-phase wall-clock is ~40 min (mem0) / ~16 min (SAGE) on the paper's setup.

**3. Score with the LLM-as-judge** (gpt-4o-mini judge) — writes
`eval_metrics_judge4omini.json` into each run dir:

```bash
bash score_openai.sh bs8
```

**4. View per-category + overall scores** (BLEU / F1 / LLM-judge):

```bash
python generate_scores.py results_openai_full_bs8 \
  --runs mem0_gpt-4o-mini/eval_metrics_judge4omini.json \
         sage_gpt-4o-mini/eval_metrics_judge4omini.json
```

**5. Regenerate the paper's LaTeX tables.** The generators read the committed
result summaries in [`paper_results/`](paper_results/). To rebuild them from *your* fresh
runs, copy the four files over the committed ones, then run the generators:

```bash
M=results_openai_full_bs8/mem0_gpt-4o-mini; S=results_openai_full_bs8/sage_gpt-4o-mini
cp $M/*_action_stats_*.json      paper_results/full_mem0_gpt-4o-mini/mem0_gpt-4o-mini_action_stats.json
cp $M/eval_metrics_judge4omini.json paper_results/full_mem0_gpt-4o-mini/
cp $S/*_action_stats_*.json      paper_results/full_sage_gpt-4o-mini/sage_gpt-4o-mini_action_stats.json
cp $S/eval_metrics_judge4omini.json paper_results/full_sage_gpt-4o-mini/

python gen_token_table.py     # write-side token / latency / cost table
python make_results_tex.py    # accuracy + efficiency tables -> paper_results/*.tex
```

Also refresh `paper_results/measured_latency.json` with the add/search/total minutes
from your run's `[TIMER]` log lines (used for the wall-clock rows). See
[`paper_results/README.md`](paper_results/README.md) for what each committed file is,
and the **Reproducing the paper tables** section below for the open-weight table.

> **Other backbones (out of scope for the main result).** Open-weight runs use the
> Ollama driver — `bash run_opensrc_full.sh sage qwen2.5:3b` then
> `bash run_opensrc_full_eval.sh sage qwen2.5:3b` (point `OLLAMA_HOST` at your server,
> default `127.0.0.1:11434`). The optional mem0g graph baseline adds `--is_graph` and
> needs `pip install kuzu` (see the parameter table).
>
> **Ollama speed knobs.** To reproduce the open-weight runs at the paper's wall-clock,
> the Ollama *server* must be tuned. The two scripts above export sensible defaults —
> `OLLAMA_NUM_PARALLEL=4`, `OLLAMA_KEEP_ALIVE=24h`, `OLLAMA_CONTEXT_LENGTH=8192`,
> `OLLAMA_FLASH_ATTENTION=true` (all env-overridable). These are read by `ollama serve`,
> so they only take effect for a server launched from the same environment; if you run a
> separate `ollama serve`, set them there. Without tuning the runs still produce the same
> answers (and the same Table 3 call counts at `batch_size=8`) — just slower.

### Key parameters

Only the parameters that matter for SAGE / mem0 local runs are listed; run
`python run_locomo_benchmark.py -h` for the full set.

| Parameter | Description | Default |
|---|---|---|
| `--technique_type` | `sage` (the gate) or `mem0` (per-write LLM routing baseline) | `mem0` |
| `--method` | `add`, `search`, or `full` (add then search) | `add` |
| `--mem0_backend` | `openai` (gpt-4o family) or `ollama` (open-weight) | `cloud` |
| `--llm_model` | Backbone model (e.g. `gpt-4o-mini`, `qwen2.5:3b`) | `llama3.2` |
| `--embedding_model` / `--embedding_dims` | Embedding model + dimensionality | `nomic-embed-text` / 768 |
| `--infer_add` | Enable inferred ADD/UPDATE/DELETE actions (required for both `sage` and the mem0 baseline) | off |
| `--batch_size` | Messages per add call. The paper's runs (gpt-4o-mini main result and open-weight Table 3) use `8`; the runner default is `2` | 2 |
| `--max_workers` | Concurrent add workers | 10 |
| `--top_k` | Memories retrieved per query at search time | 30 |
| `--action_stats_file` | Where to write per-run action counts + `usage_stats` (token/latency instrumentation) | `results/...` |
| `--qdrant_path` / `--history_db_path` / `--collection_name` | Per-run vector store / history DB / collection (a timestamp+PID is auto-appended for isolation) | `/tmp/...` |

SAGE's gate hyperparameters (`tau_0`, `tau_min`, `density_lambda`, …) are fixed in
`mem0/memory/main.py` to the values reported in the paper; compaction is disabled in
`mem0/memory/novelty_gate.py` (out of scope for this work).

## Scoring and metrics

Steps 3–4 above score and aggregate the OpenAI runs. To score a single answer file
directly (e.g. an open-weight run):

```bash
python evals.py --input_file <answers.json> --output_file <eval_metrics.json>
```

**Metrics:** BLEU and F1 (lexical overlap with the gold answer), an LLM-judge binary
correctness score, and — for write efficiency — LLM call count, token consumption,
and latency from the `usage_tracker.py` instrumentation. Overall accuracy is the
**macro** average over the four QA categories (mean of per-category means), as in the
paper.

## Reproducing the paper tables

The three generators rebuild the paper's LaTeX tables from the **committed result
summaries** under [`paper_results/`](paper_results/), so they regenerate from the repo
alone without the (gitignored, multi-GB) raw `results_*/` dirs. See
[`paper_results/README.md`](paper_results/README.md) for what each input file is.

```bash
python make_results_tex.py       # accuracy + efficiency tables (gpt-4o-mini) -> paper_results/*.tex
python gen_token_table.py        # write-side token / latency / cost table (gpt-4o-mini)
python gen_table3_llm_calls.py   # write-side LLM-call counts across open-weight models (Table 3)
```

`make_results_tex.py` and `gen_token_table.py` use the gpt-4o-mini runs (step 5 above);
`gen_table3_llm_calls.py` reads the open-weight per-model counts in
`paper_results/action_stats_analysis.csv`. The token/call generators report the
**chat-LLM** sub-totals — the tracker's top-level counts also include embedding-API
calls. SAGE is run with compaction disabled (the paper configuration).

## License

Apache License 2.0 — see [`LICENSE`](../LICENSE). This repository is a modified
fork of [mem0](https://github.com/mem0ai/mem0) (Apache-2.0); the original copyright
notice is retained in `LICENSE`.
