# SAGE: A Novelty Gate for Efficient Memory Evolution in Agentic LLMs

[![arXiv](https://img.shields.io/badge/arXiv-2605.30711-b31b1b.svg)](https://arxiv.org/abs/2605.30711)

SAGE is a long-term memory layer for LLM agents that decides what to write
**without an LLM call**. It is built as a fork of [mem0](https://github.com/mem0ai/mem0):
mem0 issues a per-write LLM call to route each candidate fact to ADD / UPDATE / DELETE / NOOP,
whereas SAGE replaces that routing with a **vector-math novelty gate** — a von
Mises–Fisher KDE novelty score compared against an adaptive per-scope threshold. The
fact-extraction step is unchanged, so SAGE is a drop-in alternative to mem0's write
path that issues far fewer write-side LLM calls and generates far fewer tokens.

## Results (LoCoMo, gpt-4o-mini)

Measured on full LoCoMo with a gpt-4o-mini backbone and a gpt-4o-mini LLM-as-judge
(`batch_size=8`), SAGE vs the mem0 baseline:

| | mem0 | SAGE |
|---|---|---|
| Write-side LLM calls | 3,217 | 2,297 |
| Write-side total tokens | 5.55M | 2.16M (**≈61% fewer**) |
| Generated (completion) tokens | 0.91M | 0.08M (**≈11× fewer**) |
| Ingestion wall-clock | 39.3 min | 15.7 min (**≈2.5× faster**) |
| LLM-judge accuracy (macro) | 53.5 | 52.2 (within **≈1.3 points**) |

Write-side counts/tokens are the chat-LLM totals (embedding-API calls excluded). All
of these numbers regenerate from vendored result summaries — see
[`evaluation/README.md`](evaluation/README.md) and
[`evaluation/paper_results/`](evaluation/paper_results/).

## Repository layout

This repo is the mem0 library with the SAGE contribution plus an evaluation harness;
the unrelated mem0 sub-projects (docs site, JS/TS SDKs, server, examples, etc.) have
been removed.

```
mem0/                       # the memory library (mem0 core + SAGE)
  memory/novelty_gate.py    #   SAGE: vMF-KDE novelty scorer + adaptive threshold
  memory/main.py            #   enable_sage write path that uses the gate
  llms/usage_tracker.py     #   write-side LLM call / token / latency instrumentation
evaluation/                 # LOCOMO benchmark harness, run scripts, table generators
  README.md                 #   how to run the benchmark and reproduce the paper tables
  paper_results/            #   vendored result summaries the tables are built from
tests/                      # tests for the library and the SAGE gate
```

## Installation

Requires **Python 3.9+**. Install the dependencies with:

```bash
pip install -r requirements.txt
```

This covers the SAGE/mem0 library, the LOCOMO benchmark harness, and scoring (the
scoring metrics pull in PyTorch via `sentence-transformers`/`bert-score`). The
benchmark runner adds the repo root to `sys.path`, so the local `mem0/` package is
used directly — no install of this repo is needed to run it. Optional baselines
(RAG, LangMem, Zep) and their extra packages are listed, commented out, at the
bottom of [`requirements.txt`](requirements.txt).

To use SAGE as a library in another project, also install it editable:

```bash
pip install -e .
```

This installs the `sage` distribution from this repo's source; the module is still
imported as `mem0` (SAGE is a modified mem0 fork, not the upstream `mem0ai` package —
do not `pip install mem0ai`, it would shadow this fork).

## Using SAGE

SAGE is enabled through the standard mem0 `MemoryConfig`; set `enable_sage` and
configure the vector store / LLM / embedder as you would for mem0:

```python
from mem0 import Memory

memory = Memory.from_config({
    "enable_sage": True,            # route writes through the novelty gate
    "sage_novelty_method": "vmf_kde",  # "vmf_kde" (default) or "gaussian_kde"
    # ... your usual vector_store / llm / embedder config ...
})

memory.add(messages, user_id="alice")          # gated write, no routing LLM call
memory.search(query="...", user_id="alice")    # retrieval is unchanged from mem0
```

With `enable_sage=False`, `Memory` behaves exactly like upstream mem0.

## Reproducing the paper

Quickstart for the headline result (full LoCoMo, gpt-4o-mini backbone + judge, SAGE
vs the mem0 baseline). Put your key in a repo-root `.env` (`OPENAI_API_KEY="sk-..."`).

Run it all in one command (steps 1–3 below), no scheduler needed:

```bash
bash reproduce_main_result.sh
```

Or run the steps individually from `evaluation/`:

```bash
# 1. Run SAGE and mem0 (add + search) at the paper's batch_size=8
bash run_abla_openai.sh mem0 gpt-4o-mini full 8
bash run_abla_openai.sh sage gpt-4o-mini full 8

# 2. Score with the gpt-4o-mini LLM-as-judge
bash score_openai.sh bs8

# 3. View per-category + overall accuracy
python generate_scores.py results_openai_full_bs8 \
  --runs mem0_gpt-4o-mini/eval_metrics_judge4omini.json \
         sage_gpt-4o-mini/eval_metrics_judge4omini.json

# 4. Regenerate the paper's efficiency + accuracy tables
python make_results_tex.py && python gen_token_table.py
```

Step 4 rebuilds the tables from the vendored summaries in
[`evaluation/paper_results/`](evaluation/paper_results/). To rebuild them from your
own fresh runs, and for the open-weight (Ollama) backbones and the optional mem0g
graph baseline, see the full step-by-step in
[`evaluation/README.md`](evaluation/README.md).

## Relationship to mem0 and license

This repository is a **modified fork of [mem0](https://github.com/mem0ai/mem0)**,
licensed under the **Apache License 2.0**. The original copyright is retained in
[`LICENSE`](LICENSE), and the upstream/derivative attribution is recorded in
[`NOTICE`](NOTICE).

## Citation

If you use SAGE in your research, please cite our paper
([arXiv:2605.30711](https://arxiv.org/abs/2605.30711)):

```bibtex
@article{wang2026sage,
  title   = {SAGE: A Novelty Gate for Efficient Memory Evolution in Agentic LLMs},
  author  = {Wang, Sijia and Brahma, Dhanajit and Henao, Ricardo},
  journal = {arXiv preprint arXiv:2605.30711},
  year    = {2026}
}
```
