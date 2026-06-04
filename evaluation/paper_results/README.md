# Vendored paper-table inputs

This directory holds the **minimal derived results** needed to regenerate the
paper's LaTeX tables from the repo alone. The raw run outputs (`results_*/`,
multi-GB) are gitignored and not shipped; the small JSON/CSV files here are the
aggregated artifacts the table generators actually read.

All numbers are for the **gpt-4o-mini** backbone with a **gpt-4o-mini LLM-as-judge**
on LoCoMo. The `full_*` runs use `batch_size=8` (the paper configuration: one shared
fact-extraction call per `add`). SAGE is run with compaction disabled.

## Contents

| Path | Used by | What it is |
|---|---|---|
| `action_stats_analysis.csv` | `gen_table3_llm_calls.py` | Per-(model×system) action counts for the 7 open-weight models (SAGE / mem0 / mem0g); source of the write-side **LLM-call** table (Table 3). |
| `full_{mem0,sage}_gpt-4o-mini/eval_metrics_judge4omini.json` | `make_results_tex.py` | Scored per-QA judge/BLEU/F1, full LoCoMo. |
| `full_{mem0,sage}_gpt-4o-mini/*_action_stats.json` | `make_results_tex.py`, `gen_token_table.py` | Add-phase `summary.usage_stats` + `action_counts` (tokens/latency/store size). |
| `measured_latency.json` | `make_results_tex.py` | Wall-clock (min) from the run logs' `[TIMER]` lines (add/search/total), full LoCoMo. |

> **Chat vs. embedding calls.** The tracker's top-level `usage_stats.llm_calls` /
> `total_tokens` also count embedding-API calls (recorded under the embedding model,
> which has zero completion tokens). `gen_token_table.py` and `make_results_tex.py`
> report the **chat model's** `per_model` sub-totals, which is what the paper uses.
>
> The `20perc_*` dirs are leftover 20% scored QA from the (now-removed) compaction
> trajectory table; no generator reads them anymore.

## Regenerate the tables

```bash
python evaluation/make_results_tex.py            # accuracy + efficiency .tex (this dir)
python evaluation/gen_token_table.py             # token / latency / cost table (stdout)
python evaluation/gen_table3_llm_calls.py        # open-weight write-side LLM-call counts (stdout)
```

`make_results_tex.py` writes `results_summary.tex` and `results_summary_twocol.tex`
into this directory (generated artifacts; safe to delete and recreate). The other two
print LaTeX rows to stdout. All three default to the files here and accept an explicit
path/glob to point at fresh runs instead.
