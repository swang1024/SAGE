#!/usr/bin/env python3
"""Regenerate the write-side LLM-call table (Table 3) from action_stats_analysis.csv.

Fair accounting:
  * Extraction LLM calls are shared by all systems (one per `add` call) and held fixed.
  * Decision stage: SAGE/sage makes 0 routing LLM calls (vMF gate) + one merge call
    per UPDATE; mem0/mem0g make one batched routing+edit call per non-empty add call.
  * Total write-side LLM calls = extraction + decision-stage calls.
  * pi_upd = UPDATE / (ADD + UPDATE), the empirical share routed to the UPDATE band.

Usage: python gen_table3_llm_calls.py [path/to/action_stats_analysis.csv]
"""
import csv
import os
import sys

# Default to the vendored analysis CSV next to this script; override via argv[1].
_HERE = os.path.dirname(os.path.abspath(__file__))
CSV = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    _HERE, "paper_results", "action_stats_analysis.csv"
)
EXTRACT = 1696  # add_api_calls: one fact-extraction LLM call per add(), identical for all systems

DISP = {
    "deepseek-r1-1.5b": "DeepSeek-R1-1.5b", "deepseek-r1-7b": "DeepSeek-R1-7b",
    "llama3.2-1b": "Llama-3.2-1b", "llama3.2-3b": "Llama-3.2-3b",
    "qwen2.5-1.5b": "Qwen2.5-1.5b", "qwen2.5-3b": "Qwen2.5-3b", "qwen2.5-7b": "Qwen2.5-7b",
}
ORDER = list(DISP)

rows = list(csv.DictReader(open(CSV)))
by = {(r["model"], r["system"]): r for r in rows}
gi = lambda r, k: int(r[k])

for m in ORDER:
    a, z, g = by[(m, "SAGE")], by[(m, "mem0")], by[(m, "mem0g")]
    au, aadd = gi(a, "update"), gi(a, "add")
    a_tot = EXTRACT + au
    pi = 100 * au / (aadd + au) if (aadd + au) else 0.0
    for label, r in (("\\method{}", a), ("mem0", z), ("mem0g", g)):
        if label == "\\method{}":
            print(f"  & {label} & \\textbf{{0}} & {au} & {EXTRACT} & \\textbf{{{a_tot}}} & --- & {pi:.1f} \\\\")
        else:
            route = gi(r, "non_empty_calls")
            tot = EXTRACT + route
            red = 100 * (tot - a_tot) / tot
            print(f"  & {label} & {route} & --- & {EXTRACT} & {tot} & {red:.0f}\\% & --- \\\\")
    print("\\midrule")
