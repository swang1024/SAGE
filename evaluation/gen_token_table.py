#!/usr/bin/env python3
"""Build the write-side token/latency comparison table from instrumented runs.

Scans action-stats JSONs (which now carry summary.usage_stats, written by the
instrumented harness) and tabulates total LLM calls, prompt/completion/total
tokens, tokens-per-call, and latency for each technique x model. Emits both a
human-readable table and ready-to-paste LaTeX rows comparing SAGE/sage vs mem0.

Usage:
  python gen_token_table.py path/to/*action_stats*.json   # explicit globs
  python gen_token_table.py            # defaults to evaluation/paper_results/full_*/
"""
import glob
import json
import os
import sys

# USD per 1K tokens (prompt, completion). Update if pricing changes.
PRICING = {
    "gpt-4o": (0.0025, 0.0100),
    "gpt-4o-mini": (0.000150, 0.000600),
}


def chat_usage(u):
    """Return chat-LLM-only usage from a usage_stats dict.

    The tracker increments one shared counter for *every* API call, and it is
    called from both the chat LLM and the embedder. So the top-level
    ``llm_calls``/``total_tokens`` also count embedding calls (recorded under the
    embedding model, which emits zero completion tokens). The paper's write-side
    numbers are the chat model only, so select the per_model entry that produces
    completion tokens and report its sub-totals. Falls back to the top-level
    aggregate if there is no per_model split (older runs).
    """
    pm = u.get("per_model") or {}
    chat = [v for v in pm.values() if v.get("completion_tokens", 0) > 0]
    if not chat:
        return u
    c = max(chat, key=lambda v: v["completion_tokens"])
    calls = c["calls"]
    return {
        "llm_calls": calls,
        "prompt_tokens": c["prompt_tokens"],
        "completion_tokens": c["completion_tokens"],
        "total_tokens": c["total_tokens"],
        "tokens_per_call": (c["total_tokens"] / calls) if calls else 0.0,
        "avg_latency_s": (c.get("total_latency", 0.0) / calls) if calls else 0.0,
    }


def infer_meta(path):
    name = os.path.basename(path).lower()
    tech = "sage" if ("sage" in name or "ang-mem" in name) else "mem0"
    model = "unknown"
    for m in ("gpt-4o-mini", "gpt-4o"):
        if m in name:
            model = m
            break
    return tech, model


def cost(model, prompt_t, completion_t):
    if model not in PRICING:
        return None
    pin, pout = PRICING[model]
    return prompt_t / 1000 * pin + completion_t / 1000 * pout


def main(argv):
    # Default to the vendored full-LOCOMO action-stats next to this script; override via argv.
    # (Only the full runs carry summary.usage_stats from the instrumented harness; the
    # 20% runs predate it.)
    _here = os.path.dirname(os.path.abspath(__file__))
    default = os.path.join(_here, "paper_results", "full_*", "*action_stats*.json")
    patterns = argv or [default]
    paths = sorted({p for pat in patterns for p in glob.glob(pat)})
    if not paths:
        print("No action-stats JSONs matched.", file=sys.stderr)
        return 1

    rows = []
    for p in paths:
        try:
            data = json.load(open(p))
        except Exception as e:  # noqa: BLE001
            print(f"skip {p}: {e}", file=sys.stderr)
            continue
        u = data.get("summary", {}).get("usage_stats")
        if not u or not u.get("llm_calls"):
            continue
        u = chat_usage(u)  # chat-LLM only; drop embedding-API calls from the counts
        tech, model = infer_meta(p)
        rows.append((tech, model, u, p))

    if not rows:
        print("No usage_stats found. Re-run with the instrumented harness.", file=sys.stderr)
        return 1

    hdr = f"{'tech':9} {'model':13} {'calls':>6} {'prompt':>9} {'compl':>9} {'total':>9} {'tok/call':>9} {'avg_lat_s':>9} {'cost$':>7}"
    print(hdr)
    print("-" * len(hdr))
    for tech, model, u, _ in sorted(rows, key=lambda r: (r[1], r[0])):
        c = cost(model, u["prompt_tokens"], u["completion_tokens"])
        print(
            f"{tech:9} {model:13} {u['llm_calls']:6d} {u['prompt_tokens']:9d} "
            f"{u['completion_tokens']:9d} {u['total_tokens']:9d} {u['tokens_per_call']:9.1f} "
            f"{u['avg_latency_s']:9.3f} {('%.3f'%c) if c is not None else 'n/a':>7}"
        )

    # LaTeX rows: pair sage vs mem0 per model, with reduction.
    print("\n% --- LaTeX rows (per model: sage then mem0, with token reduction) ---")
    by = {(t, m): u for t, m, u, _ in rows}
    models = sorted({m for _, m, _, _ in rows})
    for m in models:
        a, z = by.get(("sage", m)), by.get(("mem0", m))
        if not (a and z):
            continue
        red = 100 * (z["total_tokens"] - a["total_tokens"]) / z["total_tokens"]
        print(f"\\multirow{{2}}{{*}}{{{m}}}")
        print(f"  & \\method{{}} & {a['llm_calls']} & {a['total_tokens']:,} & {a['tokens_per_call']:.0f} & {a['avg_latency_s']:.2f} & {red:.0f}\\% \\\\")
        print(f"  & mem0 & {z['llm_calls']} & {z['total_tokens']:,} & {z['tokens_per_call']:.0f} & {z['avg_latency_s']:.2f} & --- \\\\")
        print("\\midrule")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
