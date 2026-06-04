#!/usr/bin/env python
"""Generate a LaTeX results summary (accuracy + efficiency) for the
mem0 vs SAGE (sage, no-compaction, v3-merge) comparison on LOCOMO.

Numbers are pulled directly from the scored eval files and the add-phase
usage_stats in the action-stats files. Measured wall-clock latencies (from the
SLURM .out [TIMER] lines) and store sizes (from action_counts) are embedded as
documented constants. Run from the repo root:

    python evaluation/make_results_tex.py
"""
import glob
import json
import os
from collections import defaultdict

# Inputs are vendored under evaluation/paper_results/ so the tables regenerate
# from the repo alone (the raw results_* dirs are gitignored and not shipped).
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper_results")
OUT = os.path.join(DATA, "results_summary.tex")
OUT2 = os.path.join(DATA, "results_summary_twocol.tex")

CATS = {"1": "single-hop", "2": "multi-hop", "3": "temporal", "4": "open-domain"}
CAT_ORDER = ["1", "2", "3", "4", "all"]

EVAL = {
    "mem0_full":  "full_mem0_gpt-4o-mini/eval_metrics_judge4omini.json",
    "sage_full":  "full_sage_gpt-4o-mini/eval_metrics_judge4omini.json",
}
ACTION = {
    "mem0_full": "full_mem0_gpt-4o-mini/*action_stats*.json",
    "sage_full": "full_sage_gpt-4o-mini/*action_stats*.json",
}

# Measured wall-clock (min) from the run logs' [TIMER] lines (full LOCOMO, gpt-4o-mini).
# Documented as data in paper_results/measured_latency.json; the dict below is the
# fallback if that file is absent.
_LAT_FALLBACK = {
    "mem0_full": {"add_min": 39.26, "search_min": 66.72, "total_min": 106.32},
    "sage_full": {"add_min": 15.65, "search_min": 65.46, "total_min": 81.54},
}
_lat_path = os.path.join(DATA, "measured_latency.json")
LAT = json.load(open(_lat_path)) if os.path.exists(_lat_path) else _LAT_FALLBACK


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def cat_mean(by, c, key):
    # "all" = macro average (mean of the four per-category means), matching the
    # paper's "Average" column; individual categories are plain means.
    if c == "all":
        return mean([mean(by[cc][key]) for cc in CATS])
    return mean(by[c][key])


def agg(path):
    d = json.load(open(os.path.join(DATA, path)))
    by = defaultdict(lambda: {"j": [], "b": [], "f": []})
    for _conv, items in d.items():
        for it in items:
            c = str(it.get("category"))
            if c == "5":
                continue
            for t in (c, "all"):
                by[t]["j"].append(it["llm_score"])
                by[t]["b"].append(it["bleu_score"])
                by[t]["f"].append(it["f1_score"])
    return by


def usage(patt):
    fp = sorted(glob.glob(os.path.join(DATA, patt)))[-1]
    d = json.load(open(fp))
    u = d["summary"]["usage_stats"]
    # The tracker's top-level llm_calls/total_tokens also count embedding-API calls
    # (recorded under the embedding model, which has zero completion tokens). The
    # paper's write-side numbers are the chat model only -> use the per_model entry
    # that emits completion tokens. Fall back to the aggregate if no per_model split.
    pm = u.get("per_model") or {}
    chat = [v for v in pm.values() if v.get("completion_tokens", 0) > 0]
    if chat:
        c = max(chat, key=lambda v: v["completion_tokens"])
        calls = c["calls"]
        u = {
            "llm_calls": calls,
            "prompt_tokens": c["prompt_tokens"],
            "completion_tokens": c["completion_tokens"],
            "total_tokens": c["total_tokens"],
            "avg_latency_s": (c.get("total_latency", 0.0) / calls) if calls else 0.0,
        }
    return u, d["summary"]["action_counts"]


def store_size(ac):
    return ac.get("ADD", 0) - ac.get("DELETE", 0)


def main():
    mem, sage = agg(EVAL["mem0_full"]), agg(EVAL["sage_full"])
    mu, mac = usage(ACTION["mem0_full"])
    su, sac = usage(ACTION["sage_full"])
    m_store, s_store = store_size(mac), store_size(sac)

    n = {c: (len(mem[c]["j"]) if c != "all" else sum(len(mem[cc]["j"]) for cc in CATS)) for c in CAT_ORDER}
    add_cost = lambda u: u["prompt_tokens"] / 1e6 * 0.15 + u["completion_tokens"] / 1e6 * 0.60
    eff_rows = [
        ("Write/add LLM calls",            f"{mu['llm_calls']:,}",          f"{su['llm_calls']:,}"),
        ("Add prompt tokens",              f"{mu['prompt_tokens']:,}",      f"{su['prompt_tokens']:,}"),
        ("Add completion tokens",          f"{mu['completion_tokens']:,}",  f"{su['completion_tokens']:,}"),
        ("Add total tokens",               f"{mu['total_tokens']:,}",       f"{su['total_tokens']:,}"),
        ("Avg.\\ latency / add call (s)",  f"{mu['avg_latency_s']:.2f}",    f"{su['avg_latency_s']:.2f}"),
        ("Add wall-clock (min)",           f"{LAT['mem0_full']['add_min']:.1f}",    f"{LAT['sage_full']['add_min']:.1f}"),
        ("Search wall-clock (min)",        f"{LAT['mem0_full']['search_min']:.1f}", f"{LAT['sage_full']['search_min']:.1f}"),
        ("Total wall-clock (min)",         f"{LAT['mem0_full']['total_min']:.1f}",  f"{LAT['sage_full']['total_min']:.1f}"),
        ("Final store size (memories)",    f"{m_store:,}",                  f"{s_store:,}"),
        ("Add tokens / stored memory",     f"{mu['total_tokens'] / m_store:.0f}", f"{su['total_tokens'] / s_store:.0f}"),
        ("Add API cost (USD, 4o-mini)",    f"\\${add_cost(mu):.2f}",        f"\\${add_cost(su):.2f}"),
    ]
    # --- table builders (size: "" or "\\small"; wide: span both columns via table*) ---
    def acc_table(wide=False, size=""):
        env = "table*" if wide else "table"
        t = [r"\begin{%s}[t]" % env, r"\centering"]
        if size:
            t.append(size)
        t += [
            r"\caption{Accuracy on full LOCOMO (gpt-4o-mini backbone, gpt-4o-mini "
            r"judge, $N{=}%d$ scored QA). SAGE uses compaction disabled and "
            r"the lossless-merge prompt. \textbf{Bold} marks the better value.}" % n["all"],
            r"\label{tab:locomo-accuracy}",
            r"\begin{tabular}{l r rr rr rr}", r"\toprule",
            r" & & \multicolumn{2}{c}{LLM-judge (\%)} & \multicolumn{2}{c}{BLEU} "
            r"& \multicolumn{2}{c}{F1} \\",
            r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
            r"Category & $N$ & mem0 & SAGE & mem0 & SAGE & mem0 & SAGE \\", r"\midrule",
        ]
        for c in CAT_ORDER:
            row = [CATS[c] if c != "all" else "Overall", f"{n[c]}"]
            for key in ("j", "b", "f"):
                mv, sv = cat_mean(mem, c, key) * 100, cat_mean(sage, c, key) * 100
                if mv >= sv:
                    row += [r"\textbf{%.1f}" % mv, "%.1f" % sv]
                else:
                    row += ["%.1f" % mv, r"\textbf{%.1f}" % sv]
            if c == "4":
                t.append(r"\midrule")
            if c == "all":
                row[0] = r"\textbf{Overall}"
            t.append(" & ".join(row) + r" \\")
        t += [r"\bottomrule", r"\end{tabular}", r"\end{%s}" % env]
        return t

    def eff_table(size=""):
        t = [r"\begin{table}[t]", r"\centering"]
        if size:
            t.append(size)
        t += [
            r"\caption{Efficiency on full LOCOMO (gpt-4o-mini). Add-phase token counts "
            r"and per-call latency are measured at the API boundary; wall-clock times are "
            r"from the run logs. SAGE's novelty gate replaces mem0's per-add LLM "
            r"update-reasoning with a vector-math decision, yielding far fewer generated "
            r"tokens.}",
            r"\label{tab:locomo-efficiency}",
            r"\begin{tabular}{l r r}", r"\toprule", r"Metric & mem0 & SAGE \\", r"\midrule",
        ]
        for label, m, s in eff_rows:
            t.append(f"{label} & {m} & {s} " + r"\\")
        t += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        return t

    header = [
        "% Auto-generated by evaluation/make_results_tex.py",
        "% mem0 vs SAGE (no-compaction, v3 lossless-merge prompt)",
        "% LOCOMO, gpt-4o-mini backbone, gpt-4o-mini LLM-as-judge",
        r"% Requires \usepackage{booktabs} in the preamble.",
        "",
    ]

    # ---- File 1: single-column tables, no prose (original layout) ----
    L = header + acc_table() + [""] + eff_table() + [""]
    with open(OUT, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"wrote {OUT}  ({len(L)} lines)")

    # ---- File 2: two-column-paper friendly + takeaway paragraph ----
    o_j, s_j = cat_mean(mem, "all", "j") * 100, cat_mean(sage, "all", "j") * 100
    mh_o, mh_s = cat_mean(mem, "2", "j") * 100, cat_mean(sage, "2", "j") * 100
    sh_o, sh_s = cat_mean(mem, "1", "j") * 100, cat_mean(sage, "1", "j") * 100
    tok_ratio = mu["total_tokens"] / su["total_tokens"]
    comp_ratio = mu["completion_tokens"] / su["completion_tokens"]
    add_speed = LAT["mem0_full"]["add_min"] / LAT["sage_full"]["add_min"]
    store_pct = (s_store - m_store) / m_store * 100
    takeaway = (
        r"\paragraph{Results summary.}"
        r"On full LOCOMO with a gpt-4o-mini backbone (gpt-4o-mini LLM-as-judge), "
        rf"SAGE reaches {s_j:.1f}\% LLM-judge accuracy versus mem0's {o_j:.1f}\% "
        rf"(${s_j - o_j:+.1f}$ points; Table~\ref{{tab:locomo-accuracy}}), and "
        rf"\emph{{wins}} multi-hop ({mh_s:.1f}\% vs.\ {mh_o:.1f}\%). The residual gap is "
        rf"concentrated in single-hop recall ({sh_s:.1f}\% vs.\ {sh_o:.1f}\%), where "
        r"facts dropped at ingest cannot be recovered downstream. "
        rf"SAGE attains this at {add_speed:.1f}$\times$ faster ingestion and "
        rf"{tok_ratio:.1f}$\times$ fewer write-side tokens "
        rf"({comp_ratio:.1f}$\times$ fewer \emph{{generated}} tokens), because its "
        r"novelty gate is a vector-math decision rather than a per-add LLM "
        r"update-reasoning call (Table~\ref{tab:locomo-efficiency}); the trade-off is a "
        rf"{store_pct:.0f}\% larger memory store, as compaction is left to future work."
    )
    L2 = header + [
        "% Two-column layout: wide accuracy table spans both columns (table*);",
        "% the narrower tables use single-column table with \\small.",
        "",
        takeaway,
        "",
    ] + acc_table(wide=True, size=r"\small") + [""] + eff_table(size=r"\small") + [""]
    with open(OUT2, "w") as f:
        f.write("\n".join(L2) + "\n")
    print(f"wrote {OUT2}  ({len(L2)} lines)")


if __name__ == "__main__":
    main()
