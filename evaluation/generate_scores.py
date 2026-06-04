#!/usr/bin/env python3
"""Aggregate LLM-judge / BLEU / F1 scores across one or more eval_metrics runs.

For each run it prints per-category and overall scores, then summarises the
mean / std / variance of each metric across all runs.

Convenience: point it at a results folder and it auto-discovers every
``*_eval_metrics.json`` inside, so you rarely need to list files by hand.

Examples:
  # Auto-discover every eval_metrics file in a folder
  python generate_scores.py results_full

  # Only specific runs (filenames relative to --output_folder, or absolute paths)
  python generate_scores.py results_full --runs run_a_eval_metrics.json run_b_eval_metrics.json

  # Different discovery pattern (e.g. only the gpt-4o-mini runs)
  python generate_scores.py results_openai_full --glob '*gpt-4o-mini*_eval_metrics.json'

With no arguments it falls back to the DEFAULTS block below — edit those two
values for a fixed setup, or just pass the folder on the command line.
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

# Repo root = parent of this evaluation/ directory; used to resolve relative folders.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- DEFAULTS (used only when nothing is passed on the command line) ----------
# Edit these for a fixed setup, or override with CLI args. A relative path is
# resolved against the repo root.
DEFAULT_OUTPUT_FOLDER = "results_full"
DEFAULT_GLOB = "*_eval_metrics.json"
DEFAULT_RUNS = []  # empty -> auto-discover via DEFAULT_GLOB
EXCLUDE_CATEGORY = 5  # category dropped from the aggregates (set to None to keep all)
# -----------------------------------------------------------------------------


def resolve_folder(folder):
    """Allow either an absolute path or one relative to the repo root."""
    return folder if os.path.isabs(folder) else os.path.join(REPO_ROOT, folder)


def discover_runs(folder, pattern):
    """Return eval_metrics filenames (basenames) found in folder, sorted."""
    matches = sorted(glob.glob(os.path.join(folder, pattern)))
    return [os.path.basename(p) for p in matches]


def score_run(path):
    """Compute per-category table and overall means for one eval_metrics file."""
    with open(path, "r") as f:
        data = json.load(f)

    all_items = []
    for key in data:
        all_items.extend(data[key])

    df = pd.DataFrame(all_items)
    df["category"] = pd.to_numeric(df["category"])
    if EXCLUDE_CATEGORY is not None:
        df = df[df["category"] != EXCLUDE_CATEGORY]

    by_cat = df.groupby("category").agg(
        {"bleu_score": "mean", "f1_score": "mean", "llm_score": "mean"}
    ).round(4)
    by_cat["count"] = df.groupby("category").size()
    overall = df.agg({"bleu_score": "mean", "f1_score": "mean", "llm_score": "mean"}).round(4)
    return by_cat, overall, len(df)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output_folder", nargs="?", default=DEFAULT_OUTPUT_FOLDER,
                        help=f"Folder with eval_metrics JSONs (abs, or relative to repo root). Default: {DEFAULT_OUTPUT_FOLDER}")
    parser.add_argument("--runs", nargs="+", default=DEFAULT_RUNS or None,
                        help="Explicit eval_metrics files (relative to output_folder or absolute). "
                             "If omitted, auto-discover with --glob.")
    parser.add_argument("--glob", dest="pattern", default=DEFAULT_GLOB,
                        help=f"Discovery pattern when --runs is omitted. Default: {DEFAULT_GLOB}")
    args = parser.parse_args()

    folder = resolve_folder(args.output_folder)
    if args.runs:
        runs = args.runs
    else:
        runs = discover_runs(folder, args.pattern)
        if not runs:
            parser.error(f"no files matching {args.pattern!r} in {folder}")
        print(f"Discovered {len(runs)} run(s) in {folder} matching {args.pattern!r}")

    run_means = []
    for fname in runs:
        path = fname if os.path.isabs(fname) else os.path.join(folder, fname)
        by_cat, overall, n = score_run(path)

        print(f"\n{'='*60}")
        print(f"Run: {os.path.basename(path)}")
        print(f"{'='*60}")
        print("Scores by category:")
        print(by_cat.to_string())
        print(f"\nOverall (N={n}):")
        print(f"  BLEU:         {overall['bleu_score']:.4f}")
        print(f"  F1:           {overall['f1_score']:.4f}")
        print(f"  LLM-as-judge: {overall['llm_score']:.4f}")

        run_means.append({
            "run": os.path.basename(path),
            "bleu": overall["bleu_score"],
            "f1": overall["f1_score"],
            "llm": overall["llm_score"],
        })

    if len(run_means) > 1:
        print(f"\n{'='*60}")
        print(f"VARIANCE ACROSS {len(run_means)} RUNS")
        print(f"{'='*60}")
        for metric, key in [("BLEU", "bleu"), ("F1", "f1"), ("LLM-as-judge", "llm")]:
            vals = [r[key] for r in run_means]
            print(f"  {metric:14s}: mean={np.mean(vals):.4f}  std={np.std(vals):.4f}  "
                  f"var={np.var(vals):.6f}  [min={min(vals):.4f}, max={max(vals):.4f}]")


if __name__ == "__main__":
    main()
