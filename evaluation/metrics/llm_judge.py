import argparse
import json
from collections import defaultdict

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from mem0.memory.utils import extract_json

import os

load_dotenv()

# LLM-as-judge backend is selectable so it can match the experiment's backend:
#   LLM_JUDGE_BACKEND=openai  -> real OpenAI, default model gpt-4o-mini
#   LLM_JUDGE_BACKEND=ollama  -> local Ollama OpenAI-compatible endpoint, default llama3.1:8b
# Override the model explicitly with LLM_JUDGE_MODEL. The client is built lazily on
# first use so callers (e.g. evals.py) can set these env vars before the first judge call.
DEFAULT_JUDGE_MODELS = {"openai": "gpt-4o-mini", "ollama": "llama3.1:8b"}

_JUDGE_CLIENT = None
_JUDGE_MODEL = None


def _resolve_judge():
    """Build (and cache) the judge client + model from LLM_JUDGE_BACKEND / LLM_JUDGE_MODEL."""
    global _JUDGE_CLIENT, _JUDGE_MODEL
    if _JUDGE_CLIENT is not None:
        return _JUDGE_CLIENT, _JUDGE_MODEL

    backend = os.environ.get("LLM_JUDGE_BACKEND", "openai").lower()
    if backend == "openai":
        # Reads OPENAI_API_KEY (and optional OPENAI_BASE_URL) from the environment / .env.
        _JUDGE_CLIENT = OpenAI()
    elif backend == "ollama":
        ollama_host = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434")
        if not ollama_host.startswith("http"):
            ollama_host = f"http://{ollama_host}"
        _JUDGE_CLIENT = OpenAI(
            base_url=f"{ollama_host}/v1",
            api_key="ollama",  # required by the SDK but ignored by Ollama
        )
    else:
        raise ValueError(
            f"Unsupported LLM_JUDGE_BACKEND: {backend!r} (expected 'openai' or 'ollama')"
        )

    _JUDGE_MODEL = os.environ.get("LLM_JUDGE_MODEL") or DEFAULT_JUDGE_MODELS[backend]
    return _JUDGE_CLIENT, _JUDGE_MODEL

ACCURACY_PROMPT = """
Your task is to label an answer to a question as ’CORRECT’ or ’WRONG’. You will be given the following data:
    (1) a question (posed by one user to another user), 
    (2) a ’gold’ (ground truth) answer, 
    (3) a generated answer
which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT. 

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

Now it's time for the real question:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG. 
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Just return the label CORRECT or WRONG in a json format with the key as "label".
"""


def evaluate_llm_judge(question, gold_answer, generated_answer):
    """Evaluate the generated answer against the gold answer using an LLM judge."""
    client, judge_model = _resolve_judge()
    response = client.chat.completions.create(
        model=judge_model,
        messages=[
            {
                "role": "user",
                "content": ACCURACY_PROMPT.format(
                    question=question, gold_answer=gold_answer, generated_answer=generated_answer
                ),
            }
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    label = json.loads(extract_json(response.choices[0].message.content))["label"]
    return 1 if label == "CORRECT" else 0


def main():
    """Main function to evaluate RAG results using LLM judge."""
    parser = argparse.ArgumentParser(description="Evaluate RAG results using LLM judge")
    parser.add_argument(
        "--input_file",
        type=str,
        default="results/default_run_v4_k30_new_graph.json",
        help="Path to the input dataset file",
    )

    args = parser.parse_args()

    dataset_path = args.input_file
    output_path = f"results/llm_judge_{dataset_path.split('/')[-1]}"

    with open(dataset_path, "r") as f:
        data = json.load(f)

    LLM_JUDGE = defaultdict(list)
    RESULTS = defaultdict(list)

    index = 0
    for k, v in data.items():
        for x in v:
            question = x["question"]
            gold_answer = x["answer"]
            generated_answer = x["response"]
            category = x["category"]

            # Skip category 5
            if int(category) == 5:
                continue

            # Evaluate the answer
            label = evaluate_llm_judge(question, gold_answer, generated_answer)
            LLM_JUDGE[category].append(label)

            # Store the results
            RESULTS[index].append(
                {
                    "question": question,
                    "gt_answer": gold_answer,
                    "response": generated_answer,
                    "category": category,
                    "llm_label": label,
                }
            )

            # Save intermediate results
            with open(output_path, "w") as f:
                json.dump(RESULTS, f, indent=4)

            # Print current accuracy for all categories
            print("All categories accuracy:")
            for cat, results in LLM_JUDGE.items():
                if results:  # Only print if there are results for this category
                    print(f"  Category {cat}: {np.mean(results):.4f} ({sum(results)}/{len(results)})")
            print("------------------------------------------")
        index += 1

    # Save final results
    with open(output_path, "w") as f:
        json.dump(RESULTS, f, indent=4)

    # Print final summary
    print("PATH: ", dataset_path)
    print("------------------------------------------")
    for k, v in LLM_JUDGE.items():
        print(k, np.mean(v))


if __name__ == "__main__":
    main()
