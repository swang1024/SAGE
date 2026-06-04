"""
MemorySearchLocal — local-only search backend for sage evaluation.

Mirrors the interface of search.py (MemorySearch) but uses a local
Memory.from_config() Ollama backend instead of the cloud MemoryClient.
"""
import json
import os
import time
from collections import defaultdict

from dotenv import load_dotenv
from jinja2 import Template
from openai import OpenAI  # used with Ollama's OpenAI-compatible endpoint
from prompts import ANSWER_PROMPT
from tqdm import tqdm

from mem0 import Memory
from mem0.llms.usage_tracker import USAGE_TRACKER, extract_usage

load_dotenv()


class MemorySearchLocal:
    """Search and answer using a local Ollama-backed Qdrant memory store."""

    def __init__(
        self,
        output_path="results.json",
        top_k=30,
        ollama_base_url="http://127.0.0.1:11434",
        llm_model="llama3.2",
        embedding_model="nomic-embed-text",
        embedding_dims=768,
        qdrant_path="./data/mem0_ang_eval_qdrant",
        history_db_path="./data/mem0_ang_eval_history.db",
        collection_name="mem0_ang_eval",
        backend="ollama",
    ):
        self.top_k = top_k
        self.output_path = output_path
        self.ANSWER_PROMPT = ANSWER_PROMPT
        self.backend = backend

        # Local Memory object — must mirror add_abla.py's config for this backend
        # so the search phase opens the same Qdrant store with a matching embedder.
        if backend == "openai":
            llm_cfg = {"provider": "openai", "config": {"model": llm_model}}
            embedder_cfg = {
                "provider": "openai",
                "config": {"model": embedding_model, "embedding_dims": embedding_dims},
            }
        else:
            llm_cfg = {
                "provider": "ollama",
                "config": {"model": llm_model, "ollama_base_url": ollama_base_url},
            }
            embedder_cfg = {
                "provider": "ollama",
                "config": {
                    "model": embedding_model,
                    "ollama_base_url": ollama_base_url,
                    "embedding_dims": embedding_dims,
                },
            }

        self.memory = Memory.from_config(
            {
                "llm": llm_cfg,
                "embedder": embedder_cfg,
                "vector_store": {
                    "provider": "qdrant",
                    "config": {
                        "path": qdrant_path,
                        "collection_name": collection_name,
                        "embedding_model_dims": embedding_dims,
                        "on_disk": True,
                    },
                },
                "history_db_path": history_db_path,
                "version": "v1.1",
            }
        )

        # Client for answer generation.
        self.llm_model = llm_model
        if backend == "openai":
            # Real OpenAI endpoint; reads OPENAI_API_KEY (and optional OPENAI_BASE_URL).
            self.openai_client = OpenAI()
        else:
            # Ollama-backed OpenAI-compatible endpoint.
            self.openai_client = OpenAI(
                base_url=f"{ollama_base_url}/v1",
                api_key="ollama",  # Ollama doesn't require a real key
            )
        self.results = defaultdict(list)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def search_memory(self, user_id, query, max_retries=3, retry_delay=1):
        start_time = time.time()
        retries = 0
        while retries < max_retries:
            try:
                memories = self.memory.search(query, user_id=user_id, limit=self.top_k)
                break
            except Exception as e:
                print(f"Retrying search for {user_id}...")
                retries += 1
                if retries >= max_retries:
                    raise e
                time.sleep(retry_delay)

        end_time = time.time()

        # Normalize: local Memory.search() may return a list or a dict
        items = memories.get("results", []) if isinstance(memories, dict) else memories

        semantic_memories = [
            {
                "memory": m.get("memory", ""),
                "timestamp": (m.get("metadata") or {}).get("timestamp", ""),
                "score": round(float(m.get("score", 0.0)), 2),
            }
            for m in items
        ]
        # Local backend does not support graph search
        graph_memories = None
        return semantic_memories, graph_memories, end_time - start_time

    # ------------------------------------------------------------------
    # Answer generation  (mirrors search.py lines 90-127)
    # ------------------------------------------------------------------
    def answer_question(self, speaker_1_user_id, speaker_2_user_id, question, answer, category):
        speaker_1_memories, speaker_1_graph_memories, speaker_1_memory_time = self.search_memory(
            speaker_1_user_id, question
        )
        speaker_2_memories, speaker_2_graph_memories, speaker_2_memory_time = self.search_memory(
            speaker_2_user_id, question
        )

        search_1_memory = [f"{item['timestamp']}: {item['memory']}" for item in speaker_1_memories]
        search_2_memory = [f"{item['timestamp']}: {item['memory']}" for item in speaker_2_memories]

        template = Template(self.ANSWER_PROMPT)
        answer_prompt = template.render(
            speaker_1_user_id=speaker_1_user_id.split("_")[0],
            speaker_2_user_id=speaker_2_user_id.split("_")[0],
            speaker_1_memories=json.dumps(search_1_memory, indent=4),
            speaker_2_memories=json.dumps(search_2_memory, indent=4),
            speaker_1_graph_memories=json.dumps(speaker_1_graph_memories, indent=4),
            speaker_2_graph_memories=json.dumps(speaker_2_graph_memories, indent=4),
            question=question,
        )

        t1 = time.time()
        # need to check how to write the template properly to achieve better results
        response = self.openai_client.chat.completions.create(
            model=self.llm_model,
            messages=[{"role": "user", "content": answer_prompt}],
            temperature=0.0,
        )
        t2 = time.time()
        response_time = t2 - t1
        # Record answer-generation token usage for the search phase.
        _p, _c, _t = extract_usage(response)
        USAGE_TRACKER.record(
            model=self.llm_model,
            prompt_tokens=_p,
            completion_tokens=_c,
            total_tokens=_t,
            latency=response_time,
        )
        return (
            response.choices[0].message.content,
            speaker_1_memories,
            speaker_2_memories,
            speaker_1_memory_time,
            speaker_2_memory_time,
            speaker_1_graph_memories,
            speaker_2_graph_memories,
            response_time,
        )

    # ------------------------------------------------------------------
    # Per-question processing  (mirrors search.py lines 129-169)
    # ------------------------------------------------------------------
    def process_question(self, val, speaker_a_user_id, speaker_b_user_id):
        question = val.get("question", "")
        answer = val.get("answer", "")
        category = val.get("category", -1)
        evidence = val.get("evidence", [])
        adversarial_answer = val.get("adversarial_answer", "")

        (
            response,
            speaker_1_memories,
            speaker_2_memories,
            speaker_1_memory_time,
            speaker_2_memory_time,
            speaker_1_graph_memories,
            speaker_2_graph_memories,
            response_time,
        ) = self.answer_question(speaker_a_user_id, speaker_b_user_id, question, answer, category)

        result = {
            "question": question,
            "answer": answer,
            "category": category,
            "evidence": evidence,
            "response": response,
            "adversarial_answer": adversarial_answer,
            "speaker_1_memories": speaker_1_memories,
            "speaker_2_memories": speaker_2_memories,
            "num_speaker_1_memories": len(speaker_1_memories),
            "num_speaker_2_memories": len(speaker_2_memories),
            "speaker_1_memory_time": speaker_1_memory_time,
            "speaker_2_memory_time": speaker_2_memory_time,
            "speaker_1_graph_memories": speaker_1_graph_memories,
            "speaker_2_graph_memories": speaker_2_graph_memories,
            "response_time": response_time,
        }

        # Save results after each question for fault tolerance
        with open(self.output_path, "w") as f:
            json.dump(self.results, f, indent=4)

        return result

    # ------------------------------------------------------------------
    # Dataset-level driver  (mirrors search.py lines 171-196)
    # ------------------------------------------------------------------
    def process_data_file(self, file_path):
        with open(file_path, "r") as f:
            data = json.load(f)

        # Isolate search-phase token/latency usage (the add phase already snapshotted
        # and is now torn down). Embedding (query) and answer-LLM calls both record to
        # USAGE_TRACKER; per_model keeps the embedding model separable from the LLM.
        USAGE_TRACKER.enable(True)
        USAGE_TRACKER.reset()

        for idx, item in tqdm(enumerate(data), total=len(data), desc="Processing conversations"):
            qa = item["qa"]
            conversation = item["conversation"]
            speaker_a = conversation["speaker_a"]
            speaker_b = conversation["speaker_b"]

            speaker_a_user_id = f"{speaker_a}_{idx}"
            speaker_b_user_id = f"{speaker_b}_{idx}"

            for question_item in tqdm(
                qa, total=len(qa), desc=f"Processing questions for conversation {idx}", leave=False
            ):
                result = self.process_question(question_item, speaker_a_user_id, speaker_b_user_id)
                self.results[idx].append(result)

                # Save results after each question for fault tolerance
                with open(self.output_path, "w") as f:
                    json.dump(self.results, f, indent=4)

        # Final save
        with open(self.output_path, "w") as f:
            json.dump(self.results, f, indent=4)

        # Dump search-phase usage (hard token + latency figures) next to the answers.
        usage_path = os.path.splitext(self.output_path)[0] + "_search_usage.json"
        with open(usage_path, "w") as f:
            json.dump(USAGE_TRACKER.snapshot(), f, indent=2)
        print(f"Saved search-phase usage stats to {usage_path}")
