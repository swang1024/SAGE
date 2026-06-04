import json
import os
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from tqdm import tqdm

from mem0 import Memory, MemoryClient
from mem0.llms.usage_tracker import USAGE_TRACKER

load_dotenv()


# Update custom instructions
custom_instructions = """
Generate personal memories that follow these guidelines:

1. Each memory should be self-contained with complete context, including:
   - The person's name, do not use "user" while creating memories
   - Personal details (career aspirations, hobbies, life circumstances)
   - Emotional states and reactions
   - Ongoing journeys or future plans
   - Specific dates when events occurred

2. Include meaningful personal narratives focusing on:
   - Identity and self-acceptance journeys
   - Family planning and parenting
   - Creative outlets and hobbies
   - Mental health and self-care activities
   - Career aspirations and education goals
   - Important life events and milestones

3. Make each memory rich with specific details rather than general statements
   - Include timeframes (exact dates when possible)
   - Name specific activities (e.g., "charity race for mental health" rather than just "exercise")
   - Include emotional context and personal growth elements

4. Extract memories only from user messages, not incorporating assistant responses

5. Format each memory as a paragraph with a clear narrative structure that captures the person's experience, challenges, and aspirations
"""


class MemoryADDAbla:
    """MemoryADD variant with action-level instrumentation for ablation studies."""

    def __init__(
        self,
        data_path=None,
        batch_size=2,
        is_graph=False,
        stats_output_path=None,
        backend="cloud",
        infer_add=False,
        ollama_base_url="http://127.0.0.1:11434",
        llm_model="llama3.2",
        embedding_model="nomic-embed-text",
        embedding_dims=768,
        qdrant_path="/tmp/mem0_abla_qdrant_persist",
        history_db_path="/tmp/mem0_abla_history_persist.db",
        collection_name="mem0_abla_persist",
        enable_sage=False,
        sage_novelty_method="vmf_kde",
    ):
        self.backend = backend
        self.mem0_client = None
        self.memory = None

        if self.backend == "cloud":
            self.mem0_client = MemoryClient(
                api_key=os.getenv("MEM0_API_KEY"),
                org_id=os.getenv("MEM0_ORGANIZATION_ID"),
                project_id=os.getenv("MEM0_PROJECT_ID"),
            )
            self.mem0_client.update_project(custom_instructions=custom_instructions)
        elif self.backend == "ollama":
            memory_config = {
                "llm": {
                    "provider": "ollama",
                    "config": {"model": llm_model, "ollama_base_url": ollama_base_url},
                },
                "embedder": {
                    "provider": "ollama",
                    "config": {
                        "model": embedding_model,
                        "ollama_base_url": ollama_base_url,
                        "embedding_dims": embedding_dims,
                    },
                },
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
                "enable_sage": enable_sage,
            }
            if enable_sage:
                memory_config["sage_novelty_method"] = sage_novelty_method
            if is_graph:
                # mem0g: in-process Kuzu graph store (no server); reuses the run's
                # LLM/embedder. Set only for --is_graph so plain mem0/SAGE runs keep
                # enable_graph off. Needs `pip install kuzu`.
                memory_config["graph_store"] = {
                    "provider": "kuzu",
                    "config": {"db": f"/tmp/mem0_abla_kuzu_{collection_name}"},
                }

            self.memory = Memory.from_config(
                memory_config
            )
        elif self.backend == "openai":
            memory_config = {
                "llm": {
                    "provider": "openai",
                    "config": {"model": llm_model},
                },
                "embedder": {
                    "provider": "openai",
                    "config": {
                        "model": embedding_model,
                        "embedding_dims": embedding_dims,
                    },
                },
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
                "enable_sage": enable_sage,
            }
            if enable_sage:
                memory_config["sage_novelty_method"] = sage_novelty_method
            if is_graph:
                # mem0g: in-process Kuzu graph store (no server); reuses the run's
                # LLM/embedder. Set only for --is_graph so plain mem0/SAGE runs keep
                # enable_graph off. Needs `pip install kuzu`.
                memory_config["graph_store"] = {
                    "provider": "kuzu",
                    "config": {"db": f"/tmp/mem0_abla_kuzu_{collection_name}"},
                }

            self.memory = Memory.from_config(
                memory_config
            )
        else:
            raise ValueError(f"Invalid backend: {self.backend}")

        self.batch_size = batch_size
        self.data_path = data_path
        self.data = None
        self.is_graph = is_graph
        self.infer_add = infer_add
        self.stats_output_path = stats_output_path

        self._stats_lock = threading.Lock()
        self.action_counts = Counter()
        self.per_user_action_counts = defaultdict(Counter)
        self.per_conversation_action_counts = defaultdict(Counter)
        self.api_stats = Counter()

        if data_path:
            self.load_data()

    def load_data(self):
        with open(self.data_path, "r") as f:
            self.data = json.load(f)
        return self.data

    def _record_action(self, event, user_id, conversation_idx):
        action = str(event).upper() if event is not None else "UNKNOWN"
        with self._stats_lock:
            self.action_counts[action] += 1
            self.per_user_action_counts[user_id][action] += 1
            if conversation_idx is not None:
                self.per_conversation_action_counts[str(conversation_idx)][action] += 1

    def _record_add_response(self, response, user_id, conversation_idx):
        if isinstance(response, dict):
            results = response.get("results", [])
            relations = response.get("relations", {})
        elif isinstance(response, list):
            results = response
            relations = {}
        else:
            results = []
            relations = {}

        if not results:
            self._record_action("EMPTY_RESULT", user_id, conversation_idx)

        for item in results:
            event = item.get("event", "UNKNOWN") if isinstance(item, dict) else "UNKNOWN"
            self._record_action(event, user_id, conversation_idx)

        if isinstance(relations, dict):
            added_entities = relations.get("added_entities", []) or []
            deleted_entities = relations.get("deleted_entities", []) or []
            with self._stats_lock:
                self.api_stats["graph_relations_added"] += len(added_entities)
                self.api_stats["graph_relations_deleted"] += len(deleted_entities)

    def _delete_all(self, user_id):
        if self.backend == "cloud":
            self.mem0_client.delete_all(user_id=user_id)
        else:
            self.memory.delete_all(user_id=user_id)

    def _add(self, user_id, message, metadata):
        if self.backend == "cloud":
            return self.mem0_client.add(
                message,
                user_id=user_id,
                version="v2",
                metadata=metadata,
                enable_graph=self.is_graph,
            )
        return self.memory.add(
            message,
            user_id=user_id,
            metadata=metadata,
            infer=self.infer_add,
        )

    def add_memory(self, user_id, message, metadata, retries=3, conversation_idx=None):
        for attempt in range(retries):
            try:
                response = self._add(user_id=user_id, message=message, metadata=metadata)
                with self._stats_lock:
                    self.api_stats["add_api_calls"] += 1
                    self.api_stats["messages_sent"] += len(message) if isinstance(message, list) else 1
                self._record_add_response(response, user_id, conversation_idx)
                return response
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(1)
                    continue
                with self._stats_lock:
                    self.api_stats["add_api_failures"] += 1
                raise e

    def add_memories_for_speaker(self, speaker, messages, timestamp, desc, conversation_idx):
        # this function processes all the chats from one speaker in a conversation by the end of the conversation
        for i in tqdm(range(0, len(messages), self.batch_size), desc=desc):
            batch_messages = messages[i : i + self.batch_size]
            self.add_memory(
                speaker,
                batch_messages,
                metadata={"timestamp": timestamp},
                conversation_idx=conversation_idx,
            )

    def process_conversation(self, item, idx):
        conversation = item["conversation"]
        speaker_a = conversation["speaker_a"]
        speaker_b = conversation["speaker_b"]

        speaker_a_user_id = f"{speaker_a}_{idx}"
        speaker_b_user_id = f"{speaker_b}_{idx}"

        self._delete_all(user_id=speaker_a_user_id)
        self._delete_all(user_id=speaker_b_user_id)

        for key in conversation.keys():
            if key in ["speaker_a", "speaker_b"] or "date" in key or "timestamp" in key:
                continue

            date_time_key = key + "_date_time"
            timestamp = conversation[date_time_key]
            chats = conversation[key]

            messages = []
            messages_reverse = []
            for chat in chats:
                if chat["speaker"] == speaker_a:
                    messages.append({"role": "user", "content": f"{speaker_a}: {chat['text']}"})
                    messages_reverse.append({"role": "assistant", "content": f"{speaker_a}: {chat['text']}"})
                elif chat["speaker"] == speaker_b:
                    messages.append({"role": "assistant", "content": f"{speaker_b}: {chat['text']}"})
                    messages_reverse.append({"role": "user", "content": f"{speaker_b}: {chat['text']}"})
                else:
                    raise ValueError(f"Unknown speaker: {chat['speaker']}")

            if self.backend in ("ollama", "openai") and self.infer_add:
                # Local infer-mode add can be unstable under concurrent writes; process speakers sequentially.
                print("Local infer mode detected, processing sequentially for conversation", idx)
                self.add_memories_for_speaker(
                    speaker_a_user_id, messages, timestamp, "Adding Memories for Speaker A", idx
                )
                self.add_memories_for_speaker(
                    speaker_b_user_id, messages_reverse, timestamp, "Adding Memories for Speaker B", idx
                )
            else:
                thread_a = threading.Thread(
                    target=self.add_memories_for_speaker,
                    args=(speaker_a_user_id, messages, timestamp, "Adding Memories for Speaker A", idx),
                )
                thread_b = threading.Thread(
                    target=self.add_memories_for_speaker,
                    args=(speaker_b_user_id, messages_reverse, timestamp, "Adding Memories for Speaker B", idx),
                )

                thread_a.start()
                thread_b.start()
                thread_a.join()
                thread_b.join()

        print("Messages added successfully")

    def get_action_stats(self):
        with self._stats_lock:
            return {
                "summary": {
                    "action_counts": dict(self.action_counts),
                    "api_stats": dict(self.api_stats),
                    "usage_stats": USAGE_TRACKER.snapshot(),
                },
                "per_user_action_counts": {
                    user_id: dict(counter) for user_id, counter in self.per_user_action_counts.items()
                },
                "per_conversation_action_counts": {
                    conversation_idx: dict(counter)
                    for conversation_idx, counter in self.per_conversation_action_counts.items()
                },
            }

    def dump_action_stats(self, output_path=None):
        stats = self.get_action_stats()
        path = output_path or self.stats_output_path
        if not path:
            return stats

        output_dir = os.path.dirname(path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(path, "w") as f:
            json.dump(stats, f, indent=2)
        return stats

    def process_all_conversations(self, max_workers=10):
        if not self.data:
            raise ValueError("No data loaded. Please set data_path and call load_data() first.")

        # Enable and reset token/latency tracking so the snapshot covers only the write phase.
        USAGE_TRACKER.enable(True)
        USAGE_TRACKER.reset()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.process_conversation, item, idx) for idx, item in enumerate(self.data)]
            for future in futures:
                future.result()

        stats = self.dump_action_stats()
        print("Action summary:", json.dumps(stats["summary"], indent=2))
        if self.stats_output_path:
            print(f"Saved action stats to {self.stats_output_path}")
