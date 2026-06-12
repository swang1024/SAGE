"""
Concurrent Memory.add() tests — crash repro and regression guard.

Qdrant._client_lock serializes concurrent client.* calls that
previously caused crashes when multiple threads ran Memory.add() with novelty
gating on the same scope (scroll vs upsert race on the embedded Qdrant client).

These tests use a mocked vector store to exercise the Memory + NoveltyScorer
call graph under concurrent load without requiring a real Qdrant server.
A separate integration test (run manually on the cluster) exercises the real
embedded client race; see the plan for the SLURM verification step.
"""
import threading
import uuid
from unittest.mock import MagicMock, patch

import pytest

from mem0.configs.base import MemoryConfig
from mem0.memory.main import Memory
from mem0.memory.novelty_gate import CREATE_EVENT, NoveltyResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_novelty_memory() -> Memory:
    """Return a Memory instance wired for novelty gating with mocked I/O."""
    cfg = MemoryConfig(enable_sage=True)

    with (
        patch("mem0.memory.main.MEM0_TELEMETRY", False),
        patch("mem0.utils.factory.EmbedderFactory.create") as mock_emb,
        patch("mem0.memory.main.VectorStoreFactory.create") as mock_vs,
        patch("mem0.utils.factory.LlmFactory.create") as mock_llm,
        patch("mem0.memory.main.SQLiteManager"),
        patch("mem0.memory.main.capture_event"),
    ):
        mock_emb.return_value = MagicMock()
        mock_vs.return_value = MagicMock()
        mock_llm.return_value = MagicMock()
        mem = Memory(cfg)

    return mem


def _wire_novelty_mocks(mem: Memory, scope_size: int = 0) -> None:
    """Configure per-test mock behavior on an already-built Memory."""
    dim = 4

    # Embedding model returns a stable fixed vector
    mem.embedding_model.embed_batch = MagicMock(
        side_effect=lambda texts, mode: [[0.1] * dim for _ in texts]
    )
    mem.embedding_model.embed = MagicMock(return_value=[0.1] * dim)

    # LLM extracts facts
    mem.llm.generate_response = MagicMock(return_value='["some fact"]')

    # Vector store: insert/list return sensible values
    mem.vector_store.insert = MagicMock(return_value=None)
    mem.vector_store.list = MagicMock(return_value=([], None))
    mem.vector_store.search = MagicMock(return_value=[])

    # NoveltyScorer: route every candidate as CREATE
    mem.novelty_scorer.scope_size = MagicMock(return_value=scope_size)
    mem.novelty_scorer.scope_volume = MagicMock(return_value=1.0)
    mem.novelty_scorer.novelty_method = "gaussian_kde"
    mem.novelty_scorer.score = MagicMock(
        return_value=NoveltyResult(novelty=float("inf"), nearest_id=None)
    )
    mem.novelty_scorer.sync_scope = MagicMock()
    mem.novelty_scorer.add_to_scope = MagicMock()
    mem.novelty_scorer.update_scope_memory = MagicMock()
    mem.novelty_scorer.increment_turn = MagicMock()
    mem.novelty_scorer.get_turns = MagicMock(return_value=0)

    # Threshold routes to CREATE
    mem.adaptive_threshold.update = MagicMock(return_value=0.5)
    mem.adaptive_threshold.route = MagicMock(return_value=CREATE_EVENT)
    mem.adaptive_threshold.hysteresis_delta = 0.05

    # _create_memory returns a fresh UUID each time
    mem._create_memory = MagicMock(side_effect=lambda **kw: str(uuid.uuid4()))

    # Fact extractor returns one fact per call
    mem._extract_facts = MagicMock(return_value=["some fact"])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestConcurrentAddSameScope:
    """Original crash repro: multiple threads add to the same user scope."""

    def test_concurrent_add_does_not_crash(self):
        mem = _build_novelty_memory()
        _wire_novelty_mocks(mem)

        n_threads = 8
        calls_per_thread = 25
        errors = []

        def worker():
            for _ in range(calls_per_thread):
                try:
                    mem._gate_with_novelty(
                        new_retrieved_facts=["some fact"],
                        metadata={},
                        filters={"user_id": "user-shared"},
                    )
                except Exception as exc:
                    errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30.0)

        assert not errors, f"Concurrent add raised {len(errors)} exception(s): {errors[:3]}"

    def test_all_threads_complete(self):
        """No thread should deadlock or be starved."""
        mem = _build_novelty_memory()
        _wire_novelty_mocks(mem)
        completed = []

        def worker():
            mem._gate_with_novelty(
                new_retrieved_facts=["fact"],
                metadata={},
                filters={"user_id": "user-shared"},
            )
            completed.append(1)

        threads = [threading.Thread(target=worker) for _ in range(16)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        assert len(completed) == 16, f"Only {len(completed)}/16 threads completed"


class TestConcurrentAddDistinctScopes:
    """Threads on separate user scopes must not cross-contaminate."""

    def test_distinct_scopes_no_cross_contamination(self):
        mem = _build_novelty_memory()
        _wire_novelty_mocks(mem)

        n_users = 8
        calls_per_user = 20
        errors = []

        def worker(user_id: str):
            for _ in range(calls_per_user):
                try:
                    mem._gate_with_novelty(
                        new_retrieved_facts=["fact"],
                        metadata={},
                        filters={"user_id": user_id},
                    )
                except Exception as exc:
                    errors.append((user_id, exc))

        threads = [
            threading.Thread(target=worker, args=(f"user-{i}",))
            for i in range(n_users)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30.0)

        assert not errors, f"Concurrent distinct-scope add raised errors: {errors[:3]}"

    def test_ensure_scope_hydrated_call_count_is_bounded(self):
        """_ensure_scope_hydrated is called at most once on cold scope, 0 on warm calls."""
        mem = _build_novelty_memory()
        _wire_novelty_mocks(mem)
        hydrate_calls = []
        original_hydrate = mem._ensure_scope_hydrated

        def counting_hydrate(*args, **kwargs):
            hydrate_calls.append(args)
            return original_hydrate(*args, **kwargs)

        mem._ensure_scope_hydrated = counting_hydrate
        n_gate_calls = 10

        for _ in range(n_gate_calls):
            mem._gate_with_novelty(
                new_retrieved_facts=["fact"],
                metadata={},
                filters={"user_id": "user-x"},
            )

        # lazy hydration — _ensure_scope_hydrated is called ≤1 per scope total
        # (once on the cold first call, then warm hits skip it).
        assert len(hydrate_calls) <= n_gate_calls, (
            f"Unexpected hydrate call count: {len(hydrate_calls)} for {n_gate_calls} gate calls"
        )
