from unittest.mock import MagicMock, call, patch

from mem0.configs.base import MemoryConfig
from mem0.memory.main import Memory
from mem0.memory.novelty_gate import CREATE_EVENT, UPDATE_EVENT, NoveltyResult


def _build_memory() -> Memory:
    with (
        patch("mem0.memory.main.MEM0_TELEMETRY", False),
        patch("mem0.utils.factory.EmbedderFactory.create") as mock_embedder_factory,
        patch("mem0.memory.main.VectorStoreFactory.create") as mock_vector_factory,
        patch("mem0.utils.factory.LlmFactory.create") as mock_llm_factory,
        patch("mem0.memory.main.SQLiteManager"),
        patch("mem0.memory.main.capture_event"),
    ):
        mock_embedder_factory.return_value = MagicMock()
        mock_vector_factory.return_value = MagicMock()
        mock_llm_factory.return_value = MagicMock()
        return Memory(MemoryConfig())


def test_gate_with_llm_batches_add_embeddings():
    memory = _build_memory()
    memory.embedding_model.embed_batch = MagicMock(return_value=[[0.11], [0.22]])
    memory.embedding_model.embed = MagicMock()
    memory.vector_store.search.return_value = []
    memory.llm.generate_response.return_value = '{"memory": []}'

    result = memory._gate_with_llm(
        new_retrieved_facts=["fact one", "fact two"],
        metadata={},
        filters={"user_id": "user-1"},
    )

    assert result == []
    memory.embedding_model.embed_batch.assert_called_once_with(["fact one", "fact two"], "add")
    memory.embedding_model.embed.assert_not_called()
    assert memory.vector_store.search.call_count == 2
    assert memory.vector_store.search.call_args_list[0].kwargs["vectors"] == [0.11]
    assert memory.vector_store.search.call_args_list[1].kwargs["vectors"] == [0.22]


def test_gate_with_novelty_syncs_once_after_multiple_creates():
    memory = _build_memory()
    memory.novelty_scorer = MagicMock()
    memory.novelty_scorer.scope_size.return_value = 0
    memory.novelty_scorer.scope_volume.return_value = 1.0
    memory.novelty_scorer.novelty_method = "gaussian_kde"
    # is_hydrated_fresh returns truthy MagicMock → fast path in _ensure_scope_hydrated
    memory.novelty_scorer.score.side_effect = [
        NoveltyResult(novelty=0.9, nearest_id=None),
        NoveltyResult(novelty=0.8, nearest_id=None),
    ]
    memory.adaptive_threshold = MagicMock()
    memory.adaptive_threshold.update.return_value = 0.5
    memory.adaptive_threshold.route.return_value = CREATE_EVENT
    memory.adaptive_threshold.hysteresis_delta = 0.05
    memory.embedding_model.embed_batch = MagicMock(return_value=[[0.1], [0.2]])
    memory._create_memory = MagicMock(side_effect=["m1", "m2"])
    memory._ensure_scope_hydrated = MagicMock()

    result = memory._gate_with_novelty(
        new_retrieved_facts=["fact one", "fact two"],
        metadata={},
        filters={"user_id": "user-1"},
    )

    assert result == [
        {"id": "m1", "memory": "fact one", "event": CREATE_EVENT},
        {"id": "m2", "memory": "fact two", "event": CREATE_EVENT},
    ]
    # _ensure_scope_hydrated replaces _sync_novelty_scope on the hot path; called once per _gate_with_novelty
    memory._ensure_scope_hydrated.assert_called_once_with("user-1||", {"user_id": "user-1"})
    assert memory.novelty_scorer.add_to_scope.call_count == 2
    memory.novelty_scorer.increment_turn.assert_called_once_with("user-1||")


def test_gate_with_novelty_updates_local_scope_and_syncs_once():
    memory = _build_memory()
    memory.novelty_scorer = MagicMock()
    memory.novelty_scorer.scope_size.return_value = 1
    memory.novelty_scorer.scope_volume.return_value = 1.0
    memory.novelty_scorer.novelty_method = "gaussian_kde"
    memory.novelty_scorer.score.return_value = NoveltyResult(novelty=0.7, nearest_id="existing-id")
    memory.adaptive_threshold = MagicMock()
    memory.adaptive_threshold.update.return_value = 0.5
    memory.adaptive_threshold.route.return_value = UPDATE_EVENT
    memory.adaptive_threshold.hysteresis_delta = 0.05
    memory.embedding_model.embed_batch = MagicMock(return_value=[[0.1]])
    memory.embedding_model.embed = MagicMock(return_value=[0.9])
    memory.vector_store.get.return_value = MagicMock(payload={"data": "old fact"})
    memory._merge_update_memory = MagicMock(return_value="merged fact")
    memory._update_memory = MagicMock(return_value="existing-id")
    memory._ensure_scope_hydrated = MagicMock()

    result = memory._gate_with_novelty(
        new_retrieved_facts=["incoming fact"],
        metadata={},
        filters={"user_id": "user-1"},
    )

    assert result == [
        {
            "id": "existing-id",
            "memory": "merged fact",
            "event": UPDATE_EVENT,
            "previous_memory": "old fact",
        }
    ]
    memory.novelty_scorer.update_scope_memory.assert_called_once_with(
        scope_key="user-1||",
        memory_id="existing-id",
        text="merged fact",
        embedding=[0.9],
    )
    # _ensure_scope_hydrated replaces _sync_novelty_scope on the hot path; called once per _gate_with_novelty
    memory._ensure_scope_hydrated.assert_called_once_with("user-1||", {"user_id": "user-1"})
