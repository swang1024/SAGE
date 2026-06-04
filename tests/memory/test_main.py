import logging
import threading
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from mem0.configs.base import MemoryConfig
from mem0.memory.main import AsyncMemory, Memory, _normalize_iso_timestamp_to_utc


def _setup_mocks(mocker):
    """Helper to setup common mocks for both sync and async fixtures"""
    mock_embedder = mocker.MagicMock()
    mock_embedder.return_value.embed.return_value = [0.1, 0.2, 0.3]
    mock_embedder.return_value.embed_batch.side_effect = lambda texts, action=None: [[0.1, 0.2, 0.3]] * len(texts)
    mocker.patch("mem0.utils.factory.EmbedderFactory.create", mock_embedder)

    mock_vector_store = mocker.MagicMock()
    mock_vector_store.return_value.search.return_value = []
    mocker.patch(
        "mem0.utils.factory.VectorStoreFactory.create", side_effect=[mock_vector_store.return_value, mocker.MagicMock()]
    )

    mock_llm = mocker.MagicMock()
    mocker.patch("mem0.utils.factory.LlmFactory.create", mock_llm)

    mocker.patch("mem0.memory.storage.SQLiteManager", mocker.MagicMock())

    return mock_llm, mock_vector_store


class TestAddToVectorStoreErrors:
    @pytest.fixture
    def mock_memory(self, mocker):
        """Fixture that returns a Memory instance with mocker-based mocks"""
        mock_llm, _ = _setup_mocks(mocker)

        memory = Memory()
        memory.config = mocker.MagicMock()
        memory.config.custom_fact_extraction_prompt = None
        memory.config.custom_update_memory_prompt = None
        memory.api_version = "v1.1"

        return memory

    def test_empty_llm_response_fact_extraction(self, mocker, mock_memory, caplog):
        """Test empty response from LLM during fact extraction"""
        # Setup
        mock_memory.llm.generate_response.return_value = "invalid json"  # This will trigger a JSON decode error
        mock_capture_event = mocker.MagicMock()
        mocker.patch("mem0.memory.main.capture_event", mock_capture_event)

        # Execute
        with caplog.at_level(logging.ERROR):
            result = mock_memory._add_to_vector_store(
                messages=[{"role": "user", "content": "test"}], metadata={}, filters={}, infer=True
            )

        # Verify
        assert mock_memory.llm.generate_response.call_count == 1
        assert result == []  # Should return empty list when no memories processed
        # Check for error message in any of the log records
        assert any("Error in new_retrieved_facts" in record.msg for record in caplog.records), "Expected error message not found in logs"
        assert mock_capture_event.call_count == 1

    def test_empty_llm_response_memory_actions(self, mock_memory, caplog):
        """Test empty response from LLM during memory actions"""
        # First call returns valid JSON, second call returns empty string
        mock_memory.llm.generate_response.side_effect = ['{"facts": ["test fact"]}', ""]

        # Execute
        with caplog.at_level(logging.WARNING):
            result = mock_memory._add_to_vector_store(
                messages=[{"role": "user", "content": "test"}], metadata={}, filters={}, infer=True
            )

        # Verify
        assert mock_memory.llm.generate_response.call_count == 2
        assert result == []  # Should return empty list when no memories processed
        assert "Empty response from LLM, no memories to extract" in caplog.text


@pytest.mark.asyncio
class TestAsyncAddToVectorStoreErrors:
    @pytest.fixture
    def mock_async_memory(self, mocker):
        """Fixture for AsyncMemory with mocker-based mocks"""
        mock_llm, _ = _setup_mocks(mocker)

        memory = AsyncMemory()
        memory.config = mocker.MagicMock()
        memory.config.custom_fact_extraction_prompt = None
        memory.config.custom_update_memory_prompt = None
        memory.api_version = "v1.1"

        return memory

    @pytest.mark.asyncio
    async def test_async_empty_llm_response_fact_extraction(self, mock_async_memory, caplog, mocker):
        """Test empty response in AsyncMemory._add_to_vector_store"""
        mocker.patch("mem0.utils.factory.EmbedderFactory.create", return_value=MagicMock())
        mock_async_memory.llm.generate_response.return_value = "invalid json"  # This will trigger a JSON decode error
        mock_capture_event = mocker.MagicMock()
        mocker.patch("mem0.memory.main.capture_event", mock_capture_event)

        with caplog.at_level(logging.ERROR):
            result = await mock_async_memory._add_to_vector_store(
                messages=[{"role": "user", "content": "test"}], metadata={}, effective_filters={}, infer=True
            )
        assert mock_async_memory.llm.generate_response.call_count == 1
        assert result == []
        # Check for error message in any of the log records
        assert any("Error in new_retrieved_facts" in record.msg for record in caplog.records), "Expected error message not found in logs"
        assert mock_capture_event.call_count == 1

    @pytest.mark.asyncio
    async def test_async_empty_llm_response_memory_actions(self, mock_async_memory, caplog, mocker):
        """Test empty response in AsyncMemory._add_to_vector_store"""
        mocker.patch("mem0.utils.factory.EmbedderFactory.create", return_value=MagicMock())
        mock_async_memory.llm.generate_response.side_effect = ['{"facts": ["test fact"]}', ""]
        mock_capture_event = mocker.MagicMock()
        mocker.patch("mem0.memory.main.capture_event", mock_capture_event)

        with caplog.at_level(logging.WARNING):
            result = await mock_async_memory._add_to_vector_store(
                messages=[{"role": "user", "content": "test"}], metadata={}, effective_filters={}, infer=True
            )

        assert result == []
        assert "Empty response from LLM, no memories to extract" in caplog.text
        assert mock_capture_event.call_count == 1


def _build_memory_instance(mocker, memory_cls):
    _setup_mocks(mocker)
    mocker.patch("mem0.memory.main.SQLiteManager", mocker.MagicMock())
    mocker.patch("mem0.memory.main.MEM0_TELEMETRY", False)
    memory = memory_cls()
    memory.config = mocker.MagicMock()
    memory.config.custom_fact_extraction_prompt = None
    memory.config.custom_update_memory_prompt = None
    memory.api_version = "v1.1"
    memory.vector_store = mocker.MagicMock()
    memory.db = mocker.MagicMock()
    return memory


def _assert_utc_timestamp(timestamp: str):
    parsed = datetime.fromisoformat(timestamp)
    assert parsed.tzinfo == timezone.utc
    assert parsed.utcoffset().total_seconds() == 0


def test_create_memory_uses_utc_timestamps(mocker):
    memory = _build_memory_instance(mocker, Memory)
    memory._create_memory("new memory", {"new memory": [0.1, 0.2, 0.3]}, metadata={})
    payload = memory.vector_store.insert.call_args.kwargs["payloads"][0]
    _assert_utc_timestamp(payload["created_at"])


def test_update_memory_uses_utc_timestamps(mocker):
    memory = _build_memory_instance(mocker, Memory)
    memory.vector_store.get.return_value = MagicMock(
        payload={"data": "old memory", "created_at": "2026-03-17T17:00:00-07:00"}
    )
    memory._update_memory("memory-id", "new memory", {"new memory": [0.1, 0.2, 0.3]}, metadata={})
    payload = memory.vector_store.update.call_args.kwargs["payload"]
    assert payload["created_at"] == "2026-03-18T00:00:00+00:00"
    _assert_utc_timestamp(payload["updated_at"])


@pytest.mark.asyncio
async def test_async_create_memory_uses_utc_timestamps(mocker):
    memory = _build_memory_instance(mocker, AsyncMemory)
    await memory._create_memory("new memory", {"new memory": [0.1, 0.2, 0.3]}, metadata={})
    payload = memory.vector_store.insert.call_args.kwargs["payloads"][0]
    _assert_utc_timestamp(payload["created_at"])


@pytest.mark.asyncio
async def test_async_update_memory_uses_utc_timestamps(mocker):
    memory = _build_memory_instance(mocker, AsyncMemory)
    memory.vector_store.get.return_value = MagicMock(
        payload={"data": "old memory", "created_at": "2026-03-17T17:00:00-07:00"}
    )
    await memory._update_memory("memory-id", "new memory", {"new memory": [0.1, 0.2, 0.3]}, metadata={})
    payload = memory.vector_store.update.call_args.kwargs["payload"]
    assert payload["created_at"] == "2026-03-18T00:00:00+00:00"
    _assert_utc_timestamp(payload["updated_at"])


def test_normalize_iso_timestamp_to_utc_preserves_naive_values():
    assert _normalize_iso_timestamp_to_utc("2026-03-18T00:00:00") == "2026-03-18T00:00:00"


def test_normalize_iso_timestamp_to_utc_converts_pacific():
    result = _normalize_iso_timestamp_to_utc("2026-03-17T17:00:00-07:00")
    assert result == "2026-03-18T00:00:00+00:00"


def test_normalize_iso_timestamp_to_utc_handles_none():
    assert _normalize_iso_timestamp_to_utc(None) is None


def test_normalize_iso_timestamp_to_utc_handles_empty():
    assert _normalize_iso_timestamp_to_utc("") == ""


# ---------------------------------------------------------------------------
# Consistency hardening tests
# ---------------------------------------------------------------------------

def _build_sage_memory():
    """Build a Memory instance with enable_sage=True and mocked I/O."""
    cfg = MemoryConfig(enable_sage=True)
    with (
        patch("mem0.memory.main.MEM0_TELEMETRY", False),
        patch("mem0.utils.factory.EmbedderFactory.create") as mock_emb,
        patch("mem0.memory.main.VectorStoreFactory.create") as mock_vs,
        patch("mem0.utils.factory.LlmFactory.create"),
        patch("mem0.memory.main.SQLiteManager"),
        patch("mem0.memory.main.capture_event"),
    ):
        mock_emb.return_value = MagicMock()
        mock_vs.return_value = MagicMock()
        mem = Memory(cfg)
    return mem


def _make_point(memory_id, user_id=None, agent_id=None, run_id=None):
    """Create a mock Qdrant point with appropriate payload."""
    payload = {}
    if user_id:
        payload["user_id"] = user_id
    if agent_id:
        payload["agent_id"] = agent_id
    if run_id:
        payload["run_id"] = run_id
    point = MagicMock()
    point.id = memory_id
    point.payload = payload
    return point


class TestPR3DeleteAllPerScopeInvalidation:
    def test_delete_all_broad_filter_invalidates_all_affected_scopes(self):
        """delete_all marks all touched scopes unhydrated before deletion."""
        mem = _build_sage_memory()

        # 3 memories in the same user_id but different agent_ids → 3 distinct scopes
        points = [
            _make_point("id-1", user_id="user1", agent_id="agent-a"),
            _make_point("id-2", user_id="user1", agent_id="agent-b"),
            _make_point("id-3", user_id="user1", agent_id="agent-c"),
        ]
        mem.vector_store.iter_all = MagicMock(return_value=iter(points))
        # list() is used for the delete loop; return same points
        mem.vector_store.list = MagicMock(return_value=(points, None))
        mem.vector_store.get = MagicMock(side_effect=lambda vector_id: _make_point(vector_id, user_id="user1"))
        mem.vector_store.delete = MagicMock()
        mem.db.add_history = MagicMock()

        mark_unhydrated = MagicMock()
        reset_scope = MagicMock()
        mem.novelty_scorer.mark_unhydrated = mark_unhydrated
        mem.adaptive_threshold.reset_scope = reset_scope

        mem.delete_all(user_id="user1")

        touched_keys = {call.args[0] for call in mark_unhydrated.call_args_list}
        expected_keys = {
            Memory._scope_key({"user_id": "user1", "agent_id": "agent-a"}),
            Memory._scope_key({"user_id": "user1", "agent_id": "agent-b"}),
            Memory._scope_key({"user_id": "user1", "agent_id": "agent-c"}),
        }
        assert touched_keys == expected_keys
        assert reset_scope.call_count == 3

    def test_delete_all_unmappable_filter_falls_back_to_clear_all(self):
        """When iter_all raises, delete_all falls back to clear_all_scopes."""
        mem = _build_sage_memory()
        mem.vector_store.iter_all = MagicMock(side_effect=RuntimeError("backend error"))
        mem.vector_store.list = MagicMock(return_value=([], None))

        clear_all = MagicMock()
        mem.novelty_scorer.clear_all_scopes = clear_all
        mem.novelty_scorer.mark_unhydrated = MagicMock()

        mem.delete_all(user_id="user1")

        clear_all.assert_called_once()

    def test_delete_all_empty_result_calls_coarse_fallback(self):
        """When no points match (filter returns nothing), coarse fallback is used."""
        mem = _build_sage_memory()
        mem.vector_store.iter_all = MagicMock(return_value=iter([]))
        mem.vector_store.list = MagicMock(return_value=([], None))

        clear_all = MagicMock()
        mem.novelty_scorer.clear_all_scopes = clear_all

        mem.delete_all(user_id="user-nobody")

        clear_all.assert_called_once()


class TestPR3ResetNoveltyCaches:
    def test_reset_clears_adaptive_threshold_and_scorer_together(self):
        """_reset_novelty_caches(None) clears scorer, threshold, and hydration locks atomically."""
        mem = _build_sage_memory()

        # Populate some state
        mem.novelty_scorer._scopes["key1"] = MagicMock()
        mem._hydration_locks["key1"] = threading.Lock()

        clear_scopes = MagicMock()
        reset_all = MagicMock()
        mem.novelty_scorer.clear_all_scopes = clear_scopes
        mem.adaptive_threshold.reset_all = reset_all

        mem._reset_novelty_caches()

        clear_scopes.assert_called_once()
        reset_all.assert_called_once()
        assert mem._hydration_locks == {}

    def test_reset_novelty_caches_per_scope_marks_unhydrated(self):
        """_reset_novelty_caches(scope_key) marks one scope unhydrated and resets its threshold."""
        mem = _build_sage_memory()

        mark_unhydrated = MagicMock()
        reset_scope = MagicMock()
        mem.novelty_scorer.mark_unhydrated = mark_unhydrated
        mem.adaptive_threshold.reset_scope = reset_scope

        mem._reset_novelty_caches("user1||")

        mark_unhydrated.assert_called_once_with("user1||")
        reset_scope.assert_called_once_with("user1||")

    def test_reset_novelty_caches_noop_when_sage_disabled(self):
        """_reset_novelty_caches is a no-op when enable_sage is False."""
        mem = _build_sage_memory()
        mem.enable_sage = False
        # Should not raise even though scorer may be None
        mem._reset_novelty_caches()
        mem._reset_novelty_caches("some-scope")

    def test_reset_method_uses_reset_novelty_caches(self):
        """Memory.reset() clears all novelty state via _reset_novelty_caches."""
        mem = _build_sage_memory()
        mem.novelty_scorer._scopes["sk"] = MagicMock()
        mem._hydration_locks["sk"] = threading.Lock()

        clear_scopes = MagicMock()
        reset_all = MagicMock()
        mem.novelty_scorer.clear_all_scopes = clear_scopes
        mem.adaptive_threshold.reset_all = reset_all

        mem.db = MagicMock()
        mem.db.connection = None

        with patch("mem0.memory.main.VectorStoreFactory.reset", return_value=MagicMock()):
            with patch("mem0.memory.main.capture_event"):
                mem.reset()

        clear_scopes.assert_called_once()
        reset_all.assert_called_once()
        assert mem._hydration_locks == {}


class TestPR3InvalidateNovelyScope:
    def test_invalidate_novelty_scope_public_api(self):
        """invalidate_novelty_scope marks scope unhydrated and resets threshold."""
        mem = _build_sage_memory()

        mark_unhydrated = MagicMock()
        reset_scope = MagicMock()
        mem.novelty_scorer.mark_unhydrated = mark_unhydrated
        mem.adaptive_threshold.reset_scope = reset_scope

        mem.invalidate_novelty_scope("user99||")

        mark_unhydrated.assert_called_once_with("user99||")
        reset_scope.assert_called_once_with("user99||")

    def test_invalidate_novelty_scope_noop_when_disabled(self):
        """invalidate_novelty_scope is a no-op when enable_sage is False."""
        mem = _build_sage_memory()
        mem.enable_sage = False
        mem.invalidate_novelty_scope("sk")  # should not raise


class TestPR3HydrationLockMapCap:
    def test_hydration_lock_map_capped_under_churn(self):
        """After 2000 distinct scope keys, _hydration_locks stays at or below 1024."""
        mem = _build_sage_memory()

        # Make is_hydrated_fresh always return True so we skip the slow path
        # but still trigger lock-map creation on the fast-path miss → slow-path entry
        original_fresh = mem.novelty_scorer.is_hydrated_fresh

        # We need to go through _ensure_scope_hydrated to trigger lock creation.
        # Mock the body so it's fast but lock-map code still runs.
        mem.vector_store.iter_all = MagicMock(return_value=iter([]))

        for i in range(2000):
            scope_key = f"user-{i}||"
            # The first call for each key goes through the slow path (not fresh)
            # so the lock-creation code runs. Mark it hydrated after to avoid re-run.
            mem.novelty_scorer.mark_unhydrated(scope_key)
            mem._ensure_scope_hydrated(scope_key, {"user_id": f"user-{i}"})

        assert len(mem._hydration_locks) <= 1024, (
            f"Lock map grew to {len(mem._hydration_locks)}, expected <= 1024"
        )
