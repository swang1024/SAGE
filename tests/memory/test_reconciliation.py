"""Advanced reconciliation modes and observability tests."""
import threading
from dataclasses import asdict
from unittest.mock import MagicMock, patch

import pytest

from mem0.configs.base import MemoryConfig
from mem0.memory.main import Memory
from mem0.memory.novelty_gate import NoveltyMetrics, NoveltyScorer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_sage_memory(reconciliation_mode="off"):
    cfg = MemoryConfig(enable_sage=True, novelty_reconciliation_mode=reconciliation_mode)
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


def _make_point(memory_id, user_id=None, agent_id=None, run_id=None, vec=None, data="text"):
    payload = {"data": data}
    if user_id:
        payload["user_id"] = user_id
    if agent_id:
        payload["agent_id"] = agent_id
    if run_id:
        payload["run_id"] = run_id
    p = MagicMock()
    p.id = memory_id
    p.payload = payload
    p.vector = vec
    return p


# ---------------------------------------------------------------------------
# Test: id_set reconciliation detects simultaneous add + remove
# ---------------------------------------------------------------------------

class TestIdSetReconciliation:
    def test_id_set_detects_add_and_remove_drift(self):
        """id_set mode: when live IDs differ from cache, scope is re-hydrated."""
        mem = _build_sage_memory(reconciliation_mode="id_set")
        scope_key = "user-42||"

        # Pre-populate scorer with IDs {m1, m2}
        mem.novelty_scorer.sync_scope(scope_key, ["m1", "m2"], ["a", "b"], [[0.1] * 4, [0.2] * 4])
        mem.novelty_scorer.mark_hydrated(scope_key)
        assert mem.novelty_scorer.is_hydrated_fresh(scope_key)

        # Simulate external add m3 + remove m1 (count unchanged, but set differs)
        mem.vector_store.iter_ids = MagicMock(return_value=iter(["m2", "m3"]))

        # iter_all for the subsequent cold hydration returns m2 + m3
        m2 = _make_point("m2", user_id="user-42", vec=[0.2] * 4)
        m3 = _make_point("m3", user_id="user-42", vec=[0.3] * 4)
        mem.vector_store.iter_all = MagicMock(return_value=iter([m2, m3]))

        search_filters = {"user_id": "user-42"}
        mem._ensure_scope_hydrated(scope_key, search_filters)

        # Drift should have been detected and scope re-hydrated
        metrics = mem.get_novelty_metrics()
        assert metrics["hydration_id_set_drifts"] >= 1
        # After re-hydration, scope contains m2 + m3
        ids, _, _ = mem.novelty_scorer.get_scope_snapshot(scope_key)
        assert set(ids) == {"m2", "m3"}

    def test_id_set_no_drift_stays_warm(self):
        """id_set mode: when live IDs match cache, scope stays warm (no re-hydration)."""
        mem = _build_sage_memory(reconciliation_mode="id_set")
        scope_key = "user-43||"

        mem.novelty_scorer.sync_scope(scope_key, ["m1", "m2"], ["a", "b"], [[0.1] * 4, [0.2] * 4])
        mem.novelty_scorer.mark_hydrated(scope_key)

        # Live IDs match cached IDs exactly
        mem.vector_store.iter_ids = MagicMock(return_value=iter(["m1", "m2"]))
        iter_all_mock = MagicMock()
        mem.vector_store.iter_all = iter_all_mock

        mem._ensure_scope_hydrated(scope_key, {"user_id": "user-43"})

        # No re-hydration should have run
        iter_all_mock.assert_not_called()
        metrics = mem.get_novelty_metrics()
        assert metrics["hydration_id_set_drifts"] == 0
        assert metrics["hydration_warm_hits"] >= 1


# ---------------------------------------------------------------------------
# Test: id_version reconciliation detects in-place update
# ---------------------------------------------------------------------------

class TestIdVersionReconciliation:
    def test_id_version_detects_inplace_update(self):
        """id_version mode: when payload version changes for an existing id, scope is re-hydrated."""
        mem = _build_sage_memory(reconciliation_mode="id_version")
        scope_key = "user-50||"

        mem.novelty_scorer.sync_scope(scope_key, ["m1"], ["old text"], [[0.1] * 4])
        mem.novelty_scorer.mark_hydrated(scope_key)
        # Store original version
        mem.novelty_scorer.set_id_versions(scope_key, {"m1": 111})

        # Simulate external in-place update: same ID m1, but version changed
        mem.vector_store.iter_id_versions = MagicMock(return_value=iter([("m1", 999)]))

        m1_new = _make_point("m1", user_id="user-50", vec=[0.5] * 4, data="updated text")
        mem.vector_store.iter_all = MagicMock(return_value=iter([m1_new]))

        mem._ensure_scope_hydrated(scope_key, {"user_id": "user-50"})

        metrics = mem.get_novelty_metrics()
        assert metrics["hydration_id_version_drifts"] >= 1
        # Scope re-hydrated with updated text
        ids, texts, _ = mem.novelty_scorer.get_scope_snapshot(scope_key)
        assert "m1" in ids
        assert texts[0] == "updated text"

    def test_id_version_no_drift_stays_warm(self):
        """id_version mode: when all versions match, scope stays warm."""
        mem = _build_sage_memory(reconciliation_mode="id_version")
        scope_key = "user-51||"

        mem.novelty_scorer.sync_scope(scope_key, ["m1"], ["text"], [[0.1] * 4])
        mem.novelty_scorer.mark_hydrated(scope_key)
        mem.novelty_scorer.set_id_versions(scope_key, {"m1": 42})

        mem.vector_store.iter_id_versions = MagicMock(return_value=iter([("m1", 42)]))
        iter_all_mock = MagicMock()
        mem.vector_store.iter_all = iter_all_mock

        mem._ensure_scope_hydrated(scope_key, {"user_id": "user-51"})

        iter_all_mock.assert_not_called()
        metrics = mem.get_novelty_metrics()
        assert metrics["hydration_id_version_drifts"] == 0
        assert metrics["hydration_warm_hits"] >= 1

    def test_id_version_populated_after_hydration(self):
        """After hydration in id_version mode, id_versions is set on the scope."""
        mem = _build_sage_memory(reconciliation_mode="id_version")
        scope_key = "user-52||"

        # hash and updated_at in payload are used to compute version
        m1 = MagicMock()
        m1.id = "m1"
        m1.payload = {"data": "hello", "hash": "abc123", "updated_at": "2026-01-01T00:00:00+00:00"}
        m1.vector = [0.1] * 4

        mem.vector_store.iter_id_versions = MagicMock(return_value=iter([]))  # not hydrated yet → cold
        mem.vector_store.iter_all = MagicMock(return_value=iter([m1]))

        mem._ensure_scope_hydrated(scope_key, {"user_id": "user-52"})

        versions = mem.novelty_scorer.get_id_versions(scope_key)
        assert "m1" in versions
        assert isinstance(versions["m1"], int)


# ---------------------------------------------------------------------------
# Test: off mode does not call iter_ids
# ---------------------------------------------------------------------------

class TestOffMode:
    def test_off_mode_does_not_call_iter_ids(self):
        """Default 'off' mode: reconciliation check is skipped entirely."""
        mem = _build_sage_memory(reconciliation_mode="off")
        scope_key = "user-60||"

        mem.novelty_scorer.sync_scope(scope_key, ["m1"], ["a"], [[0.1] * 4])
        mem.novelty_scorer.mark_hydrated(scope_key)

        iter_ids_mock = MagicMock()
        iter_id_versions_mock = MagicMock()
        mem.vector_store.iter_ids = iter_ids_mock
        mem.vector_store.iter_id_versions = iter_id_versions_mock

        mem._ensure_scope_hydrated(scope_key, {"user_id": "user-60"})

        iter_ids_mock.assert_not_called()
        iter_id_versions_mock.assert_not_called()
        metrics = mem.get_novelty_metrics()
        assert metrics["hydration_id_set_drifts"] == 0
        assert metrics["hydration_id_version_drifts"] == 0
        assert metrics["hydration_warm_hits"] >= 1


# ---------------------------------------------------------------------------
# Test: get_novelty_metrics increments counters
# ---------------------------------------------------------------------------

class TestGetNoveltyMetrics:
    def test_cold_miss_increments(self):
        """Cold hydration increments hydration_cold_misses."""
        mem = _build_sage_memory()
        scope_key = "user-70||"
        mem.vector_store.iter_all = MagicMock(return_value=iter([]))

        mem._ensure_scope_hydrated(scope_key, {"user_id": "user-70"})

        metrics = mem.get_novelty_metrics()
        assert metrics["hydration_cold_misses"] == 1
        assert metrics["hydration_warm_hits"] == 0

    def test_warm_hit_increments(self):
        """Second call to same fresh scope increments hydration_warm_hits."""
        mem = _build_sage_memory()
        scope_key = "user-71||"
        mem.vector_store.iter_all = MagicMock(return_value=iter([]))

        mem._ensure_scope_hydrated(scope_key, {"user_id": "user-71"})  # cold
        mem._ensure_scope_hydrated(scope_key, {"user_id": "user-71"})  # warm

        metrics = mem.get_novelty_metrics()
        assert metrics["hydration_cold_misses"] == 1
        assert metrics["hydration_warm_hits"] == 1

    def test_double_check_avoided_increments(self):
        """When two threads race for the same cold scope, the loser increments double_check_avoided."""
        mem = _build_sage_memory()
        scope_key = "user-72||"
        errors = []

        # Add a small delay on iter_all so both threads enter the lock region concurrently
        def slow_iter_all(*args, **kwargs):
            import time
            time.sleep(0.05)
            return iter([])

        mem.vector_store.iter_all = MagicMock(side_effect=slow_iter_all)

        def worker():
            try:
                mem._ensure_scope_hydrated(scope_key, {"user_id": "user-72"})
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=worker)
        t2 = threading.Thread(target=worker)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert not errors
        metrics = mem.get_novelty_metrics()
        # One thread did the cold hydration, the other was avoided by the double-check
        assert metrics["hydration_cold_misses"] + metrics["hydration_double_check_avoided"] == 2
        assert metrics["hydration_cold_misses"] == 1

    def test_incremental_counters_on_add_update_remove(self):
        """add_to_scope / update_scope_memory / remove_from_scope each increment their counter."""
        scorer = NoveltyScorer(input_dim=4, projected_dim=2, seed=42)
        scope_key = "s"
        scorer.sync_scope(scope_key, [], [], [])

        scorer.add_to_scope(scope_key, "m1", "text1", [0.1] * 4)
        assert scorer._metrics.incremental_adds == 1

        scorer.update_scope_memory(scope_key, "m1", "text1-updated", [0.2] * 4)
        assert scorer._metrics.incremental_updates == 1

        scorer.remove_from_scope(scope_key, "m1")
        assert scorer._metrics.incremental_removes == 1
        assert scorer._metrics.incremental_adds == 1  # unchanged

    def test_add_to_scope_redirected_update_not_double_counted(self):
        """add_to_scope for an existing id delegates to update_scope_memory; counts as update."""
        scorer = NoveltyScorer(input_dim=4, projected_dim=2, seed=42)
        scope_key = "s2"
        scorer.sync_scope(scope_key, ["m1"], ["text1"], [[0.1] * 4])

        scorer.add_to_scope(scope_key, "m1", "text1-new", [0.2] * 4)  # id exists → update

        # Should count as update, not an add
        assert scorer._metrics.incremental_adds == 0
        assert scorer._metrics.incremental_updates == 1

    def test_remove_idempotent_does_not_increment(self):
        """remove_from_scope for a missing id does not increment removes."""
        scorer = NoveltyScorer(input_dim=4, projected_dim=2, seed=42)
        scorer.sync_scope("s3", ["m1"], ["a"], [[0.1] * 4])
        scorer.remove_from_scope("s3", "nonexistent")
        assert scorer._metrics.incremental_removes == 0

    def test_get_novelty_metrics_returns_empty_when_disabled(self):
        """get_novelty_metrics returns {} when SAGE is disabled."""
        cfg = MemoryConfig(enable_sage=False)
        with (
            patch("mem0.memory.main.MEM0_TELEMETRY", False),
            patch("mem0.utils.factory.EmbedderFactory.create"),
            patch("mem0.memory.main.VectorStoreFactory.create"),
            patch("mem0.utils.factory.LlmFactory.create"),
            patch("mem0.memory.main.SQLiteManager"),
            patch("mem0.memory.main.capture_event"),
        ):
            mem = Memory(cfg)
        assert mem.get_novelty_metrics() == {}


# ---------------------------------------------------------------------------
# Test: telemetry event emitted on hydration
# ---------------------------------------------------------------------------

class TestMetricsTelemetryEvent:
    def test_telemetry_event_emitted_on_cold_hydration(self):
        """capture_event('mem0.novelty.metrics') is called after a cold hydration."""
        cfg = MemoryConfig(enable_sage=True)
        events = []

        def capture_event_spy(event_name, *args, **kwargs):
            events.append(event_name)

        with (
            patch("mem0.memory.main.MEM0_TELEMETRY", True),
            patch("mem0.utils.factory.EmbedderFactory.create") as mock_emb,
            patch("mem0.memory.main.VectorStoreFactory.create") as mock_vs,
            patch("mem0.utils.factory.LlmFactory.create"),
            patch("mem0.memory.main.SQLiteManager"),
            patch("mem0.memory.main.capture_event", side_effect=capture_event_spy),
        ):
            mock_emb.return_value = MagicMock()
            mock_vs.return_value = MagicMock()
            mem = Memory(cfg)

        mem.vector_store.iter_all = MagicMock(return_value=iter([]))

        with patch("mem0.memory.main.capture_event", side_effect=capture_event_spy):
            mem._ensure_scope_hydrated("user-99||", {"user_id": "user-99"})

        assert "mem0.novelty.metrics" in events

    def test_telemetry_event_not_emitted_on_warm_hit(self):
        """capture_event('mem0.novelty.metrics') is NOT called on warm hits."""
        cfg = MemoryConfig(enable_sage=True)
        events = []

        def capture_event_spy(event_name, *args, **kwargs):
            events.append(event_name)

        with (
            patch("mem0.memory.main.MEM0_TELEMETRY", True),
            patch("mem0.utils.factory.EmbedderFactory.create") as mock_emb,
            patch("mem0.memory.main.VectorStoreFactory.create") as mock_vs,
            patch("mem0.utils.factory.LlmFactory.create"),
            patch("mem0.memory.main.SQLiteManager"),
            patch("mem0.memory.main.capture_event", side_effect=capture_event_spy),
        ):
            mock_emb.return_value = MagicMock()
            mock_vs.return_value = MagicMock()
            mem = Memory(cfg)

        scope_key = "user-100||"
        # Pre-hydrate
        mem.novelty_scorer.sync_scope(scope_key, [], [], [])
        mem.novelty_scorer.mark_hydrated(scope_key)

        events.clear()
        with patch("mem0.memory.main.capture_event", side_effect=capture_event_spy):
            mem._ensure_scope_hydrated(scope_key, {"user_id": "user-100"})

        assert "mem0.novelty.metrics" not in events


# ---------------------------------------------------------------------------
# Test: NoveltyMetrics dataclass
# ---------------------------------------------------------------------------

class TestNoveltyMetricsDataclass:
    def test_all_fields_default_to_zero(self):
        m = NoveltyMetrics()
        for val in asdict(m).values():
            assert val == 0

    def test_get_metrics_returns_same_object(self):
        scorer = NoveltyScorer(input_dim=4, projected_dim=2, seed=42)
        assert scorer.get_metrics() is scorer._metrics

    def test_id_versions_round_trip(self):
        scorer = NoveltyScorer(input_dim=4, projected_dim=2, seed=42)
        scope_key = "scope-ver"
        scorer.sync_scope(scope_key, ["m1"], ["a"], [[0.1] * 4])
        scorer.set_id_versions(scope_key, {"m1": 42, "m2": 99})
        got = scorer.get_id_versions(scope_key)
        assert got == {"m1": 42, "m2": 99}

    def test_get_id_versions_missing_scope_returns_empty(self):
        scorer = NoveltyScorer(input_dim=4, projected_dim=2, seed=42)
        assert scorer.get_id_versions("nonexistent") == {}

    def test_get_scope_ids_missing_scope_returns_empty(self):
        scorer = NoveltyScorer(input_dim=4, projected_dim=2, seed=42)
        assert scorer.get_scope_ids("nonexistent") == []

    def test_get_scope_ids_returns_copy(self):
        scorer = NoveltyScorer(input_dim=4, projected_dim=2, seed=42)
        scope_key = "scope-ids"
        scorer.sync_scope(scope_key, ["m1", "m2"], ["a", "b"], [[0.1] * 4, [0.2] * 4])
        ids = scorer.get_scope_ids(scope_key)
        ids.append("injected")  # mutating the returned list must not affect the scope
        assert scorer.scope_size(scope_key) == 2
