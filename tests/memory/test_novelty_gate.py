import threading
import time

import pytest

from mem0.memory.novelty_gate import AdaptiveThreshold, NoveltyScorer


def test_novelty_scorer_higher_for_far_candidate():
    scorer = NoveltyScorer(input_dim=4, projected_dim=2, k_neighbors=3, seed=42)
    scope_key = "user-a||"
    memory_ids = ["m1", "m2", "m3"]
    memory_texts = ["a", "b", "c"]
    memory_embeddings = [
        [0.0, 0.0, 0.0, 0.0],
        [0.1, 0.1, 0.1, 0.1],
        [0.2, 0.2, 0.2, 0.2],
    ]
    scorer.sync_scope(scope_key, memory_ids, memory_texts, memory_embeddings)

    near_novelty = scorer.score(scope_key, [0.12, 0.12, 0.12, 0.12]).novelty
    far_novelty = scorer.score(scope_key, [8.0, 8.0, 8.0, 8.0]).novelty

    assert far_novelty > near_novelty


def test_novelty_scorer_higher_for_far_candidate_with_pca():
    scorer = NoveltyScorer(input_dim=4, projected_dim=2, k_neighbors=3, seed=42, reduction_method="pca")
    scope_key = "user-a||pca"
    memory_ids = ["m1", "m2", "m3"]
    memory_texts = ["a", "b", "c"]
    memory_embeddings = [
        [0.0, 0.0, 0.0, 0.0],
        [0.1, 0.1, 0.1, 0.1],
        [0.2, 0.2, 0.2, 0.2],
    ]
    scorer.sync_scope(scope_key, memory_ids, memory_texts, memory_embeddings)

    near_novelty = scorer.score(scope_key, [0.12, 0.12, 0.12, 0.12]).novelty
    far_novelty = scorer.score(scope_key, [8.0, 8.0, 8.0, 8.0]).novelty

    assert far_novelty > near_novelty


def test_vmf_novelty_scorer_higher_for_directionally_far_candidate():
    scorer = NoveltyScorer(input_dim=4, projected_dim=2, k_neighbors=3, seed=42, novelty_method="vmf_kde")
    scope_key = "user-a||vmf"
    memory_ids = ["m1", "m2", "m3"]
    memory_texts = ["a", "b", "c"]
    memory_embeddings = [
        [1.0, 0.0, 0.0, 0.0],
        [0.95, 0.1, 0.0, 0.0],
        [0.9, 0.2, 0.0, 0.0],
    ]
    scorer.sync_scope(scope_key, memory_ids, memory_texts, memory_embeddings)

    near_result = scorer.score(scope_key, [0.99, 0.01, 0.0, 0.0])
    far_result = scorer.score(scope_key, [-1.0, 0.0, 0.0, 0.0])

    assert near_result.nearest_id == "m1"
    assert far_result.novelty > near_result.novelty


def test_novelty_scorer_rejects_invalid_novelty_method():
    with pytest.raises(ValueError, match="Unsupported novelty method"):
        NoveltyScorer(input_dim=4, projected_dim=2, novelty_method="bad_method")


def test_adaptive_threshold_tracks_density():
    threshold = AdaptiveThreshold(tau_0=0.85, tau_min=0.25, density_lambda=2.0, ema_alpha=0.9, hysteresis_delta=0.15)
    scope_key = "scope-1"

    sparse = threshold.update(scope_key, memory_count=20, volume=500.0)
    dense = threshold.update(scope_key, memory_count=2000, volume=500.0)
    sparse_again = threshold.update(scope_key, memory_count=5, volume=1000.0)

    assert dense < sparse
    assert sparse_again > dense


def test_three_way_gate_routing():
    threshold = AdaptiveThreshold(hysteresis_delta=0.15)
    current_threshold = 1.0

    assert threshold.route(1.20, current_threshold) == "ADD"
    assert threshold.route(1.05, current_threshold) == "UPDATE"
    assert threshold.route(0.99, current_threshold) == "NONE"


def test_add_to_scope_appends_memory_state():
    scorer = NoveltyScorer(input_dim=3, projected_dim=2, k_neighbors=2, seed=42)
    scope_key = "scope-add"
    scorer.sync_scope(scope_key, ["m1"], ["first"], [[0.1, 0.1, 0.1]])

    scorer.add_to_scope(scope_key, "m2", "second", [0.2, 0.2, 0.2])

    ids, texts, vectors = scorer.get_scope_snapshot(scope_key)
    assert ids == ["m1", "m2"]
    assert texts == ["first", "second"]
    assert vectors.shape == (2, 3)


def test_update_scope_memory_replaces_existing_embedding_and_text():
    scorer = NoveltyScorer(input_dim=3, projected_dim=2, k_neighbors=2, seed=42)
    scope_key = "scope-update"
    scorer.sync_scope(scope_key, ["m1", "m2"], ["first", "second"], [[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]])

    scorer.update_scope_memory(scope_key, "m1", "first-updated", [0.9, 0.8, 0.7])

    ids, texts, vectors = scorer.get_scope_snapshot(scope_key)
    assert ids == ["m1", "m2"]
    assert texts[0] == "first-updated"
    assert vectors[0].tolist() == pytest.approx([0.9, 0.8, 0.7])


# --- scope mutation tests ---

def test_remove_from_scope_drops_memory():
    scorer = NoveltyScorer(input_dim=4, projected_dim=2, k_neighbors=3, seed=42)
    scope_key = "scope-remove"
    scorer.sync_scope(scope_key, ["m1", "m2", "m3"], ["a", "b", "c"],
                      [[0.1]*4, [0.2]*4, [0.3]*4])
    scorer.remove_from_scope(scope_key, "m2")
    assert scorer.scope_size(scope_key) == 2
    ids, _, _ = scorer.get_scope_snapshot(scope_key)
    assert "m2" not in ids


def test_remove_from_scope_idempotent_on_missing_id():
    scorer = NoveltyScorer(input_dim=4, projected_dim=2, seed=42)
    scope_key = "scope-idempotent"
    scorer.sync_scope(scope_key, ["m1"], ["a"], [[0.1]*4])
    scorer.remove_from_scope(scope_key, "nonexistent")  # must not raise
    assert scorer.scope_size(scope_key) == 1


def test_scope_lock_serializes_concurrent_add_to_scope():
    """2 threads each adding 50 distinct IDs: final size must be 100 with no dupes."""
    scorer = NoveltyScorer(input_dim=4, projected_dim=2, seed=42)
    scope_key = "scope-concurrent"
    errors = []

    def worker(offset):
        for i in range(50):
            try:
                mid = f"m{offset+i}"
                scorer.add_to_scope(scope_key, mid, f"text{offset+i}", [float(offset+i)*0.01]*4)
            except Exception as e:
                errors.append(e)

    t1 = threading.Thread(target=worker, args=(0,))
    t2 = threading.Thread(target=worker, args=(50,))
    t1.start(); t2.start()
    t1.join(); t2.join()
    assert not errors
    assert scorer.scope_size(scope_key) == 100
    ids, _, _ = scorer.get_scope_snapshot(scope_key)
    assert len(set(ids)) == 100


def test_score_and_add_concurrent():
    """score() and add_to_scope() interleaved across 2 threads must not crash."""
    scorer = NoveltyScorer(input_dim=4, projected_dim=2, seed=42)
    scope_key = "scope-score-race"
    scorer.sync_scope(scope_key, ["m0"], ["init"], [[0.1]*4])
    errors = []

    def adder():
        for i in range(100):
            try:
                scorer.add_to_scope(scope_key, f"add-{i}", f"text{i}", [float(i)*0.001]*4)
            except Exception as e:
                errors.append(("add", e))

    def scorer_fn():
        for _ in range(200):
            try:
                scorer.score(scope_key, [0.5]*4)
            except Exception as e:
                errors.append(("score", e))

    t1 = threading.Thread(target=adder)
    t2 = threading.Thread(target=scorer_fn)
    t1.start(); t2.start()
    t1.join(); t2.join()
    assert not errors, f"Race errors: {errors[:3]}"


def test_hydration_ttl_expires():
    scorer = NoveltyScorer(input_dim=4, projected_dim=2, seed=42, hydration_ttl_seconds=0.05)
    scope_key = "scope-ttl"
    scorer.sync_scope(scope_key, ["m1"], ["a"], [[0.1]*4])
    scorer.mark_hydrated(scope_key)
    assert scorer.is_hydrated_fresh(scope_key)
    time.sleep(0.1)
    assert not scorer.is_hydrated_fresh(scope_key)


def test_clear_all_scopes_invalidates_everything():
    scorer = NoveltyScorer(input_dim=4, projected_dim=2, seed=42)
    for i in range(3):
        scorer.sync_scope(f"scope-{i}", ["m1"], ["a"], [[0.1]*4])
        scorer.mark_hydrated(f"scope-{i}")
    scorer.clear_all_scopes()
    for i in range(3):
        assert not scorer.is_hydrated(f"scope-{i}")
        assert scorer.scope_size(f"scope-{i}") == 0


def test_adaptive_threshold_reset_scope():
    threshold = AdaptiveThreshold()
    threshold.update("scope-a", 10, 1.0)
    threshold.update("scope-b", 10, 1.0)
    threshold.reset_scope("scope-a")
    assert "scope-a" not in threshold._threshold_by_scope
    assert "scope-b" in threshold._threshold_by_scope


def test_adaptive_threshold_reset_all():
    threshold = AdaptiveThreshold()
    threshold.update("scope-a", 10, 1.0)
    threshold.update("scope-b", 10, 1.0)
    threshold.reset_all()
    assert len(threshold._threshold_by_scope) == 0
