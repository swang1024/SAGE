"""
Thread-safety tests for Qdrant._client_lock.

These tests verify that concurrent client.* calls are serialized by the lock,
preventing the scroll-vs-upsert race that caused crashes under concurrent
Memory.add() workloads with the local embedded Qdrant client.
"""
import threading
import time
import unittest
from unittest.mock import MagicMock

from qdrant_client import QdrantClient

from mem0.vector_stores.qdrant import Qdrant


def _make_qdrant() -> Qdrant:
    """Build a Qdrant instance with a mock client (no real server needed)."""
    client_mock = MagicMock(spec=QdrantClient)
    client_mock.get_collections.return_value = MagicMock(collections=[])
    return Qdrant(
        collection_name="test_col",
        embedding_model_dims=4,
        client=client_mock,
    )


class TestClientLockExists(unittest.TestCase):
    def test_client_lock_is_rlock(self):
        q = _make_qdrant()
        # RLock instances don't share a public type; use the canonical check.
        self.assertTrue(hasattr(q, "_client_lock"))
        # Verify it behaves as a reentrant lock (acquire twice without deadlock).
        q._client_lock.acquire()
        q._client_lock.acquire()
        q._client_lock.release()
        q._client_lock.release()


class TestConcurrentCallsSerialize(unittest.TestCase):
    """Verify that no two client.* calls execute simultaneously."""

    def _run_concurrent(self, n_readers=4, n_writers=4, call_delay=0.02):
        q = _make_qdrant()

        concurrency_counter = [0]
        max_concurrency = [0]
        counter_lock = threading.Lock()
        errors = []

        def _track_entry_exit(fn):
            def wrapped(*args, **kwargs):
                with counter_lock:
                    concurrency_counter[0] += 1
                    if concurrency_counter[0] > max_concurrency[0]:
                        max_concurrency[0] = concurrency_counter[0]
                try:
                    time.sleep(call_delay)
                    return fn(*args, **kwargs)
                finally:
                    with counter_lock:
                        concurrency_counter[0] -= 1

            return wrapped

        q.client.scroll.side_effect = _track_entry_exit(lambda **kw: ([], None))
        q.client.upsert.side_effect = _track_entry_exit(lambda **kw: None)

        threads = []
        for _ in range(n_readers):
            threads.append(threading.Thread(target=q.list, args=(None,)))
        for i in range(n_writers):
            threads.append(
                threading.Thread(
                    target=q.insert,
                    args=([[0.1, 0.2, 0.3, 0.4]],),
                    kwargs={"payloads": [{"data": f"mem{i}"}], "ids": [i]},
                )
            )

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        return max_concurrency[0], errors

    def test_scroll_and_upsert_do_not_overlap(self):
        max_concurrency, errors = self._run_concurrent(n_readers=4, n_writers=4)
        self.assertEqual(
            max_concurrency,
            1,
            f"Expected at most 1 concurrent client call, observed {max_concurrency}",
        )
        self.assertEqual(errors, [])

    def test_all_reader_threads_complete(self):
        """Serialization must not cause any thread to starve or deadlock."""
        q = _make_qdrant()
        q.client.scroll.return_value = ([], None)
        completed = []

        def reader():
            q.list(None)
            completed.append(1)

        threads = [threading.Thread(target=reader) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        self.assertEqual(len(completed), 8, "Not all reader threads completed")

    def test_all_writer_threads_complete(self):
        """No writer should be lost or deadlock under concurrent writes."""
        q = _make_qdrant()
        q.client.upsert.return_value = None
        completed = []

        def writer(i):
            q.insert([[0.1, 0.2, 0.3, 0.4]], payloads=[{"data": f"m{i}"}], ids=[i])
            completed.append(1)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        self.assertEqual(len(completed), 8, "Not all writer threads completed")
