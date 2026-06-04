import logging
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Literal, Optional, Sequence

import numpy as np

try:
    import faiss  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    faiss = None

CREATE_EVENT = "ADD"
UPDATE_EVENT = "UPDATE"
NOOP_EVENT = "NONE"
RANDOM_PROJECTION_METHOD = "random_projection"
PCA_PROJECTION_METHOD = "pca"
GAUSSIAN_NOVELTY_METHOD = "gaussian_kde"
VMF_NOVELTY_METHOD = "vmf_kde"


UPDATE_MERGE_PROMPT = (
    "You are merging a new fact into an existing memory about the same subject.\n"
    "Optimize for two things at once: (1) RECALL - never lose a specific detail; "
    "(2) PRECISION - keep the memory short and tightly focused so it is easy to retrieve.\n"
    "\n"
    "Existing memory: {existing}\n"
    "New information: {new}\n"
    "\n"
    "Rules:\n"
    "- Keep EVERY proper noun, person name, place, organization, number, and date "
    "from BOTH memories. Never generalize a specific "
    "(do not turn \"Rob\" into \"a colleague\").\n"
    "- Multi-value facts are ADDITIVE: merge their items into one compact list; do "
    "not drop or replace items.\n"
    "  \"James visited Italy\" + \"James visited Mexico\" -> "
    "\"James visited Italy and Mexico\".\n"
    "  \"John was offered a Nike deal\" + \"a Gatorade deal\" -> "
    "\"John was offered Nike and Gatorade deals\".\n"
    "- Only OVERWRITE when the new fact changes the SAME single-valued attribute "
    "(current employer, current city, status):\n"
    "  \"Works at Google\" + \"Works at Meta\" -> \"Works at Meta\".\n"
    "- If one memory fully contains the other, keep the more detailed one verbatim.\n"
    "- If the two facts are about DIFFERENT subjects, do NOT blend their attributes; "
    "keep each as its own clause so neither is corrupted.\n"
    "- When unsure whether two details conflict, KEEP BOTH.\n"
    "- Do not invent anything not stated in either memory.\n"
    "\n"
    "Compress wording, never content: prefer compact comma-separated lists over "
    "repetition, and stay on the single shared topic. Return ONLY the merged memory, "
    "as short as possible while retaining every specific. No explanation."
)


# Initialize logger early for util functions
logger = logging.getLogger(__name__)

@dataclass
class NoveltyMetrics:
    """Cumulative counters for novelty-gate hydration and incremental operations."""
    hydration_cold_misses: int = 0
    hydration_ttl_expirations: int = 0
    hydration_id_set_drifts: int = 0
    hydration_id_version_drifts: int = 0
    hydration_warm_hits: int = 0
    hydration_double_check_avoided: int = 0
    vectors_missing_refallback_count: int = 0
    incremental_adds: int = 0
    incremental_updates: int = 0
    incremental_removes: int = 0


@dataclass
class NoveltyResult:
    novelty: float
    nearest_id: Optional[str]


@dataclass
class _ScopeState:
    ids: List[str] = field(default_factory=list)
    texts: Dict[str, str] = field(default_factory=dict)
    vectors: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))
    unit_vectors: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))
    projected_vectors: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))
    vmf_kappa: float = 1.0
    pca_components: Optional[np.ndarray] = None
    pca_mean: Optional[np.ndarray] = None
    turns: int = 0
    index: Optional[object] = None
    # hydration state and per-scope lock
    hydrated: bool = False
    hydrated_at: float = 0.0
    lock: threading.RLock = field(default_factory=threading.RLock)
    # per-point version map for id_version reconciliation mode
    id_versions: Dict[str, int] = field(default_factory=dict)


class NoveltyScorer:
    def __init__(
        self,
        *,
        input_dim: int = 1536,
        projected_dim: int = 64,
        k_neighbors: int = 50,
        n_centroids: int = 100,
        seed: int = 42,
        reduction_method: Literal["random_projection", "pca"] = RANDOM_PROJECTION_METHOD,
        novelty_method: Literal["gaussian_kde", "vmf_kde"] = GAUSSIAN_NOVELTY_METHOD,
        vmf_kappa: Optional[float] = None,
        hydration_ttl_seconds: float = 600.0,
    ):
        """Initialize novelty scoring configuration and per-scope caches."""
        if reduction_method not in (RANDOM_PROJECTION_METHOD, PCA_PROJECTION_METHOD):
            raise ValueError(f"Unsupported reduction method: {reduction_method}")
        if novelty_method not in (GAUSSIAN_NOVELTY_METHOD, VMF_NOVELTY_METHOD):
            raise ValueError(f"Unsupported novelty method: {novelty_method}")
        if vmf_kappa is not None and vmf_kappa <= 0:
            raise ValueError("vmf_kappa must be positive when provided.")
        self.input_dim = input_dim
        self.projected_dim = projected_dim
        self.k_neighbors = k_neighbors
        self.n_centroids = n_centroids
        self.reduction_method = reduction_method
        self.novelty_method = novelty_method
        self.vmf_kappa = vmf_kappa
        self.hydration_ttl_seconds = hydration_ttl_seconds
        self._rng = np.random.default_rng(seed)
        self._projection = self._sample_projection(input_dim)
        self._scopes: Dict[str, _ScopeState] = {}
        self._metrics: NoveltyMetrics = NoveltyMetrics()

    def _normalize_rows(self, vectors: np.ndarray) -> np.ndarray:
        """L2-normalize rows so embeddings lie on the unit hypersphere."""
        if vectors.ndim != 2:
            raise ValueError("Normalization expects a 2D embedding matrix.")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        safe_norms = np.maximum(norms, 1e-12)
        return (vectors / safe_norms).astype(np.float32)

    def _estimate_vmf_kappa(self, unit_vectors: np.ndarray) -> float:
        """Estimate concentration with the Banerjee et al. approximation."""
        if unit_vectors.ndim != 2 or unit_vectors.shape[0] == 0:
            return 1.0
        dim = float(unit_vectors.shape[1])
        mean_direction = np.mean(unit_vectors, axis=0, dtype=np.float32)
        r_bar = float(np.linalg.norm(mean_direction))
        r_bar = float(np.clip(r_bar, 0.0, 1.0 - 1e-6))
        denominator = max(1.0 - (r_bar * r_bar), 1e-6)
        kappa = r_bar * (dim - (r_bar * r_bar)) / denominator
        if not math.isfinite(kappa):
            return 1.0

        max_kappa = 50.0 # practical upper bound to prevent logsumexp collapse
        return float(np.clip(kappa, 1e-3, max_kappa))

    def _sample_projection(self, input_dim: int) -> np.ndarray:
        """Sample a Gaussian random projection matrix for random-projection mode."""
        return self._rng.normal(0.0, 1.0 / math.sqrt(self.projected_dim), size=(self.projected_dim, input_dim)).astype(
            np.float32
        )

    def _fit_pca_projection(self, vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Fit PCA components and mean for one scope using an SVD-based implementation."""
        if vectors.ndim != 2:
            raise ValueError("PCA fitting expects a 2D embedding matrix.")
        mean = np.mean(vectors, axis=0, dtype=np.float32).astype(np.float32)
        centered = vectors - mean
        components = np.zeros((self.projected_dim, vectors.shape[1]), dtype=np.float32)
        if vectors.shape[0] > 1:
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
            kept = min(self.projected_dim, vt.shape[0])
            components[:kept] = vt[:kept].astype(np.float32)
        return components, mean

    def _fit_scope_projection(self, scope: _ScopeState) -> None:
        """Fit and apply the active reduction method for one scope."""
        if self.reduction_method == RANDOM_PROJECTION_METHOD:
            scope.pca_components = None
            scope.pca_mean = None
            scope.projected_vectors = self._project(scope.vectors)
            return

        components, mean = self._fit_pca_projection(scope.vectors)
        scope.pca_components = components
        scope.pca_mean = mean
        scope.projected_vectors = self._project(scope.vectors, scope=scope)

    def _ensure_projection_dim(self, input_dim: int) -> None:
        """Recreate reduction state when embedding dimensionality changes."""
        if input_dim != self.input_dim:
            self.input_dim = input_dim
            if self.reduction_method == RANDOM_PROJECTION_METHOD:
                self._projection = self._sample_projection(input_dim)
            for scope in self._scopes.values():
                if scope.vectors.size:
                    if scope.vectors.shape[1] != self.input_dim:
                        raise ValueError("Stored scope vectors have inconsistent embedding dimensions.")
                    self._fit_scope_projection(scope)
                    self._rebuild_index(scope)

    def _project(self, vectors: np.ndarray, *, scope: Optional[_ScopeState] = None) -> np.ndarray:
        """Project raw embeddings into reduced space with random projection or PCA."""
        if vectors.ndim != 2:
            raise ValueError("Projection expects a 2D embedding matrix.")
        if vectors.shape[1] != self.input_dim:
            raise ValueError("Embedding dimensionality does not match scorer input_dim.")
        if self.reduction_method == RANDOM_PROJECTION_METHOD:
            return np.matmul(vectors, self._projection.T).astype(np.float32)
        if scope is None or scope.pca_components is None or scope.pca_mean is None:
            raise ValueError("PCA projection is not fitted for this scope.")
        centered = vectors - scope.pca_mean
        return np.matmul(centered, scope.pca_components.T).astype(np.float32)

    def _scope(self, scope_key: str) -> _ScopeState:
        """Fetch or create mutable state for a memory scope."""
        return self._scopes.setdefault(scope_key, _ScopeState())

    def _rebuild_index(self, scope: _ScopeState) -> None:
        """Build a FAISS IVF index for projected vectors when FAISS is available."""
        if faiss is None or scope.projected_vectors.size == 0:
            scope.index = None
            return
        vectors = scope.projected_vectors.astype(np.float32)
        dim = vectors.shape[1]
        nlist = max(1, min(self.n_centroids, vectors.shape[0]))
        quantizer = faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)
        index.train(vectors)
        index.add(vectors)
        scope.index = index

    def sync_scope(
        self,
        scope_key: str,
        ids: Sequence[str],
        texts: Sequence[str],
        vectors: Sequence[Sequence[float]],
        *,
        force: bool = False,
    ) -> None:
        """Replace one scope's memory IDs/texts/embeddings and refresh search structures."""
        scope = self._scope(scope_key)
        with scope.lock:
            incoming_ids = list(ids)
            if not force and incoming_ids == scope.ids:
                return
            if not incoming_ids:
                scope.ids = []
                scope.texts = {}
                scope.vectors = np.zeros((0, self.input_dim), dtype=np.float32)
                scope.unit_vectors = np.zeros((0, self.input_dim), dtype=np.float32)
                scope.projected_vectors = np.zeros((0, self.projected_dim), dtype=np.float32)
                scope.vmf_kappa = 1.0
                scope.pca_components = None
                scope.pca_mean = None
                scope.index = None
                return

            vectors_np = np.asarray(vectors, dtype=np.float32)
            if vectors_np.ndim != 2:
                raise ValueError("Embeddings must be a 2D array-like structure.")
            self._ensure_projection_dim(vectors_np.shape[1])
            scope.ids = incoming_ids
            scope.texts = {memory_id: text for memory_id, text in zip(ids, texts)}
            scope.vectors = vectors_np
            scope.unit_vectors = self._normalize_rows(vectors_np)
            scope.vmf_kappa = self._estimate_vmf_kappa(scope.unit_vectors)
            self._fit_scope_projection(scope)
            self._rebuild_index(scope)

    def add_to_scope(self, scope_key: str, memory_id: str, text: str, embedding: Sequence[float]) -> None:
        """Append one memory to a scope and refresh scorer state."""
        scope = self._scope(scope_key)
        with scope.lock:
            candidate = np.asarray(embedding, dtype=np.float32)
            if candidate.ndim != 1:
                raise ValueError("Candidate embedding must be a 1D vector.")
            self._ensure_projection_dim(candidate.shape[0])

            ids = list(scope.ids)
            if memory_id in ids:
                # update_scope_memory also acquires scope.lock; RLock allows reentry
                self.update_scope_memory(scope_key, memory_id, text, embedding)
                return

            texts = [scope.texts.get(existing_id, "") for existing_id in ids]
            ids.append(memory_id)
            texts.append(text)

            if scope.vectors.size == 0:
                vectors = candidate.reshape(1, -1)
            else:
                if scope.vectors.shape[1] != candidate.shape[0]:
                    raise ValueError("Embedding dimensionality does not match scorer input_dim.")
                vectors = np.vstack((scope.vectors, candidate.reshape(1, -1)))
            # sync_scope also acquires scope.lock; RLock allows reentry
            self.sync_scope(scope_key, ids, texts, vectors, force=True)
            self._metrics.incremental_adds += 1

    def update_scope_memory(self, scope_key: str, memory_id: str, text: str, embedding: Sequence[float]) -> None:
        """Update one memory's text/embedding in a scope and refresh scorer state."""
        scope = self._scope(scope_key)
        with scope.lock:
            candidate = np.asarray(embedding, dtype=np.float32)
            if candidate.ndim != 1:
                raise ValueError("Candidate embedding must be a 1D vector.")
            self._ensure_projection_dim(candidate.shape[0])

            ids = list(scope.ids)
            if memory_id not in ids:
                self.add_to_scope(scope_key, memory_id, text, embedding)
                return

            if scope.vectors.shape[1] != candidate.shape[0]:
                raise ValueError("Embedding dimensionality does not match scorer input_dim.")

            memory_idx = ids.index(memory_id)
            texts = [scope.texts.get(existing_id, "") for existing_id in ids]
            texts[memory_idx] = text
            vectors = scope.vectors.copy()
            vectors[memory_idx] = candidate
            self.sync_scope(scope_key, ids, texts, vectors, force=True)
            self._metrics.incremental_updates += 1

    def remove_from_scope(self, scope_key: str, memory_id: str) -> None:
        """Remove one memory from a scope and refresh search structures. Idempotent if id missing."""
        scope = self._scope(scope_key)
        with scope.lock:
            if memory_id not in scope.ids:
                return
            idx = scope.ids.index(memory_id)
            new_ids = list(scope.ids)
            new_ids.pop(idx)
            new_texts = [scope.texts.get(mid, "") for mid in new_ids]
            if scope.vectors.size > 0:
                new_vectors = np.delete(scope.vectors, idx, axis=0)
            else:
                new_vectors = scope.vectors
            # sync_scope re-acquires scope.lock (RLock allows reentry)
            self.sync_scope(scope_key, new_ids, new_texts, new_vectors, force=True)
            self._metrics.incremental_removes += 1

    def scope_size(self, scope_key: str) -> int:
        """Return the number of memories currently stored in a scope."""
        scope = self._scope(scope_key)
        with scope.lock:
            return len(scope.ids)

    def scope_volume(self, scope_key: str) -> float:
        """Estimate geometric spread of projected vectors via axis-aligned hyper-rectangle volume."""
        scope = self._scope(scope_key)
        with scope.lock:
            if scope.projected_vectors.shape[0] < 2:
                return 1.0
            mins = np.min(scope.projected_vectors, axis=0)
            maxs = np.max(scope.projected_vectors, axis=0)
            span = np.maximum(maxs - mins, 1e-3)
            return float(np.exp(np.sum(np.log(span))))

    def increment_turn(self, scope_key: str) -> int:
        """Advance and return the turn counter for a scope."""
        scope = self._scope(scope_key)
        with scope.lock:
            scope.turns += 1
            return scope.turns

    def get_turns(self, scope_key: str) -> int:
        """Return the current turn counter for a scope."""
        scope = self._scope(scope_key)
        with scope.lock:
            return scope.turns

    def get_scope_snapshot(self, scope_key: str) -> tuple[List[str], List[str], np.ndarray]:
        """Return a copy of IDs, aligned texts, and embeddings for the scope."""
        scope = self._scope(scope_key)
        with scope.lock:
            texts = [scope.texts.get(memory_id, "") for memory_id in scope.ids]
            return list(scope.ids), texts, scope.vectors.copy()

    # --- Hydration lifecycle accessors ---

    def mark_hydrated(self, scope_key: str) -> None:
        scope = self._scope(scope_key)
        with scope.lock:
            scope.hydrated = True
            scope.hydrated_at = time.monotonic()

    def mark_unhydrated(self, scope_key: str) -> None:
        scope = self._scope(scope_key)
        with scope.lock:
            scope.hydrated = False

    def is_hydrated(self, scope_key: str) -> bool:
        scope = self._scopes.get(scope_key)
        if scope is None:
            return False
        with scope.lock:
            return scope.hydrated

    def hydration_age_seconds(self, scope_key: str) -> float:
        scope = self._scopes.get(scope_key)
        if scope is None:
            return float("inf")
        with scope.lock:
            if not scope.hydrated:
                return float("inf")
            return time.monotonic() - scope.hydrated_at

    def is_hydrated_fresh(self, scope_key: str) -> bool:
        scope = self._scopes.get(scope_key)
        if scope is None:
            return False
        with scope.lock:
            if not scope.hydrated:
                return False
            return (time.monotonic() - scope.hydrated_at) <= self.hydration_ttl_seconds

    def clear_all_scopes(self) -> None:
        """Discard all cached scope states (coarse invalidation)."""
        self._scopes.clear()

    # --- id_version reconciliation and observability ---

    def get_scope_ids(self, scope_key: str) -> List[str]:
        """Return a copy of the IDs currently in a scope."""
        scope = self._scopes.get(scope_key)
        if scope is None:
            return []
        with scope.lock:
            return list(scope.ids)

    def set_id_versions(self, scope_key: str, versions: Dict[str, int]) -> None:
        """Store id→version mapping for a scope (id_version reconciliation mode)."""
        scope = self._scope(scope_key)
        with scope.lock:
            scope.id_versions = dict(versions)

    def get_id_versions(self, scope_key: str) -> Dict[str, int]:
        """Return a copy of the id→version mapping for a scope."""
        scope = self._scopes.get(scope_key)
        if scope is None:
            return {}
        with scope.lock:
            return dict(scope.id_versions)

    def get_metrics(self) -> "NoveltyMetrics":
        """Return the accumulated novelty metrics (not thread-safe snapshot; best-effort)."""
        return self._metrics

    # --- Internal scoring ---

    def _knn(self, scope: _ScopeState, projected_candidate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Find nearest neighbors using FAISS when available, else a NumPy brute-force fallback."""
        if scope.projected_vectors.shape[0] == 0:
            return np.array([], dtype=np.float32), np.array([], dtype=np.int64)
        k = min(self.k_neighbors, scope.projected_vectors.shape[0])
        if scope.index is not None:
            distances, indices = scope.index.search(projected_candidate.reshape(1, -1), k)
            return distances[0], indices[0]

        diffs = scope.projected_vectors - projected_candidate
        distances = np.sum(diffs * diffs, axis=1)
        sorted_idx = np.argsort(distances)[:k]
        return distances[sorted_idx], sorted_idx

    def _score_gaussian(self, scope: _ScopeState, candidate: np.ndarray) -> NoveltyResult:
        """Compute novelty from Gaussian-kernel density in projected space."""
        projected_candidate = self._project(candidate.reshape(1, -1), scope=scope)[0]
        distances, indices = self._knn(scope, projected_candidate)
        if distances.size == 0:
            return NoveltyResult(novelty=float("inf"), nearest_id=None)

        norms = np.linalg.norm(scope.projected_vectors, axis=1)
        sigma = float(np.std(norms))
        sigma = sigma if sigma > 1e-6 else 1.0
        n_points = max(1, scope.projected_vectors.shape[0])
        bandwidth = sigma * (n_points ** (-1.0 / (self.projected_dim + 4)))
        bandwidth = max(bandwidth, 1e-6)
        kernel_scale = 1.0 / (((2.0 * math.pi * (bandwidth**2)) ** (self.projected_dim / 2.0)))
        kernels = kernel_scale * np.exp(-(distances / (2.0 * bandwidth * bandwidth)))
        density = float(np.mean(kernels))
        novelty = -math.log(max(density, 1e-12))
        nearest_idx = int(indices[0])
        nearest_id = scope.ids[nearest_idx] if nearest_idx >= 0 else None
        return NoveltyResult(novelty=novelty, nearest_id=nearest_id)

    def _score_vmf(self, scope: _ScopeState, candidate: np.ndarray) -> NoveltyResult:
        """Compute novelty from unnormalized vMF log-density on the hypersphere."""
        if scope.unit_vectors.shape[0] == 0:
            return NoveltyResult(novelty=float("inf"), nearest_id=None)

        candidate_norm = float(np.linalg.norm(candidate))
        candidate_unit = (candidate / max(candidate_norm, 1e-12)).astype(np.float32)
        similarities = np.matmul(scope.unit_vectors, candidate_unit)
        nearest_idx = int(np.argmax(similarities))
        nearest_id = scope.ids[nearest_idx]

        kappa = self.vmf_kappa if self.vmf_kappa is not None else scope.vmf_kappa
        kappa = max(float(kappa), 1e-6)
        scaled = kappa * similarities
        scaled_max = float(np.max(scaled))
        log_density = scaled_max + math.log(float(np.sum(np.exp(scaled - scaled_max))))

        # Smooth cosine similarity from log-mean-exp; invert so larger means more novel.
        smooth_similarity = (log_density - math.log(float(similarities.shape[0]))) / kappa
        smooth_similarity = float(np.clip(smooth_similarity, -1.0, 1.0))
        novelty = 1.0 - smooth_similarity
        return NoveltyResult(novelty=novelty, nearest_id=nearest_id)

    def score(self, scope_key: str, embedding: Sequence[float]) -> NoveltyResult:
        """Compute novelty score and nearest memory ID using the configured density gate."""
        scope = self._scope(scope_key)
        with scope.lock:
            if scope.projected_vectors.shape[0] == 0:
                return NoveltyResult(novelty=float("inf"), nearest_id=None)

            candidate = np.asarray(embedding, dtype=np.float32)
            if candidate.ndim != 1:
                raise ValueError("Candidate embedding must be a 1D vector.")
            self._ensure_projection_dim(candidate.shape[0])
            if self.novelty_method == VMF_NOVELTY_METHOD:
                return self._score_vmf(scope, candidate)
            return self._score_gaussian(scope, candidate)


class AdaptiveThreshold:
    def __init__(
        self,
        *,
        mode: Literal["adaptive", "fixed"] = "adaptive",
        fixed_threshold: Optional[float] = None,
        tau_0: float = 0.85,
        tau_min: float = 0.25,
        density_lambda: float = 2.0,
        ema_alpha: float = 0.9,
        hysteresis_delta: float = 0.15,
    ):
        """Initialize threshold parameters and per-scope EMA state.

        ``mode="adaptive"`` relaxes the threshold as memory density grows;
        ``mode="fixed"`` always returns ``fixed_threshold`` (which must then be
        provided).
        """
        if mode not in ("adaptive", "fixed"):
            raise ValueError(f"Unsupported threshold mode: {mode}")
        if mode == "fixed" and fixed_threshold is None:
            raise ValueError("fixed_threshold must be provided when mode='fixed'.")
        self.mode = mode
        self.fixed_threshold = fixed_threshold
        self.tau_0 = tau_0
        self.tau_min = tau_min
        self.density_lambda = density_lambda
        self.ema_alpha = ema_alpha
        self.hysteresis_delta = hysteresis_delta
        self._threshold_by_scope: Dict[str, float] = {}

    def update(self, scope_key: str, memory_count: int, volume: float) -> float:
        """Update and return the novelty threshold for a scope.

        In ``fixed`` mode the configured threshold is returned unchanged; in
        ``adaptive`` mode it relaxes with memory density, smoothed by an EMA.
        """
        if self.mode == "fixed":
            self._threshold_by_scope[scope_key] = self.fixed_threshold
            logger.info(f"[DIAG-THRESH] scope={scope_key} mode=fixed threshold={self.fixed_threshold:.4f}")
            return self.fixed_threshold
        density = float(memory_count) / max(float(volume), 1e-9)
        instantaneous = self.tau_0 * math.exp(-self.density_lambda * density) + self.tau_min
        prev = self._threshold_by_scope.get(scope_key)
        if prev is None:
            current = instantaneous
        else:
            current = self.ema_alpha * prev + (1.0 - self.ema_alpha) * instantaneous
        self._threshold_by_scope[scope_key] = current

        logger.info(f"[DIAG-THRESH] scope={scope_key} density={density:.6f} instant={instantaneous:.4f} smoothed={current:.4f}")
        return current

    def route(self, novelty: float, threshold: float) -> str:
        """Map novelty to ADD/UPDATE/NONE using threshold and hysteresis margin."""
        if novelty >= threshold + self.hysteresis_delta:
            return CREATE_EVENT
        if novelty >= threshold:
            return UPDATE_EVENT
        return NOOP_EVENT

    def reset_scope(self, scope_key: str) -> None:
        self._threshold_by_scope.pop(scope_key, None)

    def reset_all(self) -> None:
        self._threshold_by_scope.clear()
