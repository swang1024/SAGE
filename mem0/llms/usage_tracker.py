"""Thread-safe, process-global accumulator for LLM token usage and call latency.

This is evaluation/instrumentation only: it lets the evaluation harness report the
write-side token budget and latency for each backbone (e.g. gpt-4o vs gpt-4o-mini)
without threading a stats object through every call site. LLM providers call
``USAGE_TRACKER.record(...)`` after each API call; the harness calls ``reset()``
before ingestion and ``snapshot()`` afterwards.

Tracking is a no-op unless explicitly enabled (env ``MEM0_TRACK_USAGE=1`` or a call
to ``enable()``), so it adds nothing to normal library use.
"""

import os
import threading
from collections import defaultdict


class _UsageTracker:
    def __init__(self):
        self._lock = threading.Lock()
        self._enabled = os.getenv("MEM0_TRACK_USAGE", "").lower() in ("1", "true", "yes")
        self.reset()

    def enable(self, enabled=True):
        self._enabled = enabled

    @property
    def enabled(self):
        return self._enabled

    def reset(self):
        with self._lock:
            self.calls = 0
            self.failures = 0
            self.prompt_tokens = 0
            self.completion_tokens = 0
            self.total_tokens = 0
            self.total_latency = 0.0
            self.min_latency = None
            self.max_latency = None
            # per-model breakdown so e.g. gpt-4o vs gpt-4o-mini stay separable
            self.per_model = defaultdict(
                lambda: {
                    "calls": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "total_latency": 0.0,
                }
            )

    def record(
        self,
        *,
        model=None,
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
        latency=0.0,
        failed=False,
    ):
        if not self._enabled:
            return
        prompt_tokens = int(prompt_tokens or 0)
        completion_tokens = int(completion_tokens or 0)
        total_tokens = int(total_tokens or 0) or (prompt_tokens + completion_tokens)
        latency = float(latency or 0.0)
        with self._lock:
            self.calls += 1
            if failed:
                self.failures += 1
            self.prompt_tokens += prompt_tokens
            self.completion_tokens += completion_tokens
            self.total_tokens += total_tokens
            self.total_latency += latency
            self.min_latency = latency if self.min_latency is None else min(self.min_latency, latency)
            self.max_latency = latency if self.max_latency is None else max(self.max_latency, latency)
            m = self.per_model[model or "unknown"]
            m["calls"] += 1
            m["prompt_tokens"] += prompt_tokens
            m["completion_tokens"] += completion_tokens
            m["total_tokens"] += total_tokens
            m["total_latency"] += latency

    def snapshot(self):
        with self._lock:
            calls = self.calls
            ok_calls = calls - self.failures
            avg_latency = (self.total_latency / calls) if calls else 0.0
            return {
                "enabled": self._enabled,
                "llm_calls": calls,
                "llm_failures": self.failures,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
                "tokens_per_call": (self.total_tokens / ok_calls) if ok_calls else 0.0,
                "total_latency_s": round(self.total_latency, 4),
                "avg_latency_s": round(avg_latency, 4),
                "min_latency_s": round(self.min_latency, 4) if self.min_latency is not None else None,
                "max_latency_s": round(self.max_latency, 4) if self.max_latency is not None else None,
                "per_model": {k: dict(v) for k, v in self.per_model.items()},
            }


# Process-global singleton.
USAGE_TRACKER = _UsageTracker()


def extract_usage(response):
    """Pull (prompt, completion, total) token counts from an OpenAI-style response.

    Returns (0, 0, 0) when usage is unavailable so callers can record latency-only.
    """
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0, 0
    return (
        getattr(usage, "prompt_tokens", 0) or 0,
        getattr(usage, "completion_tokens", 0) or 0,
        getattr(usage, "total_tokens", 0) or 0,
    )
