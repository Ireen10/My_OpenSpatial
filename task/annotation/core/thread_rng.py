"""Per-thread RNG for annotation workers (avoids the global ``random`` module lock)."""

from __future__ import annotations

import random
import threading

_tls = threading.local()


def rng() -> random.Random:
    """Return this worker thread's ``Random`` instance (created lazily)."""
    r = getattr(_tls, "r", None)
    if r is None:
        r = random.Random()
        _tls.r = r
    return r


def seed_thread_rng(seed: int | None = None) -> random.Random:
    """Replace the thread-local RNG (optional explicit seed for tests)."""
    r = random.Random(seed)
    _tls.r = r
    return r
