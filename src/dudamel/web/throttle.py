"""Bounded failed-authentication counter for the web surface.

**This is an alerting threshold, not a brute-force control.** The web token is
a 256-bit secret; five attempts per five minutes does not meaningfully change
the cost of guessing it. What this does buy is a bound on log noise and a
signal that something is probing the surface.

The counter is deliberately keyed on failures only, and a successful
authentication clears it -- see `FailedAuthThrottle.clear`. Throttling before
validating a token would let anyone who can reach the login endpoint lock the
operator out of their own dashboard.

State is bounded: an unbounded per-client dict is a memory-exhaustion vector
for anyone with a routed IPv6 /64, so entries are LRU-evicted at
`MAX_TRACKED_CLIENTS` and swept for age on insert.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Callable

MAX_FAILURES = 5
WINDOW_SECONDS = 300.0
MAX_TRACKED_CLIENTS = 4096


class FailedAuthThrottle:
    def __init__(self, *, now: Callable[[], float] = time.monotonic) -> None:
        self._now = now
        self._failures: OrderedDict[str, list[float]] = OrderedDict()

    def _fresh(self, key: str) -> list[float]:
        cutoff = self._now() - WINDOW_SECONDS
        stamps = [t for t in self._failures.get(key, []) if t > cutoff]
        if stamps:
            self._failures[key] = stamps
            self._failures.move_to_end(key)
        else:
            self._failures.pop(key, None)
        return stamps

    def is_throttled(self, key: str) -> bool:
        return len(self._fresh(key)) >= MAX_FAILURES

    def record_failure(self, key: str) -> None:
        stamps = self._fresh(key)
        stamps.append(self._now())
        # Bound the per-key list too, not just the client count. Reaching the
        # limit is enough to throttle; beyond it, only the MAX_FAILURES most
        # recent stamps matter (is_throttled compares against the count,
        # retry_after against the oldest kept). Keeping every failure would let
        # a client that bypasses the is_throttled gate grow one list without
        # bound within a single window.
        if len(stamps) > MAX_FAILURES:
            del stamps[:-MAX_FAILURES]
        self._failures[key] = stamps
        self._failures.move_to_end(key)
        while len(self._failures) > MAX_TRACKED_CLIENTS:
            self._failures.popitem(last=False)

    def clear(self, key: str) -> None:
        self._failures.pop(key, None)

    def retry_after(self, key: str) -> int:
        stamps = self._fresh(key)
        if len(stamps) < MAX_FAILURES:
            return 0
        return max(1, int(stamps[0] + WINDOW_SECONDS - self._now()))
