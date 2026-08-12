"""Failed-authentication throttling.

The ordering property is the important one: a correct token must always
succeed, including while the client is throttled. Otherwise anyone able to
reach the login endpoint could lock the operator out of their own dashboard.
"""

from __future__ import annotations

from dudamel.web.throttle import (
    MAX_FAILURES,
    MAX_TRACKED_CLIENTS,
    WINDOW_SECONDS,
    FailedAuthThrottle,
)


def test_not_throttled_before_the_limit() -> None:
    t = FailedAuthThrottle()
    for _ in range(MAX_FAILURES - 1):
        t.record_failure("1.2.3.4")
    assert t.is_throttled("1.2.3.4") is False


def test_throttled_at_the_limit() -> None:
    t = FailedAuthThrottle()
    for _ in range(MAX_FAILURES):
        t.record_failure("1.2.3.4")
    assert t.is_throttled("1.2.3.4") is True


def test_clear_releases_a_throttled_client() -> None:
    """A correct token clears the counter even while throttled -- this is what
    keeps the operator from being locked out of their own dashboard."""
    t = FailedAuthThrottle()
    for _ in range(MAX_FAILURES):
        t.record_failure("1.2.3.4")
    t.clear("1.2.3.4")
    assert t.is_throttled("1.2.3.4") is False


def test_failures_age_out_of_the_window() -> None:
    t = FailedAuthThrottle(now=lambda: 1000.0)
    for _ in range(MAX_FAILURES):
        t.record_failure("1.2.3.4")
    assert t.is_throttled("1.2.3.4") is True
    t._now = lambda: 1000.0 + WINDOW_SECONDS + 1
    assert t.is_throttled("1.2.3.4") is False


def test_clients_are_bounded_by_lru_eviction() -> None:
    """An unbounded per-client dict is a memory-exhaustion vector for anyone
    with a routed IPv6 /64."""
    t = FailedAuthThrottle()
    for i in range(MAX_TRACKED_CLIENTS + 100):
        t.record_failure(f"client-{i}")
    assert len(t._failures) <= MAX_TRACKED_CLIENTS


def test_per_key_stamp_list_is_bounded() -> None:
    """Recording far more failures than the limit within one window must not
    grow a single key's stamp list without bound (the login endpoint gates on
    is_throttled first, but the component must bound itself regardless)."""
    t = FailedAuthThrottle(now=lambda: 1000.0)
    for _ in range(10_000):
        t.record_failure("1.2.3.4")
    assert len(t._failures["1.2.3.4"]) <= MAX_FAILURES
    assert t.is_throttled("1.2.3.4") is True
    assert 0 < t.retry_after("1.2.3.4") <= int(WINDOW_SECONDS)


def test_retry_after_is_positive_while_throttled() -> None:
    t = FailedAuthThrottle()
    for _ in range(MAX_FAILURES):
        t.record_failure("1.2.3.4")
    assert 0 < t.retry_after("1.2.3.4") <= int(WINDOW_SECONDS)
