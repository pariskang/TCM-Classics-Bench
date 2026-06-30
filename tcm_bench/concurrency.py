"""Bounded concurrent map — the shared engine for generation and evaluation.

``bounded_imap`` runs *fn* over *items* across a thread pool, keeping only
``~2 * max_workers`` calls in flight (a sliding window), and yields results as
they complete.  Because submission is bounded, a consumer that stops early
wastes at most a window of calls — important when each call is a paid LLM
request.  Order is **not** preserved.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from typing import Any


def bounded_imap(
    fn: Callable[[Any], Any],
    items: Iterable[Any],
    *,
    max_workers: int = 8,
    stats: dict | None = None,
) -> Iterator[Any]:
    """Yield ``fn(item)`` for each item as it completes.

    On exception the result is ``None`` and ``stats['errored']`` is bumped, so
    one bad call never kills the batch.  *stats* (if given) is filled with
    ``total`` / ``done`` / ``errored``.
    """
    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

    items = list(items)
    if stats is not None:
        stats.setdefault("total", len(items))
        stats.setdefault("done", 0)
        stats.setdefault("errored", 0)

    if not items:
        return

    window = max(1, max_workers) * 2
    idx = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        inflight: dict = {}
        while idx < len(items) and len(inflight) < window:
            inflight[pool.submit(fn, items[idx])] = idx
            idx += 1
        while inflight:
            finished, _ = wait(list(inflight), return_when=FIRST_COMPLETED)
            for fut in finished:
                del inflight[fut]
                try:
                    res = fut.result()
                except Exception:  # noqa: BLE001
                    res = None
                    if stats is not None:
                        stats["errored"] += 1
                if idx < len(items):  # refill the window
                    inflight[pool.submit(fn, items[idx])] = idx
                    idx += 1
                if stats is not None:
                    stats["done"] += 1
                yield res
