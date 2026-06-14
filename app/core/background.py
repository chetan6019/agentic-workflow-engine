"""Process-wide registry of fire-and-forget asyncio tasks.

Holds strong refs so tasks aren't garbage-collected mid-await (asyncio only
weakly references tasks), and lets shutdown drain them with a deadline instead
of killing them abruptly.
"""

from __future__ import annotations

import asyncio

import structlog

log = structlog.get_logger(__name__)

_TASKS: set[asyncio.Task] = set()


def spawn(coro) -> asyncio.Task:
    """Schedule a coroutine in the background, retaining a ref until it finishes."""
    task = asyncio.create_task(coro)
    _TASKS.add(task)
    task.add_done_callback(_TASKS.discard)
    return task


async def drain(timeout: float) -> None:
    """Wait up to `timeout`s for in-flight tasks to finish, then cancel stragglers."""
    pending = set(_TASKS)
    if not pending:
        return
    log.info("background_drain_start", count=len(pending))
    _done, still = await asyncio.wait(pending, timeout=timeout)
    for task in still:
        task.cancel()
    log.info("background_drain_done", finished=len(pending) - len(still), cancelled=len(still))
