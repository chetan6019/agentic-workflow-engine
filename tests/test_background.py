"""Background task registry tests (REVIEW.md R38)."""

from __future__ import annotations

import asyncio

from app.core import background


async def test_spawn_runs_and_deregisters():
    ran = asyncio.Event()

    async def work():
        ran.set()

    task = background.spawn(work())
    await task
    assert ran.is_set()
    assert task not in background._TASKS  # done-callback removed it


async def test_drain_waits_for_inflight_then_returns():
    done = []

    async def work():
        await asyncio.sleep(0.01)
        done.append(1)

    background.spawn(work())
    await background.drain(timeout=1.0)
    assert done == [1]


async def test_drain_cancels_stragglers_past_deadline():
    cancelled = []

    async def slow():
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.append(1)
            raise

    background.spawn(slow())
    await background.drain(timeout=0.05)  # deadline well before the task finishes
    await asyncio.sleep(0)  # let the cancellation propagate
    assert cancelled == [1]


async def test_drain_noop_when_idle():
    await background.drain(timeout=0.01)  # must not raise with no tasks
