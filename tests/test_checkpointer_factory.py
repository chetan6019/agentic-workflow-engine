"""init_checkpointer tests: memory no-op, postgres DSN handling, cache invalidation."""

from __future__ import annotations

from types import SimpleNamespace

from langgraph.checkpoint.memory import MemorySaver

from app.orchestration import graph


class _FakeSaver(MemorySaver):
    """Stands in for AsyncPostgresSaver: a real BaseCheckpointSaver (compile()
    validates the type) that additionally records the setup() call."""

    def __init__(self) -> None:
        super().__init__()
        self.setup_called = False

    async def setup(self) -> None:
        self.setup_called = True


class _FakeSaverCtx:
    def __init__(self, saver: _FakeSaver) -> None:
        self.saver = saver
        self.exited = False

    async def __aenter__(self) -> _FakeSaver:
        return self.saver

    async def __aexit__(self, *exc) -> None:
        self.exited = True


async def test_memory_backend_is_a_noop(monkeypatch):
    monkeypatch.setattr(graph, "get_settings",
                        lambda: SimpleNamespace(checkpointer_backend="memory"))
    before = graph._checkpointer
    await graph.init_checkpointer()
    assert graph._checkpointer is before
    assert isinstance(graph._checkpointer, MemorySaver)


async def test_postgres_backend_strips_asyncpg_and_sets_up(monkeypatch):
    monkeypatch.setattr(graph, "get_settings", lambda: SimpleNamespace(
        checkpointer_backend="postgres",
        database_url="postgresql+asyncpg://u:p@host:5432/db"))
    saver = _FakeSaver()
    ctx = _FakeSaverCtx(saver)
    captured: dict[str, str] = {}

    def fake_from_conn_string(dsn: str) -> _FakeSaverCtx:
        captured["dsn"] = dsn
        return ctx

    import langgraph.checkpoint.postgres.aio as pg_aio
    monkeypatch.setattr(pg_aio.AsyncPostgresSaver, "from_conn_string",
                        staticmethod(fake_from_conn_string))
    # Snapshot module state so this test leaves the world as it found it.
    original = graph._checkpointer
    try:
        await graph.init_checkpointer()
        # psycopg needs the plain DSN — SQLAlchemy's +asyncpg marker must be gone.
        assert captured["dsn"] == "postgresql://u:p@host:5432/db"
        assert saver.setup_called is True
        assert graph._checkpointer is saver
        # compile_graph must recompile against the new saver (cache cleared).
        assert graph.compile_graph().checkpointer is saver
        await graph.close_checkpointer()
        assert ctx.exited is True
    finally:
        graph._checkpointer = original
        graph._saver_ctx = None
        graph.compile_graph.cache_clear()


async def test_close_checkpointer_noop_without_saver():
    graph._saver_ctx = None
    await graph.close_checkpointer()  # must not raise