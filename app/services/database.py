"""PostgreSQL client used by the ``query_database`` tool.

Design goals:

* **Zero blast radius when unconfigured** - if ``DATABASE_URL`` is
  empty the client stays in a *"not connected"* state, and all
  read requests return a structured "unavailable" result instead of
  crashing the agent. This keeps demos and the test suite runnable
  without a real Postgres.
* **Bounded resources** - queries always run under an ``asyncio``
  timeout AND a Postgres-side ``statement_timeout``. A rogue LLM
  cannot pin a connection forever.
* **Read-only per session** - every acquired connection is put into
  ``READ ONLY`` transaction mode as a defense-in-depth belt on top
  of the allowlisted-query design in the tool layer.
* **Fail-lazy at startup** - if the DSN is set but Postgres is
  unreachable during application boot, we log a warning and continue
  serving. Queries will simply return "unavailable" until the DB
  comes back.
"""
from __future__ import annotations

import asyncio
from typing import Any

import asyncpg

from app.config import Settings, get_settings
from app.core.exceptions import RCAError
from app.core.logging import get_logger

log = get_logger(__name__)


class DatabaseUnavailable(RCAError):
    """Raised (or handled by the tool) when the pool is not usable."""


class DatabaseClient:
    """Thin async wrapper around an ``asyncpg`` pool."""

    def __init__(
        self,
        dsn: str,
        pool_min: int = 1,
        pool_max: int = 5,
        query_timeout_seconds: float = 5.0,
        statement_timeout_ms: int = 5000,
    ) -> None:
        self.dsn = dsn
        self.pool_min = pool_min
        self.pool_max = pool_max
        self.query_timeout_seconds = query_timeout_seconds
        self.statement_timeout_ms = statement_timeout_ms
        self._pool: asyncpg.Pool | None = None
        self._configured = bool(dsn)

    # -- Public API ------------------------------------------------------

    @property
    def is_configured(self) -> bool:
        """True iff a DATABASE_URL was provided."""
        return self._configured

    @property
    def is_connected(self) -> bool:
        """True iff a live pool exists."""
        return self._pool is not None

    async def connect(self) -> None:
        """Create the pool, if configured. Fail-lazy on connection errors."""
        if not self._configured:
            log.info("db.disabled", reason="DATABASE_URL not set")
            return
        try:
            self._pool = await asyncpg.create_pool(
                dsn=self.dsn,
                min_size=self.pool_min,
                max_size=self.pool_max,
                command_timeout=self.query_timeout_seconds,
                setup=self._setup_connection,
            )
            log.info(
                "db.connected",
                min=self.pool_min,
                max=self.pool_max,
                stmt_timeout_ms=self.statement_timeout_ms,
            )
        except Exception as exc:  # noqa: BLE001
            # Fail-lazy: keep the app running so demos don't require a
            # working database. Every subsequent fetch() will surface
            # the unavailable state to the caller.
            self._pool = None
            log.warning("db.connect_failed", error=str(exc))

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            log.info("db.closed")

    async def fetch(
        self,
        sql: str,
        *args: Any,
        timeout: float | None = None,
    ) -> list[dict[str, Any]]:
        """Run a SELECT and return rows as list-of-dict.

        Raises :class:`DatabaseUnavailable` when the pool is missing so
        the calling tool can convert that into a structured
        ``ToolResult`` rather than an exception traceback bubbling up
        to the agent.
        """
        if self._pool is None:
            raise DatabaseUnavailable(
                "database not configured or connection failed"
            )
        t = timeout or self.query_timeout_seconds
        async with self._pool.acquire() as conn:
            records = await asyncio.wait_for(conn.fetch(sql, *args), timeout=t)
        return [dict(r) for r in records]

    # -- Internals -------------------------------------------------------

    async def _setup_connection(self, conn: asyncpg.Connection) -> None:
        """Enforce read-only + statement timeout on every new session."""
        # Server-side statement timeout: a Postgres-enforced ceiling.
        await conn.execute(
            f"SET statement_timeout = {int(self.statement_timeout_ms)}"
        )
        # Read-only transactions by default; write queries would need
        # to explicitly `BEGIN READ WRITE`, which the allowlisted tool
        # never does.
        await conn.execute("SET default_transaction_read_only = ON")


class NullDatabaseClient(DatabaseClient):
    """A no-op client used when we want DB integration completely off.

    Kept as a subclass so callers only have to depend on the abstract
    behaviour: ``is_configured`` / ``is_connected`` / ``fetch``.
    """

    def __init__(self) -> None:
        super().__init__(dsn="")

    async def connect(self) -> None:  # noqa: D401
        return None

    async def close(self) -> None:
        return None

    async def fetch(self, sql: str, *args: Any, timeout: float | None = None):
        raise DatabaseUnavailable("database integration is disabled")


# ---------------------------------------------------------------------------
# Process-wide singleton
# ---------------------------------------------------------------------------

_client: DatabaseClient | None = None


def get_db_client(settings: Settings | None = None) -> DatabaseClient:
    """Return the process-scoped :class:`DatabaseClient` singleton."""
    global _client
    if _client is None:
        s = settings or get_settings()
        _client = DatabaseClient(
            dsn=s.database_url,
            pool_min=s.database_pool_min,
            pool_max=s.database_pool_max,
            query_timeout_seconds=s.database_query_timeout_seconds,
            statement_timeout_ms=s.database_statement_timeout_ms,
        )
    return _client


def set_db_client(client: DatabaseClient) -> None:
    """Test hook - swap the singleton for a fake."""
    global _client
    _client = client


def reset_db_client() -> None:
    global _client
    _client = None
