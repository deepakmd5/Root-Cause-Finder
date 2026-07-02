"""Aerospike client used by the ``query_aerospike`` tool.

Design mirrors :mod:`app.services.database`:

* **Zero blast radius when unconfigured** - if ``AEROSPIKE_HOSTS`` is
  empty (or the ``aerospike`` C-client library is not installed) the
  client stays in a *"not connected"* state, and every read returns a
  structured "unavailable" result to the calling tool. This keeps
  local dev + the test suite runnable without the C library on the
  system.
* **Bounded resources** - every read runs under an ``asyncio``
  timeout in addition to Aerospike's own client-side ``total_timeout``.
* **Read-only surface** - the client exposes only ``get``/``exists``.
  Writes are not implemented; the tool layer additionally enforces an
  allowlist of named operations so the LLM cannot construct arbitrary
  key lookups against unknown sets.
* **Fail-lazy at startup** - if the cluster is unreachable during
  application boot, we log a warning and continue serving. Reads will
  simply return "unavailable" until the cluster comes back.

The official ``aerospike`` client is a synchronous C-extension. We
wrap every call in :func:`asyncio.to_thread` so the event loop is
never blocked.
"""
from __future__ import annotations

import asyncio
from typing import Any

from app.config import Settings, get_settings
from app.core.exceptions import RCAError
from app.core.logging import get_logger

log = get_logger(__name__)


class AerospikeUnavailable(RCAError):
    """Raised (or handled by the tool) when the cluster is not usable."""


class AerospikeClient:
    """Thin async wrapper around the synchronous Aerospike client."""

    def __init__(
        self,
        hosts: str,
        namespace: str,
        username: str = "",
        password: str = "",
        total_timeout_ms: int = 1000,
        query_timeout_seconds: float = 2.0,
    ) -> None:
        # ``hosts`` is a comma-separated ``host:port`` list, e.g.
        # ``"cache-1:3000,cache-2:3000"``. Empty disables the client.
        self.hosts_raw = hosts
        self.namespace = namespace
        self.username = username
        self.password = password
        self.total_timeout_ms = total_timeout_ms
        self.query_timeout_seconds = query_timeout_seconds

        self._client: Any = None  # actual aerospike.Client, populated in connect()
        self._configured = bool(hosts and namespace)
        self._connected = False

    # -- Public API ------------------------------------------------------

    @property
    def is_configured(self) -> bool:
        """True iff AEROSPIKE_HOSTS + AEROSPIKE_NAMESPACE are set."""
        return self._configured

    @property
    def is_connected(self) -> bool:
        """True iff a live cluster connection exists."""
        return self._connected and self._client is not None

    async def connect(self) -> None:
        """Open a cluster connection, if configured. Fail-lazy on error."""
        if not self._configured:
            log.info("aerospike.disabled", reason="AEROSPIKE_HOSTS/NAMESPACE not set")
            return

        try:
            aerospike = _import_aerospike()
        except ImportError as exc:
            # Library not installed - degrade gracefully.
            log.warning("aerospike.library_missing", error=str(exc))
            return

        config: dict[str, Any] = {
            "hosts": _parse_hosts(self.hosts_raw),
            "policies": {
                "read": {"total_timeout": self.total_timeout_ms},
            },
        }

        try:
            client = aerospike.client(config)
            # ``connect`` is synchronous - hand it off to a worker thread
            # so we don't block the event loop during startup.
            if self.username or self.password:
                await asyncio.to_thread(
                    client.connect, self.username, self.password
                )
            else:
                await asyncio.to_thread(client.connect)
            self._client = client
            self._connected = True
            log.info(
                "aerospike.connected",
                hosts=self.hosts_raw,
                namespace=self.namespace,
                total_timeout_ms=self.total_timeout_ms,
            )
        except Exception as exc:  # noqa: BLE001
            # Fail-lazy: keep the app running so demos don't require a
            # working Aerospike. Every read will surface the unavailable
            # state to the caller.
            self._client = None
            self._connected = False
            log.warning("aerospike.connect_failed", error=str(exc))

    async def close(self) -> None:
        if self._client is not None:
            try:
                await asyncio.to_thread(self._client.close)
            except Exception as exc:  # noqa: BLE001
                log.warning("aerospike.close_failed", error=str(exc))
            finally:
                self._client = None
                self._connected = False
                log.info("aerospike.closed")

    async def get(
        self,
        set_name: str,
        key: str,
        *,
        namespace: str | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any] | None:
        """Read a single record. Returns ``None`` when the key is absent.

        Raises :class:`AerospikeUnavailable` when the cluster is missing
        so the calling tool can convert it into a structured
        ``ToolResult`` rather than a raw traceback.
        """
        if self._client is None or not self._connected:
            raise AerospikeUnavailable(
                "aerospike not configured or connection failed"
            )

        ns = namespace or self.namespace
        t = timeout or self.query_timeout_seconds
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(self._get_sync, ns, set_name, key),
                timeout=t,
            )
        except asyncio.TimeoutError as exc:
            raise AerospikeUnavailable(
                f"aerospike get() timed out after {t}s"
            ) from exc

    # -- Internals -------------------------------------------------------

    def _get_sync(
        self, namespace: str, set_name: str, key: str
    ) -> dict[str, Any] | None:
        """Synchronous read - runs inside asyncio.to_thread()."""
        aerospike = _import_aerospike()
        try:
            _, meta, bins = self._client.get((namespace, set_name, key))
        except aerospike.exception.RecordNotFound:
            return None
        return {"bins": dict(bins or {}), "meta": dict(meta or {})}


class NullAerospikeClient(AerospikeClient):
    """A no-op client used when integration is disabled entirely."""

    def __init__(self) -> None:
        super().__init__(hosts="", namespace="")

    async def connect(self) -> None:  # noqa: D401
        return None

    async def close(self) -> None:
        return None

    async def get(self, *args: Any, **kwargs: Any):  # noqa: ANN201
        raise AerospikeUnavailable("aerospike integration is disabled")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_hosts(raw: str) -> list[tuple[str, int]]:
    """Turn ``"host1:3000,host2:3000"`` into ``[("host1", 3000), ...]``."""
    hosts: list[tuple[str, int]] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" in chunk:
            host, port = chunk.rsplit(":", 1)
            hosts.append((host.strip(), int(port)))
        else:
            hosts.append((chunk, 3000))
    return hosts


def _import_aerospike() -> Any:
    """Deferred import so the module loads even without the C library."""
    import aerospike  # type: ignore[import-not-found]  # noqa: PLC0415

    return aerospike


# ---------------------------------------------------------------------------
# Process-wide singleton
# ---------------------------------------------------------------------------

_client: AerospikeClient | None = None


def get_aerospike_client(settings: Settings | None = None) -> AerospikeClient:
    """Return the process-scoped :class:`AerospikeClient` singleton."""
    global _client
    if _client is None:
        s = settings or get_settings()
        _client = AerospikeClient(
            hosts=s.aerospike_hosts,
            namespace=s.aerospike_namespace,
            username=s.aerospike_username,
            password=s.aerospike_password,
            total_timeout_ms=s.aerospike_total_timeout_ms,
            query_timeout_seconds=s.aerospike_query_timeout_seconds,
        )
    return _client


def set_aerospike_client(client: AerospikeClient) -> None:
    """Test hook - swap the singleton for a fake."""
    global _client
    _client = client


def reset_aerospike_client() -> None:
    global _client
    _client = None
