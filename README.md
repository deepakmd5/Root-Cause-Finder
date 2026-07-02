# Agentic AI Root Cause Finder

A production-style prototype of an autonomous Root Cause Analysis (RCA)
agent, built with **Python 3.13** and **FastAPI**.

An alert comes in -> the service **normalizes** it into a
vendor-agnostic context -> a **ReAct-style agent** iteratively calls
investigative tools (logs, metrics, deployments, traces, dependency
graph, similar-incident search) -> a **structured RCA report** is
returned with a primary hypothesis, alternates, evidence, timeline, and
prioritized remediation actions.

## How it thinks (the story behind the code)

When a production alert fires, an on-call engineer manually consults 5-6
systems: deploy history, logs, metrics, traces, dependency map,
runbook / past incidents. This service automates that loop:

```
Alert  ->  FastAPI  ->  ContextNormalizer  ->  RCAAgent (ReAct loop)
                                                    |
                                    +---------------+---------------+
                                    v               v               v
                                 LLM decides    Tool runs      State grows
                                 next step      returns        observation
                                                evidence       is fed back
                                    +---------------+---------------+
                                                    |
                                                    v
                                          Final structured RCA
```

The agent is **not** just the LLM. It is:

- **Goal** — the normalized incident context (what to investigate).
- **LLM** — decides *what to check next*, does not fabricate results.
- **Tools** — the only source of ground truth.
- **State** — the enriched context and the ReAct trace.
- **Loop** — bounded by iteration and wall-clock budgets.

The LLM is only allowed to *reason*. All evidence flows through the
tool layer. The final RCA is built from tool results, and a
post-processing guardrail refuses to emit a confident answer when
service-specific telemetry is missing.

## Highlights

- **Alert-driven workflow.** One HTTP endpoint accepts any alert shape
  (Prometheus, Datadog, PagerDuty, custom) and returns a full
  investigation.
- **Normalized context.** All sources are projected onto a single
  `NormalizedContext` before the agent sees them.
- **Pluggable LLM.** Ships with a deterministic `mock` provider (zero
  API keys, reproducible demos + tests) and an `openai` adapter. Swap
  by setting `LLM_PROVIDER`.
- **Tool registry.** Each investigative capability is a `Tool` with a
  JSON input schema advertised to the LLM. Adding a new tool is one
  file + one line of registration.
- **ReAct loop with observability.** Every thought / action /
  observation is captured in the investigation trace and exposed via
  the API. Structured `structlog` output.
- **Bounded execution.** Iteration cap + wall-clock timeout keep the
  agent safe.
- **Optional PostgreSQL access.** The `query_database` tool exposes an
  allowlisted, read-only, parameterized query surface so the agent can
  correlate alerts with the operational source of truth. The LLM
  picks a `query_name` and passes named params; it never writes SQL.
- **Optional Aerospike (NoSQL) access.** The `query_aerospike` tool
  exposes allowlisted, read-only key lookups against a hot cache (in-
  flight transaction state, policy cache, idempotency records). The
  LLM picks an `operation` from the allowlist; it never constructs
  Aerospike keys directly.
- **Fully tested.** `pytest` smoke suite runs the entire path via the
  FastAPI TestClient.

## Architecture

```
                                 +-------------------+
   Alert (JSON) --> POST /alerts | ContextNormalizer | ---> NormalizedContext
                                 +-------------------+
                                            |
                                            v
                                   +-----------------+
                                   |    RCAAgent     |     (ReAct loop)
                                   +-----------------+
                                       |         ^
                                       v         |
                                  +---------+   observation
                                  |   LLM   |
                                  +---------+
                                       |
                                       v
                                   AgentDecision
                                (use_tool | finalize)
                                       |
                                       v
                              +------------------+
                              |  ToolRegistry    |
                              +------------------+
                              | search_logs      |
                              | query_metrics    |
                              | recent_deployments|
                              | fetch_traces     |
                              | get_service_deps |
                              | find_similar_inc |
                              | query_database   |
                              | query_aerospike  |
                              +------------------+
                                       |
                                       v
                              +---------------------+
                              |  TelemetryStore     |  (in-memory seeded data)
                              |  + PostgreSQL       |  (optional, allowlisted)
                              |  + Aerospike        |  (optional, allowlisted)
                              +---------------------+
                                       |
                                       v
                                +--------------+
                                |  RCAReport   |
                                +--------------+
```

## How tool selection works

The LLM does not have hard-coded knowledge of which tool to use next.
On every ReAct iteration, the agent **advertises the current tool
catalogue** to the LLM and lets the LLM pick one:

```
                +--------------------------------------------------+
   RCAAgent --> | For each registered tool:                        |
                |   { name, description, input_schema (JSON) }     |
                +--------------------------------------------------+
                                     |
                                     v
                +--------------------------------------------------+
                | LLM sees:                                        |
                |   - the normalized incident context              |
                |   - every prior thought/action/observation       |
                |   - the tool catalogue above                     |
                | LLM returns an AgentDecision:                    |
                |   { action: "use_tool",                          |
                |     tool_name: "...", tool_input: {...} }        |
                |   or                                             |
                |   { action: "finalize", rca: {...} }             |
                +--------------------------------------------------+
                                     |
                                     v
                +--------------------------------------------------+
                | ToolRegistry.dispatch(tool_name, tool_input)     |
                |   -> Tool.run(**tool_input)                      |
                |   -> ToolResult { tool, input, summary, data }   |
                +--------------------------------------------------+
                                     |
                                     v
                +--------------------------------------------------+
                | RCAAgent._merge_into_context:                    |
                |   normalizes the ToolResult onto typed fields    |
                |   of NormalizedContext (logs, metrics, deploys,  |
                |   traces, db_records, aerospike_records, ...)    |
                | The observation is appended to the ReAct trace   |
                | and fed back into the LLM on the next step.      |
                +--------------------------------------------------+
```

Two consequences worth calling out:

- **Adding a new capability is one file + one line.** Implement a
  `Tool` subclass with a `name`, human-readable `description`, and a
  JSON `input_schema`; register it in
  `app/tools/registry.py::_build_default_registry`. The LLM sees it
  from the very next request - no prompt-template edits, no adapter
  changes.
- **The LLM never touches raw infrastructure.** Both `query_database`
  and `query_aerospike` publish an **enum of allowlisted operations**
  as part of their input schema. The LLM picks a name; the tool
  translates that name into a parameterized SQL statement or an
  Aerospike `(namespace, set, key)` triple in server-side code. There
  is no path for the LLM to smuggle a raw query or arbitrary key
  through the boundary.

## Project layout

```
app/
  main.py                     # FastAPI app factory + middleware + lifespan
  config.py                   # pydantic-settings configuration
  agents/
    rca_agent.py              # ReAct loop
    prompts.py                # system prompt
  llm/
    base.py                   # LLMAdapter protocol + factory
    mock.py                   # deterministic mock provider
    openai_adapter.py         # OpenAI chat completions
  tools/
    base.py, registry.py      # Tool ABC + registry
    logs_tool.py, metrics_tool.py, deployments_tool.py,
    traces_tool.py, dependencies_tool.py, similar_incidents_tool.py,
    database_tool.py          # allowlisted read-only PostgreSQL queries
    aerospike_tool.py         # allowlisted read-only Aerospike operations
  context/
    normalizer.py             # NormalizedContext builder
  services/
    data_store.py             # In-memory seeded telemetry
    database.py               # asyncpg pool, read-only + timeout enforced
    aerospike_client.py       # async wrapper around aerospike C-client
    investigation_service.py  # Orchestrator
  api/
    dependencies.py           # DI singletons
    routes/                   # alerts / investigations / health
  models/                     # Pydantic domain models
  core/                       # logging + exceptions
tests/
  test_smoke.py               # End-to-end tests
  test_tools.py               # Per-tool unit tests
  test_normalizer.py          # ContextNormalizer tests
  test_agent_negative.py      # Guardrail + failure-path tests
  test_database_tool.py       # DB tool + fake-client end-to-end
  test_aerospike_tool.py      # Aerospike tool + fake-client end-to-end
```

## Run it

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Default config uses the mock LLM - no API keys needed:
python run.py
# or
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000/docs for Swagger UI.

## Fire an alert

```bash
curl -s -X POST http://127.0.0.1:8000/alerts \
  -H 'Content-Type: application/json' \
  -d '{
    "title": "Payments error rate breached SLO",
    "description": "5xx errors on payments-service crossed 5%",
    "service": "payments-service",
    "environment": "production",
    "severity": "critical",
    "source": "prometheus",
    "metric_name": "error_rate",
    "metric_value": 0.19,
    "threshold": 0.05,
    "labels": {"team": "payments", "region": "ap-south-1"}
  }'
```

The response contains:

- `alert` - with server-assigned id + timestamps
- `context` - the normalized incident context, enriched by every tool
  call the agent made
- `steps[]` - the full ReAct trace (thoughts / actions / observations)
- `report` - the final structured RCA with primary hypothesis,
  alternates, timeline, and prioritized remediation

## Configuration

All settings are environment variables (see `.env.example`):

| Variable                          | Default        | Purpose                                                                          |
| --------------------------------- | -------------- | -------------------------------------------------------------------------------- |
| `LLM_PROVIDER`                    | `mock`         | `mock` or `openai`                                                               |
| `LLM_MODEL`                       | `gpt-4o-mini`  | OpenAI model id                                                                  |
| `OPENAI_API_KEY`                  | -              | Required if `LLM_PROVIDER=openai`                                                |
| `AGENT_MAX_ITERATIONS`            | `8`            | Hard cap on ReAct iterations                                                     |
| `AGENT_TIMEOUT_SECONDS`           | `60`           | Wall-clock budget per investigation                                              |
| `LOG_LEVEL`                       | `INFO`         |                                                                                  |
| `APP_ENV`                         | `development`  | Toggles console vs JSON logs                                                     |
| `DATABASE_URL`                    | *(empty)*      | Postgres DSN. Empty disables the `query_database` tool.                          |
| `DATABASE_POOL_MIN`               | `1`            | asyncpg pool minimum size                                                        |
| `DATABASE_POOL_MAX`               | `5`            | asyncpg pool maximum size                                                        |
| `DATABASE_QUERY_TIMEOUT_SECONDS`  | `5.0`          | Client-side `asyncio.wait_for` per query                                         |
| `DATABASE_STATEMENT_TIMEOUT_MS`   | `5000`         | Server-side Postgres `statement_timeout` per session                             |
| `AEROSPIKE_HOSTS`                 | *(empty)*      | Comma-separated `host:port` list. Empty disables the `query_aerospike` tool.     |
| `AEROSPIKE_NAMESPACE`             | *(empty)*      | Aerospike namespace holding the operational sets.                                |
| `AEROSPIKE_USERNAME`              | *(empty)*      | Optional username for clusters with security enabled.                            |
| `AEROSPIKE_PASSWORD`              | *(empty)*      | Optional password for clusters with security enabled.                            |
| `AEROSPIKE_TOTAL_TIMEOUT_MS`      | `1000`         | Server-side total timeout per Aerospike op.                                      |
| `AEROSPIKE_QUERY_TIMEOUT_SECONDS` | `2.0`          | Client-side `asyncio.wait_for` per Aerospike op.                                 |

## Testing

```bash
pytest -q
```

The test suite is layered on purpose - each layer catches a different
class of regression:

| File                            | Layer                | What it proves                                                                                                              |
| ------------------------------- | -------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| `tests/test_tools.py`           | Tool unit            | Every tool behaves correctly on happy path *and* empty path. Schemas advertised to the LLM are well-formed.                 |
| `tests/test_normalizer.py`      | Context builder      | Blast-radius, tags, key metrics, and summaries are derived deterministically from any incoming alert shape.                 |
| `tests/test_smoke.py`           | Full agent flow      | End-to-end: alert -> normalized context -> ReAct loop -> structured RCA, exercised via the FastAPI `TestClient`.            |
| `tests/test_agent_negative.py`  | Negative / guardrail | Empty telemetry triggers *insufficient evidence*; agent survives tool failures; iteration-budget exhaustion is caught.      |
| `tests/test_database_tool.py`   | DB tool + agent      | Allowlist enforcement, param validation, unavailable/error branches, and an end-to-end alert-with-transactionId investigation using a fake `DatabaseClient` (no live Postgres required). |
| `tests/test_aerospike_tool.py`  | Aerospike tool + agent | Operation-allowlist enforcement, missing/invalid params, unconfigured / disconnected / lookup-error branches, and an end-to-end alert-with-transactionId investigation using a fake `AerospikeClient` (no live cluster required). Also asserts that Aerospike-only evidence satisfies the service-specific guardrail. |

## Hallucination guardrails

The agent is deliberately not trusted to invent RCAs. Two independent
guardrails prevent evidence-free hypotheses from reaching the caller:

1. **In the MockLLM synthesizer** (`app/llm/mock.py`) - if no tool
   observation produced deploys, logs, metrics, or traces, the
   `FINALIZE` payload short-circuits to a low-confidence *insufficient
   evidence* report.
2. **In the agent itself** (`app/agents/rca_agent.py::_apply_guardrails`) -
   after any LLM (mock *or* OpenAI) proposes a final answer, the agent
   inspects the enriched context. If no logs / metrics / deployments /
   traces mention the alerting service - **and** no DB row or Aerospike
   record was fetched for the alert's own `transactionId` / `policyId` /
   `idempotencyKey` - the primary hypothesis is rewritten to
   *"Insufficient evidence to determine root cause"* with confidence
   `0.1`, and remediation is redirected toward *fixing the telemetry
   pipeline*, not toward a risky code rollback.

Similar-incident matches and the dependency-graph response are
excluded from this check on purpose - both always return "something"
and can't substitute for direct observation of the alerting service.
DB rows and Aerospike records count only when the lookup parameters
match the identifiers the alert itself carries (and, for Aerospike,
only when the key was actually **found** - a cache miss is context,
not evidence), so the LLM cannot dredge up an unrelated record and
claim it as evidence.

## PostgreSQL integration (optional)

The `query_database` tool gives the agent read-only access to an
operational database so it can correlate alerts against the *source of
truth* (policy / transaction state) instead of inferring everything
from logs. Two safety belts make this LLM-safe:

1. **Allowlisted named queries.** The LLM picks a `query_name` from a
   fixed dictionary and supplies named parameters. It **never writes
   SQL**. Query bodies live in `app/tools/database_tool.py::ALLOWED_QUERIES`
   and are versioned + reviewed like production code.
2. **Session-level enforcement.** Every acquired connection is put
   into `default_transaction_read_only = ON` with a
   `statement_timeout`, and every query additionally runs under an
   `asyncio.wait_for` client-side timeout.

If `DATABASE_URL` is empty, the tool cleanly reports *"database not
configured"* and the app runs happily without a Postgres. If the DSN
is set but the DB is down at boot, the app logs a warning and keeps
serving; queries return `available=false` until the DB comes back.

### Wire up a database

```bash
# 1. Start a local Postgres (any recent version).
# 2. Create a read-only role for the agent:
createuser rca_ro --pwprompt
createdb   rca_ops --owner=$(whoami)
```

Apply this insurance-flavored schema. It matches the shipping
allowlisted queries (`policy_status`, `transaction_journey`,
`recent_failed_policies_by_provider`, `policy_failure_count_by_provider`).

```sql
CREATE TABLE transactions (
    transaction_id  TEXT        PRIMARY KEY,
    state           TEXT        NOT NULL,
    workflow_id     TEXT,
    amount          NUMERIC(12,2),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE policies (
    policy_id       TEXT        PRIMARY KEY,
    transaction_id  TEXT        REFERENCES transactions(transaction_id),
    status          TEXT        NOT NULL,     -- e.g. PENDING | SUCCESS | FAILED
    provider        TEXT        NOT NULL,     -- e.g. icici | tata | bajaj
    provider_error  TEXT,
    product         TEXT,                     -- e.g. travel_insurance
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX ON policies (transaction_id);
CREATE INDEX ON policies (provider, status, created_at DESC);

-- Grant the agent role read-only access:
GRANT CONNECT ON DATABASE rca_ops TO rca_ro;
GRANT USAGE   ON SCHEMA public    TO rca_ro;
GRANT SELECT  ON ALL TABLES IN SCHEMA public TO rca_ro;
ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT SELECT ON TABLES TO rca_ro;
```

Seed a couple of rows that match the demo alert:

```sql
INSERT INTO transactions (transaction_id, state, workflow_id, amount)
VALUES ('TX123', 'COMPLETED', 'WF-42', 1200.00);

INSERT INTO policies
    (policy_id, transaction_id, status, provider, provider_error, product)
VALUES
    ('POL-ICICI-9', 'TX123', 'FAILED', 'icici', 'UPSTREAM_TIMEOUT', 'travel_insurance');
```

Point the app at it:

```bash
export DATABASE_URL="postgresql://rca_ro:secret@localhost:5432/rca_ops"
python run.py
```

`/readiness` will now report `database.configured=true` and
`database.connected=true`. Fire an alert carrying the identifier:

```bash
curl -s -X POST http://127.0.0.1:8000/alerts \
  -H 'Content-Type: application/json' \
  -d '{
    "title": "Policy issuance failed",
    "service": "payments-service",
    "severity": "high",
    "source": "custom",
    "labels": {"transactionId": "TX123", "product": "travel_insurance"}
  }' | jq '.context.db_records, .report.primary_hypothesis.supporting_evidence'
```

The agent will schedule `query_database(transaction_journey)` early
in the plan - hot-cache lookups (see the Aerospike section below) run
first when they are configured, but the DB read still precedes every
log / metric / trace call because *source of truth beats log
inference*. The returned row shows up in `context.db_records` as well
as inside the primary hypothesis' `supporting_evidence`.

### Adding a new allowlisted query

1. Add a `QueryTemplate` entry to `ALLOWED_QUERIES` in
   `app/tools/database_tool.py` with a description, parameterized SQL
   (positional `$1..$N` binds), and the ordered `params` list.
2. Optionally add a query-specific one-liner in `_summarize()` so the
   LLM's next step is informed by a human-readable summary rather than
   a raw row count.
3. That's it - the tool schema advertised to the LLM regenerates
   automatically from the allowlist.

## Aerospike integration (optional)

The `query_aerospike` tool gives the agent read-only access to a hot
NoSQL cache. Where PostgreSQL is the *source of truth*, Aerospike is
the *fastest signal* - it typically holds in-flight transaction
state, the last policy snapshot returned to downstream consumers, and
idempotency records that reveal duplicate submissions or retry
storms. Same safety model as the DB tool:

1. **Allowlisted named operations.** The LLM picks an `operation`
   from a fixed dictionary and supplies **one** named parameter that
   becomes the record key. It never constructs Aerospike keys, and
   the `(namespace, set)` mapping lives in code
   (`app/tools/aerospike_tool.py::ALLOWED_OPERATIONS`).
2. **Read-only + bounded.** The async wrapper only exposes `get()`.
   Every call runs under an `asyncio.wait_for` client-side timeout
   *and* an Aerospike-native `total_timeout_ms`. The C-extension
   client is invoked via `asyncio.to_thread` so the event loop is
   never blocked.

If `AEROSPIKE_HOSTS` is empty, the tool cleanly reports *"aerospike
not configured"* and the app runs without a cluster. If hosts are
set but the cluster is unreachable at boot, the app logs a warning
and keeps serving; lookups return `available=false` until the
cluster comes back.

The three shipping operations are:

| Operation                 | Set            | Key param         | What it tells you                                                        |
| ------------------------- | -------------- | ----------------- | ------------------------------------------------------------------------ |
| `transaction_state_get`   | `tx_state`     | `transaction_id`  | Current in-flight state, retry attempts, last error, TTL.                |
| `policy_cache_get`        | `policy_cache` | `policy_id`       | Last cached status / provider snapshot seen by consumers.                |
| `idempotency_get`         | `idempotency`  | `idempotency_key` | `attempts`, `in_flight`, and outcome - flags duplicate / retry storms.   |

### Wire up an Aerospike cluster

The official Aerospike Python client is a C extension and is
**intentionally not pinned in `requirements.txt`** so the base app
stays install-anywhere. Install it locally only when you're pointing
at a real cluster:

```bash
pip install aerospike
```

Point the app at the cluster:

```bash
export AEROSPIKE_HOSTS="cache-1.internal:3000,cache-2.internal:3000"
export AEROSPIKE_NAMESPACE="ops"
# Optional - only if security is enabled on the cluster:
# export AEROSPIKE_USERNAME=rca_ro
# export AEROSPIKE_PASSWORD=...
python run.py
```

`/readiness` will now report `aerospike.configured=true` and
`aerospike.connected=true` alongside the DB fields.

Seed a couple of records that line up with the demo alert. The
snippet below uses `aql`, but any client works - the tool only reads
the `bins` / `meta` dict shape the Python client returns natively:

```sql
-- via aql
INSERT INTO ops.tx_state (PK, state, attempts, last_error)
    VALUES ('TX123', 'PENDING', 3, 'UPSTREAM_TIMEOUT');

INSERT INTO ops.policy_cache (PK, status, provider, provider_error)
    VALUES ('POL-ICICI-9', 'FAILED', 'icici', 'UPSTREAM_TIMEOUT');

INSERT INTO ops.idempotency (PK, attempts, in_flight, outcome)
    VALUES ('IDEMP-42', 4, true, 'pending');
```

Fire an alert that carries the identifiers:

```bash
curl -s -X POST http://127.0.0.1:8000/alerts \
  -H 'Content-Type: application/json' \
  -d '{
    "title": "Policy issuance stuck",
    "service": "payments-service",
    "severity": "high",
    "source": "custom",
    "labels": {
      "transactionId":   "TX123",
      "policyId":        "POL-ICICI-9",
      "idempotencyKey":  "IDEMP-42"
    }
  }' | jq '.context.aerospike_records,
           .context.db_records,
           .report.primary_hypothesis.supporting_evidence'
```

Ordering you should observe in `steps[]`:

1. `query_aerospike(transaction_state_get)` - is the transaction
   still `PENDING`? How many `attempts`?
2. `query_aerospike(policy_cache_get)` - what did downstream last
   see for the policy?
3. `query_aerospike(idempotency_get)` - are we in a retry storm?
4. `query_database(transaction_journey)` - authoritative view.
5. `recent_deployments`, `query_metrics`, `search_logs`,
   `fetch_traces`, `get_service_dependencies`,
   `find_similar_incidents` - telemetry corroboration.
6. `FINALIZE`.

The Aerospike hits are prepended to
`primary_hypothesis.supporting_evidence`, so the RCA leads with the
freshest signal.

### Adding a new allowlisted operation

1. Add an `AerospikeOperation` entry to `ALLOWED_OPERATIONS` in
   `app/tools/aerospike_tool.py` with a `description`, the target
   `aerospike_set`, and the name of the single `key_param` the LLM
   must supply. Optionally override `namespace` if the operation
   lives outside the default namespace.
2. Optionally extend `_summarize()` with an operation-specific
   one-liner so the LLM's next step is informed by a human-readable
   summary rather than a raw bin dump.
3. That's it - the tool's input schema regenerates from the
   allowlist on the next boot; the LLM sees the new operation from
   its very next decision.

## Extending

- **Add a tool.** Create a `Tool` subclass in `app/tools/`, register it
  in `app/tools/registry.py`, and it's automatically advertised to
  every LLM.
- **Wire a real data source.** Replace `TelemetryStore` methods with
  calls to Elasticsearch / Prometheus / Jaeger / your CD system. The
  tool interfaces don't change.
- **Swap the LLM.** Add another adapter that implements the
  `LLMAdapter` protocol and route to it in `build_llm()`.
- **Persist investigations.** Replace the in-memory dict in
  `InvestigationService` with Postgres / DynamoDB / etc.
