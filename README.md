# Agentic AI Root Cause Finder

A production-style prototype of an autonomous Root Cause Analysis (RCA)
agent, built with **Python 3.13** and **FastAPI**.

An alert comes in -> the service **normalizes** it into a
vendor-agnostic context -> a **ReAct-style agent** iteratively calls
investigative tools (logs, metrics, deployments, traces, dependency
graph, similar-incident search) -> a **structured RCA report** is
returned with a primary hypothesis, alternates, evidence, timeline, and
prioritized remediation actions.

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
                              +------------------+
                                       |
                                       v
                              +------------------+
                              |  TelemetryStore  |  (in-memory seeded data)
                              +------------------+
                                       |
                                       v
                                +--------------+
                                |  RCAReport   |
                                +--------------+
```

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
    traces_tool.py, dependencies_tool.py, similar_incidents_tool.py
  context/
    normalizer.py             # NormalizedContext builder
  services/
    data_store.py             # In-memory seeded telemetry
    investigation_service.py  # Orchestrator
  api/
    dependencies.py           # DI singletons
    routes/                   # alerts / investigations / health
  models/                     # Pydantic domain models
  core/                       # logging + exceptions
tests/
  test_smoke.py               # End-to-end tests
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

| Variable                  | Default        | Purpose                                  |
| ------------------------- | -------------- | ---------------------------------------- |
| `LLM_PROVIDER`            | `mock`         | `mock` or `openai`                       |
| `LLM_MODEL`               | `gpt-4o-mini`  | OpenAI model id                          |
| `OPENAI_API_KEY`          | -              | Required if `LLM_PROVIDER=openai`        |
| `AGENT_MAX_ITERATIONS`    | `8`            | Hard cap on ReAct iterations             |
| `AGENT_TIMEOUT_SECONDS`   | `60`           | Wall-clock budget per investigation      |
| `LOG_LEVEL`               | `INFO`         |                                          |
| `APP_ENV`                 | `development`  | Toggles console vs JSON logs             |

## Testing

```bash
pytest -q
```

The suite spins up the FastAPI app in-process, POSTs an alert, and
asserts that the agent completed, produced a report, invoked the
expected tools, and enriched the normalized context.

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
