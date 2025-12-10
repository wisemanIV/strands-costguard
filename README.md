# Strands CostGuard

A Strands-native cost management library for multi-agent systems with budget enforcement, adaptive model routing, and OpenTelemetry-compatible metrics.

## Features

- **Budget Enforcement**: Define budgets at tenant, strand, workflow, and run levels with configurable limits and actions
- **Adaptive Model Routing**: Automatically route to fallback models based on budget utilization and other conditions
- **Cost Tracking**: Track and attribute costs by tenant, strand, workflow, run, model, and tool
- **OpenTelemetry Metrics**: Emit cost metrics compatible with OTel collectors for long-term storage and analysis
- **Flexible Policies**: Configure via YAML files or environment variables
- **Persistent Budget State**: Optional Valkey/Redis persistence for budget state across restarts

## Installation

```bash
pip install strands-costguard
```

For persistence support:

```bash
pip install strands-costguard[valkey]
```

## Quick Start

### Running the Examples

```bash
# Install the package in development mode
pip install -e .

# Run the basic usage example
python examples/basic_usage.py
```

### Basic Usage

```python
from strands_costguard import (
    CostGuard,
    CostGuardConfig,
    FilePolicySource,
    ModelUsage,
)

# Initialize Cost Guard
config = CostGuardConfig(
    policy_source=FilePolicySource(path="./policies"),
    enable_budget_enforcement=True,
    enable_routing=True,
    enable_metrics=True,
)

guard = CostGuard(config=config)

# Start a run
decision = guard.on_run_start(
    tenant_id="prod-tenant",
    strand_id="analytics_assistant",
    workflow_id="data_analysis",
    run_id="run-123",
)

if not decision.allowed:
    print(f"Run rejected: {decision.reason}")
else:
    # Execute your agent loop...

    # Before model calls
    model_decision = guard.before_model_call(
        run_id="run-123",
        model_name="gpt-4o",
        stage="planning",
        prompt_tokens_estimate=500,
    )

    # Use the effective model (may be downgraded)
    effective_model = model_decision.effective_model

    # After model calls
    guard.after_model_call(
        run_id="run-123",
        usage=ModelUsage.from_response(
            model_name=effective_model,
            prompt_tokens=500,
            completion_tokens=200,
        ),
    )

    # End the run
    guard.on_run_end("run-123", "completed")

# Shutdown (flushes metrics)
guard.shutdown()
```

## Configuration

### Budget Policies (budgets.yaml)

```yaml
budgets:
  - id: "tenant-default"
    scope: "tenant"
    match:
      tenant_id: "*"
    period: "monthly"
    max_cost: 1000.0
    soft_thresholds: [0.7, 0.9, 1.0]
    hard_limit: true
    on_soft_threshold_exceeded: "DOWNGRADE_MODEL"
    on_hard_limit_exceeded: "REJECT_NEW_RUNS"

  - id: "analytics-strand"
    scope: "strand"
    match:
      strand_id: "analytics_assistant"
    period: "daily"
    max_cost: 50.0
    max_runs_per_period: 1000
    max_concurrent_runs: 100
    constraints:
      max_iterations_per_run: 8
      max_tool_calls_per_run: 20
      max_model_tokens_per_run: 30000
```

### Routing Policies (routing.yaml)

```yaml
routing_policies:
  - id: "default-routing"
    match:
      strand_id: "*"
    stages:
      - stage: "planning"
        default_model: "gpt-4o-mini"
        max_tokens: 2000
      - stage: "synthesis"
        default_model: "gpt-4o"
        fallback_model: "gpt-4o-mini"
        trigger_downgrade_on:
          soft_threshold_exceeded: true
          remaining_budget_below: 5.0
```

### Pricing Table (pricing.yaml)

```yaml
pricing:
  currency: "USD"
  models:
    "gpt-4o":
      input_per_1k: 2.50
      output_per_1k: 10.00
    "gpt-4o-mini":
      input_per_1k: 0.15
      output_per_1k: 0.60
  tools:
    "web_search":
      cost_per_call: 0.01
```

## Lifecycle Hooks

Cost Guard integrates with your agent runtime via lifecycle hooks:

| Hook | When Called | Returns |
|------|-------------|---------|
| `on_run_start()` | Before starting a new run | `AdmissionDecision` |
| `on_run_end()` | After a run completes | None |
| `before_iteration()` | Before each agent loop iteration | `IterationDecision` |
| `after_iteration()` | After each iteration completes | None |
| `before_model_call()` | Before each model call | `ModelDecision` |
| `after_model_call()` | After each model call | None |
| `before_tool_call()` | Before each tool call | `ToolDecision` |
| `after_tool_call()` | After each tool call | None |

## OpenTelemetry Metrics

Cost Guard emits the following metrics:

| Metric | Type | Description |
|--------|------|-------------|
| `genai.cost.total` | Counter | Total cost in currency units |
| `genai.cost.model` | Counter | Cost per model |
| `genai.cost.tool` | Counter | Cost per tool |
| `genai.tokens.input` | Counter | Total input tokens |
| `genai.tokens.output` | Counter | Total output tokens |
| `genai.agent.iterations` | Counter | Agent loop iterations |
| `genai.agent.tool_calls` | Counter | Tool calls |
| `genai.cost.downgrade_events` | Counter | Model downgrade events |
| `genai.cost.rejection_events` | Counter | Run rejection events |

Metrics include resource attributes:
- `service.name`, `service.namespace`, `deployment.environment`
- `strands.tenant_id`, `strands.strand_id`, `strands.workflow_id`

## Budget Scopes and Priority

Budgets can be defined at multiple scopes, with higher priority scopes taking precedence:

1. **Global** (lowest priority) - Default limits for all
2. **Tenant** - Organization-level limits
3. **Strand** - Agent definition limits
4. **Workflow** (highest priority) - Specific workflow limits

When multiple budgets match, constraints are merged with more specific budgets taking priority.

## Threshold Actions

When budget soft thresholds are exceeded:

| Action | Effect |
|--------|--------|
| `LOG_ONLY` | Log warning, continue normally |
| `DOWNGRADE_MODEL` | Switch to fallback models |
| `LIMIT_CAPABILITIES` | Reduce max tokens/iterations |
| `HALT_NEW_RUNS` | Reject new runs |

When hard limits are exceeded:

| Action | Effect |
|--------|--------|
| `HALT_RUN` | Stop the current run |
| `REJECT_NEW_RUNS` | Reject new runs only |

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Type checking
mypy src/

# Linting
ruff check src/
```

## License

Apache-2.0
