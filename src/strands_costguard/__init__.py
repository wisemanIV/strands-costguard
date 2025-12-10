"""
Strand Cost Guard - Budget enforcement and adaptive routing for Strands-based multi-agent systems.

This library provides:
- Budget enforcement at tenant, strand, workflow, and run levels
- Adaptive model routing based on cost policies
- OpenTelemetry-compatible metrics emission via StrandsTelemetry
- Integration with Strands runtime lifecycle hooks
- Optional Valkey/Redis persistence for budget state

Metrics Setup:
    Cost Guard uses the global MeterProvider configured by StrandsTelemetry.
    Configure telemetry before creating a CostGuard instance:

        from strands.telemetry.config import StrandsTelemetry

        telemetry = StrandsTelemetry()
        telemetry.setup_otlp_exporter()
        telemetry.setup_meter(enable_otlp_exporter=True)

        guard = CostGuard(config=config)

Persistence Setup (optional):
    For budget state that survives restarts, configure a Valkey store:

        import valkey
        from strands_costguard.persistence import ValkeyBudgetStore

        client = valkey.Valkey(host="localhost", port=6379)
        store = ValkeyBudgetStore(client)

        config = CostGuardConfig(
            policy_source=FilePolicySource(path="./policies"),
            budget_store=store,
        )
        guard = CostGuard(config=config)
"""

from strands_costguard.core.config import CostGuardConfig
from strands_costguard.core.cost_guard import CostGuard
from strands_costguard.core.decisions import (
    AdmissionDecision,
    IterationDecision,
    ModelDecision,
    ToolDecision,
)
from strands_costguard.core.usage import IterationUsage, ModelUsage, ToolUsage
from strands_costguard.policies.budget import BudgetScope, BudgetSpec
from strands_costguard.policies.routing import RoutingPolicy, StageConfig
from strands_costguard.policies.store import FilePolicySource, PolicyStore
from strands_costguard.pricing.table import ModelPricing, PricingTable
from strands_costguard.routing.router import ModelRouter

__version__ = "0.1.0"

__all__ = [
    # Core
    "CostGuard",
    "CostGuardConfig",
    # Decisions
    "AdmissionDecision",
    "IterationDecision",
    "ModelDecision",
    "ToolDecision",
    # Usage
    "IterationUsage",
    "ModelUsage",
    "ToolUsage",
    # Policies
    "BudgetSpec",
    "BudgetScope",
    "RoutingPolicy",
    "StageConfig",
    "PolicyStore",
    "FilePolicySource",
    # Pricing
    "PricingTable",
    "ModelPricing",
    # Routing
    "ModelRouter",
]

# Optional persistence exports (requires valkey extra)
try:
    from strands_costguard.persistence import ValkeyBudgetStore  # noqa: F401

    __all__.append("ValkeyBudgetStore")
except ImportError:
    pass  # valkey not installed
