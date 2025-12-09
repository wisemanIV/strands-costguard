"""
Strand Cost Guard - Budget enforcement and adaptive routing for Strands-based multi-agent systems.

This library provides:
- Budget enforcement at tenant, strand, workflow, and run levels
- Adaptive model routing based on cost policies
- OpenTelemetry-compatible metrics emission via StrandsTelemetry
- Integration with Strands runtime lifecycle hooks

Metrics Setup:
    Cost Guard uses the global MeterProvider configured by StrandsTelemetry.
    Configure telemetry before creating a CostGuard instance:

        from strands.telemetry.config import StrandsTelemetry

        telemetry = StrandsTelemetry()
        telemetry.setup_otlp_exporter()
        telemetry.setup_meter(enable_otlp_exporter=True)

        guard = CostGuard(config=config)
"""

from strand_cost_guard.core.cost_guard import CostGuard
from strand_cost_guard.core.config import CostGuardConfig
from strand_cost_guard.core.decisions import (
    AdmissionDecision,
    IterationDecision,
    ModelDecision,
    ToolDecision,
)
from strand_cost_guard.core.usage import IterationUsage, ModelUsage, ToolUsage
from strand_cost_guard.policies.budget import BudgetSpec, BudgetScope
from strand_cost_guard.policies.routing import RoutingPolicy, StageConfig
from strand_cost_guard.policies.store import PolicyStore, FilePolicySource
from strand_cost_guard.pricing.table import PricingTable, ModelPricing
from strand_cost_guard.routing.router import ModelRouter

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
